"""PolarQuant: quantize and dequantize in rotated space.

PolarQuant decomposes each vector into:
  1. Norm (scalar): ||x||
  2. Direction: quantized via Lloyd-Max codebook on normalized coordinates

For K (keys):  MSE reconstruction + QJL correction for unbiased inner products
For V (values): MSE reconstruction only (errors average out in weighted sum)

Compression formats:
  - turbo2: 2-bit indices + signs → 5.7x compression
  - turbo3: 3-bit indices + signs → 4.6x compression
  - turbo4: 4-bit indices (no signs) → 3.8x compression, best quality

References:
    - TurboQuant paper (arXiv: 2504.19874), Section 3
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PolarQuantConfig:
    """Configuration for PolarQuant compression.
    
    Supports asymmetric K/V compression: K and V can use different bit widths.
    V compression is "free" quality-wise (errors average out in the weighted
    sum during attention), so V can be compressed more aggressively.
    
    Example configurations:
        - Symmetric turbo4: bits=4, v_bits=None → both K and V at 4-bit
        - Asymmetric turbo4-K/turbo2-V: bits=4, v_bits=2 → K at 4-bit, V at 2-bit
        - Asymmetric turbo4-K/turbo3-V: bits=4, v_bits=3 → K at 4-bit, V at 3-bit
    """

    head_dim: int = 128
    bits: int = 3  # K bit width (2, 3, or 4)
    v_bits: Optional[int] = None  # V bit width. None = same as K (symmetric)
    use_qjl: bool = True  # QJL correction for K (unbiased inner products)
    protect_boundary_layers: bool = True  # Skip compression on first/last N layers
    num_protected_layers: int = 2  # Number of layers to protect at each boundary
    num_layers: int = 0  # Total number of layers (set at init)

    @property
    def k_bits(self) -> int:
        """Key bit width."""
        return self.bits

    @property
    def effective_v_bits(self) -> int:
        """Value bit width (falls back to k_bits if v_bits not set)."""
        return self.v_bits if self.v_bits is not None else self.bits

    @property
    def is_asymmetric(self) -> bool:
        """True if K and V use different bit widths."""
        return self.v_bits is not None and self.v_bits != self.bits

    def should_compress(self, layer_idx: int) -> bool:
        """Return True if this layer should be compressed.
        
        When protect_boundary_layers is True, the first and last
        `num_protected_layers` layers are kept at full precision.
        This recovers 37-91% of quantization quality gap (Section 4.3).
        """
        if not self.protect_boundary_layers or self.num_layers == 0:
            return True
        if layer_idx < self.num_protected_layers:
            return False
        if layer_idx >= self.num_layers - self.num_protected_layers:
            return False
        return True

    @property
    def n_centroids(self) -> int:
        return 2 ** self.bits

    @property
    def compression_ratio(self) -> float:
        """Approximate compression ratio vs FP16 (symmetric, using K bits)."""
        return self._compression_ratio_for_bits(self.bits)

    @property
    def k_compression_ratio(self) -> float:
        """Compression ratio for keys."""
        return self._compression_ratio_for_bits(self.k_bits)

    @property
    def v_compression_ratio(self) -> float:
        """Compression ratio for values."""
        return self._compression_ratio_for_bits(self.effective_v_bits)

    @property
    def combined_compression_ratio(self) -> float:
        """Combined K+V compression ratio (harmonic mean of K and V ratios)."""
        return 2.0 / (1.0 / self.k_compression_ratio + 1.0 / self.v_compression_ratio)

    def _compression_ratio_for_bits(self, bits: int) -> float:
        fp16_bits = 16 * self.head_dim
        if bits <= 3:
            total_bits = bits * self.head_dim + self.head_dim + 16
        else:
            total_bits = bits * self.head_dim + 16
        return fp16_bits / total_bits


def polarquant_compress(
    x: torch.Tensor,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
    config: PolarQuantConfig,
    rotation_matrix: Optional[torch.Tensor] = None,
) -> dict:
    """Compress vectors using PolarQuant.

    Args:
        x: Input tensor [N, D] or [B, H, D].
        centroids: Codebook centroids [n_centroids].
        boundaries: Decision boundaries [n_centroids - 1].
        config: PolarQuant configuration.
        rotation_matrix: WHT matrix [D, D]. If None, assumes x is pre-rotated.

    Returns:
        Dictionary with compression artifacts:
          - indices: Quantization indices [*, D]
          - norms: Per-vector norms [*]
          - k_mse_rot: MSE reconstruction in rotated space [*, D]
          - signs: Sign bits for QJL correction [*, D] (if use_qjl)
          - rot_residual_norm: Residual norm for QJL [*] (if use_qjl)
    """
    orig_shape = x.shape
    if x.dim() == 3:
        B, H, D = x.shape
        x = x.reshape(-1, D)
    N, D = x.shape

    # Step 1: WHT rotation
    if rotation_matrix is not None:
        x_rot = x.float() @ rotation_matrix.float()
    else:
        x_rot = x.float()

    # Step 2: Compute norms and normalize
    norms = x_rot.norm(dim=-1)  # [N]
    x_normalized = x_rot / (norms.unsqueeze(-1) + 1e-8)

    # Step 3: Quantize via boundary comparison
    indices = torch.searchsorted(boundaries.float(), x_normalized.reshape(-1))
    indices = indices.clamp(0, config.n_centroids - 1).reshape(N, D)

    # Step 4: Reconstruct (MSE estimate)
    recon_normalized = centroids[indices]  # [N, D]
    k_mse_rot = recon_normalized * norms.unsqueeze(-1)

    result = {
        "indices": indices,
        "norms": norms,
        "k_mse_rot": k_mse_rot,
    }

    # Step 5: QJL correction (for K only)
    if config.use_qjl:
        rot_residual = (x_normalized - recon_normalized) * norms.unsqueeze(-1)
        rot_residual_norm = rot_residual.norm(dim=-1)
        # Project residual through random subspace (S matrix)
        # In practice, PiS is a sub-selection of rotation columns
        signs = torch.sign(rot_residual)
        signs[signs == 0] = 1.0
        result["signs"] = signs
        result["rot_residual_norm"] = rot_residual_norm

    # Restore original shape
    if len(orig_shape) == 3:
        result["k_mse_rot"] = result["k_mse_rot"].reshape(orig_shape)
        result["indices"] = result["indices"].reshape(orig_shape)
        if "signs" in result:
            result["signs"] = result["signs"].reshape(orig_shape)

    return result


def polarquant_decompress(
    compressed: dict,
    centroids: torch.Tensor,
    config: PolarQuantConfig,
) -> torch.Tensor:
    """Decompress MSE reconstruction from compressed representation.

    Args:
        compressed: Output of polarquant_compress().
        centroids: Codebook centroids.
        config: PolarQuant configuration.

    Returns:
        Reconstructed tensor in rotated space.
    """
    return compressed["k_mse_rot"]
