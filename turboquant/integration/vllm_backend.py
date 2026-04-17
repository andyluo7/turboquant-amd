"""TurboQuant FP8 Backend — turbo4 compression stored as FP8 KV cache.

Prefill: K/V → WHT rotate → PolarQuant → centroid×norm → FP8 cast → store
Decode:  Q → Q×PiT (rotate) → standard FP8 PA → out×Pi (rotate back)

This gives AITER FP8 decode speed (~48μs/layer) with TQ compression quality.
No custom attention kernel needed — uses standard FP8 paged attention.
"""
import math
import torch
import torch.nn.functional as F
from typing import Optional
from scipy.stats import norm as scipy_norm


def make_wht_matrix(d: int, device: torch.device) -> torch.Tensor:
    """Walsh-Hadamard Transform matrix (normalized, orthogonal)."""
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < d:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0) / math.sqrt(2)
    return H[:d, :d]


def make_turbo4_centroids(d: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimal 16 centroids for WHT-rotated vectors.
    
    Returns: (centroids[16], boundaries[15])
    """
    sigma = 1.0 / math.sqrt(d)
    n = 16
    centroids = []
    for i in range(n):
        lo = scipy_norm.ppf(max(i / n, 1e-10)) * sigma
        hi = scipy_norm.ppf(min((i + 1) / n, 1 - 1e-10)) * sigma
        num = sigma * (scipy_norm.pdf(lo / sigma) - scipy_norm.pdf(hi / sigma))
        den = scipy_norm.cdf(hi / sigma) - scipy_norm.cdf(lo / sigma)
        centroids.append(num / max(den, 1e-10))
    boundaries = [(centroids[i] + centroids[i + 1]) / 2 for i in range(n - 1)]
    return (torch.tensor(centroids, dtype=torch.float32),
            torch.tensor(boundaries, dtype=torch.float32))


class TurboQuantFP8State:
    """Per-layer state for turbo4→FP8 compression."""
    
    def __init__(self, head_size: int, layer_idx: int, device: torch.device):
        self.head_size = head_size
        self.layer_idx = layer_idx
        self.device = device
        
        # WHT rotation matrix (orthogonal, its own inverse)
        self.wht = make_wht_matrix(head_size, device).half()
        self.PiT = self.wht.T  # pre-rotation for Q
        self.Pi = self.wht     # post-rotation for output
        
        # Centroids and boundaries
        centroids, boundaries = make_turbo4_centroids(head_size)
        self.centroids = centroids.to(device)
        self.boundaries = boundaries.to(device)
    
    def compress_to_fp8(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress K or V tensor to FP8 via turbo4 pipeline.
        
        Args:
            x: [T, H, D] fp16/bf16 — input K or V
            
        Returns:
            fp8_data: [T, H, D] fp8_e4m3 — compressed values in rotated space
            scales: [T, H, 1] float — per-position FP8 scales
        """
        T, H, D = x.shape
        
        # Step 1: WHT rotation
        x_rot = torch.matmul(x.float(), self.wht.float())  # [T, H, D]
        
        # Step 2: Per-position norm
        norms = x_rot.norm(dim=-1, keepdim=True)  # [T, H, 1]
        x_normalized = x_rot / (norms + 1e-8)  # [T, H, D]
        
        # Step 3: Quantize to 16 centroids
        indices = torch.searchsorted(self.boundaries, x_normalized.reshape(-1))
        indices = indices.clamp(0, 15).reshape(T, H, D)
        
        # Step 4: Dequantize in rotated space
        dequantized = self.centroids[indices] * norms  # [T, H, D]
        
        # Step 5: Convert to FP8 E4M3 with per-position scale
        abs_max = dequantized.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        fp8_max = 240.0  # max representable value in FP8 E4M3
        scales = abs_max / fp8_max  # [T, H, 1]
        
        scaled = dequantized / scales
        # Cast to FP8
        fp8_data = scaled.to(torch.float8_e4m3fnuz)
        
        return fp8_data, scales
    
    def rotate_query(self, q: torch.Tensor) -> torch.Tensor:
        """Pre-rotate Q for attention in rotated space.
        
        Args:
            q: [B, num_heads, D] — raw query
        Returns:
            q_rot: [B, num_heads, D] — rotated query
        """
        return torch.matmul(q.float(), self.PiT.float()).to(q.dtype)
    
    def rotate_output(self, out: torch.Tensor) -> torch.Tensor:
        """Post-rotate attention output back to original space.
        
        Args:
            out: [B, num_heads, D] — attention output in rotated space
        Returns:
            result: [B, num_heads, D] — output in original space
        """
        return torch.matmul(out.float(), self.Pi.float()).to(out.dtype)


def turbo4_fp8_compress_and_scatter(
    x: torch.Tensor,           # [T, H, D] input K or V
    kv_cache: torch.Tensor,    # [num_blocks, block_size, H, D] FP8
    slot_mapping: torch.Tensor, # [T] int32
    kv_idx: int,               # 0=K, 1=V (for separate K/V caches)
    state: Optional[TurboQuantFP8State],
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    block_size: int = 16,
):
    """Compress K/V via turbo4 and scatter-write as FP8 to paged cache.
    
    This replaces the standard KV cache update for turbo4→FP8 mode.
    The KV cache is standard FP8 layout — no custom format needed.
    
    If state is None (boundary layer protection), writes standard FP8
    without TQ compression.
    """
    T, H, D = x.shape
    
    # Boundary layer protection: skip TQ, use standard FP8
    if state is None:
        abs_max = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        fp8_max = 240.0
        scales = abs_max / fp8_max
        fp8_data = (x.float() / scales).to(torch.float8_e4m3fnuz)
        # Fall through to scatter below
    else:
        # Turbo4 compression → FP8
        fp8_data, scales = state.compress_to_fp8(x)
    
    # Update the per-tensor/per-head scale for AITER PA
    # AITER uses a single k_scale/v_scale for the entire cache
    # We need to track the max scale across all positions
    if kv_idx == 0 and k_scale is not None:
        max_scale = scales.max().item()
        if k_scale.item() < max_scale:
            k_scale.fill_(max_scale)
    elif kv_idx == 1 and v_scale is not None:
        max_scale = scales.max().item()
        if v_scale.item() < max_scale:
            v_scale.fill_(max_scale)
    
    # Scatter to paged cache
    safe_slots = torch.clamp(slot_mapping, min=0)
    block_idx = safe_slots // block_size
    block_off = safe_slots % block_size
    
    # Write FP8 data to cache
    # Cache layout: [num_blocks, block_size, H, D] for NHD format
    kv_cache[block_idx, block_off] = fp8_data.view(T, H, D)


# ============================================================================
# Integration with vLLM attention layer
# ============================================================================

class TurboQuantFP8Config:
    """Configuration for turbo4→FP8 backend.
    
    Args:
        head_size: Attention head dimension (default: 128)
        num_layers: Total number of transformer layers
        protect_boundary_layers: Skip TQ compression on first/last N layers,
            keeping them at full FP8 precision. Default True.
        num_protected_layers: How many layers to protect at each boundary.
            Default 2 (first 2 + last 2 = 4 layers at full precision).
    """
    
    def __init__(
        self,
        head_size: int = 128,
        num_layers: int = 0,
        protect_boundary_layers: bool = True,
        num_protected_layers: int = 2,
    ):
        self.head_size = head_size
        self.num_layers = num_layers
        self.protect_boundary_layers = protect_boundary_layers
        self.num_protected_layers = num_protected_layers
        self.layer_states: dict[int, TurboQuantFP8State] = {}
    
    def should_compress(self, layer_idx: int) -> bool:
        """Check if this layer should use TQ compression.
        
        When protect_boundary_layers is True, the first and last
        `num_protected_layers` layers skip TQ and use standard FP8.
        """
        if not self.protect_boundary_layers or self.num_layers == 0:
            return True
        if layer_idx < self.num_protected_layers:
            return False
        if layer_idx >= self.num_layers - self.num_protected_layers:
            return False
        return True
    
    def get_state(self, layer_idx: int, device: torch.device) -> Optional[TurboQuantFP8State]:
        """Get per-layer TQ state, or None if layer is protected."""
        if not self.should_compress(layer_idx):
            return None  # Protected layer — use standard FP8
        if layer_idx not in self.layer_states:
            self.layer_states[layer_idx] = TurboQuantFP8State(
                self.head_size, layer_idx, device)
        return self.layer_states[layer_idx]


# Singleton config
_tq_fp8_config: Optional[TurboQuantFP8Config] = None

def get_tq_fp8_config(
    head_size: int = 128,
    num_layers: int = 0,
    protect_boundary_layers: bool = True,
    num_protected_layers: int = 2,
) -> TurboQuantFP8Config:
    """Get or create TurboQuant FP8 config.
    
    Args:
        head_size: Attention head dimension
        num_layers: Total transformer layers (needed for boundary protection)
        protect_boundary_layers: Keep first/last N layers at full FP8 precision.
            Default True. Set to False for maximum compression.
        num_protected_layers: Layers to protect at each boundary (default 2).
            With 40 layers and N=2: layers 0,1,38,39 stay at full FP8.
    """
    global _tq_fp8_config
    if _tq_fp8_config is None:
        _tq_fp8_config = TurboQuantFP8Config(
            head_size=head_size,
            num_layers=num_layers,
            protect_boundary_layers=protect_boundary_layers,
            num_protected_layers=num_protected_layers,
        )
    return _tq_fp8_config
