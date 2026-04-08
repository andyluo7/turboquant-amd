"""Lloyd-Max optimal codebook for PolarQuant.

Computes optimal quantization centroids and decision boundaries for
Gaussian-distributed coordinates (WHT-rotated vectors have approximately
i.i.d. Gaussian components with sigma = 1/sqrt(d)).

References:
    - TurboQuant paper (arXiv: 2504.19874), Section 3.1
    - Lloyd-Max quantizer: S.P. Lloyd, "Least squares quantization in PCM"
"""

import math
import torch
from typing import Optional


def solve_codebook(d: int, bits: int) -> tuple[list[float], list[float]]:
    """Compute optimal Lloyd-Max centroids and boundaries.

    Args:
        d: Head dimension (typically 128). Controls sigma = 1/sqrt(d).
        bits: Quantization bits (1-4). Produces 2^bits centroids.

    Returns:
        centroids: List of 2^bits centroid values.
        boundaries: List of 2^bits - 1 decision boundaries.
    """
    from scipy import integrate

    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)

    def pdf(x):
        return (1.0 / math.sqrt(2 * math.pi * sigma**2)) * math.exp(
            -x * x / (2 * sigma**2)
        )

    lo, hi = -6 * sigma, 6 * sigma

    # Uniform initialization
    boundaries = [lo + (hi - lo) * i / n_levels for i in range(1, n_levels)]
    centroids = [0.0] * n_levels

    for _ in range(100):  # Lloyd-Max iterations
        # Update centroids
        edges = [lo] + boundaries + [hi]
        for i in range(n_levels):
            num, _ = integrate.quad(lambda x: x * pdf(x), edges[i], edges[i + 1])
            den, _ = integrate.quad(pdf, edges[i], edges[i + 1])
            centroids[i] = num / max(den, 1e-30)
        # Update boundaries (midpoints)
        boundaries = [(centroids[i] + centroids[i + 1]) / 2 for i in range(n_levels - 1)]

    return centroids, boundaries


def make_codebook(
    d: int = 128,
    bits: int = 3,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create codebook tensors ready for GPU use.

    Args:
        d: Head dimension.
        bits: Quantization bits.
        device: Target device.
        dtype: Target dtype.

    Returns:
        centroids: Tensor of shape [2^bits].
        boundaries: Tensor of shape [2^bits - 1].
    """
    centroids, boundaries = solve_codebook(d, bits)
    return (
        torch.tensor(centroids, dtype=dtype, device=device),
        torch.tensor(boundaries, dtype=dtype, device=device),
    )
