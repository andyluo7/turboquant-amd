# TurboQuant Architecture

## Overview

TurboQuant compresses KV cache entries from FP16 (16 bits/element) to 2-4 bits/element
using online vector quantization with near-optimal distortion guarantees.

```
┌─────────────────────────────────────────────────────────────┐
│                    TurboQuant Pipeline                       │
│                                                             │
│  Input K/V ──→ WHT Rotate ──→ Normalize ──→ PolarQuant     │
│     [FP16]      [FP16]        [FP16]       [2-4 bit]       │
│                                                             │
│  PolarQuant outputs:                                        │
│    • indices [bits × D] — quantization bin                  │
│    • norm [FP16] — per-vector scale                         │
│    • signs [D bits] — QJL correction (turbo2/3 only)        │
│                                                             │
│  Decode: asymmetric attention directly on compressed data   │
│    <q, k> ≈ <q, k_mse> + ||r_k|| × √(π/2)/m × <Sq, s_k>  │
└─────────────────────────────────────────────────────────────┘
```

## Compression Formats

### turbo3 (3-bit, 4.6x compression)
- 8 Lloyd-Max centroids per coordinate
- 3-bit indices + 1-bit QJL signs + FP16 norm
- Storage: `3×D + D + 16 = 4D + 16` bits per vector (D=128 → 528 bits)
- vs FP16: `16×128 = 2048` bits → **3.9x raw, 4.6x with packing**
- QJL correction provides unbiased inner product estimation

### turbo4 (4-bit, 3.8x compression)
- 16 Lloyd-Max centroids per coordinate
- 4-bit nibble packing (2 indices per byte, no byte-spanning)
- No QJL signs needed — 16 centroids give sufficient quality
- Storage: `4×D + 16` bits per vector (D=128 → 528 bits)
- **Best quality**: +0.23% PPL vs q8_0

### turbo2 (2-bit, 5.7x compression)
- 4 centroids per coordinate
- 2-bit indices + 1-bit QJL signs
- Maximum compression, slightly lower quality

## WHT Rotation

The Walsh-Hadamard Transform (WHT) is an orthogonal rotation that transforms
structured vectors into ones with approximately i.i.d. Gaussian coordinates.

Key properties:
- **Orthogonal**: preserves inner products and norms
- **Self-inverse**: H × H = I (same matrix for forward and inverse)
- **O(n log n)** computation via butterfly structure
- **Gaussianization**: enables near-optimal scalar quantization

### Weight Fusion

The rotation cost can be **completely eliminated** by fusing H into the model weights:

```python
# Before: explicit rotation at every token
Q_rot = (hidden @ W_q.T) @ H          # rotation = 68μs/layer

# After: fused weights (one-time preprocessing)
W_q_fused = fuse(W_q, H)
Q_rot = hidden @ W_q_fused.T          # rotation = 0μs/layer
```

Verified: cosine similarity = 1.000000 between fused and explicit.

## PolarQuant

PolarQuant decomposes each vector using polar coordinates:

1. **Norm** (scalar): `||x_rot||` — captures magnitude
2. **Direction**: quantized independently per coordinate via Lloyd-Max

The Lloyd-Max codebook is optimal for Gaussian sources, and WHT rotation
ensures coordinates are approximately Gaussian. This gives distortion within
~2.7× of the information-theoretic lower bound.

## Asymmetric Attention

For K (keys), TurboQuant uses an **asymmetric** inner product estimator:

```
<q, k> ≈ <q, k_mse> + ||r_k|| × √(π/2)/m × <S@q, sign(S@r_k)>
           ─────────   ────────────────────────────────────────
           MSE term              QJL correction term
```

The MSE term uses centroid-based reconstruction. The QJL correction uses
the Johnson-Lindenstrauss projection of the residual to provide an
**unbiased** estimate of the true inner product.

For V (values), MSE-only reconstruction suffices because the weighted sum
in `softmax(scores) @ V` averages out per-vector errors.

## GPU Kernel Suite

Seven fused Triton kernels handle the full pipeline:

| Kernel | Function | Input | Output |
|--------|----------|-------|--------|
| **A** (compress_k) | K compression | FP16 K | indices + norms + signs |
| **B** (compress_v) | V compression | FP16 V | indices + norms |
| **C** (attention) | Asymmetric decode | compressed K/V + Q | attention output |
| **A'** (packed_compress) | Packed K+V | FP16 K, V | bit-packed bytes |
| **B'** (packed_attention) | Packed decode | packed bytes + Q | attention output |
| **C'** (sparse_v) | Sparse-V decode | packed + Q | output (skip zero weights) |

### Kernel A: Fused K Compression
Single kernel launch fuses: normalize → rotate (tl.dot) → quantize → gather → QJL project (tl.dot) → sign extract.
Two `tl.dot` matrix multiplies per tile. Zero intermediate tensors.

### Kernel B': Packed Attention v7
Production decode kernel with:
- Chunked dot product: 4× [BS, DQ] partial sums
- Pre-permuted sign storage for stride-4 access
- Interleaved memory loads for prefetch efficiency
- Precomputed `Q_proj × corr_scale`
- Auto-dispatch tuning table per (BH, Sk)

### Kernel C': Sparse-V v12
Skips V decompression where attention weight < 1e-6.
Reduces compute on long sequences where most V entries get negligible attention.
