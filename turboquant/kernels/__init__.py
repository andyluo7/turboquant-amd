"""Fused Triton GPU kernels for TurboQuant on AMD Instinct GPUs.

Kernel suite:
  A  (compress_k):       Fused K compression — norm + rotate + quantize + QJL
  B  (compress_v):       Fused V compression — norm + rotate + quantize (MSE only)
  C  (attention):        Asymmetric attention decode — split-K architecture
  A' (packed_compress):  Packed K+V compression — bit-packed output
  B' (packed_attention): Packed attention v7 — reads packed format directly
  C' (sparse_v):         Sparse-V v12 decode — skips near-zero attention weights

All kernels target AMD Instinct MI355X (gfx950) and MI300X (gfx942).
"""
