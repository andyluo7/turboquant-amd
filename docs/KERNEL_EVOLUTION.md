# Kernel Optimization History

## Timeline

Development on AMD Instinct MI355X (gfx950), ROCm 7.0.

### Phase 1: Basic Triton Kernels (March 2026)

| Version | Approach | Speedup vs PyTorch | Correctness |
|---------|----------|-------------------|-------------|
| v1 | Per-column loop | 0.2x (5x slower) | 100% |
| v2 | Tiled GEMM + separate norms | 1.7x | 60% (race condition) |
| v3 | Split norms kernel + GEMM | 2.3x | 100% |
| v3b | Fused norms + GEMM + quantize | 3.5x | 100% |
| **v3 final** | **+ warp/stage tuning** | **4.0x** | **100%** |

**Key learnings:**
- bf16 MFMA dot products: 2.3x speedup but 99.7% correctness (boundary mismatches). Rejected.
- BM=256 tiles: reduced occupancy on MI355X. BM=32 optimal.
- Fusing norms into GEMM tiles eliminates inter-tile synchronization.

### Phase 2: Asymmetric Attention (March 2026)

| Variant | Architecture | Performance |
|---------|-------------|-------------|
| v1 (attn) | Basic split-K | Baseline |
| v2 (splitk) | Concat kernel for short seqs | 1.3x short seq |
| unified | Adaptive dispatch (v1 long, v2 short) | Best of both |

### Phase 3: Packed Format + Production Kernels (March-April 2026)

| Version | Key Optimization | Impact |
|---------|-----------------|--------|
| v3b | Chunked K dot (4× DQ partials) | 2x register efficiency |
| v3b | Codebook gather via tl.load | Eliminated tl.where chain |
| v5b | Pre-permuted sign storage | Aligned stride-4 access |
| v6f | Interleaved memory loads | Better prefetch overlap |
| **v7** | **Precomputed Q_proj×corr_scale** | **-4 scalar muls/tile/token** |

### Phase 4: Paged Attention Integration (March-April 2026)

| Version | Feature | Notes |
|---------|---------|-------|
| v1 (paged) | Basic TQ paged attention | Working but slow |
| v4d | Split-K + block management | Production-grade |
| v8-v11 | Incremental improvements | Better memory handling |
| **v12** | **Sparse-V dequant** | Skip V where attn < 1e-6 |
| v13 | WHT-fused attention | Experimental |

### Phase 5: turbo4 Format (April 2026)

| Component | Status | Notes |
|-----------|--------|-------|
| turbo4 compress | ✅ | 4-bit nibble PolarQuant |
| turbo4→FP8 pipeline | ✅ | Compress as turbo4, store as FP8 |
| Weight fusion | ✅ | Zero rotation cost, cosine=1.0 |
| Compact KV 64B | ✅ | 2.01x FP8 capacity |
| AITER FP8 PA integration | 🔨 | Target: 48μs/layer |

## Rejected Approaches

1. **bf16 tl.dot for rotation**: Faster but 0.3% boundary mismatches
2. **BM=256 tiles**: Poor MI355X occupancy
3. **QJL for V compression**: Unnecessary — MSE sufficient for weighted sums
4. **Unfused multi-kernel pipeline**: 3+ kernel launches → memory bandwidth wasted
5. **v13 WHT-fused attention**: Higher complexity, marginal gain over v12

## Architecture Decisions

1. **Split-K for decode**: Grid = (num_heads, num_splits) — better CU utilization
2. **Auto-dispatch tuning**: Table of (BLOCK_SK, num_splits, warps) per problem size
3. **Pre-permuted signs**: Rearrange during compression for stride-4 decode access
4. **Sparse-V threshold**: 1e-6 balances skipping vs quality — validated on GSM8K
