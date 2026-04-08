# TurboQuant Fused Kernel Suite — Blog Notes

_All work done on AMD Instinct MI355X (gfx950), Tensorwave cluster, ROCm 7.0.1_
_Triton kernels targeting 256 CUs, 288 GB HBM per GPU_
_Date: March 27, 2026_

## Background

TurboQuant is a sub-4-bit KV cache compression scheme that uses:
- **MSE quantization:** Rotate K/V via orthogonal matrix Pi, quantize to 2-bit (4-level) codebook
- **QJL (Quantized Johnson-Lindenstrauss):** 1-bit sign of residual for error correction
- **Asymmetric attention:** Score = Q @ K_mse^T + Q_proj @ K_qjl^T (two-term dot product)

The original PyTorch implementation uses 8+ kernel launches per compression and standard bmm for attention. We fused everything into optimized Triton kernels for AMD MI355X.

---

## Part 1: Fused Compression Kernels (Steps A & B)

### The Problem
Original PyTorch K compression: 3 GEMMs + 5 element-wise ops = 8 kernel launches.
Profiling showed argmin codebook lookup = 58% of time at large N.

### Optimization Journey

**Step 1: Replace argmin with boundary comparison**
- For 2-bit (4 levels), `(x >= bd0) + (x >= bd1) + (x >= bd2)` gives the bin index
- Equivalent to searchsorted but works natively in Triton
- 3-14x faster for the quantization step alone

**Step 2: Rotated-domain storage (key architectural insight)**
- Original: compress in rotated space → inverse-rotate for storage → re-rotate at decode
- Insight: store K_mse in rotated space permanently. Q gets rotated once at decode (free for Sq=1).
- Saves 1 full [N,D]@[D,D] GEMM per token during compression
- Precompute PiST = Pi @ S^T — combines inverse rotation + QJL projection into single constant matrix

**Step 3: Fuse everything into one Triton kernel**
- All 6 post-rotation ops in single kernel: quantize + codebook gather + K_mse + residual + PiST GEMM via tl.dot + sign extraction + norm computation
- 2 kernel launches total: 1 rocBLAS rotation + 1 Triton post-processing

**Step 4: Fully fused single-kernel (zero rocBLAS)**
- Fuse rotation GEMM into Triton via second tl.dot — 2× [BN,D]@[D,D] per tile
- Challenge: 2× tl.dot causes register spill at w=8 (D×D=16K fp16 values × 2 = 32K regs)
- Solution: w=4 or w=2 for double-dot kernels

### Tuning on MI355X

**Sweep:** 4 variants × BN={32,64,128,256} × w={2,4,8} × s={1,2}

Key findings:
- BN=256 w=4 s=2 optimal at large N — amortizes 2× [D,D] matrix loads across 256 vectors
- w=2 optimal at medium N — less register pressure with 2× tl.dot
- Matrix preloading (loading both [D,D] matrices before computation) was SLOWER — preloading 2×16KB forces register spill for the BN×D tiles
- rocBLAS vs Triton crossover at N~100K-200K for [N,D]@[D,D] shapes

**Production dispatch strategy:**
| N range | Best approach | Config |
|---------|---------------|--------|
| ≤2K | 1-kernel (Triton) | BN=32, w=4, s=1 |
| 2K-32K | 1-kernel (Triton) | BN=64, w=2, s=2 |
| 32K-100K | 2-kernel (rocBLAS+Triton) | BN=128, w=8, s=2 |
| >200K | 1-kernel (Triton) | BN=256, w=4, s=2 |

### Results: K Compression

| Vectors | Original PyTorch (8 kernels) | Fused+Tuned | Speedup |
|---------|-----|------|---------|
| 2K | 0.117ms | 0.050ms | **2.3x** |
| 32K | 0.250ms | 0.059ms | **4.2x** |
| 131K | 0.925ms | 0.098ms | **9.4x** |
| 524K | 3.601ms | 0.271ms | **13.3x** |

### Results: V Compression
Similar approach but simpler (no QJL signs needed). 9.0-12.6x speedup.

### Failed Experiment: CUDA Stream Pipelining
Tried chunking the rotation GEMM across multiple streams and overlapping with Triton kernels.
Result: 0.36-0.68x SLOWER. Both kernels saturate all 256 CUs independently.
Chunked rocBLAS is much less efficient than single large call. Stream/event overhead adds ~0.1-0.2ms.
Lesson: on MI355X with 256 CUs, sequential execution of CU-saturating kernels beats pipelining.

---

## Part 2: Fused Asymmetric Attention (Step C)

### The Problem
TurboQuant attention is not standard: `score = Q @ K_mse^T + Q_proj @ (signs * r_norm * corr_scale)^T`
PyTorch does this as 3 separate bmm calls + element-wise ops + another bmm for V.

### Split-K Architecture (Critical for MI355X)

**Without split-K:** Grid = (BH,). For BH=8 heads, only 8 CUs active out of 256 = 3% utilization.
- Result: 5.14x at Sk=256, but **0.40x at Sk=16384** (slower than PyTorch!)

**With split-K:** Grid = (BH, num_splits). Each head's KV tokens split across multiple programs.
- BH=8 Sk=16384: **0.40x → 4.43x** (11.4x improvement)
- BH=8 Sk=32768: **6.93x**
- BH=64 Sk=16384: **8.09x**

This is the same idea as Flash-Decoding but applied to asymmetric attention.

### Concat Optimization (v2)
Precompute K_qjl = corr_scale * signs * r_norm, concat [K_mse | K_qjl] into [Sk, 2D=256], do single dot product over 2D.
- Wins 1.13-1.16x at Sk≤4096 (compute-bound: fewer reductions)
- Loses 0.85-0.99x at Sk≥8192 (bandwidth-bound: 2× wider memory footprint)

### Production Results (unified auto-dispatch wrapper)

| BH | Sk | Speedup vs PyTorch |
|----|-------|---------|
| 8 | 256 | **1.87x** |
| 8 | 32768 | **4.91x** |
| 32 | 16384 | **6.02x** |
| 64 | 16384 | **6.38x** |
| 64 | 32768 | **6.35x** |

---

## Part 3: Packed KV Cache (Steps A', B', C')

### The Idea
Instead of storing K_mse/V_mse as fp16 tensors (512 bytes/token/head), pack everything:
- 2-bit MSE indices → 4 per byte (32 bytes for D=128)
- 1-bit signs → 8 per byte (16 bytes for D=128)
- Norms stay as fp16 scalars (2 bytes each)

**Result: 512B → 86B per token/head = 6.0x compression**

At 131K context × 128 heads: 8.59GB → 1.44GB (saves 7.15GB of HBM)

### Packed Compression (A' & B')
Modified the fused compression kernels to output packed format directly.
Bit manipulation in Triton: shift + OR for packing, shift + AND + mask for unpacking.

Challenge: Triton doesn't have native bit-pack ops. Used scratch buffer for intermediate shifted values, then OR-reduced groups of 4 (for 2-bit) or 8 (for 1-bit).

### Packed Attention (C') — The Main Challenge

The kernel must:
1. Load packed uint8 bytes from HBM
2. Unpack 2-bit indices via bit shifts in registers
3. Reconstruct K/V via codebook lookup
4. Compute asymmetric attention scores
5. Online softmax + V accumulation

**Initial results:** 0.40-0.79x at most configs — **SLOWER than unpacked!**
The unpack overhead (bit shifts, masks, tl.where codebook lookup) ate the bandwidth savings.

**Tuning breakthrough:** BLOCK_SK=32 + num_splits=32 + num_warps=8
- Smaller tiles = less unpack work per program
- More splits = better CU saturation despite per-tile overhead
- Turned 0.40x slowdowns into 1.3-1.78x wins

| BH | Sk | Before tuning | After tuning |
|----|-------|------|------|
| 8 | 4096 | 0.53x ❌ | **1.29x** ✅ |
| 8 | 16384 | 0.48x ❌ | **1.36x** ✅ |
| 8 | 32768 | 0.40x ❌ | **1.35x** ✅ |
| 32 | 16384 | 0.74x ❌ | **1.68x** ✅ |
| 64 | 16384 | 0.71x ❌ | **1.78x** ✅ |

### E2E Packed Pipeline Benchmark

Full pipeline: raw K,V → compress (packed Triton) → attend (packed Triton)
vs: raw K,V → compress (PyTorch) → attend (PyTorch)

| BH | Sk | Packed total | Reference total | E2E Speedup |
|----|-------|-------------|----------------|-------------|
| 8 | 32768 | 1.41ms | 2.62ms | **1.86x** |
| 32 | 32768 | 5.33ms | 10.29ms | **1.93x** |
| 64 | 32768 | 10.84ms | 20.48ms | **1.89x** |

Correctness: cos > 0.99 across all configs.

---

## Part 4: Further Optimization (v3)

### Codebook Gather (v3a) — Verified Win
Replaced the `tl.where` chain:
```python
# Before (v2): 3 comparisons + 3 selects per element
k_recon = tl.where(k_idx==0, cb0,
          tl.where(k_idx==1, cb1,
          tl.where(k_idx==2, cb2, cb3)))

# After (v3a): single gather from 4-element codebook in HBM
k_recon = tl.load(Cb_ptr + k_idx)
```

Results (vs PyTorch ref):
| BH | Sk | v2 baseline | v3a gather | Improvement |
|----|------|------|------|-------|
| 8 | 4096 | 1.37x | **1.84x** | +34% |
| 32 | 4096 | 1.55x | **2.59x** | +67% |
| 64 | 4096 | 1.55x | **2.10x** | +35% |
| 8 | 32768 | 1.21x | **1.43x** | +18% |
| 32 | 32768 | 1.77x | **1.86x** | +5% |

Correctness: cos=1.000000 (identical to v2) ✅

### Chunked Dot Product (v3b) — Fixed & Tuned ✅
Process 4 sub-positions of each packed byte independently as [BS, DQ] chunks.
Avoids materializing full [BS, D] tensor in registers.

**Bug found & fixed:** Q chunks loaded sequentially [0..31, 32..63, ...] but packed bytes
store interleaved dims [4j, 4j+1, 4j+2, 4j+3]. Fix: stride-4 Q loading (`rdq*4 + offset`).
After fix: cos=1.000000 across all configs ✅

**Full tuning sweep results (MI355X):**

| BH | Sk | v3b tuned | Best Config |
|----|-------|------|------|
| 8 | 4096 | **1.88x** | BS=128 sp=16 w=8 |
| 8 | 16384 | **2.84x** | BS=32 sp=64 w=8 |
| 8 | 32768 | **2.43x** | BS=32 sp=64 w=8 |
| 32 | 4096 | **3.13x** | BS=16 sp=64 w=4 |
| 32 | 16384 | **2.73x** | BS=16 sp=64 w=2 |
| 32 | 32768 | **2.78x** | BS=16 sp=64 w=2 |
| 64 | 4096 | **3.03x** | BS=16 sp=32 w=2 |
| 64 | 16384 | **2.83x** | BS=16 sp=32 w=2 |
| 64 | 32768 | **2.75x** | BS=16 sp=128 w=2 |

**Key tuning insight:** BS=16 w=2 dominates at high BH — smallest tile + fewest warps = more
occupancy. At high BH, heads already provide enough parallelism to fill 256 CUs. More warps
would just increase register pressure and reduce occupancy.

---

## Complete Kernel Suite Summary

| Step | Kernel | Speedup | Memory |
|------|--------|---------|--------|
| A | K compression (prefill) | 9.4-13.3x vs PyTorch | — |
| B | V compression (prefill) | 9.0-12.6x vs PyTorch | — |
| C | Asymmetric attention (decode) | 1.87-6.38x vs PyTorch | 642 B/tok/head |
| D | V decompression (decode) | N/A — read directly | — |
| A' | K compress → packed | 9-13x vs PyTorch | 52 B/tok/head |
| B' | V compress → packed | 9-13x vs PyTorch | 34 B/tok/head |
| C' | Packed attention (decode) | **1.88-3.13x** (v3b tuned) | 86 B/tok/head |
| **E2E** | **Compress + Attend** | **1.4-1.9x** (pre-v3b) | **6.0x compression** |

## Key Lessons for AMD MI355X Kernel Development

1. **Split-K is essential for decode attention:** Without it, low head counts (BH=8) use <5% of 256 CUs.
2. **Register pressure from tl.dot:** Each [BN,D]@[D,D] dot uses ~16K fp16 registers. 2 dots → occupancy cliff at w=8. Use w=4 or w=2.
3. **rocBLAS vs Triton crossover:** rocBLAS wins at N~100K-200K for [N,D]@[D,D]. Below/above, Triton wins (launch overhead saved or amortization wins).
4. **Stream pipelining doesn't help on MI355X:** When both kernels saturate 256 CUs, sequential beats pipelining. Chunked rocBLAS is much less efficient.
5. **Matrix preloading hurts:** Loading both [D,D] matrices upfront forces register spill for BN×D tiles. Load each just before use.
6. **Codebook gather >> tl.where chain:** For 4-level codebook, `tl.load(Cb_ptr + idx)` is 30-67% faster than nested tl.where (eliminates 3 comparisons + 3 selects per element).
7. **Smaller BLOCK_SK for packed attention:** Unpack overhead per tile is fixed. Smaller tiles = less total unpack work per program, and more programs = better CU saturation.
8. **BS=16 w=2 at high BH:** When BH provides enough grid parallelism (≥32 heads), use fewest warps (w=2) to maximize occupancy. More warps = more registers = fewer wavefronts.
9. **Tuning sweep filter bugs:** Aggressive config filters (like `nsplits * BS < Sk // 4`) can exclude the best configs. Always sweep the full space or at least include the known-good configs.

## Git History (chronological)

```
50f8557  Initial TurboQuant Triton kernel (2.3x)
ae561fe  Fuse norms into GEMM (3.5x)
5a0db68  Warp/stage tuning (4.0x)
6d0df3e  Fused asymmetric attention (no split-K)
cb327ac  Split-K attention (up to 8x)
031b23e  Concat optimization (v2)
aeb7d41  Unified wrapper (auto-dispatch)
925e554  Production API (cached Q_proj)
60afddf  Fused compression kernel
54d033c  Tuned compression (BN=128 w=8)
cc8eefa  Tuned single-kernel (BN=256 w=4)
4434816  Fully fused single-kernel (zero rocBLAS)
01790a4  V compression kernel
3f6df4e  Packed KV cache compression
e2a40fa  Tuned packed compression
858cfe8  Packed attention C' (tuned)
90144d9  E2E packed benchmark
c2b789a  v3 optimizations (codebook gather + chunked dot)
97a7189  Fix v3b/v3c correctness (stride-4 Q loading)
07de724  Production v3b with auto-dispatch tuning table
```

## Files

All in `turboquant_fused/` directory (28 Python scripts).
Key production files:
- `fused_asymmetric_unified.py` — attention (decode)
- `fused_compress_single_kernel.py` — K compression
- `fused_v_compress.py` — V compression
- `packed_compress.py` — packed compression
- `packed_attention.py` — packed attention (v2)
- `packed_attn_v3_opt.py` — v3 optimizations
- `e2e_packed_benchmark.py` — E2E benchmark
