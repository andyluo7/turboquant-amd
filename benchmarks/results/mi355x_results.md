# TurboQuant Benchmark Results — AMD MI355X

**Date:** April 2026
**Hardware:** 2x AMD Instinct MI355X (gfx950), 288 GB VRAM each
**Model:** MiniMax-M2.5, TP=2
**Framework:** vLLM 0.18.0, ROCm 7.0

## Kernel-Level Performance

| Kernel | Time/Layer | Notes |
|--------|-----------|-------|
| TQ turbo3 v12 (Triton) | 843 μs | Sparse-V decode, 3-bit |
| TQ turbo4 (Triton) | 797 μs | 4-bit nibble, best quality |
| AITER FP8 PA | 48 μs | Baseline for Phase 2 |

## FP4 Paged Attention — 2× KV capacity at FP8 latency (April 2026, updated post-fix)

**Hardware:** AMD Instinct MI355X (gfx950), ROCm 7.2.x
**Config:** head_size=128, block_size=16, 100 iterations
**Goal:** Match AITER FP8 PA latency while halving KV cache bytes (2× concurrent capacity).
**Implementation:** AITER fork with native `mfma_scale_f32_16x16x128_f8f6f4` (mixed FP8×FP4) plus a fallback FP4→FP8 LUT decode path. After fixing the cross-lane shuffle reorientation in the native path (cosine 0.52 → 0.995) and re-benching, native beats LUT 1.07–1.26× across all measured S. **Dispatch threshold now 0 (always-native);** the LUT branch is kept as a safety net.

> Earlier numbers in this doc were collected with a buggy native path (incorrect MFMA→AITER lane reorientation produced cosine ≈ 0.52 at large S). The fix is in `pa_kernels_fp4.cuh` (commit `1da39d2`); all numbers below are post-fix.

### FP4 (this work, 2× capacity) vs original AITER FP8 baseline

| B | NH | NKV | SEQ    | FP8 (μs) | FP4 (μs) | ratio | result |
|---|----|-----|--------|----------|----------|-------|--------|
| 1 | 32 | 8 |  4,096 | 32.6 | 31.2 | 0.96× | ✅ 4% faster |
| 4 | 32 | 8 |  4,096 | 32.7 | 31.0 | 0.95× | ✅ 5% faster |
| 1 | 32 | 8 | 16,384 | 33.1 | 31.0 | 0.94× | ✅ 6% faster |
| 4 | 32 | 8 | 16,384 | 37.7 | 64.4 | 1.71× | ❌ 71% slower |
| 1 | 32 | 8 | 65,536 | 45.9 | 68.9 | 1.50× | ❌ 50% slower |
| 4 | 32 | 8 | 65,536 | 131.2 | 222.3 | 1.69× | ❌ 69% slower |
| 8 | 32 | 8 |  4,096 | 33.5 | 73.7 | 2.20× | ❌ 120% slower |
| 1 | 16 | 2 |  4,096 |  33.1 |  31.4 | 0.95× | ✅ 5% faster |

**Headline:** **2× KV capacity at FP8 latency achieved on 4 of 8 shapes** — small batch with context ≤ 16K, plus the GQA-light 16/2 case. Long-context (B≥1, S≥16K) and large-batch (B=8) regress 50–120%.

### Why long-context and large-batch regress (and how to fix)

FP4 native is **memory-bound at only 13.5% of HBM peak** (roofline at S=64K: 0.715 TB/s of 5.3 TB/s). Two structural gaps vs AITER FP8:

1. **No async LDS prefetch** — FP8 path overlaps HBM fetches with compute via double-buffered LDS; our FP4 fork loads K/V synchronously, so HBM latency is on the critical path.
2. **block_size=16 fragments the fetch** — every 16 tokens incurs a `block_table` indirection, capping contiguous HBM transfers. Doubling/quadrupling block_size would widen each fetch.

Scaling comparison (FP4 vs FP8, B=1, NH=32, NKV=8):

| seq_len 4K → 64K (16×) | FP8 time | FP4 native time |
|------------------------|----------|-----------------|
| Scaling | 32.6 → 45.9 μs (+41%) | 31.2 → 68.9 μs (+121%) |

**Fix priority:** (1) BLOCK_SIZE 16 → 32/64, (2) async LDS prefetch in fetch loop, (3) fuse partial+reduce two-stage kernel at large `npar`. Each is a separate optimization commit.

## KV Cache Capacity

| Config | KV Cache Tokens | Compression |
|--------|----------------|-------------|
| TurboQuant v12 | 2,585,792 | 2.01x FP8 |
| Vanilla FP8 | 1,293,104 | baseline |
| **Compact KV 64B** | **2.01x FP8** | 64-byte blocks |
| AITER FP4 PA (this work) | — (E2E pending) | 2.0× FP8 (storage layout, verified) |

## E2E Serving — Long Context (ISL=31K, OSL=1024)

### Headline: Concurrency=64

| Metric | TQ v12 | Vanilla FP8 | Winner | Delta |
|--------|--------|-------------|--------|-------|
| Output tok/s | **71.60** | 41.92 | **TQ** | **+70.8%** |
| Mean TTFT | **33.50s** | 965.24s | **TQ** | **28.8x faster** |
| Median TTFT | **13.0s** | 1261.1s | **TQ** | **97x faster** |
| P99 TTFT | **395.07s** | 1690.58s | **TQ** | **4.3x faster** |
| Failed requests | **0/640** | 1/640 | **TQ** | — |

### All Concurrency Levels (ISL=31K)

| Conc | TQ Output tok/s | Vanilla Output tok/s | TQ TTFT (s) | Vanilla TTFT (s) |
|------|-----------------|---------------------|-------------|-----------------|
| 4 | 55.63 | **90.92** | 9.97 | **9.24** |
| 8 | 62.75 | **125.65** | 10.96 | **10.89** |
| 16 | 67.91 | **150.16** | 13.95 | **12.87** |
| 32 | 70.00 | **163.72** | 20.60 | **16.06** |
| **64** | **71.60** | 41.92 | **33.50** | 965.24 |

**Crossover:** Between conc=32 and conc=64. Vanilla throughput collapses 2.3x (163→42 tok/s) while TQ holds steady (70→72 tok/s).

## E2E Serving — Short Context (ISL=1K, OSL=1024)

| Conc | TQ Output tok/s | Vanilla Output tok/s | Delta |
|------|-----------------|---------------------|-------|
| 4 | 190.67 | **241.53** | Vanilla +27% |
| 16 | 523.25 | **721.56** | Vanilla +38% |
| 64 | 1017.78 | **1756.63** | Vanilla +73% |

Vanilla wins at short context — no KV cache pressure, TQ dequant overhead hurts.

## Quality

| Benchmark | TurboQuant | FP8 Baseline | Delta |
|-----------|-----------|--------------|-------|
| GSM8K 5-shot (MiniMax-M2.5, TP=2) | **95%** | **95%** | 0% |
| turbo4 PPL vs q8_0 | — | — | +0.23% |
| turbo3 PPL vs q8_0 | — | — | +1.06% |
| WHT roundtrip max error | 2.98e-07 | — | — |
| Weight fusion cosine | 1.000000 | — | — |

## llama.cpp Results (MI300X)

| Metric | turbo4 | f16 | Delta |
|--------|--------|-----|-------|
| Prefill tok/s | **+4%** | baseline | TQ wins |
| Decode tok/s | 89% | baseline | f16 wins |
| PPL | +0.23% | baseline | negligible |
