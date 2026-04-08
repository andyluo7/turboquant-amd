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
| AITER FP8 PA | 48 μs | Target for Phase 2 |

## KV Cache Capacity

| Config | KV Cache Tokens | Compression |
|--------|----------------|-------------|
| TurboQuant v12 | 2,585,792 | 2.01x FP8 |
| Vanilla FP8 | 1,293,104 | baseline |
| **Compact KV 64B** | **2.01x FP8** | 64-byte blocks |

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
