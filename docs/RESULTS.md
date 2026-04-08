# TurboQuant Results

## Quality Benchmarks

### GSM8K (5-shot, MiniMax-M2.5, TP=2, MI355X)

| KV Cache | Accuracy | Delta vs FP8 |
|----------|----------|-------------|
| FP8 (vanilla) | 95% | — |
| **TurboQuant** | **95%** | **0%** |

### Perplexity (llama.cpp, various models)

| Format | PPL vs q8_0 | Notes |
|--------|-------------|-------|
| turbo4 (4-bit) | **+0.23%** | Best quality |
| turbo3 (3-bit) | +1.06% | Higher compression |

### Numerical Precision

| Test | Result |
|------|--------|
| WHT roundtrip max error | 2.98e-07 |
| Weight fusion cosine similarity | 1.000000 |
| turbo4 reconstruction cosine | >0.99 |

## Performance Benchmarks

### Kernel-Level (MI355X, per attention layer)

| Kernel | μs/layer | Notes |
|--------|----------|-------|
| TQ turbo3 v12 (Sparse-V) | 843 | Production Triton kernel |
| TQ turbo4 (Triton) | 797 | 4-bit nibble, fewer ALU ops |
| AITER FP8 PA | 48 | Target for AITER integration |

### KV Cache Capacity (MiniMax-M2.5, 2x MI355X, 288 GB each)

| Config | Tokens | vs FP8 |
|--------|--------|--------|
| Vanilla FP8 | 1,293,104 | 1.00x |
| **TQ Compact 64B** | **2,585,792** | **2.01x** |

### E2E Serving (ISL=31K, OSL=1024, vLLM 0.18.0)

#### The Headline: Concurrency = 64

| Metric | TurboQuant | Vanilla FP8 | Delta |
|--------|-----------|-------------|-------|
| Output throughput | **71.60 tok/s** | 41.92 tok/s | **+71%** |
| Mean TTFT | **33.5s** | 965.2s | **29x faster** |
| Median TTFT | **13.0s** | 1261.1s | **97x faster** |
| P99 TTFT | **395.1s** | 1690.6s | **4.3x faster** |
| Failed requests | **0/640** | 1/640 | — |

#### All Concurrency Levels

| Concurrency | TQ tok/s | Vanilla tok/s | Winner |
|-------------|----------|---------------|--------|
| 4 | 55.63 | **90.92** | Vanilla |
| 8 | 62.75 | **125.65** | Vanilla |
| 16 | 67.91 | **150.16** | Vanilla |
| 32 | 70.00 | **163.72** | Vanilla |
| **64** | **71.60** | 41.92 | **TQ 🏆** |

**Crossover point**: Between conc=32 and conc=64. Vanilla collapses 2.3x while TQ holds steady.

#### Short Context (ISL=1K)

| Concurrency | TQ tok/s | Vanilla tok/s | Delta |
|-------------|----------|---------------|-------|
| 4 | 190.67 | **241.53** | -21% |
| 16 | 523.25 | **721.56** | -27% |
| 64 | 1017.78 | **1756.63** | -42% |

Vanilla wins at short context (no memory pressure).

### llama.cpp (MI300X, gfx942)

| Metric | turbo4 vs f16 |
|--------|---------------|
| Prefill throughput | **+4%** (TQ wins) |
| Decode throughput | 89% of f16 |
| PPL degradation | +0.23% |

## Compression Ratios

| Format | Bits/Element | Compression vs FP16 | Quality |
|--------|-------------|---------------------|---------|
| turbo2 | 2 + signs | 5.7x | Good |
| **turbo3** | 3 + signs | **4.6x** | Very good |
| **turbo4** | 4 (nibble) | **3.8x** | **Excellent** |
| FP8 | 8 | 2.0x | Baseline |
| FP16 | 16 | 1.0x | Full precision |

## Summary

TurboQuant's value proposition:
- **At high concurrency + long context** (production workloads): 71% more throughput, 29x faster TTFT
- **Quality**: Matches FP8 on GSM8K, negligible PPL degradation
- **Capacity**: 2x more KV cache tokens in the same VRAM
- **Platform**: AMD MI355X and MI300X, ROCm 7.0+
