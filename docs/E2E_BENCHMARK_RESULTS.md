# TurboQuant E2E Benchmark Results — 2026-03-29

## Setup
- **Hardware:** AMD Instinct MI355X (gfx950), TP=2
- **Model:** MiniMax-M2.5 (MoE, 456B params)
- **Baseline:** Vanilla vLLM v0.18.0-rocm (p01-g07)
- **TurboQuant:** vLLM v0.18.0 + TQ KV cache compression (p01-g05)
- **Benchmark:** InferenceX benchmark_serving.py (random dataset, ignore-eos)
- **GPU memory:** 0.95 utilization, block-size=32

## Throughput Results

### 1k/1k (ISL=1024, OSL=1024)

| Concurrency | Baseline Out tok/s | TQ Perf Out tok/s | TQ/Base Ratio | Base TPOT (ms) | TQ TPOT (ms) |
|-------------|-------------------|-------------------|---------------|----------------|--------------|
| 4 | 262.9 | 226.9 | 0.86x | 12.42 | 16.93 |
| 8 | 545.6 | 380.1 | 0.70x | 14.24 | 20.41 |
| 16 | 930.1 | 590.2 | 0.63x | 16.68 | 26.22 |
| 32 | 1485.6 | 699.7 | 0.47x | 20.82 | 44.28 |
| 64 | 2122.2 | 802.2 | 0.38x | 29.10 | 77.37 |

### 8k/1k (ISL=8192, OSL=1024)

| Concurrency | Baseline Out tok/s | TQ Perf Out tok/s | TQ/Base Ratio | Base TPOT (ms) | TQ TPOT (ms) |
|-------------|-------------------|-------------------|---------------|----------------|--------------|
| 4 | 271.9 | 152.4 | 0.56x | 13.89 | 23.97 |
| 8 | 438.9 | 218.4 | 0.50x | 17.31 | 34.12 |
| 16 | 637.0 | 252.3 | 0.40x | 23.67 | 59.61 |
| 32 | 884.9 | 0.0 (FAILED) | — | 34.25 | — |
| 64 | 1087.5 | 0.0 (FAILED) | — | 56.17 | — |

### Detailed Baseline Metrics

**1k/1k:**
| Conc | Out tok/s | Total tok/s | TPOT (ms) | TTFT (ms) | ITL (ms) |
|------|-----------|-------------|-----------|-----------|----------|
| 4 | 262.9 | 528.5 | 12.42 | 2199.3 | 12.42 |
| 8 | 545.6 | 1087.2 | 14.24 | 109.8 | 14.25 |
| 16 | 930.1 | 1870.0 | 16.68 | 139.7 | 16.69 |
| 32 | 1485.6 | 2966.6 | 20.82 | 182.8 | 20.82 |
| 64 | 2122.2 | 4245.4 | 29.10 | 275.9 | 29.11 |

**8k/1k:**
| Conc | Out tok/s | Total tok/s | TPOT (ms) | TTFT (ms) | ITL (ms) |
|------|-----------|-------------|-----------|-----------|----------|
| 4 | 271.9 | 2444.4 | 13.89 | 443.1 | 13.89 |
| 8 | 438.9 | 3904.4 | 17.31 | 574.4 | 17.34 |
| 16 | 637.0 | 5748.9 | 23.67 | 724.3 | 23.69 |
| 32 | 884.9 | 7913.3 | 34.25 | 1052.3 | 34.30 |
| 64 | 1087.5 | 9802.2 | 56.17 | 1677.8 | 56.25 |

## GSM8K Accuracy (5-shot)

| Config | Accuracy | Delta vs Baseline |
|--------|----------|------------------|
| Baseline (vanilla) | **92.6%** | — |
| TQ Perf mode | **60.7%** | -31.9 pp |
| TQ Quality mode | **62.0%** | -30.6 pp |

## Analysis

### Throughput
- TQ perf mode is **14-62% slower** than vanilla vLLM across all concurrency levels
- The gap **widens with concurrency**: 0.86x at conc=4 → 0.38x at conc=64 (1k/1k)
- At 8k ISL, TQ fails entirely at conc≥32 (0 completed requests)
- Root cause: Triton attention kernel with 3-bit codebook dequant (21 ALU ops/element) is too expensive vs AITER native PagedAttention

### Accuracy
- 3-bit MSE quantization loses ~31 percentage points on GSM8K
- Quality mode (exact reconstruction) provides minimal improvement (+1.3pp over perf mode)
- Both TQ modes are unacceptable for math/reasoning tasks

### Why TQ Underperforms at E2E Level
1. **Kernel overhead > memory savings:** The 6x KV cache compression saves memory, but the Triton attention kernel is 1.9-7x slower than AITER PA
2. **MoE model sensitivity:** MiniMax-M2.5's MoE architecture amplifies quantization errors across expert routing
3. **No prefill benefit:** TQ only compresses during decode; prefill uses the same AITER kernels as baseline
4. **Scaling breakdown:** At high concurrency, the per-token compute overhead compounds while memory savings plateau

### TQ Quality Throughput (1k/1k ISL=1024, OSL=1024, TP=2)

| Concurrency | TQ Quality Out tok/s | Quality TPOT (ms) | Qual/Base Ratio |
|-------------|---------------------|-------------------|-----------------|
| 4 | 83.9 | 45.88 | 0.32x |
| 8 | 157.4 | 49.37 | 0.29x |
| 16 | 261.5 | 59.37 | 0.28x |
| 32 | 376.4 | 82.35 | 0.25x |
| 64 | 369.1 | 120.91 | 0.17x |

Note: TQ Quality 8k/1k failed to start server both times (RuntimeError: Engine core initialization failed).
Quality mode throughput is **2-2.5x slower** than perf mode due to exact reconstruction overhead.

### Full Three-Way Comparison (1k/1k)

| Conc | Baseline tok/s | TQ Perf tok/s | TQ Quality tok/s | Perf/Base | Qual/Base |
|------|---------------|---------------|------------------|-----------|-----------|
| 4 | 262.9 | 226.9 | 83.9 | 0.86x | 0.32x |
| 8 | 545.6 | 380.1 | 157.4 | 0.70x | 0.29x |
| 16 | 930.1 | 590.2 | 261.5 | 0.63x | 0.28x |
| 32 | 1485.6 | 699.7 | 376.4 | 0.47x | 0.25x |
| 64 | 2122.2 | 802.2 | 369.1 | 0.38x | 0.17x |

## Raw Result Files
- Baseline 1k/1k: `/workspace/results_1k1k/baseline_1k1k_tp2_conc{4,8,16,32,64}.json`
- Baseline 8k/1k: `/workspace/results_8k1k/baseline_8k1k_tp2_conc{4,8,16,32,64}.json`
- Baseline GSM8K: `/workspace/results_1k1k/gsm8k_baseline/`
- TQ Perf 1k/1k: `/tmp/tq_perf_results_1k1k/tq_perf_1k1k_tp2_conc{4,8,16,32,64}.json`
- TQ Perf 8k/1k: `/tmp/tq_perf_results_8k1k/tq_perf_8k1k_tp2_conc{4,8,16,32,64}.json`
- TQ GSM8K: `/tmp/tq_{perf,quality}_results_1k1k/gsm8k_tq_{perf,quality}/`
