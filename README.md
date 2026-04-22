# TurboQuant

**Near-optimal KV cache quantization for LLM serving on AMD GPUs**

[![Paper](https://img.shields.io/badge/arXiv-2504.19874-b31b1b.svg)](https://arxiv.org/abs/2504.19874)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

TurboQuant compresses the KV cache of large language models to 2–4 bits per element with near-optimal distortion, enabling **2× KV cache capacity** and **71% higher throughput** at high concurrency on AMD Instinct GPUs.

Based on the paper [*TurboQuant: Online Vector Quantization for KV Cache Quantization*](https://arxiv.org/abs/2504.19874) (ICLR 2026).

---

## Key Results

| | turbo4 (4-bit) | turbo3 (3-bit) | turbo2 (2-bit) |
|---|---|---|---|
| **Compression** | 3.8× vs FP16 | 4.6× vs FP16 | 5.7× vs FP16 |
| **PPL vs q8_0** | +0.23% | +1.06% | +2.8% |
| **GSM8K** | 95% (= FP8) | 95% (= FP8) | — |
| **Recommended for** | Quality-first | Balanced | Max compression |

### E2E Serving — MiniMax-M2.5, TP=2, MI355X

At **ISL=31K, concurrency=64** (the workloads that matter for production):

| Metric | TurboQuant | Vanilla FP8 | Improvement |
|--------|-----------|-------------|-------------|
| Output throughput | **71.60 tok/s** | 41.92 tok/s | **+71%** |
| Mean TTFT | **33.5s** | 965.2s | **29× faster** |
| KV cache capacity | **2.59M tokens** | 1.29M tokens | **2.0×** |
| Failed requests | **0/640** | 1/640 | — |

> TurboQuant's 2× KV capacity means more requests run simultaneously instead of queueing. At high concurrency, vanilla FP8 throughput collapses while TQ holds steady.

---

## How It Works

```
Input K/V (FP16) ──→ WHT Rotate ──→ PolarQuant ──→ Pack ──→ Compressed KV
                      │                 │              │
                      │ Gaussianize     │ Lloyd-Max    │ 2-4 bit indices
                      │ coordinates     │ quantization │ + norms + signs
                      │                 │              │
                      ▼                 ▼              ▼
                   Orthogonal        Near-optimal    4.6× smaller
                   (self-inverse)    distortion      (turbo3)
```

### Architecture

1. **WHT Rotation**: The Walsh-Hadamard Transform maps structured vectors into ones with approximately i.i.d. Gaussian coordinates. This is the key insight — Gaussian coordinates enable near-optimal *scalar* quantization.

2. **PolarQuant**: Decomposes each rotated vector into norm (scalar) + direction (quantized per-coordinate via Lloyd-Max codebook). Achieves distortion within ~2.7× of the information-theoretic lower bound.

3. **Sign Extraction** (turbo2/3): Stores 1-bit QJL correction signs for unbiased inner product estimation. Not needed for turbo4 (16 centroids give sufficient quality).

4. **Bit Packing**: Packs indices into bytes with stride-4 interleaving for efficient GPU access patterns.

### Asymmetric Attention

For keys, TurboQuant computes attention scores *directly* from compressed data:

```
<q, k> ≈ <q, k_mse> + ||r_k|| × √(π/2)/m × <S@q, sign(S@r_k)>
```

The MSE term uses centroid reconstruction; the QJL correction provides an **unbiased** inner product estimate. For values, MSE-only reconstruction suffices (errors average out in the weighted sum).

### Weight Fusion

WHT rotation cost can be **completely eliminated** by fusing the rotation matrix into model weights at load time:

```python
from turboquant.core.rotation import apply_weight_fusion

state_dict = apply_weight_fusion(state_dict, num_heads=64, num_kv_heads=8, head_size=128)
# Rotation cost: 68μs/layer → 0μs/layer
# Cosine similarity: 1.000000 (mathematically exact)
```

---

## GPU Kernel Suite

Seven fused Triton kernels handle the full pipeline on AMD Instinct GPUs:

| Kernel | File | Function |
|--------|------|----------|
| **A** | `kernels/compress_k.py` | Fused K compression (norm + rotate + quantize + QJL) |
| **B** | `kernels/compress_v.py` | Fused V compression (norm + rotate + quantize) |
| **C** | `kernels/attention.py` | Asymmetric attention decode (split-K) |
| **A'** | `kernels/packed_compress.py` | Packed K+V compression (bit-packed output) |
| **B'** | `kernels/packed_attention.py` | Packed attention v7 (production decode kernel) |
| **C'** | `kernels/sparse_v.py` | Sparse-V v12 decode (skip near-zero weights) |

### Performance

| Kernel | Time/Layer | Platform |
|--------|-----------|----------|
| TQ turbo3 v12 (Sparse-V) | 843 μs | MI355X |
| TQ turbo4 | 797 μs | MI355X |
| AITER FP8 PA (target) | 48 μs | MI355X |

---

## Platform Support

| GPU | Architecture | Status |
|-----|-------------|--------|
| AMD Instinct MI355X | gfx950 | ✅ Primary target |
| AMD Instinct MI300X | gfx942 | ✅ Tested |
| ROCm | 7.0+ | Required |

---

## Installation

```bash
# From source
git clone https://github.com/andyluo7/turboquant-amd.git
cd turboquant-amd
pip install -e ".[dev]"

# Or directly
pip install git+https://github.com/andyluo7/turboquant-amd.git
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.4.0 (with ROCm support)
- Triton ≥ 3.0.0
- AMD Instinct GPU (MI300X or MI355X)
- ROCm ≥ 7.0

---

## Quick Start

### Codebook + Rotation

```python
from turboquant.core import make_codebook, make_wht_matrix, PolarQuantConfig

# Create optimal codebook for 3-bit quantization
centroids, boundaries = make_codebook(d=128, bits=3, device="cuda")

# WHT rotation matrix
H = make_wht_matrix(128, device="cuda")  # H @ H = I (self-inverse)

# Configure compression
config = PolarQuantConfig(head_dim=128, bits=3, use_qjl=True)
print(f"Compression ratio: {config.compression_ratio:.1f}x")  # 4.6x
```

### Compress KV Cache

```python
import torch
from turboquant.core import polarquant_compress

# Simulate KV cache entries
K = torch.randn(1024, 128, device="cuda")

# Compress
compressed = polarquant_compress(K, centroids, boundaries, config, rotation_matrix=H)
# compressed["indices"]: [1024, 128] uint8
# compressed["norms"]: [1024] float
# compressed["signs"]: [1024, 128] ±1
# compressed["k_mse_rot"]: [1024, 128] MSE reconstruction
```

### Run Tests

```bash
pytest tests/ -v
```

### Run Benchmarks

```bash
# Kernel-level benchmarks (requires GPU)
python benchmarks/bench_kernels.py

# E2E serving benchmark (requires vLLM)
python benchmarks/bench_e2e.py --tq-port 8000 --vanilla-port 8001
```

---

## Integration

### vLLM

TurboQuant integrates with vLLM as a custom KV cache backend:

```bash
# Launch with TurboQuant KV cache
python -m vllm.entrypoints.openai.api_server \
    --model MiniMax-Text-01 \
    --tensor-parallel-size 2 \
    --kv-cache-dtype turboquant
```

The turbo4→FP8 pipeline compresses KV entries via PolarQuant and stores them as standard FP8, enabling use of AITER's optimized FP8 paged attention for decode.

### llama.cpp

Experimental turbo4 support via custom GGML quantization types. See [`turboquant/integration/llamacpp.py`](turboquant/integration/llamacpp.py) for details.

Results on MI300X:
- Prefill: **+4% vs f16**
- Decode: 89% of f16
- PPL: +0.23% vs q8_0

---

## Quality

| Benchmark | TurboQuant | FP8 | Notes |
|-----------|-----------|-----|-------|
| GSM8K 5-shot | **95%** | 95% | MiniMax-M2.5, TP=2, MI355X |
| WHT roundtrip | 2.98e-07 max err | — | Numerically exact |
| Weight fusion | cosine = 1.0 | — | Mathematically exact |
| turbo4 PPL | +0.23% vs q8_0 | — | Negligible |
| turbo3 PPL | +1.06% vs q8_0 | — | Very good |

---

## Repository Structure

```
turboquant-amd/
├── turboquant/
│   ├── core/                    # Codebook, rotation, quantizer
│   ├── kernels/                 # Fused Triton GPU kernels
│   ├── integration/             # vLLM + llama.cpp backends
│   └── reference/               # PyTorch reference implementation
├── benchmarks/                  # Kernel and E2E benchmarks
├── tests/                       # Unit tests
└── docs/                        # Architecture, kernel history, results
```


---

## Roadmap & Contributing

We welcome contributions! Here are the key areas where help is needed:

### 🔴 High Priority — Kernel Performance

| Task | Description | Difficulty | Impact |
|------|-------------|------------|--------|
| **AITER turbo4 PA kernel** | Fork AITER's HIP paged attention to read turbo4 (4-bit nibble) KV directly in the attention inner loop. Target: match AITER FP8 speed (~48μs/layer). See [design doc](docs/design/AITER_TURBO4_DESIGN.md). | Hard | 10x decode speedup |
| **Native FP4 MFMA on gfx950** — *further perf tuning needed* | 🟡 Correctness ✅, perf optimization 🚧. AITER PA fork uses `mfma_scale_f32_16x16x128_f8f6f4` (mixed FP8×FP4, scale=1.0) on MI355X with ROCm 7.2.0. Native path now always-on (`TQ_FP4_NATIVE_THRESHOLD=0`); shuffle reorientation bug fixed (cosine ≥ 0.995 at S=256..65536). **Native is 1.07× → 1.26× faster than the FP4→FP8 LUT path across S=4K..64K, no crossover.** **Storage: 2× KV capacity (verified, 0.5 B/elem).** Roofline: only **13.5% of HBM peak BW** at S=64K — kernel is memory-bound with significant headroom. **Remaining work:** (1) BLOCK_SIZE 16→32/64 to widen contiguous fetches, (2) profile/fuse the partial+reduce two-stage kernel at large `npar`, (3) async LDS prefetch to hide HBM latency, (4) per-block scaling for full MXFP4, (5) end-to-end vLLM model bench. See [design doc](docs/design/MXFP4_PA_KERNEL.md). | Hard | 2× KV bytes; ≥1.26× over LUT today, ~3× headroom to BW peak |
| **Triton codegen optimization** | Current Triton kernels achieve only 3% HBM bandwidth utilization on ROCm (vs AITER's 61%). Profile and optimize the generated HIP assembly. | Hard | 3-10x kernel speedup |

### 🟡 Medium Priority — Quality & Features

| Task | Description | Difficulty | Impact |
|------|-------------|------------|--------|
| ~~**Boundary layer protection**~~ | ✅ Done. `protect_boundary_layers=True` (default). Configurable via `num_protected_layers` (default 2). | Easy | Quality improvement |
| **Outlier-aware mixed precision** | Per-layer channel variance calibration → high-variance channels get more bits (paper Section 4.3). See [Hyperloom reference](https://github.com/AMD-AGI/Hyperloom/tree/main/training_optimization/turboquant). | Medium | Quality improvement |
| ~~**Asymmetric K/V compression**~~ | ✅ Done. `v_bits` parameter for independent V bit width. E.g., `k_bits=4, v_bits=2` for turbo4-K + turbo2-V. | Easy | Better compression ratio |
| **Perplexity benchmark suite** | Add wikitext-2 / wikitext-103 PPL evaluation for all turbo configs. Currently only have GSM8K accuracy. | Easy | Quality validation |
| **More model support** | Add `set_dflash_layers_to_capture` equivalent for DeepSeek-V3, Llama-4, Gemma-4 in vLLM. | Easy | Broader adoption |

### 🟢 Low Priority — Integration & Ecosystem

| Task | Description | Difficulty | Impact |
|------|-------------|------------|--------|
| **SGLang integration** | Port the vLLM backend to SGLang's attention framework. | Medium | New serving engine |
| **llama.cpp turbo4 Metal kernel** | Register LUT decode for Apple Silicon (M-series). TheTom's fork already has turbo3. | Medium | Apple Silicon support |
| **ONNX export** | Export TQ-compressed models to ONNX with custom ops. | Medium | Cross-platform |
| **Quantization-aware training** | Fine-tune with TQ compression in the loop for better quality. | Hard | Quality at lower bits |
| **Dynamic bit allocation** | Per-layer bit-width selection based on attention entropy. | Medium | Adaptive compression |

### 🔧 Infrastructure

| Task | Description | Difficulty |
|------|-------------|------------|
| **CI/CD pipeline** | GitHub Actions with MI300X self-hosted runner for kernel tests | Medium |
| **Benchmark dashboard** | Automated throughput/quality tracking across commits | Medium |
| **pip installable** | Make `pip install turboquant-amd` work cleanly | Easy |
| **Documentation** | API docs, tutorials, integration guides | Easy |

### Known Issues

| Issue | Description | Workaround |
|-------|-------------|------------|
| Triton ROCm codegen | Only 3% HBM bandwidth utilization vs AITER's 61% | Use turbo4→FP8 pipeline for production |
| `mfma_scale` zeros (resolved) | E8M0 scale byte = `0x00` is `2^-127 ≈ 0` (not 1.0) — the apparent "zeros bug" was a scale-encoding error | Pass `0x7F` (= 1.0) when no scaling is desired |
| Sparse-V v12 memory faults | Intermittent GPU memory access faults at ISL=31K conc=64 on MI355X | Root cause: Triton codegen + GPU-pair sensitivity |
| AITER broken in SGLang Docker | Public SGLang ROCm images have broken AITER imports | `pip install -e .` inside container |
| vLLM 50B compact KV NCCL crash | Non-power-of-2 slot size causes vectorized load overflow | Fixed: use 64B padded slots |

### How to Contribute

1. **Fork** the repo and create a feature branch
2. **Test** on MI300X or MI355X with ROCm 7.0+
3. **Benchmark** with the provided scripts in `benchmarks/`
4. **Submit** a PR with test results

For questions, open an issue or reach out to [@andyluo7](https://github.com/andyluo7).

---

## Citation

```bibtex
@inproceedings{turboquant2026,
    title={TurboQuant: Online Vector Quantization for KV Cache Quantization},
    year={2026},
    booktitle={International Conference on Learning Representations (ICLR)},
    url={https://arxiv.org/abs/2504.19874}
}
```

## References

- [TurboQuant paper](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [vLLM](https://github.com/vllm-project/vllm) — LLM serving engine
- [AITER](https://github.com/ROCm/aiter) — AMD inference engine with FP8 paged attention
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — C/C++ LLM inference

## License

MIT — see [LICENSE](LICENSE).
