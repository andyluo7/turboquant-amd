"""llama.cpp integration for TurboQuant KV cache quantization.

Status: Experimental — turbo4 format implemented in llama.cpp fork.

Results on AMD MI300X (gfx942):
  - turbo4 prefill:  +4% vs f16 (faster due to reduced memory bandwidth)
  - turbo4 decode:   89% of f16 throughput
  - turbo4 PPL:      +0.23% vs q8_0 (negligible quality loss)
  - turbo3 PPL:      +1.06% vs q8_0

Architecture:
  llama.cpp stores KV cache in GGML quantization format. TurboQuant adds:
    - GGML_TYPE_TQ3: 3-bit PolarQuant + QJL signs (turbo3)
    - GGML_TYPE_TQ4: 4-bit PolarQuant nibble packing (turbo4)

  Compression path:
    K/V → WHT rotate → normalize → quantize → pack indices + norm

  Attention path:
    Asymmetric: scores use MSE reconstruction + QJL correction
    Symmetric: turbo4 uses centroid dequant → standard dot product

Integration points:
  - ggml-quants.c: TQ compress/decompress functions
  - llama-kv-cache.cpp: TQ cache type registration
  - llama-graph.cpp: TQ attention op dispatch

Build instructions:
  See the turboquant branch of the llama.cpp fork for build instructions.
  Requires ROCm 6.0+ or CUDA 12.0+.

TODO:
  - [ ] Upstream PR to ggerganov/llama.cpp
  - [ ] GGML op for fused TQ attention
  - [ ] Weight fusion support (embed rotation in model weights)
"""
