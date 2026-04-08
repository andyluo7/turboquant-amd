"""TurboQuant: Near-optimal KV cache quantization for LLM serving.

Based on the paper "TurboQuant: Online Vector Quantization for KV Cache
Quantization" (arXiv: 2504.19874, ICLR 2026).

Provides fused Triton kernels for AMD Instinct MI355X/MI300X GPUs:
  - turbo3: 3-bit PolarQuant + QJL correction (4.6x compression)
  - turbo4: 4-bit PolarQuant nibble packing (3.8x compression, best quality)
"""

__version__ = "0.1.0"
