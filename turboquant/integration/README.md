# TurboQuant Integration

## vLLM Integration

### Compact KV Allocator (`vllm/compact_kv_allocator.py`)
Patches vLLM's KV cache allocator to use 50-byte slots (48 packed 3-bit + 2 norm bytes)
instead of 128-byte slots. Achieves **2.56x more KV cache capacity**.

```bash
# Inside tq-vllm container
python3 compact_kv_allocator.py --apply    # Apply patch
python3 compact_kv_allocator.py --revert   # Revert
python3 compact_kv_allocator.py --check    # Check state
```

**Results:** 2.01x FP8 capacity, 63.10 tok/s, GSM8K 95%, no NCCL crash.

### turbo4→FP8 Pipeline (`vllm/turbo4_fp8_pipeline.py`)
Full pipeline: WHT weight fusion → turbo4 compress → FP8 cast → AITER FP8 PA.
Achieves AITER decode speed (~48μs/layer) with 2x KV capacity.

```bash
python3 turbo4_fp8_pipeline.py --apply   # Apply all patches
python3 turbo4_fp8_pipeline.py --revert  # Revert
```

## AITER Integration

### MXFP4 PA Kernel (`aiter/patch_pa_mxfp4.py`)
Patches AITER's paged attention kernel to support native FP4 KV cache on gfx950.
Uses `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` for hardware FP4 decode.

**Status:** WIP — `mfma_scale` returns zeros on ROCm 7.0.0, needs 7.0.1+.

### FP4→FP8 LUT (`aiter/fp4_fp8_lut.py`)
Byte lookup table mapping FP4 E2M1 nibbles to FP8 E4M3FN bytes.
Avoids floating-point conversion — pure integer table lookup.

**Status:** LUT works, blocked by per-position E8M0 scale handling.

### turbo4 MFMA Kernel Design (`aiter/optimize_mxfp4_kernel.py`)
4-bit nibble format with 16 centroids for AITER integration.

## llama.cpp Integration

See [llamacpp.py](llamacpp.py) for build instructions and test results on MI300X/MI355X.

**PRs:**
- [TheTom/llama-cpp-turboquant#61](https://github.com/TheTom/llama-cpp-turboquant/pull/61) — MI300X + MI355X test results
- [ggml-org/llama.cpp#21570](https://github.com/ggml-org/llama.cpp/pull/21570) — gfx950 CDNA4 support
