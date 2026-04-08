# TurboQuant MI355X block_size fix + verification

## Root cause
- vLLM may allocate paged KV cache with `block_size=16`
- Older TurboQuant paged-attention wrappers could silently assume `block_size=32`
- That mismatch can make the kernel walk pages with the wrong token stride and trigger GPU memory faults at large context / high concurrency

## Workspace patches applied
- `turboquant_fused/turboquant_attn_current.py`
- `turboquant_fused/turboquant_attn_v4d.py`
- `turboquant_fused/tq_paged_attention.py`
- `turboquant_fused/tq_paged_attention_v12.py`

## Deploy targets in container
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/tq_paged_attention.py`

## Required behavior
1. Backend must compute `block_size = kv_cache.shape[2]`
2. Backend must pass that `block_size` into TQ paged attention wrappers
3. Wrapper must reject explicit mismatches against `kv_cache.shape[2]`

## Suggested verification ladder
1. Print/assert runtime `kv_cache.shape[2]`
2. Smoke test: ISL=1024, OSL=128, conc=1
3. Long-context smoke: ISL=31744, OSL=128, conc=1
4. Concurrency ramp: conc=4 -> 16 -> 32 -> 64 at ISL=31744, OSL=128
5. Full rerun: ISL=31744, OSL=1024, conc=64

## MI355X runtime notes
- Typical target: `tq-vllm` on `mi355x-p01-g05`
- Base image: `vllm/vllm-openai-rocm:v0.18.0`
- Direct node SSH requires Slurm allocation; use `k8s` hop + scheduled job
