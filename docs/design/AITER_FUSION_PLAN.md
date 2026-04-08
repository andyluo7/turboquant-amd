# TurboQuant × AITER Fusion Plan

**Goal:** Fuse 3-bit TQ dequant directly into AITER's paged attention kernel to eliminate the FP16 shadow cache materialization bottleneck.

**Current state:** Dequant-then-PA achieves 1,549 tok/s (24% of vanilla's 6,450). Our Triton TQ kernel gets 2,061 tok/s (32%). Both are far from vanilla because attention is the bottleneck.

**Target:** ≥4,000 tok/s (>60% of vanilla) by reading 3-bit TQ directly in the attention inner loop.

---

## Architecture Overview

AITER has **two** PA implementations:
1. **AOT HIP kernel** (`pa_kernels.cuh`, 2275 lines) — compiled to `.co` binaries, dispatched via `torch.ops.aiter.paged_attention_v1`. This is the production path. Template params: `cache_t`, `KV_DTYPE`, `BLOCK_SIZE`, `HEAD_SIZE`, `GQA_RATIO`.
2. **Triton PA decode** (`pa_decode.py`, 1718 lines) — 8 kernel variants (v1/v2 × wo_dot/w_dot × normal/per_token_quant). This is the fallback/research path.

**Decision: Fork the AITER Triton PA decode kernel (option B).** Rationale:
- Triton is writable/debuggable (vs 2275 lines of HIP template metaprogramming)
- AITER's Triton PA is already tuned for MI355X (it's AMD's own code)
- No need to recompile `.co` binaries or modify AITER's build system
- We already have TQ Triton expertise from the v4d/v7 packed attention kernels
- Can prototype and iterate in hours, not days

---

## Plan: 4 Phases

### Phase 1: Fork AITER Triton PA with TQ 3-bit Loads (2-3 days)

**What:** Create `tq_aiter_pa_decode.py` by forking AITER's `_paged_attn_decode_v1_w_dot_kernel` and replacing the K/V load with inline 3-bit dequant.

**Key modification — replace this:**
```python
# AITER original: load FP16/BF16/FP8 from paged cache
k_0 = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
k = k_0.to(compute_type)
```

**With this:**
```python
# TQ: load 3-bit packed uint8, dequant inline via binary tree codebook
# TQ cache layout: [num_blocks, 2(K/V), block_size, num_kv_heads, slot_bytes]
# slot_bytes = 50 (48 packed indices + 2 norm bytes)
tq_base = kv_blk_nums * stride_tq_blk + 0 * stride_tq_kv + ...
packed = tl.load(tq_cache + tq_base + byte_offsets, mask=...) 
idx = extract_3bit(packed, bit_offsets)  # binary tree codebook lookup
norm = load_fp16_norm(tq_cache + tq_base + DQ3)
k = codebook_lookup(idx, centroids) * norm  # [KV_BLK_SZ, HEAD_SZ] float16
```

**Why this works:** The 3-bit dequant (21 ALU ops/element) happens in registers, overlapping with memory latency of the next block's load. No separate dequant pass, no FP16 shadow cache.

**Files:**
- Fork: `aiter/ops/triton/_triton_kernels/attention/pa_decode.py` → `vllm/v1/attention/ops/tq_aiter_pa_decode.py`
- Keep only `v1_w_dot` variant (MiniMax-M2.5 has GQA_RATIO=6, `w_dot` is faster)
- Add: `centroids`, `stride_tq_*`, `DQ3` as kernel parameters

**Key challenges:**
- TQ cache has different layout than standard paged cache (5D: `[blocks, 2, block_size, kv_heads, slot_bytes]` vs 4D: `[blocks, kv_heads, block_size, head_size]`)
- 3-bit extraction requires byte-pair loads + bit shifts (same as our existing packed_attention.py)
- Need to handle Q rotation (pre-rotate Q before calling kernel, same as current approach)

**Validation:** Correctness test vs PyTorch reference dequant → vanilla attention on random inputs.

### Phase 2: Add Q Rotation Fusion (1 day)

**What:** Fuse Q × PiT rotation into the kernel's Q loading phase.

Currently:
```python
Q_rot = torch.matmul(q.float(), PiT).half()  # separate matmul
kernel(Q_rot, tq_cache, ...)
```

Fused:
```python
# Inside kernel, after loading Q:
q = tl.load(q_ptr + q_offs, ...)
# Apply rotation: q_rot = q @ PiT (PiT is D×D = 128×128)
# Since D=128, this is a small matmul per query token
# For w_dot variant with QUERY_GRP_SZ tokens: [GQA, D] @ [D, D]
q_rot = tl.dot(q, PiT_block, ...)
```

**Note:** This is optional — under CUDAGraphs, Q rotation is ~free. But for completeness and to avoid the separate kernel launch.

### Phase 3: Integration with vLLM TQ Backend (1 day)

**What:** Wire the fused kernel into `turboquant_attn.py`'s `_decode_perf`:

```python
def _decode_perf(self, q, kv_cache, attn_metadata, state, ...):
    # Pre-rotate Q (or fuse if Phase 2 done)
    Q_rot = torch.matmul(q.float(), state.PiT).half()
    
    # Single fused kernel: reads 3-bit TQ cache, computes attention
    output = tq_aiter_paged_attention(
        Q_rot, kv_cache, 
        attn_metadata.block_table, attn_metadata.seq_lens,
        state.centroids, self.scale,
        num_kv_heads, head_size, gqa_ratio,
    )
    
    # Rotate output back
    result = torch.matmul(output.float(), state.Pi).to(query.dtype)
    return result
```

**CUDAGraph compatibility:** Same as current — no dynamic allocations, all shapes static.

### Phase 4: Performance Tuning (2-3 days)

**What:** Optimize the fused kernel for MI355X:

1. **Block size tuning:** AITER uses KV_BLK_SZ=32/64/128. Our packed_attention v7 found BLOCK_SK=32 optimal. Sweep.
2. **Dequant optimization:** Apply the binary tree + bitcast tricks from our v4d kernel:
   - Pre-scale Q by `scale` before the loop
   - Use `exp2` instead of `exp` for softmax
   - Interleave dequant loads with compute
3. **Register pressure:** Monitor register usage — 3-bit dequant adds ~10 registers per KV element. May need to reduce QUERY_GRP_SZ or KV_BLK_SZ.
4. **Split-K for long sequences:** Use v2 (partitioned) variant for seq_len > 4096 to saturate all 256 CUs.
5. **Warp count:** AITER uses 4 warps. Our kernels found w=8 sometimes better. Sweep.

---

## Expected Performance

**Theoretical analysis:**

| Step | Current (separate) | Fused (target) |
|------|-------------------|----------------|
| Dequant 23K pages | ~2.5ms | 0 (inline) |
| Q rotation | ~0.1ms | ~0.1ms (or 0 if fused) |
| Attention kernel | ~0.3ms (AITER) | ~0.8ms (TQ dequant + attention) |
| Output rotation | ~0.1ms | ~0.1ms |
| **Total per step** | **~3.0ms** | **~1.0ms** |

The dequant cost doesn't disappear — it moves into the attention kernel's inner loop. But:
- It overlaps with memory latency (dequant is ALU, loads are memory)
- It only dequants the blocks actually accessed (not all 23K pages)
- No FP16 materialization bandwidth (saves ~1.5GB/step write + read)

**Conservative estimate:** 3x improvement → ~4,500-5,000 tok/s (70-78% of vanilla)
**Optimistic estimate:** 4x improvement → ~6,000 tok/s (~93% of vanilla)

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Register pressure from inline dequant | High | Reduce KV_BLK_SZ, use v1_wo_dot (simpler) |
| Triton PA slower than AOT HIP PA | Medium | Benchmark Triton PA vs AOT PA first (without TQ) to establish baseline |
| 3-bit extraction too expensive in inner loop | Low | Already proven in packed_attention v7 (4.7x vs PyTorch) |
| CUDAGraph compatibility issues | Low | Same pattern as current, no dynamic shapes |

---

## Pre-work: Baseline Triton PA vs AOT PA

Before building the TQ fusion, first benchmark AITER's **Triton** PA decode vs the **AOT HIP** PA decode on vanilla FP16 cache. This tells us how much headroom the Triton path has vs HIP.

If Triton PA is >2x slower than AOT PA, we may need to consider the HIP path instead (Phase 1 alternative).

---

## Files to Create

```
turboquant_fused/
├── tq_aiter_pa_decode.py      # Phase 1: Fused TQ+PA Triton kernel
├── tq_aiter_integration.py    # Phase 3: vLLM backend wiring
├── bench_tq_aiter_fused.py    # Benchmark script
└── test_tq_aiter_fused.py     # Correctness validation
```

## Timeline

| Phase | Days | Deliverable |
|-------|------|-------------|
| Pre-work | 0.5 | Triton PA vs AOT PA baseline |
| Phase 1 | 2-3 | Working fused kernel, correctness validated |
| Phase 2 | 1 | Q rotation fused (optional) |
| Phase 3 | 1 | E2E vLLM integration |
| Phase 4 | 2-3 | Tuned to >4,000 tok/s |
| **Total** | **6-8 days** | Production-ready TQ+AITER fusion |
