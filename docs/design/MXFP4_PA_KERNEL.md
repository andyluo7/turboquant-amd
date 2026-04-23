# FP4 Paged Attention Kernel — Native FP4 MFMA on gfx950

## Goal
AITER paged-attention kernel with FP4 KV-cache storage, achieving ≤ FP8 latency
with **2× KV cache capacity**.

## Key Instruction
```cpp
__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
    int32x8_t a,     // 32 bytes: A operand
    int32x8_t b,     // 32 bytes: B operand (FP4 lives in low 16 bytes)
    float4_t  c,     // accumulator
    cbsz,            // A format: 0=FP8_E4M3, 1=FP8_E5M2, 2=FP6_E2M3, 3=FP6_E3M2, 4=FP4_E2M1
    blgp,            // B format: same encoding
    opselA, scale_a, // E8M0 scale for A
    opselB, scale_b  // E8M0 scale for B
);
```

### Per-MFMA Call
- Each call processes **32 elements** per operand (A and B).
- FP4: 32 values = 16 bytes packed.
- For HEAD_SIZE=128: 128/32 = **4 calls per QK head** (same as FP8).

## Storage Layout

### KV cache
- Shape: `[num_blocks, num_kv_heads, block_size, 64]` uint8 (HND).
- 128 FP4 nibbles packed into 64 bytes per (position, head) → **2× FP8 capacity**.

### Scale tensor (separate)
- Shape: `[num_blocks, block_size, num_kv_heads]` uint8 — one E8M0 byte per
  (position, head). Overhead ≈ 0.8% of cache size.

## Q Handling
Mixed FP8 × FP4: Q is the A operand (FP8 E4M3, `cbsz=0`), K is the B operand
(FP4 E2M1, `blgp=4`). Q stays in FP8 to preserve precision; K is the
compressed cache.

## MFMA Integration
```cpp
// QK dot product, 4 calls per HEAD_SIZE=128 head.
acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
    q_arg, k_arg, acc,
    /*cbsz=*/0,   /*blgp=*/4,
    /*opselA=*/0, /*scale_a=*/127,   // 127 = E8M0 0x7F = 1.0
    /*opselB=*/0, /*scale_b=*/127);
```

## Implementation in this repo

The production path lives in
`turboquant/integration/aiter/pa_kernels_fp4.cuh`, applied to AITER via
`turboquant/integration/aiter/paged_attention_fp4_patch.py`.

The forked kernel `_paged_attention_fp4_kernel` is templated on
`bool USE_NATIVE_FP4_MFMA`; the dispatcher in `pa_v1.cuh` instantiates both
specializations and selects between them at launch time. The forked path
avoids the `if constexpr` codegen issue observed inside template kernels on
ROCm 7.2.x — each specialization is a fully separate compile target and the
unused branch in either is folded out by the optimizer.

Two QK paths share the same Q build (FP8 INTERLEAVED), V-fetch (FP4→FP8 LUT
decode), softmax, V-MFMA, and reduction:

- **Native** — K loaded as raw FP4 bytes; one
  `mfma_scale_f32_16x16x128_f8f6f4` per head with `cbsz=0/blgp=4`. Output is
  reoriented from `[M=qhead, N=token]` to AITER's `[M=token, N=qhead]` via a
  4-lane `__shfl`.
- **LUT** — FP4 nibbles decoded to FP8 E4M3 at K-fetch using a 16-entry
  compile-time LUT, then `mfma_scale_f32_16x16x128_f8f6f4` with
  `cbsz=0/blgp=0` (both operands FP8). No shuffle.

## Notes
- `__hip_cvt_fp4x2_to_halfraw2` returns zeros on ROCm 7.0.0 (gfx950); the
  byte-LUT decode path avoids it.
- The "MFMA returns zeros" report was caused by passing
  `scale_a = scale_b = 0` (E8M0 0x00 = 2^-127 ≈ 0). Pass 127 (= 0x7F = 1.0)
  for unscaled operands.
- Patching the existing `_paged_attention_kernel` template in place is
  intractable because `QK_SIZE_RATIO` is derived from `sizeof(cache_t)`; the
  forked kernel sidesteps this.
