# MXFP4 Paged Attention Kernel — Native FP4 MFMA on MI355X (gfx950)

## Goal
AITER PA kernel with FP4 KV cache storage, achieving ≤ FP8 latency with **2x KV cache capacity**.

## Key Instruction
```cpp
__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
    int32x8_t a,     // 32 bytes: Q data (padded to 256 bits)
    int32x8_t b,     // 32 bytes: K data (padded to 256 bits)
    float4_t c,      // accumulator
    cbsz,            // A format: 0=FP8_E4M3, 1=FP8_E5M2, 2=FP6_E2M3, 3=FP6_E3M2, 4=FP4_E2M1
    blgp,            // B format: same encoding
    opselA, scale_a, // E8M0 scale for A
    opselB, scale_b  // E8M0 scale for B
);
```

Found in CK: `ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp`

### Data Types (from CK dtype2conf)
| Type | cbsz/blgp | Vector Type | Size |
|------|-----------|-------------|------|
| FP8 E4M3 | 0 | int32x8_t (32B) | 32 values |
| FP8 E5M2 | 1 | int32x8_t (32B) | 32 values |
| FP6 E2M3 | 2 | pk_fp6x32_t (24B) | 32 values |
| FP6 E3M2 | 3 | int32x6_t (24B) | 32 values |
| **FP4 E2M1** | **4** | **int32x4_t (16B)** | **32 values** |

### Per-MFMA Call
- Each call processes **32 elements** per operand
- FP4: 32 values = 16 bytes of packed data
- For HEAD_SIZE=128: **128/32 = 4 MFMA calls** (same as FP8)
- FP4 operand is `int32x4_t` (16 bytes), padded to `int32x8_t` (32 bytes) via `arg256()`

## Storage Layout

### KV Cache
- Shape: `[num_blocks, block_size, num_kv_heads, 64]` uint8
- 128 FP4 nibbles packed into 64 bytes per position per head
- **Exactly 2x FP8 capacity** (FP8 uses 128 bytes per head)

### Scale Tensor (separate)
- Shape: `[num_blocks, block_size, num_kv_heads]` uint8
- 1 E8M0 byte per position per head
- Overhead: ~0.8% of cache size (negligible)
- Passed to kernel as extra pointer argument

## K/V Fetch
```cpp
// Load 16 packed FP4 bytes (32 nibbles = 32 FP4 values)
const uint8_t* k_fp4 = reinterpret_cast<const uint8_t*>(k_ptr) + byte_offset;
int32x4_t k_packed = *reinterpret_cast<const int32x4_t*>(k_fp4);
// Pad to int32x8_t for MFMA
int32x8_t k_arg = {k_packed[0], k_packed[1], k_packed[2], k_packed[3], 0, 0, 0, 0};
```

## Q Handling
Two options:
1. **Mixed FP8×FP4**: Q stays FP8 (cbsz=0), K is FP4 (blgp=4). Q conversion: bf16→FP8 (standard).
2. **Full FP4×FP4**: Both Q and K in FP4 (cbsz=4, blgp=4). Q conversion: bf16→FP4 (more quantization loss).

**Recommendation: Option 1 (mixed FP8×FP4)** — preserves Q precision, K is the compressed cache.

## Scale Passing
```cpp
// Read E8M0 scale byte from separate scale tensor
const uint8_t* k_scale_tensor = ...; // passed as kernel arg
uint8_t e8m0 = k_scale_tensor[block_idx * block_size * num_kv_heads + seq_offset * num_kv_heads + head_idx];
int32_t scale_b = (int32_t)e8m0;
int32_t scale_a = 127; // Q scale = 1.0 (FP8 data, no additional scaling)
```

## MFMA Integration
```cpp
// QK dot product: 4 MFMA calls for HEAD_SIZE=128
for (int i = 0; i < 4; i++) {
    int32x4_t k_packed = load_16_bytes(k_fp4 + i * 16);
    int32x8_t k_arg = arg256(k_packed);
    int32x8_t q_arg = ...; // Q in FP8 format
    
    acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        q_arg, k_arg, acc,
        0,  // cbsz=0: Q is FP8 E4M3
        4,  // blgp=4: K is FP4 E2M1
        0, scale_a,  // Q scale (127 = 1.0)
        0, scale_b   // K scale (from E8M0 tensor)
    );
}
```

## Performance Analysis

| Metric | FP8 PA | FP4 PA (this plan) | Improvement |
|--------|--------|-------------------|-------------|
| K/V data per head | 128 bytes | 64 bytes | **2x less HBM** |
| MFMA calls per head (QK) | 4 (16x16x32) | 4 (16x16x128) | same |
| MFMA calls per head (V) | 4 | 4 | same |
| KV cache capacity | 1x | **2x** | **2x** |
| HBM bandwidth | 1x | ~0.5x | **~2x** |
| Expected latency | 48μs | **≤48μs** | **≤1x** |

Decode is memory-bound → 2x less data load ≈ up to 2x faster.

## Implementation Steps

### Day 1: Kernel Fork + K/V Fetch
1. Fork `_paged_attention_kernel` → `_paged_attention_mxfp4_kernel`
2. Change cache_t to uint8_t, keep HEAD_SIZE=128
3. Replace K fetch: load 16B packed FP4 → `int32x4_t` → pad to `int32x8_t`
4. Replace V fetch: same pattern
5. Add scale tensor pointer as extra kernel arg (via JIT template)

### Day 2: MFMA + Q Conversion + Correctness
6. Replace QK MFMA: `gcn_mfma16x16x32_instr` → `mfma_scale_f32_16x16x128_f8f6f4`
7. Replace V accumulate MFMA: same instruction
8. Q conversion: bf16 → FP8 E4M3 (for mixed mode)
9. Scale: read from separate tensor, pass as scale_b
10. Correctness test: compare vs SDPA reference (target cos > 0.99)

### Day 3: vLLM Integration + E2E
11. Register MXFP4 backend in vLLM
12. MXFP4 compress_and_scatter (per_1x32_f4_quant → 64B cache + scale tensor)
13. CUDAGraph compatibility (pre-allocated buffers)
14. E2E serving test with MiniMax-M2.5 TP=2

## Key Findings (2026-04-07)
- `__hip_cvt_fp4x2_to_halfraw2` is **BROKEN** on gfx950 (returns zeros)
- `__builtin_amdgcn_cvt_scalef32_pk_f16_fp4` **WORKS** for FP4→FP16 conversion
- Single per-position E8M0 scale matches per-32 block scale quality (gap < 0.001 cosine)
- Non-contiguous cache view causes memory faults → use `[2, nb, bs, nkv, D]` layout
- CUDAGraph: pre-allocate all buffers, cap workspace to 128 partitions
- Patching existing kernel MFMA tiling is intractable (QK_SIZE_RATIO derived from sizeof(cache_t))
- **Native FP4 MFMA is the correct approach** — avoids all conversion/tiling issues
