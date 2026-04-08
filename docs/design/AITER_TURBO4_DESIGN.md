# AITER turbo4 Kernel Extension — Design Document

## 1. AITER PA Kernel Architecture

The `paged_attention_ll4mi_QKV_mfma16_kernel` has this structure:

### Phase 1: Load Q (lines 130-200)
- Each warp loads Q head elements into shared memory
- Layout depends on `KV_DTYPE`: FP16/BF16 (`kAuto`) vs FP8 (different layout for 32-wide MFMA)
- **turbo4 impact:** None — Q is always in native dtype

### Phase 2: Fetch K (lines 205-240)
- For each KV block, load K from paged cache via 16-byte loads
- `cache_t* k_ptr3 = k_ptr + kphysical_block_offset * kv_seq_stride`
- **16B vectorized load:** `Klocal[...] = *k_fetch_ptr_16B` (reinterpret_cast to `_B16x8*`)
- For FP8: `sizeof(cache_t)=1`, so 16B = 16 elements; uses `mfma16x16x32` FP8 instruction
- For FP16: `sizeof(cache_t)=2`, so 16B = 8 elements; uses `mfma16x16x16` FP16 instruction
- **turbo4 impact:** CRITICAL — need to load nibble-packed data + dequant before MFMA

### Phase 3: QK dot product (lines 340-410)
- `gcn_mfma16x16x32_instr<__hip_fp8_e4m3>` for FP8
- `gcn_mfma16x16x16_instr<scalar_t>` for FP16/BF16
- K scale applied to softmax scale (not per-element)
- **turbo4 impact:** After dequant, K is FP16 → use FP16 MFMA path

### Phase 4: Softmax (lines 410-590)
- Online softmax with attention logits
- **turbo4 impact:** None — same computation

### Phase 5: Fetch V + PV MFMA (lines 680-810)
- Similar to K: load from paged cache, then MFMA with softmax weights
- V scale applied post-MFMA for FP8
- **turbo4 impact:** Same as K — need nibble dequant before MFMA

### Phase 6: Output (lines 810-900)
- Write to output tensor
- **turbo4 impact:** None

## 2. Turbo4 Dequant Insertion Points

### K Dequant: Lines 221-240 (K fetch loop)

**Current FP8 K load:**
```cpp
const cache_t* k_fetch_ptr = k_ptr3 + offset1 * KX + offset2;
const _B16x8* k_fetch_ptr_16B = reinterpret_cast<const _B16x8*>(k_fetch_ptr);
Klocal[head_loop][token_depth][qkhe_depth] = *k_fetch_ptr_16B;
```

**Proposed turbo4 K load:**
```cpp
if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kTurbo4) {
    // Load nibble-packed bytes for this head element range
    // Each thread processes CONTIGUOUS_KV_ELEMS_16B_LOAD dims (16 for FP8, 8 for FP16)
    // For turbo4: load 8 bytes (16 nibbles = 16 dims) per thread
    const int dim_start = head_elem;  // starting dimension
    const int nibble_byte_start = dim_start / 2;
    const uint8_t* k_nibble_ptr = reinterpret_cast<const uint8_t*>(k_ptr3) + nibble_byte_start;
    
    // Load 8 bytes = 16 nibbles = 16 dims
    uint64_t packed = *reinterpret_cast<const uint64_t*>(k_nibble_ptr);
    
    // Load norm (2 bytes at offset 64)
    const uint8_t* norm_ptr = reinterpret_cast<const uint8_t*>(k_ptr3) + 64;
    half norm_h = *reinterpret_cast<const half*>(norm_ptr);
    float norm = __half2float(norm_h);
    
    // Extract 16 nibbles and lookup centroids
    half k_vals[16];
    for (int i = 0; i < 8; i++) {
        uint8_t byte = (packed >> (i * 8)) & 0xFF;
        int idx_lo = byte & 0xF;
        int idx_hi = (byte >> 4) & 0xF;
        k_vals[2*i]   = __float2half(centroids[idx_lo] * norm);
        k_vals[2*i+1] = __float2half(centroids[idx_hi] * norm);
    }
    
    // Store as _B16x8 (16 bytes = 8 FP16 values)
    Klocal[head_loop][token_depth][qkhe_depth] = 
        *reinterpret_cast<_B16x8*>(k_vals);
} else {
    // Original path
    Klocal[...] = *k_fetch_ptr_16B;
}
```

### V Dequant: Lines 700-740 (V fetch loop)
Same pattern as K — load nibble bytes, extract, lookup, multiply by norm.

## 3. Template Parameter Changes

### pa_v1.py — Add turbo4 dtype
```python
elif kv_cache_dtype == "turbo4":
    if query.dtype == torch.bfloat16:
        dtype = "__hip_bfloat16"
        kv_dtype = "uint8_t"  # same as FP8 in terms of C++ type
    elif query.dtype == torch.float16:
        dtype = "_Float16"
        kv_dtype = "uint8_t"
    fp8_kv_dtype = "turbo4"  # new enum value
```

### pa_common.cuh — Add turbo4 to Fp8KVCacheDataType enum
```cpp
enum class Fp8KVCacheDataType {
    kAuto = 0,
    kFp8E4M3 = 1,
    kFp8E5M2 = 2,
    kTurbo4 = 3,  // NEW: 4-bit nibble PolarQuant
};
```

### pa_kernels.cuh — Add centroids parameter
```cpp
template <...>
__inline__ __device__ void _paged_attention_kernel(
    ...,
    const float* __restrict__ centroids,  // NEW: [16] turbo4 centroids
    ...
)
```

## 4. Key Challenges

### Challenge 1: MFMA Instruction Selection
FP8 uses `mfma16x16x32` (32 elements per instruction). After turbo4 dequant, data is FP16, so we'd use `mfma16x16x16` (16 elements per instruction). This means:
- **K fetch loads fewer elements per instruction** (16 vs 32 for FP8)
- But loads fewer bytes (8 vs 16) since packed
- Net: similar throughput

### Challenge 2: Norm Load
Each position has a per-position norm (2 bytes). This is an extra memory load per position that FP8 doesn't have. Options:
- Load norm once per position, broadcast to all dims (register)
- Pre-multiply centroids by norm → 16 scaled centroids per position

### Challenge 3: Centroid Table
16 centroids × 4 bytes = 64 bytes. Options:
- **Constant memory** — fastest, but limited (64KB total)
- **Shared memory** — fast, 64 bytes per block (negligible)
- **Registers** — each thread loads all 16 (64 bytes in registers)

Recommendation: **Shared memory** — load once at kernel start, access is fast.

## 5. Performance Estimate

| Component | FP8 (current) | turbo4 (proposed) |
|-----------|--------------|-------------------|
| K bytes loaded | 128B/pos (16B × 8 threads) | 64B/pos (8B × 8 threads) + 2B norm |
| K dequant ALU | 0 (hardware FP8→FP32) | 16 shifts + 16 masks + 16 centroid lookups + 16 muls |
| K MFMA | mfma16x16x32 (FP8, 32-wide) | mfma16x16x16 (FP16, 16-wide) |
| V bytes loaded | Same as K | Same as K |
| Total data | 256B/pos | 132B/pos (48% less) |
| Extra ALU | 0 | ~128 ops/pos (shifts+masks+gathers+muls) |

**Estimated turbo4 AITER: 70-100μs/layer (1.5-2x FP8's 48.3μs)**

The 48% less data should help, but the extra ALU for nibble extraction and centroid lookup will partially offset the bandwidth savings. The key question is whether the ALU can be hidden behind the memory pipeline.

## 6. Implementation Plan

1. Add `kTurbo4` to `Fp8KVCacheDataType` enum in `pa_common.cuh`
2. Add `centroids` kernel argument in `pa_v1.cpp.jinja` and `pa_v1.cuh`
3. Add turbo4 K dequant in K fetch loop (lines 221-240)
4. Add turbo4 V dequant in V fetch loop (lines 700-740)
5. Select FP16 MFMA path when turbo4 (not FP8 MFMA)
6. Add `turbo4` handling in `pa_v1.py` (Python JIT build)
7. Test with standalone benchmark
