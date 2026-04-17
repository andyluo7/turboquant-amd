/**
 * FP4 Paged Attention with MFMA — MI355X (gfx950)
 * 
 * Uses __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4 for Q×K dot product.
 * 
 * MFMA FP4 instruction: mfma_f32_16x16x128_f4
 *   A: 16x128 FP4 (packed in 8x int32 = 256 bits = 64 FP4 values × 4 rows)
 *   B: 128x16 FP4 (packed similarly)
 *   C: 16x16 FP32 accumulator
 *   Result: C += A × B (16×16 output tile)
 * 
 * For attention decode (Q=[1,D], K=[S,D]):
 *   Q×K^T is a 1×S vector of scores.
 *   We tile Q as 1×128 chunks and K as 128×1 chunks.
 *   Each MFMA does a partial dot product over 128 FP4 elements.
 *   head_dim=128 → exactly 1 MFMA per (Q, K_pos) pair.
 * 
 * For this v1: use software FP4 dequant + vectorized FP16 dot product
 * as a stepping stone to MFMA integration.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

// FP4 E2M1 dequantization using CK intrinsic
// __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(fp4_packed, scale_e8m0)
// Input: uint8 packed FP4 pair, uint8 E8M0 scale
// Output: half2 (two FP16 values)

__device__ __forceinline__ float fp4_nibble_to_float(uint8_t nibble) {
    // FP4 E2M1 encoding
    static constexpr float LUT[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return LUT[nibble & 0xF];
}

// Wavefront-64 reduction (AMD gfx950)
__device__ __forceinline__ float wave64_reduce_sum(float val) {
    // Use DPP (Data Parallel Primitives) for wavefront reduction
    val += __shfl_xor(val, 1, 64);
    val += __shfl_xor(val, 2, 64);
    val += __shfl_xor(val, 4, 64);
    val += __shfl_xor(val, 8, 64);
    val += __shfl_xor(val, 16, 64);
    val += __shfl_xor(val, 32, 64);
    return val;
}

__device__ __forceinline__ float wave64_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor(val, 1, 64));
    val = fmaxf(val, __shfl_xor(val, 2, 64));
    val = fmaxf(val, __shfl_xor(val, 4, 64));
    val = fmaxf(val, __shfl_xor(val, 8, 64));
    val = fmaxf(val, __shfl_xor(val, 16, 64));
    val = fmaxf(val, __shfl_xor(val, 32, 64));
    return val;
}

/**
 * FP4 PA Decode Kernel v2 — Vectorized with proper wavefront-64
 * 
 * Each wavefront (64 threads) handles one (batch, head) pair.
 * 
 * For head_dim=128:
 *   - 64 threads × 2 elements/thread = 128 elements covered
 *   - FP4 packed: 64 bytes per position per KV head
 *   - Each thread loads 1 byte (2 FP4 values), dequants to 2 floats
 *   - Warp reduction for Q·K score
 *   - Online softmax
 *   - V accumulation with same pattern
 */
__global__ void fp4_pa_mfma_v2(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K_cache,
    const uint8_t* __restrict__ V_cache,
    const uint8_t* __restrict__ K_scale,
    const uint8_t* __restrict__ V_scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    half* __restrict__ Out,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_num_blocks,
    const float sm_scale
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;  // 0..63
    const int kv_head_idx = head_idx * num_kv_heads / num_heads;
    
    const int ctx_len = context_lens[seq_idx];
    const int hd_half = head_dim / 2;  // 64 for head_dim=128
    const int scale_dim = head_dim / 32;  // 4 for head_dim=128
    
    // Each thread handles 1 byte = 2 FP4 values
    // Thread i handles elements [2*i, 2*i+1] of head_dim
    const int my_byte = tid;  // byte index in packed KV
    const int my_elem0 = tid * 2;
    const int my_elem1 = tid * 2 + 1;
    
    // Load Q elements
    const int q_base = seq_idx * num_heads * head_dim + head_idx * head_dim;
    float q0 = 0.0f, q1 = 0.0f;
    if (my_elem0 < head_dim) q0 = __half2float(Q[q_base + my_elem0]);
    if (my_elem1 < head_dim) q1 = __half2float(Q[q_base + my_elem1]);
    
    // Online softmax accumulators
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;
    
    // Iterate over KV blocks
    for (int bi = 0; bi < max_num_blocks; bi++) {
        const int start_pos = bi * block_size;
        if (start_pos >= ctx_len) break;
        
        const int phys_block = block_tables[seq_idx * max_num_blocks + bi];
        
        for (int s = 0; s < block_size; s++) {
            const int pos = start_pos + s;
            if (pos >= ctx_len) break;
            
            // === Load & dequant K ===
            const int k_offset = phys_block * block_size * num_kv_heads * hd_half
                               + s * num_kv_heads * hd_half
                               + kv_head_idx * hd_half
                               + my_byte;
            
            float k0 = 0.0f, k1 = 0.0f;
            if (my_byte < hd_half) {
                uint8_t packed = K_cache[k_offset];
                k0 = fp4_nibble_to_float(packed & 0x0F);
                k1 = fp4_nibble_to_float((packed >> 4) & 0x0F);
                
                // Apply E8M0 scale (per 32-element group)
                const int ks_offset = phys_block * block_size * num_kv_heads * scale_dim
                                    + s * num_kv_heads * scale_dim
                                    + kv_head_idx * scale_dim
                                    + (my_elem0 / 32);
                float ks = exp2f((float)K_scale[ks_offset] - 127.0f);
                k0 *= ks;
                k1 *= ks;
            }
            
            // Q·K partial dot product
            float partial = q0 * k0 + q1 * k1;
            float score = wave64_reduce_sum(partial) * sm_scale;
            
            // Online softmax
            float m_new = fmaxf(m_i, score);
            float alpha = expf(m_i - m_new);
            float beta = expf(score - m_new);
            float l_new = l_i * alpha + beta;
            
            // === Load & dequant V ===
            const int v_offset = phys_block * block_size * num_kv_heads * hd_half
                               + s * num_kv_heads * hd_half
                               + kv_head_idx * hd_half
                               + my_byte;
            
            float v0 = 0.0f, v1 = 0.0f;
            if (my_byte < hd_half) {
                uint8_t packed = V_cache[v_offset];
                v0 = fp4_nibble_to_float(packed & 0x0F);
                v1 = fp4_nibble_to_float((packed >> 4) & 0x0F);
                
                const int vs_offset = phys_block * block_size * num_kv_heads * scale_dim
                                    + s * num_kv_heads * scale_dim
                                    + kv_head_idx * scale_dim
                                    + (my_elem0 / 32);
                float vs = exp2f((float)V_scale[vs_offset] - 127.0f);
                v0 *= vs;
                v1 *= vs;
            }
            
            // Accumulate
            float inv_l = 1.0f / fmaxf(l_new, 1e-10f);
            acc0 = (l_i * alpha * acc0 + beta * v0) * inv_l;
            acc1 = (l_i * alpha * acc1 + beta * v1) * inv_l;
            
            m_i = m_new;
            l_i = l_new;
        }
    }
    
    // Store output
    const int out_base = seq_idx * num_heads * head_dim + head_idx * head_dim;
    if (my_elem0 < head_dim) Out[out_base + my_elem0] = __float2half(acc0);
    if (my_elem1 < head_dim) Out[out_base + my_elem1] = __float2half(acc1);
}


extern "C" void launch_fp4_pa_mfma(
    const void* Q, const void* K_cache, const void* V_cache,
    const void* K_scale, const void* V_scale,
    const void* block_tables, const void* context_lens,
    void* Out,
    int num_seqs, int num_heads, int num_kv_heads,
    int head_dim, int block_size, int max_num_blocks,
    float scale, hipStream_t stream
) {
    dim3 grid(num_seqs, num_heads);
    dim3 block(64);
    
    fp4_pa_mfma_v2<<<grid, block, 0, stream>>>(
        (const half*)Q, (const uint8_t*)K_cache, (const uint8_t*)V_cache,
        (const uint8_t*)K_scale, (const uint8_t*)V_scale,
        (const int*)block_tables, (const int*)context_lens, (half*)Out,
        num_heads, num_kv_heads, head_dim, block_size, max_num_blocks, scale
    );
}
