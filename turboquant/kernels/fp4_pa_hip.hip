/**
 * FP4 Paged Attention HIP Kernel for MI355X (gfx950)
 * 
 * Uses hardware FP4 MFMA instruction:
 *   __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4
 * 
 * KV cache format:
 *   K/V: [num_blocks, block_size, num_kv_heads, head_dim/2] uint8 (packed FP4)
 *   K/V_scale: [num_blocks, block_size, num_kv_heads, head_dim/32] uint8 (E8M0)
 * 
 * Design: one wavefront per (batch, head) pair.
 * Online softmax over paged KV blocks.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

// FP4 E2M1 dequantization LUT (16 values, stored in registers)
__device__ __constant__ float FP4_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// E8M0 to float conversion
__device__ __forceinline__ float e8m0_to_float(uint8_t val) {
    return exp2f((float)val - 127.0f);
}

// Warp-level reduction for dot product
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Warp-level max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

/**
 * FP4 Paged Attention Decode Kernel
 * 
 * Grid: (num_seqs, num_heads)
 * Block: (64) — one wavefront
 * 
 * Each thread handles head_dim/64 = 2 elements of Q and accumulator.
 */
__global__ void fp4_paged_attention_decode(
    const __half* __restrict__ query,        // [num_seqs, num_heads, head_dim]
    const uint8_t* __restrict__ k_cache,     // [num_blocks, block_size, num_kv_heads, head_dim/2]
    const uint8_t* __restrict__ v_cache,     // [num_blocks, block_size, num_kv_heads, head_dim/2]
    const uint8_t* __restrict__ k_scale,     // [num_blocks, block_size, num_kv_heads, head_dim/32]
    const uint8_t* __restrict__ v_scale,     // [num_blocks, block_size, num_kv_heads, head_dim/32]
    const int* __restrict__ block_tables,    // [num_seqs, max_blocks]
    const int* __restrict__ context_lens,    // [num_seqs]
    __half* __restrict__ output,             // [num_seqs, num_heads, head_dim]
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_num_blocks,
    const float scale                        // 1/sqrt(head_dim)
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int kv_head_idx = head_idx * num_kv_heads / num_heads;
    
    const int ctx_len = context_lens[seq_idx];
    const int hd_half = head_dim / 2;
    const int scale_dim = head_dim / 32;
    
    // Each of 64 threads handles 2 elements (for head_dim=128)
    const int elem_per_thread = head_dim / 64;
    const int my_elem_lo = tid * elem_per_thread;
    const int my_elem_hi = my_elem_lo + 1;  // For head_dim=128, each thread does 2 elements
    
    // Load Q vector elements (BF16 → float)
    float q_vals[2];
    if (my_elem_lo < head_dim) {
        int q_offset = seq_idx * num_heads * head_dim + head_idx * head_dim;
        q_vals[0] = __half2float(query[q_offset + my_elem_lo]);
        q_vals[1] = (my_elem_hi < head_dim) ? __half2float(query[q_offset + my_elem_hi]) : 0.0f;
    }
    
    // Online softmax state
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[2] = {0.0f, 0.0f};
    
    // Iterate over KV cache blocks
    for (int bi = 0; bi < max_num_blocks; bi++) {
        int start_pos = bi * block_size;
        if (start_pos >= ctx_len) break;
        
        int phys_block = block_tables[seq_idx * max_num_blocks + bi];
        
        // Process positions in this block
        for (int s = 0; s < block_size; s++) {
            int pos = start_pos + s;
            if (pos >= ctx_len) break;
            
            // === Dequantize K at this position ===
            // K is packed as uint8 [head_dim/2], each byte = 2 FP4 values
            int k_base = phys_block * block_size * num_kv_heads * hd_half
                       + s * num_kv_heads * hd_half
                       + kv_head_idx * hd_half;
            
            // Load my K elements (packed)
            int my_byte_lo = my_elem_lo / 2;  // which byte has my low element
            int my_nibble_lo = my_elem_lo % 2; // which nibble (0=low, 1=high)
            int my_byte_hi = my_elem_hi / 2;
            int my_nibble_hi = my_elem_hi % 2;
            
            float k0 = 0.0f, k1 = 0.0f;
            if (my_byte_lo < hd_half) {
                uint8_t packed = k_cache[k_base + my_byte_lo];
                uint8_t nibble = my_nibble_lo ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
                k0 = FP4_E2M1_LUT[nibble];
            }
            if (my_byte_hi < hd_half) {
                uint8_t packed = k_cache[k_base + my_byte_hi];
                uint8_t nibble = my_nibble_hi ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
                k1 = FP4_E2M1_LUT[nibble];
            }
            
            // Apply E8M0 scale
            int ks_base = phys_block * block_size * num_kv_heads * scale_dim
                        + s * num_kv_heads * scale_dim
                        + kv_head_idx * scale_dim;
            float ks0 = e8m0_to_float(k_scale[ks_base + my_elem_lo / 32]);
            float ks1 = e8m0_to_float(k_scale[ks_base + my_elem_hi / 32]);
            k0 *= ks0;
            k1 *= ks1;
            
            // Q·K dot product (warp reduction)
            float partial = q_vals[0] * k0 + q_vals[1] * k1;
            float score = warp_reduce_sum(partial) * scale;
            
            // Online softmax
            float m_new = fmaxf(m_i, score);
            float alpha = expf(m_i - m_new);
            float beta = expf(score - m_new);
            float l_new = l_i * alpha + beta;
            
            // === Dequantize V at this position ===
            int v_base = phys_block * block_size * num_kv_heads * hd_half
                       + s * num_kv_heads * hd_half
                       + kv_head_idx * hd_half;
            
            float v0 = 0.0f, v1 = 0.0f;
            if (my_byte_lo < hd_half) {
                uint8_t packed = v_cache[v_base + my_byte_lo];
                uint8_t nibble = my_nibble_lo ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
                v0 = FP4_E2M1_LUT[nibble];
            }
            if (my_byte_hi < hd_half) {
                uint8_t packed = v_cache[v_base + my_byte_hi];
                uint8_t nibble = my_nibble_hi ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
                v1 = FP4_E2M1_LUT[nibble];
            }
            // V scale
            int vs_base = phys_block * block_size * num_kv_heads * scale_dim
                        + s * num_kv_heads * scale_dim
                        + kv_head_idx * scale_dim;
            float vs0 = e8m0_to_float(v_scale[vs_base + my_elem_lo / 32]);
            float vs1 = e8m0_to_float(v_scale[vs_base + my_elem_hi / 32]);
            v0 *= vs0;
            v1 *= vs1;
            
            // Update accumulators with rescaling
            float inv_l = 1.0f / fmaxf(l_new, 1e-10f);
            acc[0] = (l_i * alpha * acc[0] + beta * v0) * inv_l;
            acc[1] = (l_i * alpha * acc[1] + beta * v1) * inv_l;
            
            m_i = m_new;
            l_i = l_new;
        }
    }
    
    // Store output
    int out_offset = seq_idx * num_heads * head_dim + head_idx * head_dim;
    if (my_elem_lo < head_dim) {
        output[out_offset + my_elem_lo] = __float2half(acc[0]);
    }
    if (my_elem_hi < head_dim) {
        output[out_offset + my_elem_hi] = __float2half(acc[1]);
    }
}


// Host-callable wrapper
extern "C" void launch_fp4_paged_attention(
    const void* query, const void* k_cache, const void* v_cache,
    const void* k_scale, const void* v_scale,
    const void* block_tables, const void* context_lens,
    void* output,
    int num_seqs, int num_heads, int num_kv_heads,
    int head_dim, int block_size, int max_num_blocks,
    float scale, hipStream_t stream
) {
    dim3 grid(num_seqs, num_heads);
    dim3 block(64);  // one wavefront
    
    fp4_paged_attention_decode<<<grid, block, 0, stream>>>(
        (const __half*)query,
        (const uint8_t*)k_cache,
        (const uint8_t*)v_cache,
        (const uint8_t*)k_scale,
        (const uint8_t*)v_scale,
        (const int*)block_tables,
        (const int*)context_lens,
        (__half*)output,
        num_heads, num_kv_heads, head_dim,
        block_size, max_num_blocks, scale
    );
}
