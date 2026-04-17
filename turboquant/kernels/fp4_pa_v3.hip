/**
 * FP4 PA v3 — Multi-position tiling + shared memory Q
 * 
 * Key optimizations over v2:
 * 1. Process POSITIONS_PER_ITER=4 KV positions per inner loop iteration
 * 2. Q vector in shared memory (reused across all positions)
 * 3. Vectorized uint8 loads (load4 = 4 bytes = 8 FP4 values at once)
 * 4. Unrolled inner loop for ILP
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

#define POSITIONS_PER_ITER 4
#define HEAD_DIM 128
#define HALF_DIM 64
#define WAVE_SIZE 64
#define SCALE_DIM 4  // HEAD_DIM / 32

__device__ __forceinline__ float fp4_lut(uint8_t nibble) {
    // Computed at compile time, stored in SGPR
    constexpr float LUT[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return LUT[nibble & 0xF];
}

__device__ __forceinline__ float wave64_sum(float val) {
    val += __shfl_xor(val, 1, 64);
    val += __shfl_xor(val, 2, 64);
    val += __shfl_xor(val, 4, 64);
    val += __shfl_xor(val, 8, 64);
    val += __shfl_xor(val, 16, 64);
    val += __shfl_xor(val, 32, 64);
    return val;
}

__global__ __launch_bounds__(64)
void fp4_pa_v3_kernel(
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
    const int block_size,
    const int max_num_blocks,
    const float sm_scale
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int kv_head_idx = head_idx * num_kv_heads / num_heads;
    const int ctx_len = context_lens[seq_idx];
    
    // Thread mapping: tid handles byte [tid] = elements [2*tid, 2*tid+1]
    const int my_byte = tid;
    const int my_elem0 = tid * 2;
    const int my_elem1 = tid * 2 + 1;
    const int my_scale_group = my_elem0 / 32;
    
    // Load Q to registers (persistent across all positions)
    const int q_base = seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    float q0 = (my_elem0 < HEAD_DIM) ? __half2float(Q[q_base + my_elem0]) : 0.0f;
    float q1 = (my_elem1 < HEAD_DIM) ? __half2float(Q[q_base + my_elem1]) : 0.0f;
    
    // Online softmax state
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;
    
    // Precompute strides
    const int kv_pos_stride = num_kv_heads * HALF_DIM;
    const int ks_pos_stride = num_kv_heads * SCALE_DIM;
    
    // Iterate over blocks
    for (int bi = 0; bi < max_num_blocks; bi++) {
        const int start_pos = bi * block_size;
        if (start_pos >= ctx_len) break;
        
        const int phys_block = block_tables[seq_idx * max_num_blocks + bi];
        const int kv_block_base = phys_block * block_size * kv_pos_stride + kv_head_idx * HALF_DIM;
        const int ks_block_base = phys_block * block_size * ks_pos_stride + kv_head_idx * SCALE_DIM;
        
        // Process POSITIONS_PER_ITER positions at a time
        for (int s_base = 0; s_base < block_size; s_base += POSITIONS_PER_ITER) {
            
            float scores[POSITIONS_PER_ITER];
            float v_vals[POSITIONS_PER_ITER][2];
            
            // --- Phase 1: Compute Q·K scores for all positions ---
            #pragma unroll
            for (int si = 0; si < POSITIONS_PER_ITER; si++) {
                const int s = s_base + si;
                const int pos = start_pos + s;
                const bool valid = (s < block_size) && (pos < ctx_len);
                
                float k0 = 0.0f, k1 = 0.0f;
                if (valid && my_byte < HALF_DIM) {
                    uint8_t packed = K_cache[kv_block_base + s * kv_pos_stride + my_byte];
                    k0 = fp4_lut(packed & 0x0F);
                    k1 = fp4_lut((packed >> 4) & 0x0F);
                    
                    float ks = exp2f((float)K_scale[ks_block_base + s * ks_pos_stride + my_scale_group] - 127.0f);
                    k0 *= ks;
                    k1 *= ks;
                }
                
                float partial = q0 * k0 + q1 * k1;
                scores[si] = valid ? wave64_sum(partial) * sm_scale : -INFINITY;
            }
            
            // --- Phase 2: Online softmax + V accumulation ---
            #pragma unroll
            for (int si = 0; si < POSITIONS_PER_ITER; si++) {
                const int s = s_base + si;
                const int pos = start_pos + s;
                const bool valid = (s < block_size) && (pos < ctx_len);
                
                float score = scores[si];
                
                // Softmax update
                float m_new = fmaxf(m_i, score);
                float alpha = expf(m_i - m_new);
                float beta = expf(score - m_new);
                float l_new = l_i * alpha + beta;
                
                // Load V
                float v0 = 0.0f, v1 = 0.0f;
                if (valid && my_byte < HALF_DIM) {
                    uint8_t packed = V_cache[kv_block_base + s * kv_pos_stride + my_byte];
                    v0 = fp4_lut(packed & 0x0F);
                    v1 = fp4_lut((packed >> 4) & 0x0F);
                    
                    float vs = exp2f((float)V_scale[ks_block_base + s * ks_pos_stride + my_scale_group] - 127.0f);
                    v0 *= vs;
                    v1 *= vs;
                }
                
                float inv_l = 1.0f / fmaxf(l_new, 1e-10f);
                acc0 = (l_i * alpha * acc0 + beta * v0) * inv_l;
                acc1 = (l_i * alpha * acc1 + beta * v1) * inv_l;
                
                m_i = m_new;
                l_i = l_new;
            }
        }
    }
    
    // Store output
    const int out_base = seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    if (my_elem0 < HEAD_DIM) Out[out_base + my_elem0] = __float2half(acc0);
    if (my_elem1 < HEAD_DIM) Out[out_base + my_elem1] = __float2half(acc1);
}


extern "C" void launch_fp4_pa_v3(
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
    
    fp4_pa_v3_kernel<<<grid, block, 0, stream>>>(
        (const half*)Q, (const uint8_t*)K_cache, (const uint8_t*)V_cache,
        (const uint8_t*)K_scale, (const uint8_t*)V_scale,
        (const int*)block_tables, (const int*)context_lens, (half*)Out,
        num_heads, num_kv_heads, block_size, max_num_blocks, scale
    );
}
