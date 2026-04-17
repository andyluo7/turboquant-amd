/**
 * FP4 PA v7 — 8-wave Split-K + preloaded scales + optimized dequant
 * 
 * Changes from v6:
 * 1. 8 wavefronts (512 threads) — 8-way sequence split
 * 2. Preload block_tables into shared memory
 * 3. Fused dequant: combine nibble extract + scale in fewer ops
 * 4. Precompute scale per group (4 scales for head_dim=128)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

#define HEAD_DIM 128
#define HALF_DIM 64
#define NUM_WAVES 8
#define WAVE_SIZE 64
#define BLOCK_THREADS (NUM_WAVES * WAVE_SIZE)  // 512
#define SCALE_DIM 4

// Compact FP4 dequant — branchless, uses FMA
__device__ __forceinline__ void fp4_unpack2(uint8_t packed, float scale, float& out0, float& out1) {
    // Nibble extraction
    const int lo = packed & 0x0F;
    const int hi = (packed >> 4) & 0x0F;
    
    // FP4 E2M1 LUT (inlined constants for SGPR)
    constexpr float L[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    out0 = L[lo] * scale;
    out1 = L[hi] * scale;
}

__global__ __launch_bounds__(512)
void fp4_pa_v7_kernel(
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
    const int wave_id = tid / WAVE_SIZE;
    const int lane_id = tid % WAVE_SIZE;
    const int kv_head_idx = head_idx * num_kv_heads / num_heads;
    const int ctx_len = context_lens[seq_idx];
    
    // Shared memory for cross-wave reduction
    __shared__ float s_m[NUM_WAVES];
    __shared__ float s_l[NUM_WAVES];
    __shared__ float s_acc[NUM_WAVES][HEAD_DIM];
    // Preloaded block table
    __shared__ int s_block_table[256]; // max 256 blocks
    
    // Cooperative block table preload
    for (int i = tid; i < max_num_blocks && i < 256; i += BLOCK_THREADS) {
        s_block_table[i] = block_tables[seq_idx * max_num_blocks + i];
    }
    __syncthreads();
    
    // Q in registers
    const int my_elem0 = lane_id * 2;
    const int my_elem1 = lane_id * 2 + 1;
    const int q_base = seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    float q0 = (my_elem0 < HEAD_DIM) ? __half2float(Q[q_base + my_elem0]) : 0.0f;
    float q1 = (my_elem1 < HEAD_DIM) ? __half2float(Q[q_base + my_elem1]) : 0.0f;
    const int my_scale_group = my_elem0 / 32;
    
    // 8-way sequence split
    const int positions_per_wave = (ctx_len + NUM_WAVES - 1) / NUM_WAVES;
    const int my_start = wave_id * positions_per_wave;
    const int my_end = min(my_start + positions_per_wave, ctx_len);
    
    const int kv_pos_stride = num_kv_heads * HALF_DIM;
    const int ks_pos_stride = num_kv_heads * SCALE_DIM;
    
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;
    
    for (int pos = my_start; pos < my_end; pos++) {
        const int block_idx = pos / block_size;
        const int block_off = pos % block_size;
        const int phys_block = s_block_table[block_idx]; // From shared memory!
        
        const int kv_offset = phys_block * block_size * kv_pos_stride
                            + block_off * kv_pos_stride
                            + kv_head_idx * HALF_DIM;
        const int ks_offset = phys_block * block_size * ks_pos_stride
                            + block_off * ks_pos_stride
                            + kv_head_idx * SCALE_DIM;
        
        // Load K scale (per-32 group, only need 1 for my 2 elements)
        float k_scale_val = exp2f((float)K_scale[ks_offset + my_scale_group] - 127.0f);
        
        // Load & dequant K (fused)
        float k0, k1;
        if (lane_id < HALF_DIM) {
            fp4_unpack2(K_cache[kv_offset + lane_id], k_scale_val, k0, k1);
        } else {
            k0 = 0.0f; k1 = 0.0f;
        }
        
        // Q·K dot product with wave-64 reduction
        float partial = q0 * k0 + q1 * k1;
        partial += __shfl_xor(partial, 1, 64);
        partial += __shfl_xor(partial, 2, 64);
        partial += __shfl_xor(partial, 4, 64);
        partial += __shfl_xor(partial, 8, 64);
        partial += __shfl_xor(partial, 16, 64);
        partial += __shfl_xor(partial, 32, 64);
        float score = partial * sm_scale;
        
        // Online softmax
        float m_new = fmaxf(m_i, score);
        float alpha = expf(m_i - m_new);
        float beta = expf(score - m_new);
        float l_new = l_i * alpha + beta;
        
        // Load V scale
        float v_scale_val = exp2f((float)V_scale[ks_offset + my_scale_group] - 127.0f);
        
        // Load & dequant V
        float v0, v1;
        if (lane_id < HALF_DIM) {
            fp4_unpack2(V_cache[kv_offset + lane_id], v_scale_val, v0, v1);
        } else {
            v0 = 0.0f; v1 = 0.0f;
        }
        
        // Accumulate
        float inv_l = 1.0f / fmaxf(l_new, 1e-10f);
        acc0 = (l_i * alpha * acc0 + beta * v0) * inv_l;
        acc1 = (l_i * alpha * acc1 + beta * v1) * inv_l;
        
        m_i = m_new;
        l_i = l_new;
    }
    
    // === Cross-wave reduction ===
    if (lane_id == 0) {
        s_m[wave_id] = m_i;
        s_l[wave_id] = l_i;
    }
    if (my_elem0 < HEAD_DIM) s_acc[wave_id][my_elem0] = acc0;
    if (my_elem1 < HEAD_DIM) s_acc[wave_id][my_elem1] = acc1;
    __syncthreads();
    
    // Wave 0 reduces
    if (wave_id == 0) {
        float final_m = s_m[0];
        #pragma unroll
        for (int w = 1; w < NUM_WAVES; w++)
            final_m = fmaxf(final_m, s_m[w]);
        
        float final_l = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WAVES; w++)
            final_l += s_l[w] * expf(s_m[w] - final_m);
        
        float out0 = 0.0f, out1 = 0.0f;
        float inv_fl = 1.0f / fmaxf(final_l, 1e-10f);
        #pragma unroll
        for (int w = 0; w < NUM_WAVES; w++) {
            float scale_w = s_l[w] * expf(s_m[w] - final_m) * inv_fl;
            if (my_elem0 < HEAD_DIM) out0 += s_acc[w][my_elem0] * scale_w;
            if (my_elem1 < HEAD_DIM) out1 += s_acc[w][my_elem1] * scale_w;
        }
        
        const int out_base = seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
        if (my_elem0 < HEAD_DIM) Out[out_base + my_elem0] = __float2half(out0);
        if (my_elem1 < HEAD_DIM) Out[out_base + my_elem1] = __float2half(out1);
    }
}


extern "C" void launch_fp4_pa_v7(
    const void* Q, const void* K_cache, const void* V_cache,
    const void* K_scale, const void* V_scale,
    const void* block_tables, const void* context_lens,
    void* Out,
    int num_seqs, int num_heads, int num_kv_heads,
    int head_dim, int block_size, int max_num_blocks,
    float scale, hipStream_t stream
) {
    dim3 grid(num_seqs, num_heads);
    dim3 block(512);
    
    fp4_pa_v7_kernel<<<grid, block, 0, stream>>>(
        (const half*)Q, (const uint8_t*)K_cache, (const uint8_t*)V_cache,
        (const uint8_t*)K_scale, (const uint8_t*)V_scale,
        (const int*)block_tables, (const int*)context_lens, (half*)Out,
        num_heads, num_kv_heads, block_size, max_num_blocks, scale
    );
}
