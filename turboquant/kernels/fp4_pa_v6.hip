/**
 * FP4 PA v6 — Split-K across wavefronts
 * 
 * Key change: 4 wavefronts split the sequence dimension.
 * Each wavefront handles ctx_len/4 positions independently.
 * Final reduction combines partial softmax results.
 * 
 * This reduces per-wavefront work by 4x for long sequences.
 * Short sequences (S<64) may not benefit.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

#define HEAD_DIM 128
#define HALF_DIM 64
#define NUM_WAVES 4
#define WAVE_SIZE 64
#define BLOCK_SIZE_T (NUM_WAVES * WAVE_SIZE)
#define SCALE_DIM 4

__device__ __forceinline__ float fp4_d(uint8_t n) {
    constexpr float L[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return L[n & 0xF];
}

__global__ __launch_bounds__(256)
void fp4_pa_v6_kernel(
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
    __shared__ float s_m[NUM_WAVES];      // max per wave
    __shared__ float s_l[NUM_WAVES];      // sum per wave
    __shared__ float s_acc[NUM_WAVES][HEAD_DIM]; // partial outputs per wave
    
    const int my_elem0 = lane_id * 2;
    const int my_elem1 = lane_id * 2 + 1;
    
    // Load Q
    const int q_base = seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    float q0 = (my_elem0 < HEAD_DIM) ? __half2float(Q[q_base + my_elem0]) : 0.0f;
    float q1 = (my_elem1 < HEAD_DIM) ? __half2float(Q[q_base + my_elem1]) : 0.0f;
    
    // === Split-K: each wave handles a chunk of the sequence ===
    const int total_positions = ctx_len;
    const int positions_per_wave = (total_positions + NUM_WAVES - 1) / NUM_WAVES;
    const int my_start = wave_id * positions_per_wave;
    const int my_end = min(my_start + positions_per_wave, total_positions);
    
    const int kv_pos_stride = num_kv_heads * HALF_DIM;
    const int ks_pos_stride = num_kv_heads * SCALE_DIM;
    
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;
    
    // Process my chunk of positions
    for (int pos = my_start; pos < my_end; pos++) {
        const int block_idx = pos / block_size;
        const int block_off = pos % block_size;
        const int phys_block = block_tables[seq_idx * max_num_blocks + block_idx];
        
        const int kv_offset = phys_block * block_size * kv_pos_stride
                            + block_off * kv_pos_stride
                            + kv_head_idx * HALF_DIM;
        const int ks_offset = phys_block * block_size * ks_pos_stride
                            + block_off * ks_pos_stride
                            + kv_head_idx * SCALE_DIM;
        
        // Load & dequant K
        float k0 = 0.0f, k1 = 0.0f;
        if (lane_id < HALF_DIM) {
            uint8_t packed = K_cache[kv_offset + lane_id];
            float ks = exp2f((float)K_scale[ks_offset + (my_elem0 / 32)] - 127.0f);
            k0 = fp4_d(packed & 0x0F) * ks;
            k1 = fp4_d((packed >> 4) & 0x0F) * ks;
        }
        
        // Q·K dot product
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
        
        // Load & dequant V
        float v0 = 0.0f, v1 = 0.0f;
        if (lane_id < HALF_DIM) {
            uint8_t packed = V_cache[kv_offset + lane_id];
            float vs = exp2f((float)V_scale[ks_offset + (my_elem0 / 32)] - 127.0f);
            v0 = fp4_d(packed & 0x0F) * vs;
            v1 = fp4_d((packed >> 4) & 0x0F) * vs;
        }
        
        float inv_l = 1.0f / fmaxf(l_new, 1e-10f);
        acc0 = (l_i * alpha * acc0 + beta * v0) * inv_l;
        acc1 = (l_i * alpha * acc1 + beta * v1) * inv_l;
        
        m_i = m_new;
        l_i = l_new;
    }
    
    // === Cross-wave reduction ===
    // Each wave writes its partial result to shared memory
    if (lane_id == 0) {
        s_m[wave_id] = m_i;
        s_l[wave_id] = l_i;
    }
    // All lanes write their acc elements
    if (my_elem0 < HEAD_DIM) s_acc[wave_id][my_elem0] = acc0;
    if (my_elem1 < HEAD_DIM) s_acc[wave_id][my_elem1] = acc1;
    __syncthreads();
    
    // Wave 0 does the final reduction
    if (wave_id == 0) {
        // Combine partial softmax results from all waves
        float final_m = s_m[0];
        for (int w = 1; w < NUM_WAVES; w++) {
            final_m = fmaxf(final_m, s_m[w]);
        }
        
        float final_l = 0.0f;
        for (int w = 0; w < NUM_WAVES; w++) {
            final_l += s_l[w] * expf(s_m[w] - final_m);
        }
        
        // Combine V accumulators with rescaling
        float out0 = 0.0f, out1 = 0.0f;
        for (int w = 0; w < NUM_WAVES; w++) {
            float scale_w = s_l[w] * expf(s_m[w] - final_m) / fmaxf(final_l, 1e-10f);
            if (my_elem0 < HEAD_DIM) out0 += s_acc[w][my_elem0] * scale_w;
            if (my_elem1 < HEAD_DIM) out1 += s_acc[w][my_elem1] * scale_w;
        }
        
        const int out_base = seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
        if (my_elem0 < HEAD_DIM) Out[out_base + my_elem0] = __float2half(out0);
        if (my_elem1 < HEAD_DIM) Out[out_base + my_elem1] = __float2half(out1);
    }
}


extern "C" void launch_fp4_pa_v6(
    const void* Q, const void* K_cache, const void* V_cache,
    const void* K_scale, const void* V_scale,
    const void* block_tables, const void* context_lens,
    void* Out,
    int num_seqs, int num_heads, int num_kv_heads,
    int head_dim, int block_size, int max_num_blocks,
    float scale, hipStream_t stream
) {
    dim3 grid(num_seqs, num_heads);
    dim3 block(256);
    
    fp4_pa_v6_kernel<<<grid, block, 0, stream>>>(
        (const half*)Q, (const uint8_t*)K_cache, (const uint8_t*)V_cache,
        (const uint8_t*)K_scale, (const uint8_t*)V_scale,
        (const int*)block_tables, (const int*)context_lens, (half*)Out,
        num_heads, num_kv_heads, block_size, max_num_blocks, scale
    );
}
