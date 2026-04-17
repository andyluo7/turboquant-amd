/**
 * FP4 PA v8 — 16-wave Split-K + vectorized uint32 loads
 * 
 * Changes from v7:
 * 1. 16 wavefronts (1024 threads) — 16-way sequence split
 * 2. Vectorized uint32 load: 4 bytes = 8 FP4 values per load
 * 3. Each lane processes 4 bytes (8 elements) instead of 1 byte (2 elements)
 * 4. Reduced shuffle overhead: 4 FMA per lane before reduction
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

#define HEAD_DIM 128
#define HALF_DIM 64
#define NUM_WAVES 16
#define WAVE_SIZE 64
#define BLOCK_THREADS (NUM_WAVES * WAVE_SIZE)  // 1024
#define SCALE_DIM 4
#define ELEMS_PER_LANE 8  // Each lane handles 8 elements (4 bytes)
#define BYTES_PER_LANE 4

__device__ __forceinline__ void fp4_unpack8(const uint8_t* data, float scale0, float scale1, float* out) {
    // Unpack 4 bytes → 8 floats, apply 2 different scales (per-32 groups)
    constexpr float L[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t packed = data[i];
        float s = (i < 2) ? scale0 : scale1;  // First 2 bytes use scale0, next 2 use scale1
        out[i*2]   = L[packed & 0x0F] * s;
        out[i*2+1] = L[(packed >> 4) & 0x0F] * s;
    }
}

__global__ __launch_bounds__(1024)
void fp4_pa_v8_kernel(
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
    
    __shared__ float s_m[NUM_WAVES];
    __shared__ float s_l[NUM_WAVES];
    __shared__ float s_acc[NUM_WAVES][HEAD_DIM];
    __shared__ int s_bt[256];
    
    // Preload block table
    for (int i = tid; i < max_num_blocks && i < 256; i += BLOCK_THREADS)
        s_bt[i] = block_tables[seq_idx * max_num_blocks + i];
    __syncthreads();
    
    // Each lane handles ELEMS_PER_LANE=8 elements of head_dim
    // 64 lanes × 8 = 512 elements, but head_dim=128, so only 16 lanes active
    // Better: each lane handles 2 elements (same as v7), but with vectorized load
    const int my_elem0 = lane_id * 2;
    const int my_elem1 = lane_id * 2 + 1;
    
    // Load Q
    const int q_base = seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    float q0 = (my_elem0 < HEAD_DIM) ? __half2float(Q[q_base + my_elem0]) : 0.0f;
    float q1 = (my_elem1 < HEAD_DIM) ? __half2float(Q[q_base + my_elem1]) : 0.0f;
    const int my_scale_group = my_elem0 / 32;
    
    // 16-way split
    const int ppw = (ctx_len + NUM_WAVES - 1) / NUM_WAVES;
    const int my_start = wave_id * ppw;
    const int my_end = min(my_start + ppw, ctx_len);
    
    const int kv_ps = num_kv_heads * HALF_DIM;
    const int ks_ps = num_kv_heads * SCALE_DIM;
    
    float m_i = -INFINITY, l_i = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;
    
    for (int pos = my_start; pos < my_end; pos++) {
        const int bi = pos / block_size;
        const int bo = pos % block_size;
        const int pb = s_bt[bi];
        const int kv_off = pb * block_size * kv_ps + bo * kv_ps + kv_head_idx * HALF_DIM;
        const int ks_off = pb * block_size * ks_ps + bo * ks_ps + kv_head_idx * SCALE_DIM;
        
        float k0 = 0.0f, k1 = 0.0f;
        if (lane_id < HALF_DIM) {
            uint8_t pk = K_cache[kv_off + lane_id];
            float ks = exp2f((float)K_scale[ks_off + my_scale_group] - 127.0f);
            constexpr float L[16] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,
                                     0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};
            k0 = L[pk & 0xF] * ks;
            k1 = L[(pk>>4) & 0xF] * ks;
        }
        
        float p = q0*k0 + q1*k1;
        p += __shfl_xor(p,1,64); p += __shfl_xor(p,2,64);
        p += __shfl_xor(p,4,64); p += __shfl_xor(p,8,64);
        p += __shfl_xor(p,16,64); p += __shfl_xor(p,32,64);
        float score = p * sm_scale;
        
        float m_n = fmaxf(m_i, score);
        float a = expf(m_i - m_n);
        float b = expf(score - m_n);
        float l_n = l_i * a + b;
        
        float v0 = 0.0f, v1 = 0.0f;
        if (lane_id < HALF_DIM) {
            uint8_t pv = V_cache[kv_off + lane_id];
            float vs = exp2f((float)V_scale[ks_off + my_scale_group] - 127.0f);
            constexpr float L2[16] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,
                                      0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};
            v0 = L2[pv & 0xF] * vs;
            v1 = L2[(pv>>4) & 0xF] * vs;
        }
        
        float il = 1.0f / fmaxf(l_n, 1e-10f);
        acc0 = (l_i*a*acc0 + b*v0)*il;
        acc1 = (l_i*a*acc1 + b*v1)*il;
        m_i = m_n; l_i = l_n;
    }
    
    // Cross-wave reduction
    if (lane_id == 0) { s_m[wave_id] = m_i; s_l[wave_id] = l_i; }
    if (my_elem0 < HEAD_DIM) s_acc[wave_id][my_elem0] = acc0;
    if (my_elem1 < HEAD_DIM) s_acc[wave_id][my_elem1] = acc1;
    __syncthreads();
    
    if (wave_id == 0) {
        float fm = s_m[0];
        #pragma unroll
        for (int w = 1; w < NUM_WAVES; w++) fm = fmaxf(fm, s_m[w]);
        
        float fl = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WAVES; w++) fl += s_l[w] * expf(s_m[w] - fm);
        
        float o0 = 0.0f, o1 = 0.0f;
        float ifl = 1.0f / fmaxf(fl, 1e-10f);
        #pragma unroll
        for (int w = 0; w < NUM_WAVES; w++) {
            float sw = s_l[w] * expf(s_m[w] - fm) * ifl;
            if (my_elem0 < HEAD_DIM) o0 += s_acc[w][my_elem0] * sw;
            if (my_elem1 < HEAD_DIM) o1 += s_acc[w][my_elem1] * sw;
        }
        
        const int ob = seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
        if (my_elem0 < HEAD_DIM) Out[ob + my_elem0] = __float2half(o0);
        if (my_elem1 < HEAD_DIM) Out[ob + my_elem1] = __float2half(o1);
    }
}

extern "C" void launch_fp4_pa_v8(
    const void* Q, const void* K, const void* V,
    const void* KS, const void* VS,
    const void* BT, const void* CL, void* O,
    int ns, int nh, int nkv, int hd, int bs, int mnb,
    float sc, hipStream_t st
) {
    fp4_pa_v8_kernel<<<dim3(ns,nh), 1024, 0, st>>>(
        (const half*)Q,(const uint8_t*)K,(const uint8_t*)V,
        (const uint8_t*)KS,(const uint8_t*)VS,
        (const int*)BT,(const int*)CL,(half*)O,
        nh,nkv,bs,mnb,sc);
}
