/**
 * FP4 PA v4 — Multi-threadgroup with shared memory KV tiling
 * 
 * Key optimizations:
 * 1. 256 threads (4 wavefronts) per (batch, head)
 * 2. Cooperative KV loading: all threads load KV into shared memory
 * 3. Process TILE_S=16 positions per outer iteration
 * 4. Q stays in registers; K/V tiles in LDS
 * 5. Reduce warp shuffle overhead by computing dot products in LDS
 * 
 * Memory access pattern:
 *   - Global → LDS: 256 threads cooperatively load 16 positions × 64 bytes = 1024 bytes
 *   - LDS → Registers: each thread reads its Q-matching elements
 *   - This gives 4x more threads for global memory coalescing
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

#define HEAD_DIM 128
#define HALF_DIM 64
#define TILE_S 16       // KV positions per shared memory tile
#define BLOCK_SIZE_T 256 // threads per block (4 wavefronts)
#define SCALE_DIM 4

__device__ __forceinline__ float fp4_dequant(uint8_t nibble) {
    constexpr float LUT[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return LUT[nibble & 0xF];
}

// Cross-wavefront reduction using shared memory
__device__ float block_reduce_sum(float val, float* smem, int tid) {
    // Intra-wavefront reduction
    val += __shfl_xor(val, 1, 64);
    val += __shfl_xor(val, 2, 64);
    val += __shfl_xor(val, 4, 64);
    val += __shfl_xor(val, 8, 64);
    val += __shfl_xor(val, 16, 64);
    val += __shfl_xor(val, 32, 64);
    
    // Only need wave0's result for scores (all waves compute same Q·K)
    return val;
}

__global__ __launch_bounds__(256)
void fp4_pa_v4_kernel(
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
    const int wave_id = tid / 64;
    const int lane_id = tid % 64;
    const int kv_head_idx = head_idx * num_kv_heads / num_heads;
    const int ctx_len = context_lens[seq_idx];
    
    // Shared memory for KV tile: [TILE_S][HALF_DIM] floats for K, same for V
    // Plus scales: [TILE_S][SCALE_DIM]
    __shared__ float s_k[TILE_S][HALF_DIM * 2];  // Unpacked K: [TILE_S][HEAD_DIM]
    __shared__ float s_v[TILE_S][HALF_DIM * 2];  // Unpacked V: [TILE_S][HEAD_DIM]
    
    // Load Q to registers (all threads have same Q)
    const int q_base = seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    // Each thread loads different Q elements for the dot product
    // lane_id maps to elements [2*lane_id, 2*lane_id+1]
    const int my_elem0 = lane_id * 2;
    const int my_elem1 = lane_id * 2 + 1;
    float q0 = (my_elem0 < HEAD_DIM) ? __half2float(Q[q_base + my_elem0]) : 0.0f;
    float q1 = (my_elem1 < HEAD_DIM) ? __half2float(Q[q_base + my_elem1]) : 0.0f;
    
    // Online softmax per thread (each thread accumulates its own V elements)
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;
    
    const int kv_pos_stride = num_kv_heads * HALF_DIM;
    const int ks_pos_stride = num_kv_heads * SCALE_DIM;
    
    for (int bi = 0; bi < max_num_blocks; bi++) {
        const int start_pos = bi * block_size;
        if (start_pos >= ctx_len) break;
        
        const int phys_block = block_tables[seq_idx * max_num_blocks + bi];
        const int kv_base = phys_block * block_size * kv_pos_stride + kv_head_idx * HALF_DIM;
        const int ks_base = phys_block * block_size * ks_pos_stride + kv_head_idx * SCALE_DIM;
        
        // Process block_size positions in tiles of TILE_S
        for (int s_tile = 0; s_tile < block_size; s_tile += TILE_S) {
            
            // === Phase 1: Cooperative KV load + dequant into shared memory ===
            // 256 threads load TILE_S × HALF_DIM bytes = 16 × 64 = 1024 bytes total
            // Each thread loads 1024/256 = 4 bytes = 8 FP4 values
            const int total_bytes = TILE_S * HALF_DIM;
            const int bytes_per_thread = (total_bytes + BLOCK_SIZE_T - 1) / BLOCK_SIZE_T;
            
            for (int b = 0; b < bytes_per_thread; b++) {
                const int flat_idx = tid * bytes_per_thread + b;
                if (flat_idx >= total_bytes) break;
                
                const int s_local = flat_idx / HALF_DIM;
                const int byte_idx = flat_idx % HALF_DIM;
                const int s = s_tile + s_local;
                const int pos = start_pos + s;
                
                if (s < block_size && pos < ctx_len) {
                    // Load K byte, dequant to 2 floats
                    uint8_t k_packed = K_cache[kv_base + s * kv_pos_stride + byte_idx];
                    float k0 = fp4_dequant(k_packed & 0x0F);
                    float k1 = fp4_dequant((k_packed >> 4) & 0x0F);
                    
                    // Apply E8M0 scale
                    int scale_group = (byte_idx * 2) / 32;
                    float ks = exp2f((float)K_scale[ks_base + s * ks_pos_stride + scale_group] - 127.0f);
                    s_k[s_local][byte_idx * 2] = k0 * ks;
                    s_k[s_local][byte_idx * 2 + 1] = k1 * ks;
                    
                    // Load V byte, dequant
                    uint8_t v_packed = V_cache[kv_base + s * kv_pos_stride + byte_idx];
                    float v0 = fp4_dequant(v_packed & 0x0F);
                    float v1 = fp4_dequant((v_packed >> 4) & 0x0F);
                    float vs = exp2f((float)V_scale[ks_base + s * ks_pos_stride + scale_group] - 127.0f);
                    s_v[s_local][byte_idx * 2] = v0 * vs;
                    s_v[s_local][byte_idx * 2 + 1] = v1 * vs;
                } else {
                    s_k[s_local][byte_idx * 2] = 0.0f;
                    s_k[s_local][byte_idx * 2 + 1] = 0.0f;
                    s_v[s_local][byte_idx * 2] = 0.0f;
                    s_v[s_local][byte_idx * 2 + 1] = 0.0f;
                }
            }
            __syncthreads();
            
            // === Phase 2: Q·K scores + softmax + V accumulation ===
            // Each lane computes partial dot product from its Q elements
            for (int si = 0; si < TILE_S; si++) {
                const int pos = start_pos + s_tile + si;
                const bool valid = (s_tile + si < block_size) && (pos < ctx_len);
                
                // Read K from shared memory (my elements)
                float k0 = s_k[si][my_elem0];
                float k1 = (my_elem1 < HEAD_DIM) ? s_k[si][my_elem1] : 0.0f;
                
                // Partial dot product
                float partial = q0 * k0 + q1 * k1;
                
                // Wave-level reduction for score
                partial += __shfl_xor(partial, 1, 64);
                partial += __shfl_xor(partial, 2, 64);
                partial += __shfl_xor(partial, 4, 64);
                partial += __shfl_xor(partial, 8, 64);
                partial += __shfl_xor(partial, 16, 64);
                partial += __shfl_xor(partial, 32, 64);
                float score = valid ? partial * sm_scale : -INFINITY;
                
                // Online softmax
                float m_new = fmaxf(m_i, score);
                float alpha = expf(m_i - m_new);
                float beta = expf(score - m_new);
                float l_new = l_i * alpha + beta;
                
                // Read V from shared memory
                float v0 = s_v[si][my_elem0];
                float v1 = (my_elem1 < HEAD_DIM) ? s_v[si][my_elem1] : 0.0f;
                
                float inv_l = 1.0f / fmaxf(l_new, 1e-10f);
                acc0 = (l_i * alpha * acc0 + beta * v0) * inv_l;
                acc1 = (l_i * alpha * acc1 + beta * v1) * inv_l;
                
                m_i = m_new;
                l_i = l_new;
            }
            __syncthreads();
        }
    }
    
    // Store output — only wave 0 writes (all waves compute same result)
    if (wave_id == 0) {
        const int out_base = seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
        if (my_elem0 < HEAD_DIM) Out[out_base + my_elem0] = __float2half(acc0);
        if (my_elem1 < HEAD_DIM) Out[out_base + my_elem1] = __float2half(acc1);
    }
}


extern "C" void launch_fp4_pa_v4(
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
    
    fp4_pa_v4_kernel<<<grid, block, 0, stream>>>(
        (const half*)Q, (const uint8_t*)K_cache, (const uint8_t*)V_cache,
        (const uint8_t*)K_scale, (const uint8_t*)V_scale,
        (const int*)block_tables, (const int*)context_lens, (half*)Out,
        num_heads, num_kv_heads, block_size, max_num_blocks, scale
    );
}
