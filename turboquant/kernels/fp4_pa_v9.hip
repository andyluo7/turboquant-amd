/**
 * FP4 PA v9 — Grid-level Split-K (multiple blocks per head)
 * 
 * Launch grid: (num_seqs, num_heads, NUM_SPLITS)
 * Each block handles ctx_len/NUM_SPLITS positions.
 * Partial results written to workspace, then reduced in a second kernel.
 * 
 * This bypasses the 1024 thread/block limit by using multiple blocks.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

#define HEAD_DIM 128
#define HALF_DIM 64
#define WAVE_SIZE 64
#define BLOCK_THREADS 1024  // 16 waves
#define NUM_WAVES_PER_BLOCK 16

// Kernel 1: Compute partial attention per split
__global__ __launch_bounds__(1024)
void fp4_pa_v9_partial(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K_cache,
    const uint8_t* __restrict__ V_cache,
    const uint8_t* __restrict__ K_scale,
    const uint8_t* __restrict__ V_scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    float* __restrict__ partial_m,     // [num_seqs, num_heads, num_splits]
    float* __restrict__ partial_l,     // [num_seqs, num_heads, num_splits]
    float* __restrict__ partial_acc,   // [num_seqs, num_heads, num_splits, HEAD_DIM]
    const int num_heads,
    const int num_kv_heads,
    const int block_size,
    const int max_num_blocks,
    const float sm_scale,
    const int num_splits
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int tid = threadIdx.x;
    const int wave_id = tid / WAVE_SIZE;
    const int lane_id = tid % WAVE_SIZE;
    const int kv_head_idx = head_idx * num_kv_heads / num_heads;
    const int ctx_len = context_lens[seq_idx];
    
    __shared__ float s_m[NUM_WAVES_PER_BLOCK];
    __shared__ float s_l[NUM_WAVES_PER_BLOCK];
    __shared__ float s_acc[NUM_WAVES_PER_BLOCK][HEAD_DIM];
    __shared__ int s_bt[256];
    
    for (int i = tid; i < max_num_blocks && i < 256; i += BLOCK_THREADS)
        s_bt[i] = block_tables[seq_idx * max_num_blocks + i];
    __syncthreads();
    
    const int my_e0 = lane_id * 2, my_e1 = lane_id * 2 + 1;
    const int q_base = seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    float q0 = (my_e0 < HEAD_DIM) ? __half2float(Q[q_base + my_e0]) : 0.0f;
    float q1 = (my_e1 < HEAD_DIM) ? __half2float(Q[q_base + my_e1]) : 0.0f;
    const int my_sg = my_e0 / 32;
    
    // Grid-level split: this block handles [split_start, split_end)
    const int total_per_split = (ctx_len + num_splits - 1) / num_splits;
    const int split_start = split_idx * total_per_split;
    const int split_end = min(split_start + total_per_split, ctx_len);
    
    // Intra-block wave split
    const int wave_total = split_end - split_start;
    const int ppw = (wave_total + NUM_WAVES_PER_BLOCK - 1) / NUM_WAVES_PER_BLOCK;
    const int w_start = split_start + wave_id * ppw;
    const int w_end = min(w_start + ppw, split_end);
    
    const int kv_ps = num_kv_heads * HALF_DIM;
    const int ks_ps = num_kv_heads * (HEAD_DIM/32);
    
    constexpr float L[16] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,
                              0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};
    
    float m_i = -INFINITY, l_i = 0.0f, a0 = 0.0f, a1 = 0.0f;
    
    for (int pos = w_start; pos < w_end; pos++) {
        const int pb = s_bt[pos / block_size];
        const int bo = pos % block_size;
        const int kv_off = pb*block_size*kv_ps + bo*kv_ps + kv_head_idx*HALF_DIM;
        const int ks_off = pb*block_size*ks_ps + bo*ks_ps + kv_head_idx*(HEAD_DIM/32);
        
        float ks_v = exp2f((float)K_scale[ks_off + my_sg] - 127.0f);
        float k0=0, k1=0;
        if (lane_id < HALF_DIM) {
            uint8_t pk = K_cache[kv_off + lane_id];
            k0 = L[pk&0xF]*ks_v; k1 = L[(pk>>4)&0xF]*ks_v;
        }
        float p = q0*k0+q1*k1;
        p+=__shfl_xor(p,1,64); p+=__shfl_xor(p,2,64); p+=__shfl_xor(p,4,64);
        p+=__shfl_xor(p,8,64); p+=__shfl_xor(p,16,64); p+=__shfl_xor(p,32,64);
        float sc = p * sm_scale;
        
        float mn=fmaxf(m_i,sc), al=expf(m_i-mn), be=expf(sc-mn), ln=l_i*al+be;
        
        float vs_v = exp2f((float)V_scale[ks_off + my_sg] - 127.0f);
        float v0=0, v1=0;
        if (lane_id < HALF_DIM) {
            uint8_t pv = V_cache[kv_off + lane_id];
            v0 = L[pv&0xF]*vs_v; v1 = L[(pv>>4)&0xF]*vs_v;
        }
        float il = 1.0f/fmaxf(ln,1e-10f);
        a0=(l_i*al*a0+be*v0)*il; a1=(l_i*al*a1+be*v1)*il;
        m_i=mn; l_i=ln;
    }
    
    // Intra-block wave reduction
    if (lane_id==0) { s_m[wave_id]=m_i; s_l[wave_id]=l_i; }
    if (my_e0<HEAD_DIM) s_acc[wave_id][my_e0]=a0;
    if (my_e1<HEAD_DIM) s_acc[wave_id][my_e1]=a1;
    __syncthreads();
    
    if (wave_id==0) {
        float fm=s_m[0];
        for(int w=1;w<NUM_WAVES_PER_BLOCK;w++) fm=fmaxf(fm,s_m[w]);
        float fl=0;
        for(int w=0;w<NUM_WAVES_PER_BLOCK;w++) fl+=s_l[w]*expf(s_m[w]-fm);
        float o0=0,o1=0,ifl=1.0f/fmaxf(fl,1e-10f);
        for(int w=0;w<NUM_WAVES_PER_BLOCK;w++) {
            float sw=s_l[w]*expf(s_m[w]-fm)*ifl;
            if(my_e0<HEAD_DIM) o0+=s_acc[w][my_e0]*sw;
            if(my_e1<HEAD_DIM) o1+=s_acc[w][my_e1]*sw;
        }
        
        // Write partial results
        const int pidx = seq_idx*num_heads*num_splits + head_idx*num_splits + split_idx;
        if (lane_id==0) { partial_m[pidx]=fm; partial_l[pidx]=fl; }
        if (my_e0<HEAD_DIM) partial_acc[pidx*HEAD_DIM+my_e0]=o0;
        if (my_e1<HEAD_DIM) partial_acc[pidx*HEAD_DIM+my_e1]=o1;
    }
}

// Kernel 2: Reduce partial results
__global__ __launch_bounds__(64)
void fp4_pa_v9_reduce(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    half* __restrict__ Out,
    const int num_heads,
    const int num_splits
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int lane_id = threadIdx.x;
    const int my_e0 = lane_id*2, my_e1 = lane_id*2+1;
    
    const int base = seq_idx*num_heads*num_splits + head_idx*num_splits;
    
    float fm = partial_m[base];
    for (int s=1; s<num_splits; s++) fm = fmaxf(fm, partial_m[base+s]);
    
    float fl = 0;
    for (int s=0; s<num_splits; s++) fl += partial_l[base+s]*expf(partial_m[base+s]-fm);
    
    float o0=0, o1=0, ifl=1.0f/fmaxf(fl,1e-10f);
    for (int s=0; s<num_splits; s++) {
        float sw = partial_l[base+s]*expf(partial_m[base+s]-fm)*ifl;
        int ai = (base+s)*HEAD_DIM;
        if (my_e0<HEAD_DIM) o0 += partial_acc[ai+my_e0]*sw;
        if (my_e1<HEAD_DIM) o1 += partial_acc[ai+my_e1]*sw;
    }
    
    const int ob = seq_idx*num_heads*HEAD_DIM + head_idx*HEAD_DIM;
    if (my_e0<HEAD_DIM) Out[ob+my_e0] = __float2half(o0);
    if (my_e1<HEAD_DIM) Out[ob+my_e1] = __float2half(o1);
}


extern "C" void launch_fp4_pa_v9(
    const void* Q, const void* K, const void* V,
    const void* KS, const void* VS,
    const void* BT, const void* CL, void* O,
    void* ws_m, void* ws_l, void* ws_acc,
    int ns, int nh, int nkv, int hd, int bs, int mnb,
    float sc, int num_splits, hipStream_t st
) {
    // Kernel 1: partial attention
    fp4_pa_v9_partial<<<dim3(ns,nh,num_splits), 1024, 0, st>>>(
        (const half*)Q,(const uint8_t*)K,(const uint8_t*)V,
        (const uint8_t*)KS,(const uint8_t*)VS,
        (const int*)BT,(const int*)CL,
        (float*)ws_m,(float*)ws_l,(float*)ws_acc,
        nh,nkv,bs,mnb,sc,num_splits);
    
    // Kernel 2: reduce
    fp4_pa_v9_reduce<<<dim3(ns,nh), 64, 0, st>>>(
        (const float*)ws_m,(const float*)ws_l,(const float*)ws_acc,
        (half*)O, nh, num_splits);
}
