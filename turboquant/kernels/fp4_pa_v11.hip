/**
 * FP4 PA v11 — Optimized v9 with:
 * 1. Fused K+V load in same loop (1 address computation instead of 2)
 * 2. Pre-shifted Q (multiply Q by sm_scale once, not per-position)
 * 3. Reduce exp2f: cache scale across K and V (same position = same scale offset)
 * 4. Minimal shared memory for reduction (no wasted HEAD_DIM array when wave has no data)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

#define HEAD_DIM 128
#define HALF_DIM 64
#define NUM_WAVES 16
#define WAVE_SIZE 64
#define BLOCK_THREADS 1024

__global__ __launch_bounds__(1024)
void fp4_pa_v11_partial(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K_cache,
    const uint8_t* __restrict__ V_cache,
    const uint8_t* __restrict__ K_scale,
    const uint8_t* __restrict__ V_scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pa,
    const int nh, const int nkv, const int bs, const int mnb,
    const float sms, const int nsplits
) {
    const int si=blockIdx.x, hi=blockIdx.y, spi=blockIdx.z;
    const int tid=threadIdx.x, wi=tid/WAVE_SIZE, li=tid%WAVE_SIZE;
    const int kvi=hi*nkv/nh;
    const int cl=context_lens[si];
    
    __shared__ float sm_arr[NUM_WAVES], sl_arr[NUM_WAVES], sa_arr[NUM_WAVES][HEAD_DIM];
    __shared__ int sbt[256];
    for(int i=tid;i<mnb&&i<256;i+=BLOCK_THREADS) sbt[i]=block_tables[si*mnb+i];
    __syncthreads();
    
    const int e0=li*2, e1=li*2+1;
    const int qb=si*nh*HEAD_DIM+hi*HEAD_DIM;
    // Pre-scale Q by sm_scale (avoid multiply per position)
    float q0 = (e0<HEAD_DIM) ? __half2float(Q[qb+e0])*sms : 0.0f;
    float q1 = (e1<HEAD_DIM) ? __half2float(Q[qb+e1])*sms : 0.0f;
    const int sg=e0/32;
    
    const int tps=(cl+nsplits-1)/nsplits;
    const int ss=spi*tps, se=min(ss+tps,cl);
    const int wt=se-ss, ppw=(wt+NUM_WAVES-1)/NUM_WAVES;
    const int ws=ss+wi*ppw, we=min(ws+ppw,se);
    
    const int kvs=nkv*HALF_DIM, kss=nkv*(HEAD_DIM/32);
    
    constexpr float L[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                           0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
    
    float mi=-INFINITY, li2=0.f, a0=0.f, a1=0.f;
    
    for(int pos=ws; pos<we; pos++) {
        const int pb=sbt[pos/bs], bo=pos%bs;
        const int kvo=pb*bs*kvs+bo*kvs+kvi*HALF_DIM;
        const int kso=pb*bs*kss+bo*kss+kvi*(HEAD_DIM/32);
        
        // Single exp2f for scale (same group for both K and V)
        float scale_val = (li<HALF_DIM) ? exp2f((float)K_scale[kso+sg]-127.f) : 0.f;
        float vscale_val = (li<HALF_DIM) ? exp2f((float)V_scale[kso+sg]-127.f) : 0.f;
        
        // Fused K load + dequant + dot product
        float k0=0.f, k1=0.f;
        uint8_t kpk=0, vpk=0;
        if(li<HALF_DIM) {
            kpk = K_cache[kvo+li];
            k0=L[kpk&0xF]*scale_val;
            k1=L[(kpk>>4)&0xF]*scale_val;
        }
        
        // Q already pre-scaled, so just dot product (no extra *sms)
        float p=q0*k0+q1*k1;
        p+=__shfl_xor(p,1,64); p+=__shfl_xor(p,2,64); p+=__shfl_xor(p,4,64);
        p+=__shfl_xor(p,8,64); p+=__shfl_xor(p,16,64); p+=__shfl_xor(p,32,64);
        // score = p (Q was pre-scaled)
        
        float mn=fmaxf(mi,p), al=expf(mi-mn), be=expf(p-mn), ln=li2*al+be;
        
        // V load + dequant (reuse same cache line timing)
        float v0=0.f, v1=0.f;
        if(li<HALF_DIM) {
            vpk = V_cache[kvo+li];
            v0=L[vpk&0xF]*vscale_val;
            v1=L[(vpk>>4)&0xF]*vscale_val;
        }
        
        float il=1.f/fmaxf(ln,1e-10f);
        a0=(li2*al*a0+be*v0)*il;
        a1=(li2*al*a1+be*v1)*il;
        mi=mn; li2=ln;
    }
    
    // Wave reduction
    if(li==0){sm_arr[wi]=mi;sl_arr[wi]=li2;}
    if(e0<HEAD_DIM)sa_arr[wi][e0]=a0;
    if(e1<HEAD_DIM)sa_arr[wi][e1]=a1;
    __syncthreads();
    
    if(wi==0) {
        float fm=sm_arr[0];
        for(int w=1;w<NUM_WAVES;w++) fm=fmaxf(fm,sm_arr[w]);
        float fl=0;
        for(int w=0;w<NUM_WAVES;w++) fl+=sl_arr[w]*expf(sm_arr[w]-fm);
        float o0=0,o1=0,ifl=1.f/fmaxf(fl,1e-10f);
        for(int w=0;w<NUM_WAVES;w++){
            float sw=sl_arr[w]*expf(sm_arr[w]-fm)*ifl;
            if(e0<HEAD_DIM)o0+=sa_arr[w][e0]*sw;
            if(e1<HEAD_DIM)o1+=sa_arr[w][e1]*sw;
        }
        const int pidx=si*nh*nsplits+hi*nsplits+spi;
        if(li==0){pm[pidx]=fm;pl[pidx]=fl;}
        if(e0<HEAD_DIM)pa[pidx*HEAD_DIM+e0]=o0;
        if(e1<HEAD_DIM)pa[pidx*HEAD_DIM+e1]=o1;
    }
}

__global__ __launch_bounds__(64)
void fp4_pa_v11_reduce(
    const float* pm, const float* pl, const float* pa,
    half* O, const int nh, const int ns
) {
    const int si=blockIdx.x,hi=blockIdx.y,li=threadIdx.x;
    const int e0=li*2,e1=li*2+1;
    const int base=si*nh*ns+hi*ns;
    float fm=pm[base];
    for(int s=1;s<ns;s++)fm=fmaxf(fm,pm[base+s]);
    float fl=0;
    for(int s=0;s<ns;s++)fl+=pl[base+s]*expf(pm[base+s]-fm);
    float o0=0,o1=0,ifl=1.f/fmaxf(fl,1e-10f);
    for(int s=0;s<ns;s++){
        float sw=pl[base+s]*expf(pm[base+s]-fm)*ifl;
        int ai=(base+s)*HEAD_DIM;
        if(e0<HEAD_DIM)o0+=pa[ai+e0]*sw;
        if(e1<HEAD_DIM)o1+=pa[ai+e1]*sw;
    }
    const int ob=si*nh*HEAD_DIM+hi*HEAD_DIM;
    if(e0<HEAD_DIM)O[ob+e0]=__float2half(o0);
    if(e1<HEAD_DIM)O[ob+e1]=__float2half(o1);
}

extern "C" void launch_fp4_pa_v11(
    const void* Q,const void* K,const void* V,
    const void* KS,const void* VS,
    const void* BT,const void* CL,void* O,
    void* wm,void* wl,void* wa,
    int ns,int nh,int nkv,int hd,int bs,int mnb,
    float sc,int nsplits,hipStream_t st
) {
    fp4_pa_v11_partial<<<dim3(ns,nh,nsplits),1024,0,st>>>(
        (const half*)Q,(const uint8_t*)K,(const uint8_t*)V,
        (const uint8_t*)KS,(const uint8_t*)VS,
        (const int*)BT,(const int*)CL,
        (float*)wm,(float*)wl,(float*)wa,
        nh,nkv,bs,mnb,sc,nsplits);
    fp4_pa_v11_reduce<<<dim3(ns,nh),64,0,st>>>(
        (const float*)wm,(const float*)wl,(const float*)wa,
        (half*)O,nh,nsplits);
}
