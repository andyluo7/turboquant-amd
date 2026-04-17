import torch, ctypes, math, time
dev = "cuda:0"
lib = ctypes.CDLL("/tmp/fp4_pa_v9.so")

def bench(B,NH,NKV,HD,BS,SEQ,ns):
    NB=max(SEQ//BS+4,16);MB=SEQ//BS
    q=torch.randn(B,NH,HD,dtype=torch.float16,device=dev)
    kc=torch.randint(0,255,(NB,BS,NKV,HD//2),dtype=torch.uint8,device=dev)
    vc=torch.randint(0,255,(NB,BS,NKV,HD//2),dtype=torch.uint8,device=dev)
    ks=torch.full((NB,BS,NKV,HD//32),127,dtype=torch.uint8,device=dev)
    vs=torch.full((NB,BS,NKV,HD//32),127,dtype=torch.uint8,device=dev)
    bt=torch.arange(MB,device=dev).unsqueeze(0).expand(B,-1).contiguous().int()
    cl=torch.full((B,),SEQ,dtype=torch.int32,device=dev)
    out=torch.empty_like(q)
    wm=torch.empty(B*NH*ns,dtype=torch.float32,device=dev)
    wl=torch.empty(B*NH*ns,dtype=torch.float32,device=dev)
    wa=torch.empty(B*NH*ns*HD,dtype=torch.float32,device=dev)
    s=torch.cuda.current_stream().cuda_stream
    a=[ctypes.c_void_p(t.data_ptr()) for t in [q,kc,vc,ks,vs,bt,cl,out,wm,wl,wa]]
    a+=[B,NH,NKV,HD,BS,MB,ctypes.c_float(1.0/math.sqrt(HD)),ns,ctypes.c_void_p(s)]
    for _ in range(10):lib.launch_fp4_pa_v9(*a)
    torch.cuda.synchronize()
    t0=time.time()
    for _ in range(500):lib.launch_fp4_pa_v9(*a)
    torch.cuda.synchronize()
    return (time.time()-t0)/500*1e6

print("=== Optimal split count sweep (v9) ===")
for SEQ in [256, 512, 1024, 2048, 4096]:
    results = []
    for ns in [2, 4, 8, 12, 16, 24, 32]:
        u = bench(1,16,2,128,16,SEQ,ns)
        results.append((ns, u))
    best = min(results, key=lambda x: x[1])
    line = f"  S={SEQ:5d}: "
    for ns, u in results:
        marker = " *" if ns == best[0] else ""
        line += f"{ns}={u:.1f}{marker}  "
    print(line)
