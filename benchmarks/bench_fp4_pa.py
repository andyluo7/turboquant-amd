import torch, ctypes, math, time
dev = "cuda:0"

lib9 = ctypes.CDLL("/tmp/fp4_pa_v9.so")
lib8 = ctypes.CDLL("/tmp/fp4_pa_v8.so")

def bench_v9(B, NH, NKV, HD, BS, SEQ, num_splits=4):
    NB = max(SEQ//BS+4,16); MB = SEQ//BS
    q=torch.randn(B,NH,HD,dtype=torch.float16,device=dev)
    kc=torch.randint(0,255,(NB,BS,NKV,HD//2),dtype=torch.uint8,device=dev)
    vc=torch.randint(0,255,(NB,BS,NKV,HD//2),dtype=torch.uint8,device=dev)
    ks=torch.full((NB,BS,NKV,HD//32),127,dtype=torch.uint8,device=dev)
    vs=torch.full((NB,BS,NKV,HD//32),127,dtype=torch.uint8,device=dev)
    bt=torch.arange(MB,device=dev).unsqueeze(0).expand(B,-1).contiguous().int()
    cl=torch.full((B,),SEQ,dtype=torch.int32,device=dev)
    out=torch.empty_like(q)
    # Workspace
    ws_m = torch.empty(B*NH*num_splits, dtype=torch.float32, device=dev)
    ws_l = torch.empty(B*NH*num_splits, dtype=torch.float32, device=dev)
    ws_acc = torch.empty(B*NH*num_splits*HD, dtype=torch.float32, device=dev)
    s=torch.cuda.current_stream().cuda_stream
    a=[ctypes.c_void_p(t.data_ptr()) for t in [q,kc,vc,ks,vs,bt,cl,out,ws_m,ws_l,ws_acc]]
    a+=[B,NH,NKV,HD,BS,MB,ctypes.c_float(1.0/math.sqrt(HD)),num_splits,ctypes.c_void_p(s)]
    for _ in range(10): lib9.launch_fp4_pa_v9(*a)
    torch.cuda.synchronize()
    t0=time.time()
    for _ in range(200): lib9.launch_fp4_pa_v9(*a)
    torch.cuda.synchronize()
    return (time.time()-t0)/200*1e6, out.abs().sum().item()>0

def bench_v8(B, NH, NKV, HD, BS, SEQ):
    NB = max(SEQ//BS+4,16); MB = SEQ//BS
    q=torch.randn(B,NH,HD,dtype=torch.float16,device=dev)
    kc=torch.randint(0,255,(NB,BS,NKV,HD//2),dtype=torch.uint8,device=dev)
    vc=torch.randint(0,255,(NB,BS,NKV,HD//2),dtype=torch.uint8,device=dev)
    ks=torch.full((NB,BS,NKV,HD//32),127,dtype=torch.uint8,device=dev)
    vs=torch.full((NB,BS,NKV,HD//32),127,dtype=torch.uint8,device=dev)
    bt=torch.arange(MB,device=dev).unsqueeze(0).expand(B,-1).contiguous().int()
    cl=torch.full((B,),SEQ,dtype=torch.int32,device=dev)
    out=torch.empty_like(q)
    s=torch.cuda.current_stream().cuda_stream
    a=[ctypes.c_void_p(t.data_ptr()) for t in [q,kc,vc,ks,vs,bt,cl,out]]
    a+=[B,NH,NKV,HD,BS,MB,ctypes.c_float(1.0/math.sqrt(HD)),ctypes.c_void_p(s)]
    for _ in range(10): lib8.launch_fp4_pa_v8(*a)
    torch.cuda.synchronize()
    t0=time.time()
    for _ in range(200): lib8.launch_fp4_pa_v8(*a)
    torch.cuda.synchronize()
    return (time.time()-t0)/200*1e6, out.abs().sum().item()>0

print("=== v9 (grid split-K) vs v8 (16-wave) ===")
for SEQ in [256, 1024, 4096]:
    u8,_ = bench_v8(1,16,2,128,16,SEQ)
    for ns in [2, 4, 8]:
        u9,n9 = bench_v9(1,16,2,128,16,SEQ,ns)
        print(f"  S={SEQ:5d} splits={ns}: v8={u8:7.1f}us  v9={u9:7.1f}us  {u8/u9:.2f}x  nz={n9}")
