#!/usr/bin/env python3
"""Test the FP4 PA HIP kernel on MI355X."""
import torch
import ctypes
import math
import time
import os

# Load the compiled kernel
lib = ctypes.CDLL("/tmp/fp4_pa.so")

def fp4_pa_hip(query, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens):
    batch, num_heads, head_dim = query.shape
    _, block_size, num_kv_heads, hd_half = k_cache.shape
    max_blocks = block_tables.shape[1]
    scale = 1.0 / math.sqrt(head_dim)
    
    output = torch.empty_like(query)
    
    stream = torch.cuda.current_stream().cuda_stream
    
    lib.launch_fp4_paged_attention(
        ctypes.c_void_p(query.data_ptr()),
        ctypes.c_void_p(k_cache.data_ptr()),
        ctypes.c_void_p(v_cache.data_ptr()),
        ctypes.c_void_p(k_scale.data_ptr()),
        ctypes.c_void_p(v_scale.data_ptr()),
        ctypes.c_void_p(block_tables.data_ptr()),
        ctypes.c_void_p(context_lens.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(batch),
        ctypes.c_int(num_heads),
        ctypes.c_int(num_kv_heads),
        ctypes.c_int(head_dim),
        ctypes.c_int(block_size),
        ctypes.c_int(max_blocks),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    torch.cuda.synchronize()
    return output


def test():
    device = 'cuda:0'
    print("=== FP4 PA HIP Kernel Test ===")
    
    configs = [
        (1, 16, 2, 128, 16, 256),
        (4, 16, 2, 128, 16, 256),
        (1, 16, 2, 128, 16, 1024),
        (8, 32, 4, 128, 16, 512),
    ]
    
    for batch, nh, nkv, hd, bs, seq in configs:
        nb = 64
        mb = seq // bs
        
        query = torch.randn(batch, nh, hd, dtype=torch.float16, device=device)
        k_cache = torch.randint(0, 255, (nb, bs, nkv, hd//2), dtype=torch.uint8, device=device)
        v_cache = torch.randint(0, 255, (nb, bs, nkv, hd//2), dtype=torch.uint8, device=device)
        k_scale = torch.full((nb, bs, nkv, hd//32), 127, dtype=torch.uint8, device=device)
        v_scale = torch.full((nb, bs, nkv, hd//32), 127, dtype=torch.uint8, device=device)
        bt = torch.arange(mb, device=device).unsqueeze(0).expand(batch, -1).contiguous().int()
        cl = torch.full((batch,), seq, dtype=torch.int32, device=device)
        
        out = fp4_pa_hip(query, k_cache, v_cache, k_scale, v_scale, bt, cl)
        nz = out.abs().sum().item() > 0
        
        # Benchmark
        for _ in range(5):
            fp4_pa_hip(query, k_cache, v_cache, k_scale, v_scale, bt, cl)
        torch.cuda.synchronize()
        
        t0 = time.time()
        iters = 100
        for _ in range(iters):
            fp4_pa_hip(query, k_cache, v_cache, k_scale, v_scale, bt, cl)
        torch.cuda.synchronize()
        us = (time.time() - t0) / iters * 1e6
        
        print(f"  B={batch} H={nh} KVH={nkv} S={seq}: {us:.1f}µs, nz={nz}")
    
    # SDPA reference
    print("\n=== BF16 SDPA Reference ===")
    for batch, nh, hd, seq in [(1, 16, 128, 256), (1, 16, 128, 1024)]:
        q = torch.randn(batch, nh, 1, hd, dtype=torch.float16, device=device)
        k = torch.randn(batch, nh, seq, hd, dtype=torch.float16, device=device)
        v = torch.randn(batch, nh, seq, hd, dtype=torch.float16, device=device)
        for _ in range(10):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        us = (time.time() - t0) / 100 * 1e6
        print(f"  SDPA B={batch} H={nh} S={seq}: {us:.1f}µs")
    
    print("\n✅ FP4 PA HIP kernel test complete!")


if __name__ == "__main__":
    test()
