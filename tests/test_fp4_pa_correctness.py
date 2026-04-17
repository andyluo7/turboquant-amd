#!/usr/bin/env python3
"""Verify FP4 PA v9 kernel correctness against PyTorch reference."""
import torch
import ctypes
import math
import numpy as np

dev = "cuda:0"

# === PyTorch Reference (from fp4_pa_reference.py) ===

def fp4_unpack(packed):
    """Unpack uint8 → two FP4 E2M1 values."""
    LUT = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=torch.float32, device=packed.device)
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    val_lo = LUT[lo.long()]
    val_hi = LUT[hi.long()]
    result = torch.stack([val_lo, val_hi], dim=-1)
    return result.reshape(*packed.shape[:-1], packed.shape[-1] * 2)

def e8m0_to_float(scale_u8):
    return torch.pow(2.0, scale_u8.float() - 127.0)

def fp4_pa_reference(query, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens):
    batch, num_heads, head_dim = query.shape
    _, block_size, num_kv_heads, hd_half = k_cache.shape
    heads_per_group = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    output = torch.zeros_like(query, dtype=torch.float32)
    
    for b in range(batch):
        ctx_len = context_lens[b].item()
        for h in range(num_heads):
            kv_h = h // heads_per_group
            q = query[b, h].float()
            keys, values = [], []
            for pos in range(ctx_len):
                bi = pos // block_size
                bo = pos % block_size
                pb = block_tables[b, bi].item()
                
                k_packed = k_cache[pb, bo, kv_h]
                k_unpacked = fp4_unpack(k_packed)
                ks = k_scale[pb, bo, kv_h]
                ks_float = e8m0_to_float(ks)
                k_scaled = k_unpacked * ks_float.repeat_interleave(32)[:head_dim]
                keys.append(k_scaled)
                
                v_packed = v_cache[pb, bo, kv_h]
                v_unpacked = fp4_unpack(v_packed)
                vs = v_scale[pb, bo, kv_h]
                vs_float = e8m0_to_float(vs)
                v_scaled = v_unpacked * vs_float.repeat_interleave(32)[:head_dim]
                values.append(v_scaled)
            
            if not keys:
                continue
            K = torch.stack(keys)
            V = torch.stack(values)
            scores = (q @ K.T) * scale
            attn = torch.softmax(scores, dim=-1)
            output[b, h] = attn @ V
    
    return output.to(query.dtype)

# === HIP Kernel ===
lib = ctypes.CDLL("/tmp/fp4_pa_v9.so")

def fp4_pa_hip(query, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens, nsplits=4):
    batch, num_heads, head_dim = query.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    max_blocks = block_tables.shape[1]
    output = torch.empty_like(query)
    wm = torch.empty(batch*num_heads*nsplits, dtype=torch.float32, device=dev)
    wl = torch.empty(batch*num_heads*nsplits, dtype=torch.float32, device=dev)
    wa = torch.empty(batch*num_heads*nsplits*head_dim, dtype=torch.float32, device=dev)
    s = torch.cuda.current_stream().cuda_stream
    a = [ctypes.c_void_p(t.data_ptr()) for t in [query, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens, output, wm, wl, wa]]
    a += [batch, num_heads, num_kv_heads, head_dim, block_size, max_blocks,
          ctypes.c_float(1.0/math.sqrt(head_dim)), nsplits, ctypes.c_void_p(s)]
    lib.launch_fp4_pa_v9(*a)
    torch.cuda.synchronize()
    return output

# === Test ===
print("=== FP4 PA v9 Correctness Verification ===\n")

configs = [
    (1, 4, 1, 128, 16, 32, "Small: B=1 H=4 S=32"),
    (1, 4, 1, 128, 16, 64, "Medium: B=1 H=4 S=64"),
    (1, 16, 2, 128, 16, 128, "GQA: B=1 H=16 KVH=2 S=128"),
    (2, 4, 1, 128, 16, 64, "Batch: B=2 H=4 S=64"),
    (1, 16, 2, 128, 16, 256, "Large: B=1 H=16 S=256"),
]

all_pass = True
for batch, nh, nkv, hd, bs, seq, label in configs:
    torch.manual_seed(42)
    nb = max(seq//bs + 4, 16)
    mb = seq // bs
    
    query = torch.randn(batch, nh, hd, dtype=torch.float16, device=dev)
    k_cache = torch.randint(0, 255, (nb, bs, nkv, hd//2), dtype=torch.uint8, device=dev)
    v_cache = torch.randint(0, 255, (nb, bs, nkv, hd//2), dtype=torch.uint8, device=dev)
    k_scale = torch.full((nb, bs, nkv, hd//32), 127, dtype=torch.uint8, device=dev)  # scale=1.0
    v_scale = torch.full((nb, bs, nkv, hd//32), 127, dtype=torch.uint8, device=dev)
    block_tables = torch.arange(mb, device=dev).unsqueeze(0).expand(batch, -1).contiguous().int()
    context_lens = torch.full((batch,), seq, dtype=torch.int32, device=dev)
    
    # Reference
    ref = fp4_pa_reference(query, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens)
    
    # HIP kernel
    hip = fp4_pa_hip(query, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens)
    
    # Compare
    ref_f = ref.float()
    hip_f = hip.float()
    
    max_diff = (ref_f - hip_f).abs().max().item()
    mean_diff = (ref_f - hip_f).abs().mean().item()
    
    # Cosine similarity
    cos = torch.nn.functional.cosine_similarity(
        ref_f.reshape(-1).unsqueeze(0),
        hip_f.reshape(-1).unsqueeze(0)
    ).item()
    
    # Relative error
    ref_norm = ref_f.norm().item()
    rel_err = (ref_f - hip_f).norm().item() / max(ref_norm, 1e-10) * 100
    
    status = "✅" if cos > 0.99 and rel_err < 5.0 else "❌"
    if status == "❌":
        all_pass = False
    
    print(f"{label}")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Cosine sim: {cos:.6f}")
    print(f"  Rel error: {rel_err:.2f}%")
    print(f"  Status: {status}")
    print()

print(f"{'='*50}")
print(f"Overall: {'ALL PASS ✅' if all_pass else 'SOME FAILED ❌'}")
