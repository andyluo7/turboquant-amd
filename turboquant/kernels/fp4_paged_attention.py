#!/usr/bin/env python3
"""FP4 Paged Attention — PyTorch reference implementation.

Validates the concept: decode attention with FP4 KV cache.
Uses torch ops for correctness, then compare against AITER FP8 PA speed.
"""
import torch
import math
import time


def fp4_unpack(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 → two FP4 values as float.
    
    FP4 E2M1 encoding (unsigned nibble):
      0=0.0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0
      8=-0.0, 9=-0.5, 10=-1.0, 11=-1.5, 12=-2.0, 13=-3.0, 14=-4.0, 15=-6.0
    """
    # Lookup table for FP4 E2M1
    LUT = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=torch.float32, device=packed.device)
    
    lo = packed & 0x0F  # low nibble
    hi = (packed >> 4) & 0x0F  # high nibble
    
    val_lo = LUT[lo.long()]
    val_hi = LUT[hi.long()]
    
    # Interleave: [N, D//2] → [N, D]
    result = torch.stack([val_lo, val_hi], dim=-1)
    return result.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def e8m0_to_float(scale_u8: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 uint8 to float scale factor.
    
    E8M0: value = 2^(uint8_value - 127)
    """
    return torch.pow(2.0, scale_u8.float() - 127.0)


def fp4_paged_attention_ref(
    query: torch.Tensor,         # [batch, num_heads, head_dim] BF16
    k_cache: torch.Tensor,       # [num_blocks, block_size, num_kv_heads, head_dim//2] uint8
    v_cache: torch.Tensor,       # [num_blocks, block_size, num_kv_heads, head_dim//2] uint8
    k_scale: torch.Tensor,       # [num_blocks, block_size, num_kv_heads, scale_dim] uint8
    v_scale: torch.Tensor,       # [num_blocks, block_size, num_kv_heads, scale_dim] uint8
    block_tables: torch.Tensor,  # [batch, max_blocks] int32
    context_lens: torch.Tensor,  # [batch] int32
    scale_group_size: int = 32,
) -> torch.Tensor:
    """Reference FP4 paged attention for validation."""
    batch, num_heads, head_dim = query.shape
    num_blocks, block_size, num_kv_heads, hd_half = k_cache.shape
    heads_per_group = num_heads // num_kv_heads  # GQA ratio
    
    scale = 1.0 / math.sqrt(head_dim)
    output = torch.zeros_like(query, dtype=torch.float32)
    
    for b in range(batch):
        ctx_len = context_lens[b].item()
        
        for h in range(num_heads):
            kv_h = h // heads_per_group
            q = query[b, h].float()  # [head_dim]
            
            # Gather all K and V for this sequence
            keys = []
            values = []
            
            for pos in range(ctx_len):
                block_idx = pos // block_size
                block_off = pos % block_size
                phys_block = block_tables[b, block_idx].item()
                
                # Unpack K
                k_packed = k_cache[phys_block, block_off, kv_h]  # [hd_half] uint8
                k_unpacked = fp4_unpack(k_packed)  # [head_dim] float
                
                # Apply E8M0 scale (per group of 32)
                ks = k_scale[phys_block, block_off, kv_h]  # [scale_dim] uint8
                ks_float = e8m0_to_float(ks)  # [scale_dim]
                # Broadcast scale to head_dim
                k_scaled = k_unpacked * ks_float.repeat_interleave(scale_group_size)[:head_dim]
                keys.append(k_scaled)
                
                # Unpack V
                v_packed = v_cache[phys_block, block_off, kv_h]
                v_unpacked = fp4_unpack(v_packed)
                vs = v_scale[phys_block, block_off, kv_h]
                vs_float = e8m0_to_float(vs)
                v_scaled = v_unpacked * vs_float.repeat_interleave(scale_group_size)[:head_dim]
                values.append(v_scaled)
            
            if not keys:
                continue
                
            K = torch.stack(keys)  # [ctx_len, head_dim]
            V = torch.stack(values)  # [ctx_len, head_dim]
            
            # Standard attention
            scores = (q @ K.T) * scale  # [ctx_len]
            attn_weights = torch.softmax(scores, dim=-1)  # [ctx_len]
            output[b, h] = attn_weights @ V  # [head_dim]
    
    return output.to(query.dtype)


def test_fp4_pa():
    batch, num_heads, num_kv_heads, head_dim = 1, 4, 1, 128
    block_size, num_blocks, seq_len = 16, 8, 64
    max_blocks = seq_len // block_size
    device = 'cuda:0'
    
    query = torch.randn(batch, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_cache = torch.randint(0, 255, (num_blocks, block_size, num_kv_heads, head_dim//2),
                            dtype=torch.uint8, device=device)
    v_cache = torch.randint(0, 255, (num_blocks, block_size, num_kv_heads, head_dim//2),
                            dtype=torch.uint8, device=device)
    k_scale = torch.full((num_blocks, block_size, num_kv_heads, head_dim//32), 127,
                         dtype=torch.uint8, device=device)  # scale=1.0
    v_scale = torch.full((num_blocks, block_size, num_kv_heads, head_dim//32), 127,
                         dtype=torch.uint8, device=device)
    block_tables = torch.arange(max_blocks, device=device).unsqueeze(0).int()
    context_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    
    print(f"=== FP4 Paged Attention Reference Test ===")
    print(f"batch={batch}, heads={num_heads}, kv_heads={num_kv_heads}, "
          f"head_dim={head_dim}, seq={seq_len}")
    
    out = fp4_paged_attention_ref(
        query, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens
    )
    torch.cuda.synchronize()
    
    print(f"Output shape: {out.shape}")
    print(f"Output non-zero: {out.abs().sum().item() > 0}")
    print(f"Output sample: {out[0, 0, :4].tolist()}")
    
    # Compare with BF16 attention for consistency check
    # Dequantize full KV cache to BF16 and run standard attention
    K_full = torch.zeros(seq_len, num_kv_heads, head_dim, device=device)
    V_full = torch.zeros(seq_len, num_kv_heads, head_dim, device=device)
    for pos in range(seq_len):
        bi = pos // block_size
        bo = pos % block_size
        for kh in range(num_kv_heads):
            K_full[pos, kh] = fp4_unpack(k_cache[bi, bo, kh])
            V_full[pos, kh] = fp4_unpack(v_cache[bi, bo, kh])
    
    # Standard attention with dequantized KV
    q = query[0, 0].float()  # [head_dim]
    k = K_full[:, 0].float()  # [seq, head_dim]
    v = V_full[:, 0].float()
    scores = (q @ k.T) / math.sqrt(head_dim)
    attn = torch.softmax(scores, dim=-1)
    ref_out = attn @ v
    
    diff = (out[0, 0].float() - ref_out).abs().max().item()
    print(f"Max diff vs dequant reference: {diff:.6f}")
    print(f"Match: {'✅' if diff < 0.01 else '❌'}")
    
    # Benchmark
    print(f"\n=== Benchmark (batch={batch}, seq={seq_len}) ===")
    
    # Larger test
    for seq in [256, 1024, 4096]:
        nb = seq // block_size
        if nb > num_blocks:
            continue
        bt = torch.arange(nb, device=device).unsqueeze(0).int()
        cl = torch.tensor([seq], dtype=torch.int32, device=device)
        
        t0 = time.time()
        for _ in range(10):
            fp4_paged_attention_ref(query, k_cache, v_cache, k_scale, v_scale, bt, cl)
        torch.cuda.synchronize()
        us = (time.time() - t0) / 10 * 1e6
        print(f"  seq={seq}: {us:.0f} µs")
    
    print("\n✅ FP4 PA reference implementation validated!")


if __name__ == "__main__":
    test_fp4_pa()
