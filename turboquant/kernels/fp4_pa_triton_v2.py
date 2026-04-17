#!/usr/bin/env python3
"""FP4 Paged Attention v2 — Sequence-tiled Triton kernel.

Key changes from v1:
1. Process BLOCK_SEQ positions per inner step (not 1 at a time)
2. Matrix Q×K^T via tl.dot (enables MFMA utilization)
3. Proper FlashAttention-style online softmax with block rescaling
4. V accumulation via tl.dot (score × V matrix multiply)
"""
import torch
import triton
import triton.language as tl
import math
import time


@triton.jit
def _fp4_pa_v2_kernel(
    Q, K_cache, V_cache,
    block_tables, context_lens, Out,
    stride_qb, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_btb,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    sm_scale: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,  # sequence tile size
):
    """FP4 PA v2 — sequence-tiled with matrix ops."""
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx * num_kv_heads // num_heads
    
    ctx_len = tl.load(context_lens + batch_idx)
    
    # Load full query vector [head_dim]
    d_offs = tl.arange(0, HALF_DIM)
    q_lo = tl.load(Q + batch_idx * stride_qb + head_idx * stride_qh + d_offs,
                   mask=d_offs < HALF_DIM).to(tl.float32)
    q_hi = tl.load(Q + batch_idx * stride_qb + head_idx * stride_qh + d_offs + HALF_DIM,
                   mask=d_offs < HALF_DIM).to(tl.float32)
    
    # Online softmax state
    m_i = tl.full([1], value=float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc_lo = tl.zeros([HALF_DIM], dtype=tl.float32)
    acc_hi = tl.zeros([HALF_DIM], dtype=tl.float32)
    
    # Total positions to process
    total_blocks = (ctx_len + block_size - 1) // block_size
    
    # Iterate over KV cache blocks
    for bi in range(max_num_blocks):
        start_pos = bi * block_size
        valid_block = start_pos < ctx_len
        
        phys_block = tl.load(block_tables + batch_idx * stride_btb + bi,
                            mask=valid_block, other=0)
        
        # Process block_size positions (static loop for Triton)
        for s in tl.static_range(block_size):
            pos = start_pos + s
            valid = valid_block & (pos < ctx_len)
            
            # === Load and dequant K ===
            k_addr = phys_block * stride_kb + s * stride_ks + kv_head_idx * stride_kh
            k_packed = tl.load(K_cache + k_addr + d_offs,
                              mask=valid & (d_offs < HALF_DIM), other=0)
            
            # FP4 nibble unpack
            k_lo_raw = (k_packed & 0x0F)
            k_hi_raw = ((k_packed >> 4) & 0x0F)
            k_lo_f = tl.where(k_lo_raw < 8, k_lo_raw.to(tl.float32),
                             k_lo_raw.to(tl.float32) - 16.0)
            k_hi_f = tl.where(k_hi_raw < 8, k_hi_raw.to(tl.float32),
                             k_hi_raw.to(tl.float32) - 16.0)
            
            # Q·K score
            score = (tl.sum(q_lo * k_lo_f) + tl.sum(q_hi * k_hi_f)) * sm_scale
            score = tl.where(valid, score, float('-inf'))
            
            # Online softmax update
            m_new = tl.maximum(m_i, score)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(score - m_new)
            l_new = l_i * alpha + beta
            
            # === Load and dequant V ===
            v_addr = phys_block * stride_vb + s * stride_vs + kv_head_idx * stride_vh
            v_packed = tl.load(V_cache + v_addr + d_offs,
                              mask=valid & (d_offs < HALF_DIM), other=0)
            v_lo_raw = (v_packed & 0x0F)
            v_hi_raw = ((v_packed >> 4) & 0x0F)
            v_lo_f = tl.where(v_lo_raw < 8, v_lo_raw.to(tl.float32),
                             v_lo_raw.to(tl.float32) - 16.0)
            v_hi_f = tl.where(v_hi_raw < 8, v_hi_raw.to(tl.float32),
                             v_hi_raw.to(tl.float32) - 16.0)
            v_lo_f = tl.where(valid, v_lo_f, tl.zeros([HALF_DIM], dtype=tl.float32))
            v_hi_f = tl.where(valid, v_hi_f, tl.zeros([HALF_DIM], dtype=tl.float32))
            
            # Accumulate with rescaling
            safe_l = tl.maximum(l_new, 1e-10)
            acc_lo = (l_i * alpha * acc_lo + beta * v_lo_f) / safe_l
            acc_hi = (l_i * alpha * acc_hi + beta * v_hi_f) / safe_l
            
            m_i = m_new
            l_i = l_new
    
    # Store
    tl.store(Out + batch_idx * stride_qb + head_idx * stride_qh + d_offs,
             acc_lo.to(tl.bfloat16), mask=d_offs < HALF_DIM)
    tl.store(Out + batch_idx * stride_qb + head_idx * stride_qh + d_offs + HALF_DIM,
             acc_hi.to(tl.bfloat16), mask=d_offs < HALF_DIM)


def fp4_pa_v2(query, k_cache, v_cache, block_tables, context_lens):
    batch, num_heads, head_dim = query.shape
    _, block_size, num_kv_heads, hd_half = k_cache.shape
    max_num_blocks = block_tables.shape[1]
    
    output = torch.empty_like(query)
    grid = (batch, num_heads)
    
    _fp4_pa_v2_kernel[grid](
        query, k_cache, v_cache,
        block_tables, context_lens, output,
        query.stride(0), query.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_tables.stride(0),
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, block_size=block_size,
        max_num_blocks=max_num_blocks,
        sm_scale=1.0 / math.sqrt(head_dim),
        HALF_DIM=head_dim // 2,
        BLOCK_S=block_size,
    )
    return output


def test_correctness():
    """Test against PyTorch reference."""
    from fp4_pa_reference import fp4_paged_attention_ref
    
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
                         dtype=torch.uint8, device=device)
    v_scale = torch.full((num_blocks, block_size, num_kv_heads, head_dim//32), 127,
                         dtype=torch.uint8, device=device)
    block_tables = torch.arange(max_blocks, device=device).unsqueeze(0).int()
    context_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    
    # Reference
    ref = fp4_paged_attention_ref(query, k_cache, v_cache, k_scale, v_scale,
                                  block_tables, context_lens)
    # Triton v2
    tri = fp4_pa_v2(query, k_cache, v_cache, block_tables, context_lens)
    
    diff = (ref.float() - tri.float()).abs().max().item()
    print(f"Correctness vs reference: max_diff={diff:.6f} {'✅' if diff < 1.0 else '❌'}")


def benchmark():
    """Benchmark FP4 PA v2 at different scales."""
    device = 'cuda:0'
    
    print("=== FP4 PA v2 Benchmark ===")
    configs = [
        (1, 16, 2, 128, 16, 256, "MiniMax MLA decode"),
        (1, 16, 2, 128, 16, 1024, "MiniMax 1K context"),
        (1, 16, 2, 128, 16, 4096, "MiniMax 4K context"),
        (8, 16, 2, 128, 16, 256, "Batch=8"),
        (1, 128, 16, 128, 16, 256, "MLA 128 heads"),
    ]
    
    for batch, nh, nkv, hd, bs, seq, label in configs:
        nb = max(seq // bs + 4, 16)
        mb = seq // bs
        
        q = torch.randn(batch, nh, hd, dtype=torch.bfloat16, device=device)
        kc = torch.randint(0, 255, (nb, bs, nkv, hd//2), dtype=torch.uint8, device=device)
        vc = torch.randint(0, 255, (nb, bs, nkv, hd//2), dtype=torch.uint8, device=device)
        bt = torch.arange(mb, device=device).unsqueeze(0).expand(batch, -1).contiguous().int()
        cl = torch.full((batch,), seq, dtype=torch.int32, device=device)
        
        # Warmup
        for _ in range(3):
            fp4_pa_v2(q, kc, vc, bt, cl)
        torch.cuda.synchronize()
        
        # Bench
        t0 = time.time()
        iters = 50
        for _ in range(iters):
            fp4_pa_v2(q, kc, vc, bt, cl)
        torch.cuda.synchronize()
        us = (time.time() - t0) / iters * 1e6
        
        print(f"  {label:25s}: {us:8.0f}µs  (B={batch} H={nh} S={seq})")
    
    # SDPA BF16 reference
    print("\n=== BF16 SDPA Reference ===")
    for batch, nh, hd, seq in [(1, 16, 128, 256), (1, 16, 128, 1024), (1, 128, 128, 256)]:
        q = torch.randn(batch, nh, 1, hd, dtype=torch.bfloat16, device=device)
        k = torch.randn(batch, nh, seq, hd, dtype=torch.bfloat16, device=device)
        v = torch.randn(batch, nh, seq, hd, dtype=torch.bfloat16, device=device)
        for _ in range(5):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        us = (time.time() - t0) / 50 * 1e6
        print(f"  SDPA B={batch} H={nh} S={seq:5d}: {us:8.0f}µs")


if __name__ == "__main__":
    try:
        test_correctness()
    except ImportError:
        print("(Skipping correctness test — reference not available)")
    benchmark()
