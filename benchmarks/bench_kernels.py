#!/usr/bin/env python3
"""End-to-end TurboQuant Packed KV Cache Benchmark — v7 Edition.

Measures the full pipeline:
  Packed:   raw K,V -> compress_k/v (packed, permuted signs) -> packed_attention v7
  Unpacked: raw K,V -> compress_k/v (unpacked PyTorch) -> PyTorch attention

Uses v7 attention kernel: interleaved loads + chunked dot + pre-permuted QJL
signs + fused corr_scale. Auto-dispatch from tuning table.

Reports: compress time, attention time, total time, memory usage, speedups.
"""
import math, time, sys
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def solve_codebook(d, bits):
    from scipy import integrate
    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)
    def pdf(x):
        return (1.0/(math.sqrt(2*math.pi*sigma**2)))*math.exp(-x*x/(2*sigma**2))
    lo, hi = -3.5*sigma, 3.5*sigma
    centroids = [lo+(hi-lo)*(i+0.5)/n_levels for i in range(n_levels)]
    for _ in range(200):
        boundaries = [(centroids[i]+centroids[i+1])/2.0 for i in range(n_levels-1)]
        edges = [lo*3]+boundaries+[hi*3]
        new_c = []
        for i in range(n_levels):
            a, b = edges[i], edges[i+1]
            num, _ = integrate.quad(lambda x: x*pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            new_c.append(num/den if den > 1e-15 else centroids[i])
        if max(abs(new_c[i]-centroids[i]) for i in range(n_levels)) < 1e-10:
            break
        centroids = new_c
    return torch.tensor(centroids, dtype=torch.float32)

def unpack_2bit(packed, D):
    N = packed.shape[0]; p = packed.to(torch.int32)
    return torch.stack([p&3, (p>>2)&3, (p>>4)&3, (p>>6)&3], dim=-1).reshape(N, D).to(torch.int8)

def unpack_1bit(packed, D):
    N = packed.shape[0]; p = packed.to(torch.int32).unsqueeze(-1)
    bits = torch.arange(8, device=packed.device)
    return ((p >> bits) & 1).reshape(N, D).to(torch.int8)

def permute_signs_for_chunked(signs_packed, D=128):
    """Reorder sign bits: original[c, c+4, c+8, ...] -> chunk c in contiguous bytes."""
    DE = D // 8; DQ = D // 4; shape = signs_packed.shape
    flat = signs_packed.reshape(-1, DE); N = flat.shape[0]; device = flat.device
    p = flat.to(torch.int32).unsqueeze(-1)
    all_bits = ((p >> torch.arange(8, device=device)) & 1).reshape(N, D)
    perm_bits = torch.zeros_like(all_bits)
    for c in range(4):
        perm_bits[:, c*DQ:(c+1)*DQ] = all_bits[:, torch.arange(c, D, 4, device=device)]
    perm_bytes = torch.zeros(N, DE, dtype=torch.uint8, device=device)
    for b in range(DE):
        val = torch.zeros(N, dtype=torch.int32, device=device)
        for bit in range(8):
            val |= perm_bits[:, b*8+bit].to(torch.int32) << bit
        perm_bytes[:, b] = val.to(torch.uint8)
    return perm_bytes.reshape(shape)


# ===========================================================================
# Packed Compression Kernels (from packed_compress.py)
# ===========================================================================
@triton.jit
def _fully_packed_k(
    K_ptr, PiT_ptr, PiST_ptr,
    cb0, cb1, cb2, cb3, bd0, bd1, bd2,
    Scratch_ptr,
    Kidx_ptr, Ksigns_ptr, Krnorm_ptr, Knorm_ptr,
    N_total,
    D: tl.constexpr, DQ: tl.constexpr, DE: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rd = tl.arange(0, D)
    n_mask = rn < N_total
    k = tl.load(K_ptr + rn[:, None]*D + rd[None, :], mask=n_mask[:, None], other=0.0).to(tl.float32)
    k_sq = tl.sum(k*k, axis=1)
    k_norm_val = tl.sqrt(k_sq)
    k_normalized = k * (1.0/(k_norm_val+1e-8))[:, None]
    tl.store(Knorm_ptr + rn, k_norm_val.to(tl.float16), mask=n_mask)
    pit = tl.load(PiT_ptr + rd[:, None]*D + rd[None, :]).to(tl.float32)
    rotated = tl.dot(k_normalized.to(tl.float16), pit.to(tl.float16)).to(tl.float32)
    idx = (rotated >= bd0).to(tl.int32) + (rotated >= bd1).to(tl.int32) + (rotated >= bd2).to(tl.int32)
    recon = tl.where(idx==0, cb0, tl.where(idx==1, cb1, tl.where(idx==2, cb2, cb3)))
    rot_resid = (rotated - recon) * k_norm_val[:, None]
    r_norm_sq = tl.sum(rot_resid*rot_resid, axis=1)
    tl.store(Krnorm_ptr + rn, tl.sqrt(r_norm_sq).to(tl.float16), mask=n_mask)
    pist = tl.load(PiST_ptr + rd[:, None]*D + rd[None, :]).to(tl.float32)
    proj = tl.dot(rot_resid.to(tl.float16), pist.to(tl.float16)).to(tl.float32)
    sign_bits = (proj >= 0).to(tl.int32)
    shift_2bit = ((rd % 4) * 2).to(tl.int32)
    shifted_idx = idx << shift_2bit[None, :]
    tl.store(Scratch_ptr + rn[:, None]*D + rd[None, :], shifted_idx, mask=n_mask[:, None])
    rq = tl.arange(0, DQ)
    p0 = tl.load(Scratch_ptr + rn[:, None]*D + (rq*4+0)[None, :], mask=n_mask[:, None], other=0)
    p1 = tl.load(Scratch_ptr + rn[:, None]*D + (rq*4+1)[None, :], mask=n_mask[:, None], other=0)
    p2 = tl.load(Scratch_ptr + rn[:, None]*D + (rq*4+2)[None, :], mask=n_mask[:, None], other=0)
    p3 = tl.load(Scratch_ptr + rn[:, None]*D + (rq*4+3)[None, :], mask=n_mask[:, None], other=0)
    tl.store(Kidx_ptr + rn[:, None]*DQ + rq[None, :], (p0|p1|p2|p3).to(tl.int8), mask=n_mask[:, None])
    shift_1bit = (rd % 8).to(tl.int32)
    shifted_signs = sign_bits << shift_1bit[None, :]
    tl.store(Scratch_ptr + rn[:, None]*D + rd[None, :], shifted_signs, mask=n_mask[:, None])
    re = tl.arange(0, DE)
    ps = tl.load(Scratch_ptr + rn[:, None]*D + (re*8+0)[None, :], mask=n_mask[:, None], other=0)
    ps = ps | tl.load(Scratch_ptr + rn[:, None]*D + (re*8+1)[None, :], mask=n_mask[:, None], other=0)
    ps = ps | tl.load(Scratch_ptr + rn[:, None]*D + (re*8+2)[None, :], mask=n_mask[:, None], other=0)
    ps = ps | tl.load(Scratch_ptr + rn[:, None]*D + (re*8+3)[None, :], mask=n_mask[:, None], other=0)
    ps = ps | tl.load(Scratch_ptr + rn[:, None]*D + (re*8+4)[None, :], mask=n_mask[:, None], other=0)
    ps = ps | tl.load(Scratch_ptr + rn[:, None]*D + (re*8+5)[None, :], mask=n_mask[:, None], other=0)
    ps = ps | tl.load(Scratch_ptr + rn[:, None]*D + (re*8+6)[None, :], mask=n_mask[:, None], other=0)
    ps = ps | tl.load(Scratch_ptr + rn[:, None]*D + (re*8+7)[None, :], mask=n_mask[:, None], other=0)
    tl.store(Ksigns_ptr + rn[:, None]*DE + re[None, :], ps.to(tl.int8), mask=n_mask[:, None])


@triton.jit
def _fully_packed_v(
    V_ptr, PiT_ptr, bd0, bd1, bd2,
    Scratch_ptr, Vidx_ptr, Vnorm_ptr,
    N_total,
    D: tl.constexpr, DQ: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rd = tl.arange(0, D)
    n_mask = rn < N_total
    v = tl.load(V_ptr + rn[:, None]*D + rd[None, :], mask=n_mask[:, None], other=0.0).to(tl.float32)
    v_sq = tl.sum(v*v, axis=1)
    v_norm_val = tl.sqrt(v_sq)
    v_normalized = v * (1.0/(v_norm_val+1e-8))[:, None]
    tl.store(Vnorm_ptr + rn, v_norm_val.to(tl.float16), mask=n_mask)
    pit = tl.load(PiT_ptr + rd[:, None]*D + rd[None, :]).to(tl.float32)
    rotated = tl.dot(v_normalized.to(tl.float16), pit.to(tl.float16)).to(tl.float32)
    idx = (rotated >= bd0).to(tl.int32) + (rotated >= bd1).to(tl.int32) + (rotated >= bd2).to(tl.int32)
    shift_2bit = ((rd % 4) * 2).to(tl.int32)
    shifted_idx = idx << shift_2bit[None, :]
    tl.store(Scratch_ptr + rn[:, None]*D + rd[None, :], shifted_idx, mask=n_mask[:, None])
    rq = tl.arange(0, DQ)
    p0 = tl.load(Scratch_ptr + rn[:, None]*D + (rq*4+0)[None, :], mask=n_mask[:, None], other=0)
    p1 = tl.load(Scratch_ptr + rn[:, None]*D + (rq*4+1)[None, :], mask=n_mask[:, None], other=0)
    p2 = tl.load(Scratch_ptr + rn[:, None]*D + (rq*4+2)[None, :], mask=n_mask[:, None], other=0)
    p3 = tl.load(Scratch_ptr + rn[:, None]*D + (rq*4+3)[None, :], mask=n_mask[:, None], other=0)
    tl.store(Vidx_ptr + rn[:, None]*DQ + rq[None, :], (p0|p1|p2|p3).to(tl.int8), mask=n_mask[:, None])


# ===========================================================================
# v7 Attention Kernel (production)
# ===========================================================================
@triton.jit
def _packed_attn_v7(
    Q_ptr, Q_proj_cs_ptr,
    Kidx_ptr, Ksigns_perm_ptr, Krnorm_ptr, Knorm_ptr,
    Vidx_ptr, Vnorm_ptr, Cb_ptr,
    Out_ptr, Lse_ptr, Sk, num_splits,
    D: tl.constexpr, DQ: tl.constexpr, DE: tl.constexpr,
    SDE: tl.constexpr, BLOCK_SK: tl.constexpr,
):
    head_id = tl.program_id(0)
    split_id = tl.program_id(1)
    tokens_per_split = (Sk + num_splits - 1) // num_splits
    sk_start = split_id * tokens_per_split
    sk_end = tl.minimum(sk_start + tokens_per_split, Sk)

    rdq = tl.arange(0, DQ)
    rd = tl.arange(0, D)

    q0 = tl.load(Q_ptr + head_id*D + rdq*4 + 0).to(tl.float32)
    q1 = tl.load(Q_ptr + head_id*D + rdq*4 + 1).to(tl.float32)
    q2 = tl.load(Q_ptr + head_id*D + rdq*4 + 2).to(tl.float32)
    q3 = tl.load(Q_ptr + head_id*D + rdq*4 + 3).to(tl.float32)

    qp0 = tl.load(Q_proj_cs_ptr + head_id*D + rdq*4 + 0).to(tl.float32)
    qp1 = tl.load(Q_proj_cs_ptr + head_id*D + rdq*4 + 1).to(tl.float32)
    qp2 = tl.load(Q_proj_cs_ptr + head_id*D + rdq*4 + 2).to(tl.float32)
    qp3 = tl.load(Q_proj_cs_ptr + head_id*D + rdq*4 + 3).to(tl.float32)

    sign_byte_in_chunk = (rdq // 8).to(tl.int32)
    sign_bit_in_byte = (rdq % 8).to(tl.int32)
    packed_col_full = (rd // 4).to(tl.int32)
    shift_2bit_full = ((rd % 4) * 2).to(tl.int32)

    m_prev = float('-inf')
    l_prev = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    for tile_start in range(sk_start, sk_end, BLOCK_SK):
        tile_end = tl.minimum(tile_start + BLOCK_SK, sk_end)
        rsk = tile_start + tl.arange(0, BLOCK_SK)
        sk_mask = rsk < tile_end

        # Phase 1: Batch all loads
        kidx_packed = tl.load(Kidx_ptr + head_id*Sk*DQ + rsk[:, None]*DQ + rdq[None, :],
                              mask=sk_mask[:, None], other=0).to(tl.int32)
        s0_raw = tl.load(Ksigns_perm_ptr + head_id*Sk*DE + rsk[:, None]*DE + sign_byte_in_chunk[None, :],
                         mask=sk_mask[:, None], other=0).to(tl.int32)
        s1_raw = tl.load(Ksigns_perm_ptr + head_id*Sk*DE + rsk[:, None]*DE + SDE + sign_byte_in_chunk[None, :],
                         mask=sk_mask[:, None], other=0).to(tl.int32)
        s2_raw = tl.load(Ksigns_perm_ptr + head_id*Sk*DE + rsk[:, None]*DE + 2*SDE + sign_byte_in_chunk[None, :],
                         mask=sk_mask[:, None], other=0).to(tl.int32)
        s3_raw = tl.load(Ksigns_perm_ptr + head_id*Sk*DE + rsk[:, None]*DE + 3*SDE + sign_byte_in_chunk[None, :],
                         mask=sk_mask[:, None], other=0).to(tl.int32)
        k_norms = tl.load(Knorm_ptr + head_id*Sk + rsk, mask=sk_mask, other=0.0).to(tl.float32)
        k_rnorms = tl.load(Krnorm_ptr + head_id*Sk + rsk, mask=sk_mask, other=0.0).to(tl.float32)

        # Phase 2: K MSE score (codebook gather)
        kr0 = tl.load(Cb_ptr + (kidx_packed & 3))
        kr1 = tl.load(Cb_ptr + ((kidx_packed >> 2) & 3))
        kr2 = tl.load(Cb_ptr + ((kidx_packed >> 4) & 3))
        kr3 = tl.load(Cb_ptr + ((kidx_packed >> 6) & 3))
        score_mse = (tl.sum(kr0*q0[None,:],axis=1)+tl.sum(kr1*q1[None,:],axis=1)+
                     tl.sum(kr2*q2[None,:],axis=1)+tl.sum(kr3*q3[None,:],axis=1))*k_norms

        # Phase 3: Chunked QJL score (corr_scale already in qp)
        sf0=((s0_raw>>sign_bit_in_byte[None,:])&1).to(tl.float32)*2.0-1.0
        sqjl0=tl.sum(qp0[None,:]*sf0*k_rnorms[:,None],axis=1)
        sf1=((s1_raw>>sign_bit_in_byte[None,:])&1).to(tl.float32)*2.0-1.0
        sqjl1=tl.sum(qp1[None,:]*sf1*k_rnorms[:,None],axis=1)
        sf2=((s2_raw>>sign_bit_in_byte[None,:])&1).to(tl.float32)*2.0-1.0
        sqjl2=tl.sum(qp2[None,:]*sf2*k_rnorms[:,None],axis=1)
        sf3=((s3_raw>>sign_bit_in_byte[None,:])&1).to(tl.float32)*2.0-1.0
        sqjl3=tl.sum(qp3[None,:]*sf3*k_rnorms[:,None],axis=1)

        score=tl.where(sk_mask,score_mse+sqjl0+sqjl1+sqjl2+sqjl3,float('-inf'))

        m_new=tl.maximum(m_prev,tl.max(score,axis=0))
        alpha=tl.exp(m_prev-m_new); p=tl.exp(score-m_new)
        l_new=alpha*l_prev+tl.sum(p,axis=0)

        vidx_raw=tl.load(Vidx_ptr+head_id*Sk*DQ+rsk[:,None]*DQ+packed_col_full[None,:],
                         mask=sk_mask[:,None],other=0).to(tl.int32)
        v_idx_full=(vidx_raw>>shift_2bit_full[None,:])&3
        v_recon_full=tl.load(Cb_ptr+v_idx_full)
        v_norms=tl.load(Vnorm_ptr+head_id*Sk+rsk,mask=sk_mask,other=0.0).to(tl.float32)
        v_mse=v_recon_full*v_norms[:,None]

        acc=alpha*acc+tl.sum(p[:,None]*v_mse,axis=0)
        m_prev=m_new; l_prev=l_new

    tl.store(Out_ptr+head_id*num_splits*D+split_id*D+rd,acc)
    tl.store(Lse_ptr+head_id*num_splits+split_id,m_prev+tl.log(l_prev))


@triton.jit
def _splitk_reduce(Out_ptr,Lse_ptr,Final_ptr,num_splits,D:tl.constexpr,NS:tl.constexpr):
    head_id=tl.program_id(0); rd=tl.arange(0,D)
    lses=tl.load(Lse_ptr+head_id*num_splits+tl.arange(0,NS),mask=tl.arange(0,NS)<num_splits,other=float('-inf'))
    max_lse=tl.max(lses,axis=0); acc=tl.zeros([D],dtype=tl.float32); total_w=0.0
    for s in range(NS):
        if s<num_splits:
            w=tl.exp(tl.load(Lse_ptr+head_id*num_splits+s)-max_lse)
            acc+=w*tl.load(Out_ptr+head_id*num_splits*D+s*D+rd); total_w+=w
    tl.store(Final_ptr+head_id*D+rd,(acc/total_w).to(tl.float16))


# ===========================================================================
# v7 Auto-dispatch tuning table
# ===========================================================================
_TUNING_TABLE = [
    (16,  1024,  32,  16,  4,  2),
    (16,  4096, 128,  16,  8,  1),
    (16, 16384,  32,  64,  8,  2),
    (16, 99999,  64,  64,  8,  2),
    (48,  1024,  16,  16,  4,  2),
    (48,  4096,  16,  64,  4,  1),
    (48, 16384,  16,  64,  2,  2),
    (48, 99999,  16,  64,  2,  1),
    (99999,  1024,  16,  16,  2,  1),
    (99999,  4096,  16,  32,  2,  2),
    (99999, 16384,  16,  32,  2,  2),
    (99999, 99999,  16, 128,  2,  1),
]

def _get_config(BH, Sk):
    for max_bh, max_sk, bs, sp, w, s in _TUNING_TABLE:
        if BH <= max_bh and Sk <= max_sk:
            sp = min(sp, max(1, Sk // bs))
            return bs, sp, w, s
    return 16, 64, 2, 1


# ===========================================================================
# Wrapper functions
# ===========================================================================
def compress_k_packed(K, PiT, PiST, centroids, boundaries):
    N, D = K.shape
    cb = centroids.tolist(); bd = boundaries.tolist()
    scratch = torch.empty(N, D, dtype=torch.int32, device=K.device)
    k_idx = torch.empty(N, D//4, dtype=torch.int8, device=K.device)
    k_signs = torch.empty(N, D//8, dtype=torch.int8, device=K.device)
    k_rnorm = torch.empty(N, dtype=torch.float16, device=K.device)
    k_norm = torch.empty(N, dtype=torch.float16, device=K.device)
    if N <= 2048: BN, nw, ns = 32, 4, 1
    elif N <= 32768: BN, nw, ns = 128, 4, 1
    else: BN, nw, ns = 64, 2, 2
    grid = ((N+BN-1)//BN,)
    _fully_packed_k[grid](K, PiT, PiST, cb[0],cb[1],cb[2],cb[3], bd[0],bd[1],bd[2],
        scratch, k_idx, k_signs, k_rnorm, k_norm, N, D=D, DQ=D//4, DE=D//8, BN=BN,
        num_warps=nw, num_stages=ns)
    return k_idx.to(torch.uint8), k_signs.to(torch.uint8), k_rnorm, k_norm

def compress_v_packed(V, PiT, centroids, boundaries):
    N, D = V.shape
    bd = boundaries.tolist()
    scratch = torch.empty(N, D, dtype=torch.int32, device=V.device)
    v_idx = torch.empty(N, D//4, dtype=torch.int8, device=V.device)
    v_norm = torch.empty(N, dtype=torch.float16, device=V.device)
    if N <= 32768: BN, nw, ns = 64, 8, 2
    else: BN, nw, ns = 128, 8, 1
    grid = ((N+BN-1)//BN,)
    _fully_packed_v[grid](V, PiT, bd[0],bd[1],bd[2],
        scratch, v_idx, v_norm, N, D=D, DQ=D//4, BN=BN, num_warps=nw, num_stages=ns)
    return v_idx.to(torch.uint8), v_norm

def packed_attention_v7(Q, Q_proj, ki, ks_perm, kr, kn, vi, vn, centroids, corr_scale):
    """v7 packed attention with auto-dispatch and precomputed Q_proj*corr_scale."""
    BH, D = Q.shape; Sk = kr.shape[1]
    DQ = D//4; DE = D//8; SDE = DQ//8
    BS, nsplits, nw, ns = _get_config(BH, Sk)
    cb = centroids.contiguous()
    Q_proj_cs = (Q_proj.float() * corr_scale).to(Q_proj.dtype)
    out_splits = torch.empty(BH, nsplits, D, dtype=torch.float32, device=Q.device)
    lse_splits = torch.empty(BH, nsplits, dtype=torch.float32, device=Q.device)
    _packed_attn_v7[(BH, nsplits)](
        Q, Q_proj_cs, ki, ks_perm, kr, kn, vi, vn, cb,
        out_splits, lse_splits, Sk, nsplits,
        D=D, DQ=DQ, DE=DE, SDE=SDE, BLOCK_SK=BS,
        num_warps=nw, num_stages=ns)
    NS_pad = 1
    while NS_pad < nsplits: NS_pad *= 2
    final = torch.empty(BH, D, dtype=torch.float16, device=Q.device)
    _splitk_reduce[(BH,)](out_splits, lse_splits, final, nsplits, D=D, NS=NS_pad, num_warps=4)
    return final

def ref_attention(Q, Q_proj, K_mse, signs_f, k_rnorm, V_mse, cs):
    score_mse = torch.bmm(Q.unsqueeze(1), K_mse.transpose(1,2)).squeeze(1)
    K_qjl = signs_f * k_rnorm.unsqueeze(-1) * cs
    score_qjl = torch.bmm(Q_proj.unsqueeze(1), K_qjl.transpose(1,2)).squeeze(1)
    attn = F.softmax(score_mse + score_qjl, dim=-1)
    return torch.bmm(attn.unsqueeze(1), V_mse).squeeze(1)

def compress_unpacked_ref(K, V, PiT, ST, centroids, boundaries, corr_scale):
    N, D = K.shape
    k_norm = torch.norm(K, dim=-1, keepdim=True)
    K_n = K / (k_norm + 1e-8)
    K_rot = K_n @ PiT
    k_idx = torch.searchsorted(boundaries, K_rot.reshape(-1)).reshape(N, D).clamp(0, 3)
    K_mse = centroids[k_idx.long()] * k_norm
    residual = (K_rot - centroids[k_idx.long()]) * k_norm
    k_rnorm = torch.norm(residual, dim=-1)
    proj = residual @ (PiT @ ST).T
    signs_f = torch.sign(proj); signs_f[signs_f == 0] = 1.0
    v_norm = torch.norm(V, dim=-1, keepdim=True)
    V_n = V / (v_norm + 1e-8)
    V_rot = V_n @ PiT
    v_idx = torch.searchsorted(boundaries, V_rot.reshape(-1)).reshape(N, D).clamp(0, 3)
    V_mse = centroids[v_idx.long()] * v_norm
    return K_mse.half(), signs_f.half(), k_rnorm.half(), V_mse.half()


# ===========================================================================
# E2E Benchmark
# ===========================================================================
def run():
    device = "cuda"; D = 128; bits = 3; mse_bits = bits - 1; seed = 42

    gen = torch.Generator(device=device); gen.manual_seed(seed)
    G = torch.randn(D, D, device=device, generator=gen)
    Q_orth, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R)); diag_sign[diag_sign == 0] = 1.0
    Pi = Q_orth * diag_sign.unsqueeze(0); PiT = Pi.T.contiguous()
    gen2 = torch.Generator(device=device); gen2.manual_seed(seed+10000)
    S = torch.randn(D, D, device=device, generator=gen2); ST = S.T.contiguous()
    PiST = (Pi @ ST).contiguous()

    centroids = solve_codebook(D, mse_bits).to(device)
    boundaries = ((centroids[:-1] + centroids[1:]) / 2.0).contiguous()
    corr_scale = math.sqrt(math.pi / 2) / math.sqrt(D)

    print("=" * 80)
    print("TurboQuant E2E Packed KV Cache Benchmark v7 — MI355X")
    print("=" * 80)
    print(f"D={D}, bits={bits}, device={torch.cuda.get_device_name(0) or 'MI355X'}")
    print(f"Attention: v7 (interleaved loads + chunked dot + pre-permuted QJL + fused corr_scale)")
    print()

    warmup, iters = 25, 200

    # ====== CORRECTNESS ======
    print("=== Correctness (v7 packed Triton attn vs PyTorch attn, same compressed data) ===")
    for BH in [8, 32, 64]:
        for Sk in [4096, 16384, 32768]:
            K_raw = torch.randn(BH, Sk, D, device=device, dtype=torch.float32)
            V_raw = torch.randn(BH, Sk, D, device=device, dtype=torch.float32)
            Q = torch.randn(BH, D, dtype=torch.float16, device=device) * 0.1
            Q_proj = (Q.float() @ ST.float()).half()

            ki_l, ks_l, kr_l, kn_l, vi_l, vn_l = [], [], [], [], [], []
            for h in range(BH):
                ki, ks, kr, kn = compress_k_packed(K_raw[h], PiT, PiST, centroids, boundaries)
                vi, vn = compress_v_packed(V_raw[h], PiT, centroids, boundaries)
                ki_l.append(ki); ks_l.append(ks); kr_l.append(kr); kn_l.append(kn)
                vi_l.append(vi); vn_l.append(vn)
            ki_t=torch.stack(ki_l); ks_t=torch.stack(ks_l)
            kr_t=torch.stack(kr_l); kn_t=torch.stack(kn_l)
            vi_t=torch.stack(vi_l); vn_t=torch.stack(vn_l)

            # Permute signs for v7
            ks_perm = permute_signs_for_chunked(ks_t, D)

            # v7 attention
            packed_out = packed_attention_v7(Q, Q_proj, ki_t, ks_perm, kr_t, kn_t,
                                            vi_t, vn_t, centroids, corr_scale)
            # PyTorch reference (unpack same data)
            kidx = unpack_2bit(ki_t.reshape(-1, D//4), D).reshape(BH, Sk, D)
            ksig = unpack_1bit(ks_t.reshape(-1, D//8), D).reshape(BH, Sk, D)
            vidx = unpack_2bit(vi_t.reshape(-1, D//4), D).reshape(BH, Sk, D)
            Km = centroids[kidx.long()] * kn_t.unsqueeze(-1)
            sf = ksig.float() * 2 - 1
            Vm = centroids[vidx.long()] * vn_t.unsqueeze(-1)
            ref_out = ref_attention(Q.float(), Q_proj.float(), Km.float(), sf.float(),
                                   kr_t.float(), Vm.float(), corr_scale)

            cs_val = F.cosine_similarity(packed_out.float().reshape(1,-1),
                                         ref_out.float().reshape(1,-1)).item()
            bs,sp,w,s = _get_config(BH, Sk)
            print(f"  BH={BH:>3} Sk={Sk:>5}: cos={cs_val:.6f} {'PASS' if cs_val>0.99 else 'FAIL'}  (BS={bs} sp={sp} w={w})")

    # ====== MEMORY ======
    print(f"\n=== Memory Savings ===")
    fp16_bytes = D * 2 * 2  # K + V in fp16
    packed_bytes = D//4 + D//8 + 2 + 2 + D//4 + 2  # ki + ks + kr + kn + vi + vn
    ratio = fp16_bytes / packed_bytes
    print(f"  Per token/head: {fp16_bytes}B (fp16) -> {packed_bytes}B (packed) = {ratio:.1f}x compression")
    for ctx in [4096, 16384, 32768, 131072]:
        for nh in [64, 128]:
            fp16_gb = ctx * nh * fp16_bytes / 1e9
            pack_gb = ctx * nh * packed_bytes / 1e9
            if fp16_gb > 0.5:
                print(f"  ctx={ctx:>6} heads={nh:>3}: {fp16_gb:.2f}GB -> {pack_gb:.2f}GB (saves {fp16_gb-pack_gb:.2f}GB)")

    # ====== BENCHMARK ======
    print(f"\n=== E2E Benchmark: Compress + Attend ===")
    print(f"{'BH':>4} {'Sk':>6} | {'pack_comp':>10} {'pack_attn':>10} {'pack_tot':>10} | {'ref_comp':>10} {'ref_attn':>10} {'ref_tot':>10} | {'E2E spdup':>9} {'attn spdup':>10}")
    print("-" * 115)

    for BH in [8, 32, 64]:
        for Sk in [4096, 16384, 32768]:
            K_raw = torch.randn(BH, Sk, D, device=device, dtype=torch.float32)
            V_raw = torch.randn(BH, Sk, D, device=device, dtype=torch.float32)
            Q = torch.randn(BH, D, dtype=torch.float16, device=device) * 0.1
            Q_proj = (Q.float() @ ST.float()).half()

            # Warmup packed pipeline
            for _ in range(warmup):
                ki_l, ks_l, kr_l, kn_l, vi_l, vn_l = [], [], [], [], [], []
                for h in range(BH):
                    ki, ks, kr, kn = compress_k_packed(K_raw[h], PiT, PiST, centroids, boundaries)
                    vi, vn = compress_v_packed(V_raw[h], PiT, centroids, boundaries)
                    ki_l.append(ki); ks_l.append(ks); kr_l.append(kr); kn_l.append(kn)
                    vi_l.append(vi); vn_l.append(vn)
                ki_t=torch.stack(ki_l); ks_t=torch.stack(ks_l)
                kr_t=torch.stack(kr_l); kn_t=torch.stack(kn_l)
                vi_t=torch.stack(vi_l); vn_t=torch.stack(vn_l)
                ks_perm = permute_signs_for_chunked(ks_t, D)
                packed_attention_v7(Q, Q_proj, ki_t, ks_perm, kr_t, kn_t,
                                   vi_t, vn_t, centroids, corr_scale)
            torch.cuda.synchronize()

            # Bench compress only (includes sign permutation)
            t0 = time.perf_counter()
            for _ in range(iters):
                ki_l, ks_l, kr_l, kn_l, vi_l, vn_l = [], [], [], [], [], []
                for h in range(BH):
                    ki, ks, kr, kn = compress_k_packed(K_raw[h], PiT, PiST, centroids, boundaries)
                    vi, vn = compress_v_packed(V_raw[h], PiT, centroids, boundaries)
                    ki_l.append(ki); ks_l.append(ks); kr_l.append(kr); kn_l.append(kn)
                    vi_l.append(vi); vn_l.append(vn)
                ki_t=torch.stack(ki_l); ks_t=torch.stack(ks_l)
                kr_t=torch.stack(kr_l); kn_t=torch.stack(kn_l)
                vi_t=torch.stack(vi_l); vn_t=torch.stack(vn_l)
                ks_perm = permute_signs_for_chunked(ks_t, D)
            torch.cuda.synchronize()
            pack_comp_ms = (time.perf_counter()-t0)/iters*1000

            # Bench attend only
            for _ in range(warmup):
                packed_attention_v7(Q, Q_proj, ki_t, ks_perm, kr_t, kn_t,
                                   vi_t, vn_t, centroids, corr_scale)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                packed_attention_v7(Q, Q_proj, ki_t, ks_perm, kr_t, kn_t,
                                   vi_t, vn_t, centroids, corr_scale)
            torch.cuda.synchronize()
            pack_attn_ms = (time.perf_counter()-t0)/iters*1000
            pack_total = pack_comp_ms + pack_attn_ms

            # Reference pipeline
            for _ in range(warmup): compress_unpacked_ref(K_raw[0], V_raw[0], PiT, ST, centroids, boundaries, corr_scale)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                Km_l, sf_l, krn_l, Vm_l = [], [], [], []
                for h in range(BH):
                    Km, sf, krn, Vm = compress_unpacked_ref(K_raw[h], V_raw[h], PiT, ST, centroids, boundaries, corr_scale)
                    Km_l.append(Km); sf_l.append(sf); krn_l.append(krn); Vm_l.append(Vm)
                Km_t=torch.stack(Km_l); sf_t=torch.stack(sf_l)
                krn_t=torch.stack(krn_l); Vm_t=torch.stack(Vm_l)
            torch.cuda.synchronize()
            ref_comp_ms = (time.perf_counter()-t0)/iters*1000

            for _ in range(warmup):
                ref_attention(Q.float(), Q_proj.float(), Km_t.float(), sf_t.float(),
                             krn_t.float(), Vm_t.float(), corr_scale)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                ref_attention(Q.float(), Q_proj.float(), Km_t.float(), sf_t.float(),
                             krn_t.float(), Vm_t.float(), corr_scale)
            torch.cuda.synchronize()
            ref_attn_ms = (time.perf_counter()-t0)/iters*1000
            ref_total = ref_comp_ms + ref_attn_ms

            e2e_sp = ref_total / pack_total if pack_total > 0 else 0
            attn_sp = ref_attn_ms / pack_attn_ms if pack_attn_ms > 0 else 0
            print(f"{BH:>4} {Sk:>6} | {pack_comp_ms:>9.3f}ms {pack_attn_ms:>9.3f}ms {pack_total:>9.3f}ms | "
                  f"{ref_comp_ms:>9.3f}ms {ref_attn_ms:>9.3f}ms {ref_total:>9.3f}ms | {e2e_sp:>8.2f}x  {attn_sp:>8.2f}x")

    print("\nDone!")

if __name__ == "__main__":
    run()
