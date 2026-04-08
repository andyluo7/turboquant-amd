"""TQ Paged Attention v12 — Sparse V Dequant.

Key idea from turboquant_plus: Skip V dequantization for positions where
softmax attention weight < threshold (1e-6). At long context, 90%+ of 
weights are negligible → eliminates ~50% of V dequant work.

Changes from v11 single-pass kernel:
- After QK softmax, compute max attention weight per KV position
- Skip V dequant+accumulate for positions below threshold
- All K dequant still runs (needed for attention weights)

Test: standalone benchmark comparing v11 (no skip) vs v12 (sparse V)
"""
import torch
import triton
import triton.language as tl
import time

D = 128
DQ3 = D * 3 // 8  # 48
CD = 32
SPARSE_V_THRESHOLD = 1e-6


@triton.jit
def _tq_v12_sparse_v_kernel(
    out, q_ptr, tq_cache_ptr, centroids_ptr,
    blk_tables_ptr, seq_lens_ptr, scale,
    stride_o_s, stride_o_nh,
    stride_q_s, stride_q_nh,
    stride_tq_blk, stride_tq_kv, stride_tq_pos, stride_tq_head,
    stride_bt_s,
    KV_BLK_SZ: tl.constexpr, KV_BLK_SZ_POW2: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr, QUERY_GRP_SZ_POW2: tl.constexpr,
    DQ3_C: tl.constexpr, CD_C: tl.constexpr,
    SPARSE_THRESH: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    log2e: tl.constexpr = 1.4426950408889634
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_kv_blks = tl.cdiv(seq_len, KV_BLK_SZ)
    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)
    cd = tl.arange(0, CD_C)

    qb = seq_idx*stride_q_s + (kv_head_idx*QUERY_GRP_SZ + grp_offs[:,None])*stride_q_nh
    gm = grp_offs[:,None] < QUERY_GRP_SZ

    q0 = (tl.load(q_ptr+qb+cd[None,:], mask=gm, other=0.)*scale).to(tl.float16)
    q1 = (tl.load(q_ptr+qb+(CD_C+cd)[None,:], mask=gm, other=0.)*scale).to(tl.float16)
    q2 = (tl.load(q_ptr+qb+(2*CD_C+cd)[None,:], mask=gm, other=0.)*scale).to(tl.float16)
    q3 = (tl.load(q_ptr+qb+(3*CD_C+cd)[None,:], mask=gm, other=0.)*scale).to(tl.float16)

    # Precompute bit extraction offsets (same for K and V)
    d0=cd;      bp0=d0*3;bi0=bp0>>3;bo0=bp0&7;nb0=tl.minimum(bi0+1,DQ3_C-1);nv0=(bi0+1<DQ3_C)
    d1=CD_C+cd; bp1=d1*3;bi1=bp1>>3;bo1=bp1&7;nb1=tl.minimum(bi1+1,DQ3_C-1);nv1=(bi1+1<DQ3_C)
    d2=2*CD_C+cd;bp2=d2*3;bi2=bp2>>3;bo2=bp2&7;nb2=tl.minimum(bi2+1,DQ3_C-1);nv2=(bi2+1<DQ3_C)
    d3=3*CD_C+cd;bp3=d3*3;bi3=bp3>>3;bo3=bp3&7;nb3=tl.minimum(bi3+1,DQ3_C-1);nv3=(bi3+1<DQ3_C)

    a0=tl.zeros([QUERY_GRP_SZ_POW2,CD_C],dtype=tl.float32)
    a1=tl.zeros([QUERY_GRP_SZ_POW2,CD_C],dtype=tl.float32)
    a2=tl.zeros([QUERY_GRP_SZ_POW2,CD_C],dtype=tl.float32)
    a3=tl.zeros([QUERY_GRP_SZ_POW2,CD_C],dtype=tl.float32)
    max_logit=tl.zeros([QUERY_GRP_SZ_POW2],dtype=tl.float32)+float("-inf")
    exp_sum=tl.zeros([QUERY_GRP_SZ_POW2],dtype=tl.float32)
    blk_tbl_ptr = blk_tables_ptr + seq_idx*stride_bt_s

    for b in range(num_kv_blks):
        kv_blk_num = tl.load(blk_tbl_ptr + b)
        blk_seq_offs = b*KV_BLK_SZ + blk_offs
        pos_mask = (blk_seq_offs < seq_len) & (blk_offs < KV_BLK_SZ)
        k_base = kv_blk_num*stride_tq_blk + kv_head_idx*stride_tq_head
        v_base = kv_blk_num*stride_tq_blk + stride_tq_kv + kv_head_idx*stride_tq_head
        norm_offs = blk_offs*stride_tq_pos + DQ3_C

        # === K dequant (always needed for attention weights) ===
        n0k=tl.load(tq_cache_ptr+k_base+norm_offs,mask=pos_mask,other=0).to(tl.int32)
        n1k=tl.load(tq_cache_ptr+k_base+norm_offs+1,mask=pos_mask,other=0).to(tl.int32)
        to0=blk_offs[:,None]*stride_tq_pos+bi0[None,:]; tm0=pos_mask[:,None]&(d0[None,:]<HEAD_SZ)
        kr0a=tl.load(tq_cache_ptr+k_base+to0,mask=tm0,other=0).to(tl.int32)
        kr0b=tl.load(tq_cache_ptr+k_base+blk_offs[:,None]*stride_tq_pos+nb0[None,:],mask=tm0&nv0[None,:],other=0).to(tl.int32)
        to1=blk_offs[:,None]*stride_tq_pos+bi1[None,:]; tm1=pos_mask[:,None]&(d1[None,:]<HEAD_SZ)
        kr1a=tl.load(tq_cache_ptr+k_base+to1,mask=tm1,other=0).to(tl.int32)
        kr1b=tl.load(tq_cache_ptr+k_base+blk_offs[:,None]*stride_tq_pos+nb1[None,:],mask=tm1&nv1[None,:],other=0).to(tl.int32)
        to2=blk_offs[:,None]*stride_tq_pos+bi2[None,:]; tm2=pos_mask[:,None]&(d2[None,:]<HEAD_SZ)
        kr2a=tl.load(tq_cache_ptr+k_base+to2,mask=tm2,other=0).to(tl.int32)
        kr2b=tl.load(tq_cache_ptr+k_base+blk_offs[:,None]*stride_tq_pos+nb2[None,:],mask=tm2&nv2[None,:],other=0).to(tl.int32)
        to3=blk_offs[:,None]*stride_tq_pos+bi3[None,:]; tm3=pos_mask[:,None]&(d3[None,:]<HEAD_SZ)
        kr3a=tl.load(tq_cache_ptr+k_base+to3,mask=tm3,other=0).to(tl.int32)
        kr3b=tl.load(tq_cache_ptr+k_base+blk_offs[:,None]*stride_tq_pos+nb3[None,:],mask=tm3&nv3[None,:],other=0).to(tl.int32)

        k_norm=((n0k|(n1k<<8)).to(tl.int16)).to(tl.float16,bitcast=True).to(tl.float32)
        ik0=(((kr0a|(kr0b<<8))>>bo0[None,:])&7); kc0=(tl.load(centroids_ptr+ik0)*k_norm[:,None]).to(tl.float16)
        ik1=(((kr1a|(kr1b<<8))>>bo1[None,:])&7); kc1=(tl.load(centroids_ptr+ik1)*k_norm[:,None]).to(tl.float16)
        ik2=(((kr2a|(kr2b<<8))>>bo2[None,:])&7); kc2=(tl.load(centroids_ptr+ik2)*k_norm[:,None]).to(tl.float16)
        ik3=(((kr3a|(kr3b<<8))>>bo3[None,:])&7); kc3=(tl.load(centroids_ptr+ik3)*k_norm[:,None]).to(tl.float16)

        # QK dot (4 chunks)
        qk = tl.dot(q0,tl.trans(kc0),out_dtype=tl.float32)
        qk += tl.dot(q1,tl.trans(kc1),out_dtype=tl.float32)
        qk += tl.dot(q2,tl.trans(kc2),out_dtype=tl.float32)
        qk += tl.dot(q3,tl.trans(kc3),out_dtype=tl.float32)
        qk = tl.where((grp_offs[:,None]<QUERY_GRP_SZ)&(blk_seq_offs[None,:]<seq_len),qk,float("-inf"))

        # Online softmax
        ml_new = tl.maximum(tl.max(qk,axis=1), max_logit)
        p = tl.math.exp2((qk-ml_new[:,None])*log2e)
        alpha = tl.math.exp2((max_logit-ml_new)*log2e)
        a0*=alpha[:,None]; a1*=alpha[:,None]; a2*=alpha[:,None]; a3*=alpha[:,None]

        # === SPARSE V: Check if ANY query in the group has significant weight ===
        # max_p_per_pos: [KV_BLK_SZ_POW2] — max attention weight across GQA group
        max_p_per_pos = tl.max(p, axis=0)  # max across query heads
        # any_significant: scalar — is there ANY position worth dequanting?
        any_significant = tl.max(max_p_per_pos) > SPARSE_THRESH

        # Only dequant V if at least one position has significant weight
        # (Triton doesn't support `if` on runtime values, so we use masking)
        # We mask V loads with the per-position significance check
        v_mask = max_p_per_pos > SPARSE_THRESH  # [KV_BLK_SZ_POW2]
        v_pos_mask = pos_mask & v_mask  # combine with position validity

        # === V dequant (SPARSE — only for significant positions) ===
        n0v=tl.load(tq_cache_ptr+v_base+norm_offs,mask=v_pos_mask,other=0).to(tl.int32)
        n1v=tl.load(tq_cache_ptr+v_base+norm_offs+1,mask=v_pos_mask,other=0).to(tl.int32)

        vm0=v_pos_mask[:,None]&(d0[None,:]<HEAD_SZ)
        vr0a=tl.load(tq_cache_ptr+v_base+to0,mask=vm0,other=0).to(tl.int32)
        vr0b=tl.load(tq_cache_ptr+v_base+blk_offs[:,None]*stride_tq_pos+nb0[None,:],mask=vm0&nv0[None,:],other=0).to(tl.int32)
        vm1=v_pos_mask[:,None]&(d1[None,:]<HEAD_SZ)
        vr1a=tl.load(tq_cache_ptr+v_base+to1,mask=vm1,other=0).to(tl.int32)
        vr1b=tl.load(tq_cache_ptr+v_base+blk_offs[:,None]*stride_tq_pos+nb1[None,:],mask=vm1&nv1[None,:],other=0).to(tl.int32)
        vm2=v_pos_mask[:,None]&(d2[None,:]<HEAD_SZ)
        vr2a=tl.load(tq_cache_ptr+v_base+to2,mask=vm2,other=0).to(tl.int32)
        vr2b=tl.load(tq_cache_ptr+v_base+blk_offs[:,None]*stride_tq_pos+nb2[None,:],mask=vm2&nv2[None,:],other=0).to(tl.int32)
        vm3=v_pos_mask[:,None]&(d3[None,:]<HEAD_SZ)
        vr3a=tl.load(tq_cache_ptr+v_base+to3,mask=vm3,other=0).to(tl.int32)
        vr3b=tl.load(tq_cache_ptr+v_base+blk_offs[:,None]*stride_tq_pos+nb3[None,:],mask=vm3&nv3[None,:],other=0).to(tl.int32)

        v_norm=((n0v|(n1v<<8)).to(tl.int16)).to(tl.float16,bitcast=True).to(tl.float32)
        p16 = p.to(tl.float16)
        iv0=(((vr0a|(vr0b<<8))>>bo0[None,:])&7); a0+=tl.dot(p16,(tl.load(centroids_ptr+iv0)*v_norm[:,None]).to(tl.float16),out_dtype=tl.float32)
        iv1=(((vr1a|(vr1b<<8))>>bo1[None,:])&7); a1+=tl.dot(p16,(tl.load(centroids_ptr+iv1)*v_norm[:,None]).to(tl.float16),out_dtype=tl.float32)
        iv2=(((vr2a|(vr2b<<8))>>bo2[None,:])&7); a2+=tl.dot(p16,(tl.load(centroids_ptr+iv2)*v_norm[:,None]).to(tl.float16),out_dtype=tl.float32)
        iv3=(((vr3a|(vr3b<<8))>>bo3[None,:])&7); a3+=tl.dot(p16,(tl.load(centroids_ptr+iv3)*v_norm[:,None]).to(tl.float16),out_dtype=tl.float32)

        exp_sum = exp_sum*alpha + tl.sum(p, axis=1)
        max_logit = ml_new

    inv = 1.0 / exp_sum
    a0*=inv[:,None]; a1*=inv[:,None]; a2*=inv[:,None]; a3*=inv[:,None]
    ob = seq_idx*stride_o_s + (kv_head_idx*QUERY_GRP_SZ + grp_offs[:,None])*stride_o_nh
    om = grp_offs[:,None] < QUERY_GRP_SZ
    tl.store(out+ob+cd[None,:], a0.to(out.dtype.element_ty), mask=om)
    tl.store(out+ob+(CD_C+cd)[None,:], a1.to(out.dtype.element_ty), mask=om)
    tl.store(out+ob+(2*CD_C+cd)[None,:], a2.to(out.dtype.element_ty), mask=om)
    tl.store(out+ob+(3*CD_C+cd)[None,:], a3.to(out.dtype.element_ty), mask=om)


# === v11 kernel (no sparse V, for comparison) ===
@triton.jit
def _tq_v11_single_kernel(
    out, q_ptr, tq_cache_ptr, centroids_ptr,
    blk_tables_ptr, seq_lens_ptr, scale,
    stride_o_s, stride_o_nh,
    stride_q_s, stride_q_nh,
    stride_tq_blk, stride_tq_kv, stride_tq_pos, stride_tq_head,
    stride_bt_s,
    KV_BLK_SZ: tl.constexpr, KV_BLK_SZ_POW2: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr, QUERY_GRP_SZ_POW2: tl.constexpr,
    DQ3_C: tl.constexpr, CD_C: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    log2e: tl.constexpr = 1.4426950408889634
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_kv_blks = tl.cdiv(seq_len, KV_BLK_SZ)
    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)
    cd = tl.arange(0, CD_C)
    qb = seq_idx*stride_q_s + (kv_head_idx*QUERY_GRP_SZ + grp_offs[:,None])*stride_q_nh
    gm = grp_offs[:,None] < QUERY_GRP_SZ
    q0 = (tl.load(q_ptr+qb+cd[None,:], mask=gm, other=0.)*scale).to(tl.float16)
    q1 = (tl.load(q_ptr+qb+(CD_C+cd)[None,:], mask=gm, other=0.)*scale).to(tl.float16)
    q2 = (tl.load(q_ptr+qb+(2*CD_C+cd)[None,:], mask=gm, other=0.)*scale).to(tl.float16)
    q3 = (tl.load(q_ptr+qb+(3*CD_C+cd)[None,:], mask=gm, other=0.)*scale).to(tl.float16)
    d0=cd;      bp0=d0*3;bi0=bp0>>3;bo0=bp0&7;nb0=tl.minimum(bi0+1,DQ3_C-1);nv0=(bi0+1<DQ3_C)
    d1=CD_C+cd; bp1=d1*3;bi1=bp1>>3;bo1=bp1&7;nb1=tl.minimum(bi1+1,DQ3_C-1);nv1=(bi1+1<DQ3_C)
    d2=2*CD_C+cd;bp2=d2*3;bi2=bp2>>3;bo2=bp2&7;nb2=tl.minimum(bi2+1,DQ3_C-1);nv2=(bi2+1<DQ3_C)
    d3=3*CD_C+cd;bp3=d3*3;bi3=bp3>>3;bo3=bp3&7;nb3=tl.minimum(bi3+1,DQ3_C-1);nv3=(bi3+1<DQ3_C)
    a0=tl.zeros([QUERY_GRP_SZ_POW2,CD_C],dtype=tl.float32)
    a1=tl.zeros([QUERY_GRP_SZ_POW2,CD_C],dtype=tl.float32)
    a2=tl.zeros([QUERY_GRP_SZ_POW2,CD_C],dtype=tl.float32)
    a3=tl.zeros([QUERY_GRP_SZ_POW2,CD_C],dtype=tl.float32)
    max_logit=tl.zeros([QUERY_GRP_SZ_POW2],dtype=tl.float32)+float("-inf")
    exp_sum=tl.zeros([QUERY_GRP_SZ_POW2],dtype=tl.float32)
    blk_tbl_ptr = blk_tables_ptr + seq_idx*stride_bt_s
    for b in range(num_kv_blks):
        kv_blk_num = tl.load(blk_tbl_ptr + b)
        blk_seq_offs = b*KV_BLK_SZ + blk_offs
        pos_mask = (blk_seq_offs < seq_len) & (blk_offs < KV_BLK_SZ)
        k_base = kv_blk_num*stride_tq_blk + kv_head_idx*stride_tq_head
        v_base = kv_blk_num*stride_tq_blk + stride_tq_kv + kv_head_idx*stride_tq_head
        norm_offs = blk_offs*stride_tq_pos + DQ3_C
        n0k=tl.load(tq_cache_ptr+k_base+norm_offs,mask=pos_mask,other=0).to(tl.int32)
        n1k=tl.load(tq_cache_ptr+k_base+norm_offs+1,mask=pos_mask,other=0).to(tl.int32)
        n0v=tl.load(tq_cache_ptr+v_base+norm_offs,mask=pos_mask,other=0).to(tl.int32)
        n1v=tl.load(tq_cache_ptr+v_base+norm_offs+1,mask=pos_mask,other=0).to(tl.int32)
        to0=blk_offs[:,None]*stride_tq_pos+bi0[None,:]; tm0=pos_mask[:,None]&(d0[None,:]<HEAD_SZ)
        kr0a=tl.load(tq_cache_ptr+k_base+to0,mask=tm0,other=0).to(tl.int32)
        kr0b=tl.load(tq_cache_ptr+k_base+blk_offs[:,None]*stride_tq_pos+nb0[None,:],mask=tm0&nv0[None,:],other=0).to(tl.int32)
        to1=blk_offs[:,None]*stride_tq_pos+bi1[None,:]; tm1=pos_mask[:,None]&(d1[None,:]<HEAD_SZ)
        kr1a=tl.load(tq_cache_ptr+k_base+to1,mask=tm1,other=0).to(tl.int32)
        kr1b=tl.load(tq_cache_ptr+k_base+blk_offs[:,None]*stride_tq_pos+nb1[None,:],mask=tm1&nv1[None,:],other=0).to(tl.int32)
        to2=blk_offs[:,None]*stride_tq_pos+bi2[None,:]; tm2=pos_mask[:,None]&(d2[None,:]<HEAD_SZ)
        kr2a=tl.load(tq_cache_ptr+k_base+to2,mask=tm2,other=0).to(tl.int32)
        kr2b=tl.load(tq_cache_ptr+k_base+blk_offs[:,None]*stride_tq_pos+nb2[None,:],mask=tm2&nv2[None,:],other=0).to(tl.int32)
        to3=blk_offs[:,None]*stride_tq_pos+bi3[None,:]; tm3=pos_mask[:,None]&(d3[None,:]<HEAD_SZ)
        kr3a=tl.load(tq_cache_ptr+k_base+to3,mask=tm3,other=0).to(tl.int32)
        kr3b=tl.load(tq_cache_ptr+k_base+blk_offs[:,None]*stride_tq_pos+nb3[None,:],mask=tm3&nv3[None,:],other=0).to(tl.int32)
        vr0a=tl.load(tq_cache_ptr+v_base+to0,mask=tm0,other=0).to(tl.int32)
        vr0b=tl.load(tq_cache_ptr+v_base+blk_offs[:,None]*stride_tq_pos+nb0[None,:],mask=tm0&nv0[None,:],other=0).to(tl.int32)
        vr1a=tl.load(tq_cache_ptr+v_base+to1,mask=tm1,other=0).to(tl.int32)
        vr1b=tl.load(tq_cache_ptr+v_base+blk_offs[:,None]*stride_tq_pos+nb1[None,:],mask=tm1&nv1[None,:],other=0).to(tl.int32)
        vr2a=tl.load(tq_cache_ptr+v_base+to2,mask=tm2,other=0).to(tl.int32)
        vr2b=tl.load(tq_cache_ptr+v_base+blk_offs[:,None]*stride_tq_pos+nb2[None,:],mask=tm2&nv2[None,:],other=0).to(tl.int32)
        vr3a=tl.load(tq_cache_ptr+v_base+to3,mask=tm3,other=0).to(tl.int32)
        vr3b=tl.load(tq_cache_ptr+v_base+blk_offs[:,None]*stride_tq_pos+nb3[None,:],mask=tm3&nv3[None,:],other=0).to(tl.int32)
        k_norm=((n0k|(n1k<<8)).to(tl.int16)).to(tl.float16,bitcast=True).to(tl.float32)
        ik0=(((kr0a|(kr0b<<8))>>bo0[None,:])&7); kc0=(tl.load(centroids_ptr+ik0)*k_norm[:,None]).to(tl.float16)
        ik1=(((kr1a|(kr1b<<8))>>bo1[None,:])&7); kc1=(tl.load(centroids_ptr+ik1)*k_norm[:,None]).to(tl.float16)
        ik2=(((kr2a|(kr2b<<8))>>bo2[None,:])&7); kc2=(tl.load(centroids_ptr+ik2)*k_norm[:,None]).to(tl.float16)
        ik3=(((kr3a|(kr3b<<8))>>bo3[None,:])&7); kc3=(tl.load(centroids_ptr+ik3)*k_norm[:,None]).to(tl.float16)
        qk = tl.dot(q0,tl.trans(kc0),out_dtype=tl.float32)
        qk += tl.dot(q1,tl.trans(kc1),out_dtype=tl.float32)
        qk += tl.dot(q2,tl.trans(kc2),out_dtype=tl.float32)
        qk += tl.dot(q3,tl.trans(kc3),out_dtype=tl.float32)
        qk = tl.where((grp_offs[:,None]<QUERY_GRP_SZ)&(blk_seq_offs[None,:]<seq_len),qk,float("-inf"))
        ml_new = tl.maximum(tl.max(qk,axis=1), max_logit)
        p = tl.math.exp2((qk-ml_new[:,None])*log2e)
        alpha = tl.math.exp2((max_logit-ml_new)*log2e)
        a0*=alpha[:,None]; a1*=alpha[:,None]; a2*=alpha[:,None]; a3*=alpha[:,None]
        v_norm=((n0v|(n1v<<8)).to(tl.int16)).to(tl.float16,bitcast=True).to(tl.float32)
        p16 = p.to(tl.float16)
        iv0=(((vr0a|(vr0b<<8))>>bo0[None,:])&7); a0+=tl.dot(p16,(tl.load(centroids_ptr+iv0)*v_norm[:,None]).to(tl.float16),out_dtype=tl.float32)
        iv1=(((vr1a|(vr1b<<8))>>bo1[None,:])&7); a1+=tl.dot(p16,(tl.load(centroids_ptr+iv1)*v_norm[:,None]).to(tl.float16),out_dtype=tl.float32)
        iv2=(((vr2a|(vr2b<<8))>>bo2[None,:])&7); a2+=tl.dot(p16,(tl.load(centroids_ptr+iv2)*v_norm[:,None]).to(tl.float16),out_dtype=tl.float32)
        iv3=(((vr3a|(vr3b<<8))>>bo3[None,:])&7); a3+=tl.dot(p16,(tl.load(centroids_ptr+iv3)*v_norm[:,None]).to(tl.float16),out_dtype=tl.float32)
        exp_sum = exp_sum*alpha + tl.sum(p, axis=1)
        max_logit = ml_new
    inv = 1.0 / exp_sum
    a0*=inv[:,None]; a1*=inv[:,None]; a2*=inv[:,None]; a3*=inv[:,None]
    ob = seq_idx*stride_o_s + (kv_head_idx*QUERY_GRP_SZ + grp_offs[:,None])*stride_o_nh
    om = grp_offs[:,None] < QUERY_GRP_SZ
    tl.store(out+ob+cd[None,:], a0.to(out.dtype.element_ty), mask=om)
    tl.store(out+ob+(CD_C+cd)[None,:], a1.to(out.dtype.element_ty), mask=om)
    tl.store(out+ob+(2*CD_C+cd)[None,:], a2.to(out.dtype.element_ty), mask=om)
    tl.store(out+ob+(3*CD_C+cd)[None,:], a3.to(out.dtype.element_ty), mask=om)


def benchmark():
    """Compare v11 (no sparse) vs v12 (sparse V) at various sequence lengths."""
    import math
    torch.manual_seed(42)
    device = 'cuda'
    
    num_kv_heads = 4
    gqa_ratio = 6
    num_heads = num_kv_heads * gqa_ratio
    head_size = D
    block_size = 32
    scale = 1.0 / math.sqrt(head_size)
    slot_bytes = D  # 128 bytes per slot (48 packed + 2 norm + padding)
    
    centroids = torch.tensor([-1.5, -0.8, -0.3, -0.05, 0.05, 0.3, 0.8, 1.5],
                             dtype=torch.float32, device=device)
    
    KBP = triton.next_power_of_2(block_size)
    GRP = triton.next_power_of_2(gqa_ratio)
    
    print(f"{'Config':>20s} | {'v11 (ms)':>10s} | {'v12 sparse (ms)':>16s} | {'Speedup':>8s} | {'cos_sim':>8s}")
    print("-" * 75)
    
    for ns, slen in [(1, 1024), (1, 4096), (1, 16384), (1, 32768),
                     (8, 4096), (32, 4096), (64, 4096), (192, 4096)]:
        num_blocks = (slen + block_size - 1) // block_size
        total_blocks = ns * num_blocks + 16
        
        q = torch.randn(ns, num_heads, head_size, dtype=torch.float16, device=device)
        # TQ cache: [num_blocks, 2(K/V), block_size, num_kv_heads, slot_bytes] uint8
        tq_cache = torch.randint(0, 256, (total_blocks, 2, block_size, num_kv_heads, slot_bytes),
                                 dtype=torch.uint8, device=device)
        # Write valid FP16 norms at bytes 48-49
        for blk in range(min(total_blocks, 100)):  # just first 100 blocks
            for kv in range(2):
                for pos in range(block_size):
                    for h in range(num_kv_heads):
                        norm_val = torch.tensor([0.5], dtype=torch.float16)
                        norm_bytes = norm_val.view(torch.uint8)
                        tq_cache[blk, kv, pos, h, DQ3] = norm_bytes[0]
                        tq_cache[blk, kv, pos, h, DQ3+1] = norm_bytes[1]
        
        block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0).expand(ns, -1).contiguous()
        if ns > 1:
            offsets = torch.arange(ns, device=device, dtype=torch.int32) * num_blocks
            block_tables = block_tables + offsets.unsqueeze(1)
        
        seq_lens = torch.full((ns,), slen, dtype=torch.int32, device=device)
        
        # Warmup + benchmark v11
        out11 = torch.empty_like(q)
        for _ in range(3):
            _tq_v11_single_kernel[(ns, num_kv_heads)](
                out11, q, tq_cache, centroids, block_tables, seq_lens, scale,
                out11.stride(0), out11.stride(1), q.stride(0), q.stride(1),
                tq_cache.stride(0), tq_cache.stride(1), tq_cache.stride(2), tq_cache.stride(3),
                block_tables.stride(0),
                KV_BLK_SZ=block_size, KV_BLK_SZ_POW2=KBP, HEAD_SZ=head_size,
                QUERY_GRP_SZ=gqa_ratio, QUERY_GRP_SZ_POW2=GRP, DQ3_C=DQ3, CD_C=CD)
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        ITERS = 20
        for _ in range(ITERS):
            _tq_v11_single_kernel[(ns, num_kv_heads)](
                out11, q, tq_cache, centroids, block_tables, seq_lens, scale,
                out11.stride(0), out11.stride(1), q.stride(0), q.stride(1),
                tq_cache.stride(0), tq_cache.stride(1), tq_cache.stride(2), tq_cache.stride(3),
                block_tables.stride(0),
                KV_BLK_SZ=block_size, KV_BLK_SZ_POW2=KBP, HEAD_SZ=head_size,
                QUERY_GRP_SZ=gqa_ratio, QUERY_GRP_SZ_POW2=GRP, DQ3_C=DQ3, CD_C=CD)
        torch.cuda.synchronize()
        v11_ms = (time.perf_counter() - t0) / ITERS * 1000
        
        # Warmup + benchmark v12
        out12 = torch.empty_like(q)
        for _ in range(3):
            _tq_v12_sparse_v_kernel[(ns, num_kv_heads)](
                out12, q, tq_cache, centroids, block_tables, seq_lens, scale,
                out12.stride(0), out12.stride(1), q.stride(0), q.stride(1),
                tq_cache.stride(0), tq_cache.stride(1), tq_cache.stride(2), tq_cache.stride(3),
                block_tables.stride(0),
                KV_BLK_SZ=block_size, KV_BLK_SZ_POW2=KBP, HEAD_SZ=head_size,
                QUERY_GRP_SZ=gqa_ratio, QUERY_GRP_SZ_POW2=GRP, DQ3_C=DQ3, CD_C=CD,
                SPARSE_THRESH=SPARSE_V_THRESHOLD)
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        for _ in range(ITERS):
            _tq_v12_sparse_v_kernel[(ns, num_kv_heads)](
                out12, q, tq_cache, centroids, block_tables, seq_lens, scale,
                out12.stride(0), out12.stride(1), q.stride(0), q.stride(1),
                tq_cache.stride(0), tq_cache.stride(1), tq_cache.stride(2), tq_cache.stride(3),
                block_tables.stride(0),
                KV_BLK_SZ=block_size, KV_BLK_SZ_POW2=KBP, HEAD_SZ=head_size,
                QUERY_GRP_SZ=gqa_ratio, QUERY_GRP_SZ_POW2=GRP, DQ3_C=DQ3, CD_C=CD,
                SPARSE_THRESH=SPARSE_V_THRESHOLD)
        torch.cuda.synchronize()
        v12_ms = (time.perf_counter() - t0) / ITERS * 1000
        
        # Correctness check
        cos = torch.nn.functional.cosine_similarity(
            out11.flatten().float(), out12.flatten().float(), dim=0).item()
        
        speedup = v11_ms / v12_ms
        print(f"  B={ns:>3d},S={slen:>5d} | {v11_ms:>10.3f} | {v12_ms:>16.3f} | {speedup:>7.2f}x | {cos:>8.4f}")


if __name__ == "__main__":
    benchmark()
