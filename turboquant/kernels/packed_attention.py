#!/usr/bin/env python3
"""TurboQuant Packed Attention v7: production kernel for MI355X.

Reads packed 2-bit indices + 1-bit signs (pre-permuted to stride-4 order),
unpacks in registers, reconstructs K_mse/V_mse via codebook gather,
computes asymmetric attention with fully chunked score computation.

Split-K architecture: grid = (num_heads, num_splits)

Optimizations (cumulative):
  v3b: Chunked K dot product: 4x [BS, DQ] partial sums instead of [BS, D]
  v3b: Codebook gather via tl.load(Cb_ptr + idx) instead of tl.where chain
  v3b: Stride-4 Q loading to match packed byte interleaving
  v5b: Pre-permuted sign storage: QJL score also chunked to 4x [BS, DQ]
       (signs reordered during compression via permute_signs_for_chunked)
  v6f: Interleaved loads: batch all K idx + 4 sign chunk loads + norms
       before any compute, giving memory controller more prefetch time
  v7:  Precomputed Q_proj * corr_scale — eliminates 4 scalar multiplies
       per tile per token inside the inner loop
  Auto-dispatch: selects optimal (BLOCK_SK, num_splits, warps) per (BH, Sk)

Tuned on AMD Instinct MI355X (gfx950, 256 CUs).

Requires signs in permuted order — use compress_k_permuted() from packed_compress.py.
"""

import math, time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def solve_codebook(d, bits):
    from scipy import integrate
    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)
    def pdf(x):
        return (1.0 / math.sqrt(2 * math.pi * sigma**2)) * math.exp(-x*x / (2*sigma**2))
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
    for _ in range(200):
        boundaries = [(centroids[i] + centroids[i+1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_c = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            new_c.append(num / den if den > 1e-15 else centroids[i])
        if max(abs(new_c[i] - centroids[i]) for i in range(n_levels)) < 1e-10:
            break
        centroids = new_c
    return torch.tensor(centroids, dtype=torch.float32)


# ===========================================================================
# Unpack utilities (used for testing)
# ===========================================================================
def unpack_2bit(packed, D):
    N = packed.shape[0]
    p = packed.to(torch.int32)
    return torch.stack([p & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3], dim=-1).reshape(N, D).to(torch.int8)

def unpack_1bit(packed, D):
    N = packed.shape[0]
    p = packed.to(torch.int32).unsqueeze(-1)
    bits = torch.arange(8, device=packed.device)
    return ((p >> bits) & 1).reshape(N, D).to(torch.int8)


# ===========================================================================
# v7: Interleaved Loads + Chunked Dot + Pre-Permuted Signs + Fused corr_scale
# ===========================================================================
@triton.jit
def _packed_attn_v7(
    Q_ptr,             # [BH, D] fp16
    Q_proj_cs_ptr,     # [BH, D] fp16 — Q_proj * corr_scale (precomputed)
    Kidx_ptr,          # [BH, Sk, DQ] uint8 — packed 2-bit K indices
    Ksigns_perm_ptr,   # [BH, Sk, DE] uint8 — packed 1-bit K signs (PERMUTED)
    Krnorm_ptr,        # [BH, Sk] fp16
    Knorm_ptr,         # [BH, Sk] fp16
    Vidx_ptr,          # [BH, Sk, DQ] uint8 — packed 2-bit V indices
    Vnorm_ptr,         # [BH, Sk] fp16
    Cb_ptr,            # [4] fp32 — codebook centroids
    Out_ptr,           # [BH, num_splits, D] fp32
    Lse_ptr,           # [BH, num_splits] fp32
    Sk,
    num_splits,
    D: tl.constexpr,
    DQ: tl.constexpr,     # D // 4 = 32
    DE: tl.constexpr,     # D // 8 = 16
    SDE: tl.constexpr,    # DQ // 8 = 4 (sign bytes per chunk)
    BLOCK_SK: tl.constexpr,
):
    head_id = tl.program_id(0)
    split_id = tl.program_id(1)
    tokens_per_split = (Sk + num_splits - 1) // num_splits
    sk_start = split_id * tokens_per_split
    sk_end = tl.minimum(sk_start + tokens_per_split, Sk)

    rdq = tl.arange(0, DQ)
    rd = tl.arange(0, D)

    # Stride-4 Q loading to match packed byte interleaving
    q0 = tl.load(Q_ptr + head_id*D + rdq*4 + 0).to(tl.float32)
    q1 = tl.load(Q_ptr + head_id*D + rdq*4 + 1).to(tl.float32)
    q2 = tl.load(Q_ptr + head_id*D + rdq*4 + 2).to(tl.float32)
    q3 = tl.load(Q_ptr + head_id*D + rdq*4 + 3).to(tl.float32)

    # Stride-4 Q_proj*corr_scale loading for chunked QJL score (corr_scale precomputed)
    qp0 = tl.load(Q_proj_cs_ptr + head_id*D + rdq*4 + 0).to(tl.float32)
    qp1 = tl.load(Q_proj_cs_ptr + head_id*D + rdq*4 + 1).to(tl.float32)
    qp2 = tl.load(Q_proj_cs_ptr + head_id*D + rdq*4 + 2).to(tl.float32)
    qp3 = tl.load(Q_proj_cs_ptr + head_id*D + rdq*4 + 3).to(tl.float32)

    # Sign unpack indices for permuted layout:
    # Within each chunk of DQ=32 dims, sign byte = rdq // 8, bit = rdq % 8
    sign_byte_in_chunk = (rdq // 8).to(tl.int32)   # [DQ]
    sign_bit_in_byte = (rdq % 8).to(tl.int32)       # [DQ]

    # V unpack maps (full D, unchanged)
    packed_col_full = (rd // 4).to(tl.int32)
    shift_2bit_full = ((rd % 4) * 2).to(tl.int32)

    # Online softmax state
    m_prev = float('-inf')
    l_prev = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    for tile_start in range(sk_start, sk_end, BLOCK_SK):
        tile_end = tl.minimum(tile_start + BLOCK_SK, sk_end)
        rsk = tile_start + tl.arange(0, BLOCK_SK)
        sk_mask = rsk < tile_end

        # === PHASE 1: Batch all loads (K idx + 4 sign chunks + norms) ===
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

        # === PHASE 2: Compute K MSE score from registers ===
        kr0 = tl.load(Cb_ptr + (kidx_packed & 3))
        kr1 = tl.load(Cb_ptr + ((kidx_packed >> 2) & 3))
        kr2 = tl.load(Cb_ptr + ((kidx_packed >> 4) & 3))
        kr3 = tl.load(Cb_ptr + ((kidx_packed >> 6) & 3))

        score_mse = (tl.sum(kr0 * q0[None, :], axis=1) +
                     tl.sum(kr1 * q1[None, :], axis=1) +
                     tl.sum(kr2 * q2[None, :], axis=1) +
                     tl.sum(kr3 * q3[None, :], axis=1)) * k_norms

        # === PHASE 3: Compute chunked QJL score (corr_scale already in qp) ===
        sf0 = ((s0_raw >> sign_bit_in_byte[None, :]) & 1).to(tl.float32) * 2.0 - 1.0
        sqjl0 = tl.sum(qp0[None, :] * sf0 * k_rnorms[:, None], axis=1)

        sf1 = ((s1_raw >> sign_bit_in_byte[None, :]) & 1).to(tl.float32) * 2.0 - 1.0
        sqjl1 = tl.sum(qp1[None, :] * sf1 * k_rnorms[:, None], axis=1)

        sf2 = ((s2_raw >> sign_bit_in_byte[None, :]) & 1).to(tl.float32) * 2.0 - 1.0
        sqjl2 = tl.sum(qp2[None, :] * sf2 * k_rnorms[:, None], axis=1)

        sf3 = ((s3_raw >> sign_bit_in_byte[None, :]) & 1).to(tl.float32) * 2.0 - 1.0
        sqjl3 = tl.sum(qp3[None, :] * sf3 * k_rnorms[:, None], axis=1)

        score = tl.where(sk_mask, score_mse + sqjl0 + sqjl1 + sqjl2 + sqjl3, float('-inf'))

        # === Online softmax ===
        m_new = tl.maximum(m_prev, tl.max(score, axis=0))
        alpha = tl.exp(m_prev - m_new)
        p = tl.exp(score - m_new)
        l_new = alpha * l_prev + tl.sum(p, axis=0)

        # === V: full [BS, D] reconstruction via indirect gather ===
        vidx_raw = tl.load(Vidx_ptr + head_id*Sk*DQ + rsk[:, None]*DQ + packed_col_full[None, :],
                           mask=sk_mask[:, None], other=0).to(tl.int32)
        v_idx_full = (vidx_raw >> shift_2bit_full[None, :]) & 3
        v_recon_full = tl.load(Cb_ptr + v_idx_full)
        v_norms = tl.load(Vnorm_ptr + head_id*Sk + rsk, mask=sk_mask, other=0.0).to(tl.float32)
        v_mse = v_recon_full * v_norms[:, None]

        acc = alpha * acc + tl.sum(p[:, None] * v_mse, axis=0)
        m_prev = m_new
        l_prev = l_new

    out_offset = head_id * num_splits * D + split_id * D
    tl.store(Out_ptr + out_offset + rd, acc)
    tl.store(Lse_ptr + head_id*num_splits + split_id, m_prev + tl.log(l_prev))


@triton.jit
def _splitk_reduce(
    Out_ptr,    # [BH, num_splits, D] fp32
    Lse_ptr,    # [BH, num_splits] fp32
    Final_ptr,  # [BH, D] fp16
    num_splits,
    D: tl.constexpr,
    NS: tl.constexpr,
):
    head_id = tl.program_id(0)
    rd = tl.arange(0, D)
    rs = tl.arange(0, NS)

    lses = tl.load(Lse_ptr + head_id * num_splits + rs,
                   mask=rs < num_splits, other=float('-inf'))
    max_lse = tl.max(lses, axis=0)

    acc = tl.zeros([D], dtype=tl.float32)
    total_w = 0.0
    for s in range(NS):
        if s < num_splits:
            lse_s = tl.load(Lse_ptr + head_id * num_splits + s)
            w = tl.exp(lse_s - max_lse)
            out_s = tl.load(Out_ptr + head_id * num_splits * D + s * D + rd)
            acc += w * out_s
            total_w += w

    acc = acc / total_w
    tl.store(Final_ptr + head_id * D + rd, acc.to(tl.float16))


# ===========================================================================
# Auto-dispatch: select optimal (BLOCK_SK, num_splits, warps) per (BH, Sk)
# Tuned on MI355X (gfx950, 256 CUs), March 2026
# ===========================================================================
# Tuning table: (BH_threshold, Sk_threshold) -> (BLOCK_SK, num_splits, num_warps, num_stages)
# Searched over BS={16,32,64,128} x sp={4..128} x w={2,4,8} x s={1,2}
_TUNING_TABLE = [
    # (max_BH, max_Sk, BLOCK_SK, num_splits, num_warps, num_stages)
    # BH <= 16
    (16,  1024,  32,  16,  4,  2),   # 1.86-1.88x
    (16,  4096, 128,  16,  8,  1),   # 1.88x
    (16, 16384,  32,  64,  8,  2),   # 2.84x
    (16, 99999,  64,  64,  8,  2),   # re-tuned for v7: +28% over BS=32

    # BH 17-48
    (48,  1024,  16,  16,  4,  2),   # 1.87x
    (48,  4096,  16,  64,  4,  1),   # 3.11-3.13x
    (48, 16384,  16,  64,  2,  2),   # 2.73x
    (48, 99999,  16,  64,  2,  1),   # 2.78x

    # BH >= 49
    (99999,  1024,  16,  16,  2,  1),   # 1.87x
    (99999,  4096,  16,  32,  2,  2),   # 3.03x
    (99999, 16384,  16,  32,  2,  2),   # 2.83x
    (99999, 99999,  16, 128,  2,  1),   # 2.75x
]


def _get_config(BH, Sk):
    """Look up tuned config from table."""
    for max_bh, max_sk, bs, sp, w, s in _TUNING_TABLE:
        if BH <= max_bh and Sk <= max_sk:
            # Clamp num_splits so we don't have more splits than tiles
            max_splits = max(1, Sk // bs)
            sp = min(sp, max_splits)
            return bs, sp, w, s
    # Fallback
    return 16, 64, 2, 1


# ===========================================================================
# Reference: unpacked attention (for correctness comparison)
# ===========================================================================
def attention_unpacked_ref(Q, Q_proj, K_mse_rot, signs_float, k_r_norm, V_mse_rot, corr_scale):
    """PyTorch reference with unpacked data."""
    BH, D = Q.shape

    score_mse = torch.bmm(Q.unsqueeze(1), K_mse_rot.transpose(1, 2)).squeeze(1)
    K_qjl = signs_float * k_r_norm.unsqueeze(-1) * corr_scale
    score_qjl = torch.bmm(Q_proj.unsqueeze(1), K_qjl.transpose(1, 2)).squeeze(1)
    scores = score_mse + score_qjl

    attn = F.softmax(scores, dim=-1)
    output = torch.bmm(attn.unsqueeze(1), V_mse_rot).squeeze(1)
    return output


# ===========================================================================
# Wrapper
# ===========================================================================
def packed_attention(Q, Q_proj, k_idx_packed, k_signs_packed, k_r_norm, k_norm,
                     v_idx_packed, v_norm, centroids, corr_scale,
                     num_splits=None, block_sk=None, num_warps=None, num_stages=None):
    """Packed attention with split-K and auto-dispatch.

    Uses v7 kernel: interleaved loads + chunked dot + pre-permuted QJL signs
    + precomputed Q_proj*corr_scale (saves 4 multiplies per tile per token).
    Signs must be in permuted (stride-4) order — use compress_k_permuted()
    from packed_compress.py, or permute_signs_for_chunked() on existing signs.
    Auto-selects (BLOCK_SK, num_splits, warps, stages) from tuning table
    unless overridden by explicit arguments.
    """
    BH, D = Q.shape
    Sk = k_r_norm.shape[1]
    DQ = D // 4
    DE = D // 8

    # Auto-dispatch from tuning table
    bs_auto, sp_auto, w_auto, s_auto = _get_config(BH, Sk)
    BS = block_sk if block_sk is not None else bs_auto
    nsplits = num_splits if num_splits is not None else sp_auto
    nw = num_warps if num_warps is not None else w_auto
    ns = num_stages if num_stages is not None else s_auto

    SDE = DQ // 8  # sign bytes per chunk = 4
    cb_ptr = centroids.contiguous()

    # Precompute Q_proj * corr_scale (optimization G+I: saves 4 muls/tile)
    Q_proj_cs = (Q_proj.float() * corr_scale).to(Q_proj.dtype)

    out_splits = torch.empty(BH, nsplits, D, dtype=torch.float32, device=Q.device)
    lse_splits = torch.empty(BH, nsplits, dtype=torch.float32, device=Q.device)

    _packed_attn_v7[(BH, nsplits)](
        Q, Q_proj_cs,
        k_idx_packed, k_signs_packed, k_r_norm, k_norm,
        v_idx_packed, v_norm,
        cb_ptr,
        out_splits, lse_splits,
        Sk, nsplits,
        D=D, DQ=DQ, DE=DE, SDE=SDE, BLOCK_SK=BS,
        num_warps=nw, num_stages=ns)

    # Reduce splits
    NS_pad = 1
    while NS_pad < nsplits:
        NS_pad *= 2

    final = torch.empty(BH, D, dtype=torch.float16, device=Q.device)
    _splitk_reduce[(BH,)](
        out_splits, lse_splits, final,
        nsplits, D=D, NS=NS_pad,
        num_warps=4)

    return final


def run():
    device = "cuda"; D = 128
    bits = 3; mse_bits = bits - 1; seed = 42

    gen = torch.Generator(device=device); gen.manual_seed(seed)
    G = torch.randn(D, D, device=device, generator=gen)
    Q_orth, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R)); diag_sign[diag_sign == 0] = 1.0
    Pi = (Q_orth * diag_sign.unsqueeze(0)); PiT = Pi.T.contiguous()
    gen2 = torch.Generator(device=device); gen2.manual_seed(seed + 10000)
    S = torch.randn(D, D, device=device, generator=gen2); ST = S.T.contiguous()

    centroids = solve_codebook(D, mse_bits).to(device)
    corr_scale = math.sqrt(math.pi / 2) / math.sqrt(D)

    from packed_compress import permute_signs_for_chunked
    print(f"Packed Attention v7 (production) — MI355X")
    print(f"D={D}, bits={bits}")
    print(f"Device: {torch.cuda.get_device_name(0) or 'MI355X'}\n")

    warmup, iters = 20, 200

    # === Correctness ===
    print("=== Correctness ===")
    for BH in [8, 32, 64]:
        for Sk in [256, 4096, 16384, 32768]:
            k_idx_packed = torch.randint(0, 256, (BH, Sk, D//4), dtype=torch.uint8, device=device)
            k_signs_packed = torch.randint(0, 256, (BH, Sk, D//8), dtype=torch.uint8, device=device)
            k_r_norm = torch.rand(BH, Sk, dtype=torch.float16, device=device) * 0.1
            k_norm = torch.rand(BH, Sk, dtype=torch.float16, device=device) + 0.5
            v_idx_packed = torch.randint(0, 256, (BH, Sk, D//4), dtype=torch.uint8, device=device)
            v_norm = torch.rand(BH, Sk, dtype=torch.float16, device=device) + 0.5
            Q = torch.randn(BH, D, dtype=torch.float16, device=device) * 0.1
            Q_proj = torch.randn(BH, D, dtype=torch.float16, device=device) * 0.1

            # Unpack for reference
            k_idx = unpack_2bit(k_idx_packed.reshape(-1, D//4), D).reshape(BH, Sk, D)
            k_signs = unpack_1bit(k_signs_packed.reshape(-1, D//8), D).reshape(BH, Sk, D)
            v_idx = unpack_2bit(v_idx_packed.reshape(-1, D//4), D).reshape(BH, Sk, D)

            K_mse_rot = centroids[k_idx.long()] * k_norm.unsqueeze(-1)
            signs_float = k_signs.float() * 2 - 1
            V_mse_rot = centroids[v_idx.long()] * v_norm.unsqueeze(-1)

            ref = attention_unpacked_ref(Q.float(), Q_proj.float(),
                                        K_mse_rot.float(), signs_float, k_r_norm.float(),
                                        V_mse_rot.float(), corr_scale)

            # Permute signs for v5b kernel
            k_signs_perm = permute_signs_for_chunked(k_signs_packed, D)
            packed_out = packed_attention(Q, Q_proj, k_idx_packed, k_signs_perm,
                                         k_r_norm, k_norm, v_idx_packed, v_norm,
                                         centroids, corr_scale)

            bs, sp, w, s = _get_config(BH, Sk)
            cs = F.cosine_similarity(ref.reshape(1, -1), packed_out.float().reshape(1, -1)).item()
            status = "PASS" if cs > 0.99 else "FAIL"
            print(f"  BH={BH:>3} Sk={Sk:>5}: cos={cs:.6f} {status}  (BS={bs} sp={sp} w={w})")

    # === Benchmark ===
    print(f"\n=== Benchmark (auto-dispatch) ===")
    print(f"{'BH':>4} {'Sk':>6} {'v6f_ms':>9} {'ref_ms':>9} {'speedup':>8} {'config':>20}")
    print("-" * 60)

    for BH in [8, 32, 64]:
        for Sk in [256, 1024, 4096, 16384, 32768]:
            k_idx_packed = torch.randint(0, 256, (BH, Sk, D//4), dtype=torch.uint8, device=device)
            k_signs_packed = torch.randint(0, 256, (BH, Sk, D//8), dtype=torch.uint8, device=device)
            k_r_norm = torch.rand(BH, Sk, dtype=torch.float16, device=device) * 0.1
            k_norm = torch.rand(BH, Sk, dtype=torch.float16, device=device) + 0.5
            v_idx_packed = torch.randint(0, 256, (BH, Sk, D//4), dtype=torch.uint8, device=device)
            v_norm = torch.rand(BH, Sk, dtype=torch.float16, device=device) + 0.5
            Q = torch.randn(BH, D, dtype=torch.float16, device=device) * 0.1
            Q_proj = torch.randn(BH, D, dtype=torch.float16, device=device) * 0.1

            # v5b kernel (permuted signs)
            k_signs_perm = permute_signs_for_chunked(k_signs_packed, D)
            for _ in range(warmup):
                packed_attention(Q, Q_proj, k_idx_packed, k_signs_perm,
                               k_r_norm, k_norm, v_idx_packed, v_norm,
                               centroids, corr_scale)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                packed_attention(Q, Q_proj, k_idx_packed, k_signs_perm,
                               k_r_norm, k_norm, v_idx_packed, v_norm,
                               centroids, corr_scale)
            torch.cuda.synchronize()
            packed_ms = (time.perf_counter()-t0)/iters*1000

            # PyTorch reference
            k_idx = unpack_2bit(k_idx_packed.reshape(-1, D//4), D).reshape(BH, Sk, D)
            k_signs = unpack_1bit(k_signs_packed.reshape(-1, D//8), D).reshape(BH, Sk, D)
            v_idx = unpack_2bit(v_idx_packed.reshape(-1, D//4), D).reshape(BH, Sk, D)
            K_mse_rot = centroids[k_idx.long()] * k_norm.unsqueeze(-1)
            signs_float = k_signs.float() * 2 - 1
            V_mse_rot = centroids[v_idx.long()] * v_norm.unsqueeze(-1)

            for _ in range(warmup):
                attention_unpacked_ref(Q.float(), Q_proj.float(),
                                     K_mse_rot.float(), signs_float, k_r_norm.float(),
                                     V_mse_rot.float(), corr_scale)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                attention_unpacked_ref(Q.float(), Q_proj.float(),
                                     K_mse_rot.float(), signs_float, k_r_norm.float(),
                                     V_mse_rot.float(), corr_scale)
            torch.cuda.synchronize()
            ref_ms = (time.perf_counter()-t0)/iters*1000

            bs, sp, w, s = _get_config(BH, Sk)
            sp_val = ref_ms / packed_ms if packed_ms > 0 else 0
            print(f"{BH:>4} {Sk:>6} {packed_ms:>8.4f}ms {ref_ms:>8.4f}ms {sp_val:>7.2f}x"
                  f"  BS={bs} sp={sp} w={w}")

    print("\nDone!")

if __name__ == "__main__":
    run()
