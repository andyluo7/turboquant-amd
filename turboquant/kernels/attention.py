#!/usr/bin/env python3
"""Unified TurboQuant Fused Asymmetric Attention for MI355X.

Adaptive dispatch:
  - Sk <= SK_THRESHOLD (4096): v2 concat kernel (fewer reductions)
  - Sk > SK_THRESHOLD: v1 split-K kernel (less bandwidth)
  - Sq == 1: decode kernels (split-K)
  - Sq > 1: general tiled kernel

Auto-selects optimal num_splits based on BH and Sk.
"""

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SK_THRESHOLD = 0   # v1 separate loads everywhere
BK_DEFAULT = 256


# ===========================================================================
# Kernels — v1 Split-K (separate loads, 2 dot products)
# ===========================================================================
@triton.jit
def _splitk_v1_partial(
    Q_ptr, Q_proj_ptr, K_mse_ptr, Signs_ptr, R_norm_ptr, V_mse_ptr,
    M_ptr, L_ptr, Acc_ptr,
    sm_scale, corr_scale, Sk,
    D: tl.constexpr, BK: tl.constexpr, num_splits: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_split = tl.program_id(1)
    split_size = (Sk + num_splits - 1) // num_splits
    k_begin = pid_split * split_size
    k_end = min(k_begin + split_size, Sk)
    rd = tl.arange(0, D)
    q = tl.load(Q_ptr + pid_bh * D + rd).to(tl.float32)
    q_proj = tl.load(Q_proj_ptr + pid_bh * D + rd).to(tl.float32)
    kv_base = pid_bh * Sk
    m_i = float('-inf'); l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for k_start in range(k_begin, k_end, BK):
        rk = k_start + tl.arange(0, BK)
        k_mask = rk < k_end
        k_mse = tl.load(K_mse_ptr + (kv_base + rk[:, None]) * D + rd[None, :],
                        mask=k_mask[:, None], other=0.0).to(tl.float32)
        term1 = tl.sum(q[None, :] * k_mse, axis=1)
        signs = tl.load(Signs_ptr + (kv_base + rk[:, None]) * D + rd[None, :],
                        mask=k_mask[:, None], other=0).to(tl.float32)
        qjl_ip = tl.sum(q_proj[None, :] * signs, axis=1)
        r_norm = tl.load(R_norm_ptr + kv_base + rk, mask=k_mask, other=0.0).to(tl.float32)
        scores = (term1 + corr_scale * qjl_ip * r_norm) * sm_scale
        scores = tl.where(k_mask, scores, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_new = alpha * l_i + tl.sum(p, axis=0)
        acc = acc * alpha
        v_mse = tl.load(V_mse_ptr + (kv_base + rk[:, None]) * D + rd[None, :],
                        mask=k_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(p[:, None] * v_mse, axis=0)
        m_i = m_new; l_i = l_new
    split_off = pid_bh * num_splits + pid_split
    tl.store(M_ptr + split_off, m_i)
    tl.store(L_ptr + split_off, l_i)
    tl.store(Acc_ptr + split_off * D + rd, acc)


# ===========================================================================
# Kernels — v2 Split-K Concat (single dot product over 2D)
# ===========================================================================
@triton.jit
def _splitk_v2_partial(
    Q_combined_ptr, K_combined_ptr, V_mse_ptr,
    M_ptr, L_ptr, Acc_ptr,
    sm_scale, Sk,
    D: tl.constexpr, D2: tl.constexpr, BK: tl.constexpr,
    num_splits: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_split = tl.program_id(1)
    split_size = (Sk + num_splits - 1) // num_splits
    k_begin = pid_split * split_size
    k_end = min(k_begin + split_size, Sk)
    rd2 = tl.arange(0, D2)
    rd = tl.arange(0, D)
    q_comb = tl.load(Q_combined_ptr + pid_bh * D2 + rd2).to(tl.float32)
    kv_base = pid_bh * Sk
    m_i = float('-inf'); l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for k_start in range(k_begin, k_end, BK):
        rk = k_start + tl.arange(0, BK)
        k_mask = rk < k_end
        k_comb = tl.load(K_combined_ptr + (kv_base + rk[:, None]) * D2 + rd2[None, :],
                         mask=k_mask[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(q_comb[None, :] * k_comb, axis=1) * sm_scale
        scores = tl.where(k_mask, scores, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_new = alpha * l_i + tl.sum(p, axis=0)
        acc = acc * alpha
        v_mse = tl.load(V_mse_ptr + (kv_base + rk[:, None]) * D + rd[None, :],
                        mask=k_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(p[:, None] * v_mse, axis=0)
        m_i = m_new; l_i = l_new
    split_off = pid_bh * num_splits + pid_split
    tl.store(M_ptr + split_off, m_i)
    tl.store(L_ptr + split_off, l_i)
    tl.store(Acc_ptr + split_off * D + rd, acc)


# ===========================================================================
# Reduce (shared)
# ===========================================================================
@triton.jit
def _splitk_reduce(
    M_ptr, L_ptr, Acc_ptr, O_ptr,
    D: tl.constexpr, num_splits: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    rd = tl.arange(0, D)
    m_global = float('-inf')
    for s in range(num_splits):
        m_global = tl.maximum(m_global, tl.load(M_ptr + pid_bh * num_splits + s))
    l_global = 0.0
    acc_global = tl.zeros([D], dtype=tl.float32)
    for s in range(num_splits):
        m_s = tl.load(M_ptr + pid_bh * num_splits + s)
        l_s = tl.load(L_ptr + pid_bh * num_splits + s)
        acc_s = tl.load(Acc_ptr + (pid_bh * num_splits + s) * D + rd)
        alpha = tl.exp(m_s - m_global)
        l_global += alpha * l_s
        acc_global += alpha * acc_s
    tl.store(O_ptr + pid_bh * D + rd, acc_global / l_global)


# ===========================================================================
# General kernel (Sq > 1) — v1 style (no split-K, tl.dot based)
# ===========================================================================
@triton.jit
def _fused_general(
    Q_ptr, Q_proj_ptr, K_mse_ptr, Signs_ptr, R_norm_ptr, V_mse_ptr, O_ptr,
    sm_scale, corr_scale,
    stride_q_bh, stride_q_sq, stride_k_bh, stride_k_sk, stride_n_bh,
    Sq, Sk,
    D: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    q_start = pid_q * BQ
    Q_bh = Q_ptr + pid_bh * stride_q_bh
    QP_bh = Q_proj_ptr + pid_bh * stride_q_bh
    K_bh = K_mse_ptr + pid_bh * stride_k_bh
    S_bh = Signs_ptr + pid_bh * stride_k_bh
    N_bh = R_norm_ptr + pid_bh * stride_n_bh
    V_bh = V_mse_ptr + pid_bh * stride_k_bh
    O_bh = O_ptr + pid_bh * stride_q_bh
    rq = q_start + tl.arange(0, BQ)
    rd = tl.arange(0, D)
    q_mask = rq < Sq
    q = tl.load(Q_bh + rq[:, None] * stride_q_sq + rd[None, :],
                mask=q_mask[:, None], other=0.0).to(tl.float32)
    q_proj = tl.load(QP_bh + rq[:, None] * stride_q_sq + rd[None, :],
                     mask=q_mask[:, None], other=0.0).to(tl.float32)
    m_i = tl.full([BQ], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BQ], dtype=tl.float32)
    acc = tl.zeros([BQ, D], dtype=tl.float32)
    for k_start in range(0, Sk, BK):
        rk = k_start + tl.arange(0, BK)
        k_mask = rk < Sk
        k_mse = tl.load(K_bh + rk[:, None] * stride_k_sk + rd[None, :],
                        mask=k_mask[:, None], other=0.0).to(tl.float32)
        term1 = tl.dot(q.to(tl.float16), tl.trans(k_mse.to(tl.float16))).to(tl.float32)
        signs = tl.load(S_bh + rk[:, None] * stride_k_sk + rd[None, :],
                        mask=k_mask[:, None], other=0).to(tl.float32)
        qjl_ip = tl.dot(q_proj.to(tl.float16), tl.trans(signs.to(tl.float16))).to(tl.float32)
        r_norm = tl.load(N_bh + rk, mask=k_mask, other=0.0).to(tl.float32)
        scores = (term1 + corr_scale * qjl_ip * r_norm[None, :]) * sm_scale
        scores = tl.where(k_mask[None, :], scores, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v_mse = tl.load(V_bh + rk[:, None] * stride_k_sk + rd[None, :],
                        mask=k_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.dot(p.to(tl.float16), v_mse.to(tl.float16)).to(tl.float32)
        m_i = m_new; l_i = l_new
    acc = acc / l_i[:, None]
    tl.store(O_bh + rq[:, None] * stride_q_sq + rd[None, :], acc, mask=q_mask[:, None])


# ===========================================================================
# Public API
# ===========================================================================
def _choose_num_splits(BH: int, Sk: int, BK: int = BK_DEFAULT) -> int:
    """Heuristic: fill ~256 CUs without over-splitting."""
    if Sk <= BK:
        return 1
    target_programs = max(256, BH * 4)
    num_splits = max(1, target_programs // BH)
    # Clamp: each split must handle at least BK tokens
    max_splits = Sk // BK
    num_splits = min(num_splits, max_splits)
    # Round to power of 2 for even work distribution
    p2 = 1
    while p2 * 2 <= num_splits:
        p2 *= 2
    return max(1, p2)


class FusedAsymmetricAttention:
    """Drop-in replacement for TurboQuant's asymmetric_attention_scores().

    Usage:
        faa = FusedAsymmetricAttention(D=128, S=S_matrix)

        # Decode (Sq=1):
        output = faa.forward(Q, K_mse, signs, r_norm, V_mse)

        # Prefill (Sq>1):
        output = faa.forward(Q, K_mse, signs, r_norm, V_mse)

    Q:      [B, H, Sq, D] or [BH, Sq, D] fp32
    K_mse:  [B, H, Sk, D] or [BH, Sk, D] fp16
    signs:  [B, H, Sk, D] or [BH, Sk, D] int8 {-1, +1}
    r_norm: [B, H, Sk] or [BH, Sk] fp16
    V_mse:  [B, H, Sk, D] or [BH, Sk, D] fp16
    S:      [D, D] fp32 — QJL projection matrix

    Returns: [same shape as Q] fp32
    """

    def __init__(self, D: int = 128, S: torch.Tensor = None, sk_threshold: int = SK_THRESHOLD):
        self.D = D
        self.D2 = D * 2
        self.S = S
        self.sm_scale = 1.0 / math.sqrt(D)
        self.corr_scale = math.sqrt(math.pi / 2) / D
        self.sk_threshold = sk_threshold

    def forward(self, Q, K_mse, signs, r_norm, V_mse, Q_proj=None):
        """Run fused asymmetric attention.

        Args:
            Q:      [B,H,Sq,D] or [BH,Sq,D] fp32
            K_mse:  [B,H,Sk,D] or [BH,Sk,D] fp16
            signs:  [B,H,Sk,D] or [BH,Sk,D] int8
            r_norm: [B,H,Sk] or [BH,Sk] fp16
            V_mse:  [B,H,Sk,D] or [BH,Sk,D] fp16
            Q_proj: [B,H,Sq,D] or [BH,Sq,D] fp32 (optional, pre-computed Q @ S^T)
                    If None, computed internally (adds ~0.02-0.03ms overhead).
                    For production: compute once after QKV projection and pass in.
        """
        # Flatten to [BH, Sq, D]
        if Q.dim() == 4:
            B, H, Sq, D = Q.shape
            Q = Q.reshape(B * H, Sq, D)
            K_mse = K_mse.reshape(B * H, K_mse.shape[-2], D)
            signs = signs.reshape(B * H, signs.shape[-2], D)
            r_norm = r_norm.reshape(B * H, r_norm.shape[-1])
            V_mse = V_mse.reshape(B * H, V_mse.shape[-2], D)
            if Q_proj is not None:
                Q_proj = Q_proj.reshape(B * H, Sq, D)
        else:
            B, H = None, None
            Sq = Q.shape[1]
            D = Q.shape[2]

        BH = Q.shape[0]
        Sk = K_mse.shape[1]

        # Project Q through S (skip if pre-computed)
        if Q_proj is None:
            Q_proj = (Q.reshape(-1, D).float() @ self.S.T).reshape(BH, Sq, D).contiguous()
        else:
            Q_proj = Q_proj.float().contiguous()
        Q = Q.float().contiguous()

        if Sq == 1:
            out = self._decode(Q.squeeze(1), Q_proj.squeeze(1), K_mse, signs, r_norm, V_mse, BH, Sk)
            out = out.unsqueeze(1)
        else:
            out = self._prefill(Q, Q_proj, K_mse, signs, r_norm, V_mse, BH, Sq, Sk)

        if B is not None:
            out = out.reshape(B, H, Sq, D)
        return out

    def _decode(self, Q, Q_proj, K_mse, signs, r_norm, V_mse, BH, Sk):
        """Decode path (Sq=1): adaptive v1/v2 + split-K."""
        D = self.D
        BK = BK_DEFAULT
        num_splits = _choose_num_splits(BH, Sk, BK)

        M = torch.empty(BH, num_splits, dtype=torch.float32, device=Q.device)
        L = torch.empty(BH, num_splits, dtype=torch.float32, device=Q.device)
        Acc = torch.empty(BH, num_splits, D, dtype=torch.float32, device=Q.device)
        O = torch.empty(BH, D, dtype=torch.float32, device=Q.device)

        if Sk <= self.sk_threshold:
            # v2 concat: single dot product over 2D
            K_qjl = (self.corr_scale * signs.float() * r_norm.float().unsqueeze(-1)).to(torch.float16)
            K_combined = torch.cat([K_mse, K_qjl], dim=-1).contiguous()
            Q_combined = torch.cat([Q, Q_proj], dim=-1).contiguous()

            _splitk_v2_partial[(BH, num_splits)](
                Q_combined, K_combined, V_mse,
                M, L, Acc,
                self.sm_scale, Sk,
                D=D, D2=self.D2, BK=BK, num_splits=num_splits,
                num_warps=4, num_stages=2)
        else:
            # v1 separate: 2 dot products, less bandwidth
            _splitk_v1_partial[(BH, num_splits)](
                Q, Q_proj, K_mse, signs, r_norm, V_mse,
                M, L, Acc,
                self.sm_scale, self.corr_scale, Sk,
                D=D, BK=BK, num_splits=num_splits,
                num_warps=4, num_stages=2)

        _splitk_reduce[(BH,)](M, L, Acc, O, D=D, num_splits=num_splits, num_warps=4)
        return O

    def _prefill(self, Q, Q_proj, K_mse, signs, r_norm, V_mse, BH, Sq, Sk):
        """Prefill path (Sq>1): general tiled kernel."""
        D = self.D
        BQ = min(Sq, 16)
        BK = min(Sk, 32)

        O = torch.empty(BH, Sq, D, dtype=torch.float32, device=Q.device)
        grid = ((Sq + BQ - 1) // BQ, BH)

        stride_q_bh = Sq * D; stride_q_sq = D
        stride_k_bh = Sk * D; stride_k_sk = D
        stride_n_bh = Sk

        _fused_general[grid](
            Q, Q_proj, K_mse, signs, r_norm, V_mse, O,
            self.sm_scale, self.corr_scale,
            stride_q_bh, stride_q_sq, stride_k_bh, stride_k_sk, stride_n_bh,
            Sq, Sk, D=D, BQ=BQ, BK=BK,
            num_warps=4, num_stages=2)
        return O


# ===========================================================================
# Reference
# ===========================================================================
def ref_attn(Q, Q_proj, K_mse, signs, r_norm, V_mse, sm_scale, corr_scale):
    t1 = torch.matmul(Q.float(), K_mse.float().transpose(-1, -2))
    qjl = torch.matmul(Q_proj.float(), signs.float().transpose(-1, -2))
    scores = (t1 + corr_scale * qjl * r_norm.float().unsqueeze(-2)) * sm_scale
    return F.softmax(scores, dim=-1) @ V_mse.float()


# ===========================================================================
# Benchmark
# ===========================================================================
def run():
    import time
    device = "cuda"; D = 128
    sm_scale = 1.0 / math.sqrt(D)
    corr_scale = math.sqrt(math.pi / 2) / D
    S = torch.randn(D, D, device=device, dtype=torch.float32)

    faa = FusedAsymmetricAttention(D=D, S=S)

    print(f"Unified Fused Asymmetric Attention — MI355X")
    print(f"Device: {torch.cuda.get_device_name(0) or 'MI355X'}")
    print(f"SK_THRESHOLD={SK_THRESHOLD} (v2 concat below, v1 separate above)\n")

    warmup, iters = 30, 300

    print(f"{'Config':>35} {'Ref':>9} {'w/proj':>9} {'cached':>9} {'proj/R':>7} {'cache/R':>7} {'#sp':>5} {'cos':>8}")
    print("-" * 110)

    configs = [
        # Decode
        (1, 8, 1, 256), (1, 8, 1, 1024), (1, 8, 1, 4096),
        (1, 8, 1, 8192), (1, 8, 1, 16384), (1, 8, 1, 32768),
        (1, 32, 1, 256), (1, 32, 1, 1024), (1, 32, 1, 4096),
        (1, 32, 1, 16384), (1, 32, 1, 32768),
        (1, 64, 1, 4096), (1, 64, 1, 16384), (1, 64, 1, 32768),
        (2, 32, 1, 4096), (2, 32, 1, 16384),
        # Prefill
        (1, 8, 64, 256), (1, 8, 64, 1024),
        (1, 32, 64, 256), (1, 32, 64, 4096),
    ]

    for B, H, Sq, Sk in configs:
        BH = B * H
        label = f"B={B} H={H:>2} Sq={Sq:>3} Sk={Sk:>5}"

        try:
            Q = torch.randn(B, H, Sq, D, device=device, dtype=torch.float32)
            K_mse = torch.randn(B, H, Sk, D, device=device, dtype=torch.float16)
            V_mse = torch.randn(B, H, Sk, D, device=device, dtype=torch.float16)
            signs = (torch.randint(0, 2, (B, H, Sk, D), device=device) * 2 - 1).to(torch.int8)
            r_norm = torch.rand(B, H, Sk, device=device, dtype=torch.float16) * 0.5

            # Pre-compute Q_proj (simulating production: done once at QKV projection)
            Q_proj = (Q.reshape(-1, D).float() @ S.T).reshape(B, H, Sq, D).contiguous()

            # Reference (PyTorch, Q_proj already computed — fair comparison)
            ref = ref_attn(Q, Q_proj, K_mse, signs, r_norm, V_mse, sm_scale, corr_scale)

            for _ in range(warmup):
                ref_attn(Q, Q_proj, K_mse, signs, r_norm, V_mse, sm_scale, corr_scale)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                ref_attn(Q, Q_proj, K_mse, signs, r_norm, V_mse, sm_scale, corr_scale)
            torch.cuda.synchronize()
            ref_ms = (time.perf_counter()-t0)/iters*1000

            # Mode 1: with projection (Q_proj computed inside)
            for _ in range(warmup):
                faa.forward(Q, K_mse, signs, r_norm, V_mse)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                faa.forward(Q, K_mse, signs, r_norm, V_mse)
            torch.cuda.synchronize()
            proj_ms = (time.perf_counter()-t0)/iters*1000

            # Mode 2: cached Q_proj (production path)
            for _ in range(warmup):
                faa.forward(Q, K_mse, signs, r_norm, V_mse, Q_proj=Q_proj)
            torch.cuda.synchronize()

            out = faa.forward(Q, K_mse, signs, r_norm, V_mse, Q_proj=Q_proj)
            cs = F.cosine_similarity(ref.reshape(-1, D).float(), out.reshape(-1, D).float()).mean().item()

            t0 = time.perf_counter()
            for _ in range(iters):
                faa.forward(Q, K_mse, signs, r_norm, V_mse, Q_proj=Q_proj)
            torch.cuda.synchronize()
            cached_ms = (time.perf_counter()-t0)/iters*1000

            sp_proj = ref_ms / proj_ms
            sp_cached = ref_ms / cached_ms
            nsplits = _choose_num_splits(BH, Sk) if Sq == 1 else 0
            ok = "✅" if cs > 0.999 else "⚠️"

            print(f"{label:>35} {ref_ms:>8.4f}ms {proj_ms:>8.4f}ms {cached_ms:>8.4f}ms "
                  f"{sp_proj:>6.2f}x {sp_cached:>6.2f}x {nsplits:>5} {cs:>8.6f}{ok}")
        except Exception as e:
            print(f"{label:>35} FAIL: {str(e)[:60]}")

    # Summary
    print(f"\n{'='*110}")
    print("API Usage (production — cached Q_proj):")
    print("  faa = FusedAsymmetricAttention(D=128, S=qjl_matrix)")
    print("  Q_proj = Q @ S.T   # compute once after QKV projection")
    print("  output = faa.forward(Q, K_mse, signs, r_norm, V_mse, Q_proj=Q_proj)")
    print()
    print("API Usage (convenience — auto-computes Q_proj):")
    print("  output = faa.forward(Q, K_mse, signs, r_norm, V_mse)")
    print()
    print("  # Accepts [B,H,Sq,D] or [BH,Sq,D] inputs")

if __name__ == "__main__":
    run()
