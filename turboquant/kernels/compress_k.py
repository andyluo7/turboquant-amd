#!/usr/bin/env python3
"""Fully fused TurboQuant compression: single Triton kernel, zero rocBLAS.

Fuses ALL operations into one kernel:
  1. Normalize K
  2. Rotate: K_norm @ Pi^T  (tl.dot [BN, D] @ [D, D])
  3. Quantize via boundary comparison
  4. Centroid gather
  5. k_mse_rot = recon * norm
  6. rot_residual = (rotated - recon) * norm
  7. r_norm = ||rot_residual||
  8. projected = rot_residual @ PiST  (tl.dot [BN, D] @ [D, D])
  9. signs = sign(projected)

One kernel launch. Two tl.dot calls per tile. No intermediate tensors.
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
# Fully fused: norm + rotate + quant + residual + PiST GEMM + sign + rnorm
# ===========================================================================
@triton.jit
def _fully_fused_compress(
    K_ptr,           # [N, D] fp32 — raw K vectors
    PiT_ptr,         # [D, D] fp32 — Pi^T (rotation matrix)
    PiST_ptr,        # [D, D] fp32 — Pi @ S^T (combined matrix)
    cb0, cb1, cb2, cb3,
    bd0, bd1, bd2,
    Kmse_rot_ptr,    # [N, D] fp16 — output
    Signs_ptr,       # [N, D] int8 — output
    Rnorm_ptr,       # [N] fp16 — output
    N_total,
    D: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    vec_start = pid * BN
    rn = vec_start + tl.arange(0, BN)
    rd = tl.arange(0, D)
    n_mask = rn < N_total

    # --- Load K tile [BN, D] ---
    k = tl.load(K_ptr + rn[:, None] * D + rd[None, :],
                mask=n_mask[:, None], other=0.0).to(tl.float32)

    # --- Normalize: K_norm = K / ||K|| ---
    k_sq = tl.sum(k * k, axis=1)  # [BN]
    k_norm_val = tl.sqrt(k_sq)    # [BN] — the vector norms
    k_inv_norm = 1.0 / (k_norm_val + 1e-8)  # [BN]
    k_normalized = k * k_inv_norm[:, None]   # [BN, D]

    # --- Rotate: rotated = K_norm @ Pi^T  [BN, D] @ [D, D] → [BN, D] ---
    pit = tl.load(PiT_ptr + rd[:, None] * D + rd[None, :]).to(tl.float32)  # [D, D]
    rotated = tl.dot(k_normalized.to(tl.float16), pit.to(tl.float16)).to(tl.float32)

    # --- Quantize via boundary comparison ---
    idx = (rotated >= bd0).to(tl.int32) + (rotated >= bd1).to(tl.int32) + (rotated >= bd2).to(tl.int32)
    recon = tl.where(idx == 0, cb0,
            tl.where(idx == 1, cb1,
            tl.where(idx == 2, cb2, cb3)))

    # --- k_mse_rot = recon * norm ---
    k_mse_rot = recon * k_norm_val[:, None]
    tl.store(Kmse_rot_ptr + rn[:, None] * D + rd[None, :],
             k_mse_rot.to(tl.float16), mask=n_mask[:, None])

    # --- rot_residual = (rotated - recon) * norm ---
    rot_resid = (rotated - recon) * k_norm_val[:, None]

    # --- r_norm = ||rot_residual|| ---
    r_norm_sq = tl.sum(rot_resid * rot_resid, axis=1)
    tl.store(Rnorm_ptr + rn, tl.sqrt(r_norm_sq).to(tl.float16), mask=n_mask)

    # --- projected = rot_resid @ PiST  [BN, D] @ [D, D] → [BN, D] ---
    pist = tl.load(PiST_ptr + rd[:, None] * D + rd[None, :]).to(tl.float32)
    proj = tl.dot(rot_resid.to(tl.float16), pist.to(tl.float16)).to(tl.float32)

    # --- signs = sign(projected) ---
    signs = tl.where(proj >= 0, tl.full([BN, D], 1, dtype=tl.int8),
                                tl.full([BN, D], -1, dtype=tl.int8))
    tl.store(Signs_ptr + rn[:, None] * D + rd[None, :], signs, mask=n_mask[:, None])


# ===========================================================================
# Baselines
# ===========================================================================
@triton.jit
def _post_rotation_only(
    Rotated_ptr, Norms_ptr, PiST_ptr,
    cb0, cb1, cb2, cb3, bd0, bd1, bd2,
    Kmse_rot_ptr, Signs_ptr, Rnorm_ptr,
    N_total,
    D: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0)
    vec_start = pid * BN
    rn = vec_start + tl.arange(0, BN)
    rd = tl.arange(0, D)
    n_mask = rn < N_total
    rot = tl.load(Rotated_ptr + rn[:, None] * D + rd[None, :],
                  mask=n_mask[:, None], other=0.0).to(tl.float32)
    norms = tl.load(Norms_ptr + rn, mask=n_mask, other=1.0).to(tl.float32)
    idx = (rot >= bd0).to(tl.int32) + (rot >= bd1).to(tl.int32) + (rot >= bd2).to(tl.int32)
    recon = tl.where(idx == 0, cb0, tl.where(idx == 1, cb1, tl.where(idx == 2, cb2, cb3)))
    k_mse_rot = recon * norms[:, None]
    tl.store(Kmse_rot_ptr + rn[:, None] * D + rd[None, :], k_mse_rot.to(tl.float16), mask=n_mask[:, None])
    rot_resid = (rot - recon) * norms[:, None]
    r_norm_sq = tl.sum(rot_resid * rot_resid, axis=1)
    tl.store(Rnorm_ptr + rn, tl.sqrt(r_norm_sq).to(tl.float16), mask=n_mask)
    pist = tl.load(PiST_ptr + rd[:, None] * D + rd[None, :]).to(tl.float32)
    proj = tl.dot(rot_resid.to(tl.float16), pist.to(tl.float16)).to(tl.float32)
    signs = tl.where(proj >= 0, tl.full([BN, D], 1, dtype=tl.int8), tl.full([BN, D], -1, dtype=tl.int8))
    tl.store(Signs_ptr + rn[:, None] * D + rd[None, :], signs, mask=n_mask[:, None])


def compress_two_kernel(K, PiT, PiST, centroids, boundaries):
    """Two-kernel: rocBLAS rotation + Triton post-rotation."""
    N, D = K.shape
    norms = torch.norm(K, dim=-1)
    K_norm = K / (norms.unsqueeze(-1) + 1e-8)
    rotated = K_norm @ PiT

    k_mse_rot = torch.empty(N, D, dtype=torch.float16, device=K.device)
    signs = torch.empty(N, D, dtype=torch.int8, device=K.device)
    r_norm = torch.empty(N, dtype=torch.float16, device=K.device)

    cb = centroids.tolist(); bd = boundaries.tolist()
    BN, nw, ns = (128, 8, 2) if N > 8192 else (32, 4, 1)
    grid = ((N + BN - 1) // BN,)
    _post_rotation_only[grid](
        rotated, norms, PiST,
        cb[0], cb[1], cb[2], cb[3], bd[0], bd[1], bd[2],
        k_mse_rot, signs, r_norm,
        N, D=D, BN=BN, num_warps=nw, num_stages=ns)
    return k_mse_rot, signs, r_norm


def compress_one_kernel(K, PiT, PiST, centroids, boundaries, BN=None, nw=None, ns=None):
    """Single kernel: everything fused.
    
    Tuned configs from MI355X sweep (v1 kernel, D=128):
      N ≤ 2K:   BN=32,  w=4, s=1
      N ≤ 32K:  BN=64,  w=2, s=2
      N > 32K:  BN=256, w=4, s=2
    """
    N, D = K.shape
    k_mse_rot = torch.empty(N, D, dtype=torch.float16, device=K.device)
    signs = torch.empty(N, D, dtype=torch.int8, device=K.device)
    r_norm = torch.empty(N, dtype=torch.float16, device=K.device)

    cb = centroids.tolist(); bd = boundaries.tolist()

    # Auto-tune if not specified
    if BN is None:
        if N <= 2048:
            BN, nw, ns = 32, 4, 1
        elif N <= 32768:
            BN, nw, ns = 64, 2, 2
        else:
            BN, nw, ns = 256, 4, 2

    grid = ((N + BN - 1) // BN,)
    _fully_fused_compress[grid](
        K, PiT, PiST,
        cb[0], cb[1], cb[2], cb[3], bd[0], bd[1], bd[2],
        k_mse_rot, signs, r_norm,
        N, D=D, BN=BN, num_warps=nw, num_stages=ns)
    return k_mse_rot, signs, r_norm


def run():
    device = "cuda"; D = 128
    bits = 3; mse_bits = bits - 1; seed = 42

    gen = torch.Generator(device=device); gen.manual_seed(seed)
    G = torch.randn(D, D, device=device, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R)); diag_sign[diag_sign == 0] = 1.0
    Pi = (Q * diag_sign.unsqueeze(0))
    PiT = Pi.T.contiguous()

    gen2 = torch.Generator(device=device); gen2.manual_seed(seed + 10000)
    S = torch.randn(D, D, device=device, generator=gen2)
    ST = S.T.contiguous()
    PiST = (Pi @ ST).contiguous()

    centroids = solve_codebook(D, mse_bits).to(device)
    boundaries = ((centroids[:-1] + centroids[1:]) / 2.0).contiguous()

    print(f"Fully Fused TurboQuant Compression — MI355X")
    print(f"D={D}, bits={bits}")
    print(f"Device: {torch.cuda.get_device_name(0) or 'MI355X'}\n")

    warmup, iters = 20, 200

    # Correctness
    print("Correctness:")
    K_test = torch.randn(131072, D, device=device, dtype=torch.float32)
    ref_mse, ref_signs, ref_rnorm = compress_two_kernel(K_test, PiT, PiST, centroids, boundaries)
    fused_mse, fused_signs, fused_rnorm = compress_one_kernel(K_test, PiT, PiST, centroids, boundaries)
    cs = F.cosine_similarity(ref_mse.float().reshape(1,-1), fused_mse.float().reshape(1,-1)).item()
    sm = (ref_signs == fused_signs).float().mean().item()
    rn_diff = (ref_rnorm.float() - fused_rnorm.float()).abs().max().item()
    print(f"  cos={cs:.8f} sign_match={sm:.6f} rnorm_diff={rn_diff:.2e} {'✅' if cs>0.999 and sm>0.99 else '❌'}\n")

    # Tune the fully fused kernel
    print("Tuning fully fused kernel at N=131072:")
    print(f"  {'BN':>4} {'w':>3} {'s':>3} {'ms':>10} {'tag':>4}")
    best_ms = float('inf'); best_cfg = ""
    for BN in [32, 64, 128]:
        for nw in [4, 8]:
            for ns in [1, 2, 3]:
                try:
                    for _ in range(warmup):
                        compress_one_kernel(K_test, PiT, PiST, centroids, boundaries, BN, nw, ns)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        compress_one_kernel(K_test, PiT, PiST, centroids, boundaries, BN, nw, ns)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter()-t0)/iters*1000
                    tag = ""
                    if ms < best_ms: best_ms = ms; best_cfg = f"BN={BN} w={nw} s={ns}"; tag = " ★"
                    print(f"  {BN:>4} {nw:>3} {ns:>3} {ms:>9.4f}ms{tag}")
                except Exception as e:
                    print(f"  {BN:>4} {nw:>3} {ns:>3} FAIL: {str(e)[:50]}")
    print(f"  → Best: {best_cfg} = {best_ms:.4f}ms\n")

    # Also tune at N=524288
    print("Tuning fully fused kernel at N=524288:")
    K_big = torch.randn(524288, D, device=device, dtype=torch.float32)
    best_ms_big = float('inf'); best_cfg_big = ""
    for BN in [64, 128]:
        for nw in [4, 8]:
            for ns in [1, 2]:
                try:
                    for _ in range(warmup):
                        compress_one_kernel(K_big, PiT, PiST, centroids, boundaries, BN, nw, ns)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        compress_one_kernel(K_big, PiT, PiST, centroids, boundaries, BN, nw, ns)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter()-t0)/iters*1000
                    tag = ""
                    if ms < best_ms_big: best_ms_big = ms; best_cfg_big = f"BN={BN} w={nw} s={ns}"; tag = " ★"
                    print(f"  {BN:>4} {nw:>3} {ns:>3} {ms:>9.4f}ms{tag}")
                except Exception as e:
                    print(f"  {BN:>4} {nw:>3} {ns:>3} FAIL: {str(e)[:50]}")
    print(f"  → Best: {best_cfg_big} = {best_ms_big:.4f}ms\n")

    # Final comparison
    print(f"{'N':>8} {'2-kernel':>12} {'1-kernel':>12} {'speedup':>8}")
    print("-" * 50)
    for N in [32, 2048, 8192, 32768, 131072, 524288]:
        K = torch.randn(N, D, device=device, dtype=torch.float32)

        for _ in range(warmup): compress_two_kernel(K, PiT, PiST, centroids, boundaries)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): compress_two_kernel(K, PiT, PiST, centroids, boundaries)
        torch.cuda.synchronize()
        two_ms = (time.perf_counter()-t0)/iters*1000

        BN_use = 128; nw_use = 8; ns_use = 2
        for _ in range(warmup): compress_one_kernel(K, PiT, PiST, centroids, boundaries, BN_use, nw_use, ns_use)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): compress_one_kernel(K, PiT, PiST, centroids, boundaries, BN_use, nw_use, ns_use)
        torch.cuda.synchronize()
        one_ms = (time.perf_counter()-t0)/iters*1000

        sp = two_ms / one_ms
        print(f"{N:>8} {two_ms:>11.4f}ms {one_ms:>11.4f}ms {sp:>7.2f}x")

    print("\nDone!")

if __name__ == "__main__":
    run()
