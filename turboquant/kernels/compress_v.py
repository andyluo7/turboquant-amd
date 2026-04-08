#!/usr/bin/env python3
"""TurboQuant Step B: Fused V compression kernel for MI355X.

V uses MSE-only quantization (no QJL correction).
Single Triton kernel: norm + rotate + quantize + gather + scale.

Outputs V_mse_rot in rotated space (same as K compression).
During decode, attention output = softmax(scores) @ V_mse_rot,
then inverse-rotate the final result once.
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
# Fully fused V compression: single kernel
# ===========================================================================
@triton.jit
def _fused_v_compress(
    V_ptr,           # [N, D] fp32 — raw V vectors
    PiT_ptr,         # [D, D] fp32 — rotation matrix Pi^T
    cb0, cb1, cb2, cb3,
    bd0, bd1, bd2,
    Vmse_rot_ptr,    # [N, D] fp16 — output: V_mse in rotated space
    N_total,
    D: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rd = tl.arange(0, D)
    n_mask = rn < N_total

    # Load V tile [BN, D]
    v = tl.load(V_ptr + rn[:, None] * D + rd[None, :],
                mask=n_mask[:, None], other=0.0).to(tl.float32)

    # Normalize
    v_sq = tl.sum(v * v, axis=1)          # [BN]
    v_norm = tl.sqrt(v_sq)                 # [BN]
    v_normalized = v * (1.0 / (v_norm + 1e-8))[:, None]  # [BN, D]

    # Rotate: [BN, D] @ [D, D]
    pit = tl.load(PiT_ptr + rd[:, None] * D + rd[None, :]).to(tl.float32)
    rotated = tl.dot(v_normalized.to(tl.float16), pit.to(tl.float16)).to(tl.float32)

    # Quantize via boundary comparison
    idx = (rotated >= bd0).to(tl.int32) + (rotated >= bd1).to(tl.int32) + (rotated >= bd2).to(tl.int32)
    recon = tl.where(idx == 0, cb0,
            tl.where(idx == 1, cb1,
            tl.where(idx == 2, cb2, cb3)))

    # V_mse_rot = recon * norms
    v_mse_rot = recon * v_norm[:, None]
    tl.store(Vmse_rot_ptr + rn[:, None] * D + rd[None, :],
             v_mse_rot.to(tl.float16), mask=n_mask[:, None])


# ===========================================================================
# 2-kernel baseline: rocBLAS rotation + Triton post-rotation
# ===========================================================================
@triton.jit
def _v_post_rotation(
    Rotated_ptr, Norms_ptr,
    cb0, cb1, cb2, cb3, bd0, bd1, bd2,
    Vmse_rot_ptr,
    N_total,
    D: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rd = tl.arange(0, D)
    n_mask = rn < N_total

    rot = tl.load(Rotated_ptr + rn[:, None] * D + rd[None, :],
                  mask=n_mask[:, None], other=0.0).to(tl.float32)
    norms = tl.load(Norms_ptr + rn, mask=n_mask, other=1.0).to(tl.float32)

    idx = (rot >= bd0).to(tl.int32) + (rot >= bd1).to(tl.int32) + (rot >= bd2).to(tl.int32)
    recon = tl.where(idx == 0, cb0,
            tl.where(idx == 1, cb1,
            tl.where(idx == 2, cb2, cb3)))

    v_mse_rot = recon * norms[:, None]
    tl.store(Vmse_rot_ptr + rn[:, None] * D + rd[None, :],
             v_mse_rot.to(tl.float16), mask=n_mask[:, None])


# ===========================================================================
# Wrappers
# ===========================================================================
def v_compress_pytorch(V, PiT, centroids, boundaries):
    """PyTorch reference: rotation + argmin."""
    N, D = V.shape
    norms = torch.norm(V, dim=-1, keepdim=True)
    V_norm = V / (norms + 1e-8)
    rotated = V_norm @ PiT
    # argmin codebook lookup (original method)
    diffs = rotated.unsqueeze(-1) - centroids
    indices = diffs.abs().argmin(dim=-1)
    recon = centroids[indices.long()]
    v_mse_rot = recon * norms
    return v_mse_rot.to(torch.float16)


def v_compress_pytorch_ss(V, PiT, centroids, boundaries):
    """PyTorch with searchsorted (optimized baseline)."""
    N, D = V.shape
    norms = torch.norm(V, dim=-1, keepdim=True)
    V_norm = V / (norms + 1e-8)
    rotated = V_norm @ PiT
    indices = torch.searchsorted(boundaries, rotated)
    recon = centroids[indices.long()]
    v_mse_rot = recon * norms
    return v_mse_rot.to(torch.float16)


def v_compress_two_kernel(V, PiT, centroids, boundaries, BN=128, nw=8, ns=2):
    """2-kernel: rocBLAS rotation + Triton quantize."""
    N, D = V.shape
    norms = torch.norm(V, dim=-1)
    V_norm = V / (norms.unsqueeze(-1) + 1e-8)
    rotated = V_norm @ PiT

    v_mse_rot = torch.empty(N, D, dtype=torch.float16, device=V.device)
    cb = centroids.tolist(); bd = boundaries.tolist()
    grid = ((N + BN - 1) // BN,)
    _v_post_rotation[grid](
        rotated, norms,
        cb[0], cb[1], cb[2], cb[3], bd[0], bd[1], bd[2],
        v_mse_rot, N, D=D, BN=BN, num_warps=nw, num_stages=ns)
    return v_mse_rot


def v_compress_one_kernel(V, PiT, centroids, boundaries, BN=None, nw=None, ns=None):
    """Single kernel: everything fused."""
    N, D = V.shape
    v_mse_rot = torch.empty(N, D, dtype=torch.float16, device=V.device)
    cb = centroids.tolist(); bd = boundaries.tolist()

    if BN is None:
        # Tuned on MI355X (D=128, 2-bit MSE, single tl.dot):
        #   Only 1 GEMM → w=8 works great (no register spill from 2nd dot)
        #   BN=64 w=8 at medium N, BN=128 w=8 at large N
        if N <= 8192:
            BN, nw, ns = 64, 8, 2
        elif N <= 131072:
            BN, nw, ns = 64, 8, 2
        else:
            BN, nw, ns = 128, 8, 2

    grid = ((N + BN - 1) // BN,)
    _fused_v_compress[grid](
        V, PiT,
        cb[0], cb[1], cb[2], cb[3], bd[0], bd[1], bd[2],
        v_mse_rot, N, D=D, BN=BN, num_warps=nw, num_stages=ns)
    return v_mse_rot


def run():
    device = "cuda"; D = 128
    bits = 3; mse_bits = bits - 1; seed = 42

    gen = torch.Generator(device=device); gen.manual_seed(seed)
    G = torch.randn(D, D, device=device, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R)); diag_sign[diag_sign == 0] = 1.0
    Pi = (Q * diag_sign.unsqueeze(0))
    PiT = Pi.T.contiguous()

    centroids = solve_codebook(D, mse_bits).to(device)
    boundaries = ((centroids[:-1] + centroids[1:]) / 2.0).contiguous()

    print(f"TurboQuant Step B: V Compression — MI355X")
    print(f"D={D}, bits={bits}, levels={len(centroids)}")
    print(f"Device: {torch.cuda.get_device_name(0) or 'MI355X'}\n")

    warmup, iters = 25, 300

    # ====== Correctness ======
    print("Correctness check (N=131072):")
    V_test = torch.randn(131072, D, device=device, dtype=torch.float32)
    ref = v_compress_pytorch_ss(V_test, PiT, centroids, boundaries)

    for name, fn in [
        ("2-kernel", lambda: v_compress_two_kernel(V_test, PiT, centroids, boundaries)),
        ("1-kernel", lambda: v_compress_one_kernel(V_test, PiT, centroids, boundaries)),
    ]:
        out = fn()
        cs = F.cosine_similarity(ref.float().reshape(1,-1), out.float().reshape(1,-1)).item()
        diff = (ref.float() - out.float()).abs().max().item()
        print(f"  {name}: cos={cs:.8f} max_diff={diff:.2e} {'✅' if cs > 0.999 else '❌'}")
    print()

    # ====== Tuning sweep for 1-kernel ======
    print("Tuning 1-kernel at N=131072:")
    print(f"  {'BN':>4} {'w':>3} {'s':>3} {'ms':>10}")
    best_ms = float('inf'); best_cfg = ""
    for BN in [32, 64, 128, 256]:
        for nw in [2, 4, 8]:
            for ns in [1, 2]:
                try:
                    for _ in range(warmup):
                        v_compress_one_kernel(V_test, PiT, centroids, boundaries, BN, nw, ns)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        v_compress_one_kernel(V_test, PiT, centroids, boundaries, BN, nw, ns)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter()-t0)/iters*1000
                    tag = ""
                    if ms < best_ms: best_ms = ms; best_cfg = f"BN={BN} w={nw} s={ns}"; tag = " ★"
                    if ms < best_ms * 1.5:
                        print(f"  {BN:>4} {nw:>3} {ns:>3} {ms:>9.4f}ms{tag}")
                except:
                    pass
    print(f"  → Best: {best_cfg} = {best_ms:.4f}ms\n")

    # Tune at N=524288
    print("Tuning 1-kernel at N=524288:")
    V_big = torch.randn(524288, D, device=device, dtype=torch.float32)
    best_ms2 = float('inf'); best_cfg2 = ""
    for BN in [64, 128, 256]:
        for nw in [2, 4, 8]:
            for ns in [1, 2]:
                try:
                    for _ in range(warmup):
                        v_compress_one_kernel(V_big, PiT, centroids, boundaries, BN, nw, ns)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        v_compress_one_kernel(V_big, PiT, centroids, boundaries, BN, nw, ns)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter()-t0)/iters*1000
                    tag = ""
                    if ms < best_ms2: best_ms2 = ms; best_cfg2 = f"BN={BN} w={nw} s={ns}"; tag = " ★"
                    if ms < best_ms2 * 1.5:
                        print(f"  {BN:>4} {nw:>3} {ns:>3} {ms:>9.4f}ms{tag}")
                except:
                    pass
    print(f"  → Best: {best_cfg2} = {best_ms2:.4f}ms\n")

    # Also tune 2-kernel at both sizes
    print("Tuning 2-kernel (post-rotation only) at N=131072:")
    best_ms3 = float('inf'); best_cfg3 = ""
    for BN in [32, 64, 128, 256]:
        for nw in [2, 4, 8]:
            for ns in [1, 2]:
                try:
                    for _ in range(warmup):
                        v_compress_two_kernel(V_test, PiT, centroids, boundaries, BN, nw, ns)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        v_compress_two_kernel(V_test, PiT, centroids, boundaries, BN, nw, ns)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter()-t0)/iters*1000
                    tag = ""
                    if ms < best_ms3: best_ms3 = ms; best_cfg3 = f"BN={BN} w={nw} s={ns}"; tag = " ★"
                    if ms < best_ms3 * 1.5:
                        print(f"  {BN:>4} {nw:>3} {ns:>3} {ms:>9.4f}ms{tag}")
                except:
                    pass
    print(f"  → Best: {best_cfg3} = {best_ms3:.4f}ms\n")

    # ====== Final comparison ======
    print(f"{'N':>8} {'PyTorch':>10} {'PT+SS':>10} {'2-kern':>10} {'1-kern':>10} "
          f"{'vs_PT':>8} {'vs_SS':>8} {'1v2':>6}")
    print("-" * 80)

    for N in [32, 2048, 8192, 32768, 131072, 524288]:
        V = torch.randn(N, D, device=device, dtype=torch.float32)

        # PyTorch original (argmin)
        for _ in range(warmup): v_compress_pytorch(V, PiT, centroids, boundaries)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): v_compress_pytorch(V, PiT, centroids, boundaries)
        torch.cuda.synchronize()
        pt_ms = (time.perf_counter()-t0)/iters*1000

        # PyTorch + searchsorted
        for _ in range(warmup): v_compress_pytorch_ss(V, PiT, centroids, boundaries)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): v_compress_pytorch_ss(V, PiT, centroids, boundaries)
        torch.cuda.synchronize()
        ss_ms = (time.perf_counter()-t0)/iters*1000

        # 2-kernel (best config)
        for _ in range(warmup): v_compress_two_kernel(V, PiT, centroids, boundaries)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): v_compress_two_kernel(V, PiT, centroids, boundaries)
        torch.cuda.synchronize()
        two_ms = (time.perf_counter()-t0)/iters*1000

        # 1-kernel (auto-tuned)
        for _ in range(warmup): v_compress_one_kernel(V, PiT, centroids, boundaries)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): v_compress_one_kernel(V, PiT, centroids, boundaries)
        torch.cuda.synchronize()
        one_ms = (time.perf_counter()-t0)/iters*1000

        vs_pt = pt_ms / one_ms
        vs_ss = ss_ms / one_ms
        one_v_two = two_ms / one_ms

        print(f"{N:>8} {pt_ms:>9.4f}ms {ss_ms:>9.4f}ms {two_ms:>9.4f}ms {one_ms:>9.4f}ms "
              f"{vs_pt:>7.2f}x {vs_ss:>7.2f}x {one_v_two:>5.2f}x")

    print("\nDone!")

if __name__ == "__main__":
    run()
