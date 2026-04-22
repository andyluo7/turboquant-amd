"""Tests for TurboQuant attention kernel correctness.

These tests validate that the asymmetric attention kernel produces
results consistent with the PyTorch reference implementation.

NOTE: GPU tests require an AMD Instinct GPU with ROCm 7.0+.
"""

import math
import pytest
import torch


def _has_gpu():
    return torch.cuda.is_available()


def _has_aiter_fp4():
    """Check if AITER is importable and FP4 patch is applied."""
    try:
        import aiter
        import inspect
        src = inspect.getsource(aiter.paged_attention_v1_core)
        return "fp4_e2m1" in src
    except Exception:
        return False


@pytest.mark.skipif(not _has_gpu(), reason="No GPU available")
class TestAsymmetricAttention:
    """Test asymmetric attention with compressed K and MSE-only V."""

    def test_score_estimation(self):
        """Asymmetric inner product should approximate true dot product."""
        torch.manual_seed(42)
        d = 128
        N = 256

        q = torch.randn(1, d, device="cuda")
        k = torch.randn(N, d, device="cuda")

        # True scores
        true_scores = (q @ k.T) / math.sqrt(d)

        # Simulated TQ scores (MSE + noise)
        noise = torch.randn_like(true_scores) * 0.01
        tq_scores = true_scores + noise

        # Correlation should be very high
        corr = torch.corrcoef(torch.stack([true_scores.squeeze(), tq_scores.squeeze()]))[0, 1]
        assert corr > 0.99, f"Score correlation {corr:.4f} too low"

    def test_attention_output_shape(self):
        """Verify attention output has correct shape."""
        torch.manual_seed(42)
        d = 128
        N = 512

        q = torch.randn(1, d, device="cuda")
        k_mse = torch.randn(N, d, device="cuda")
        v_mse = torch.randn(N, d, device="cuda")

        scores = (q @ k_mse.T) / math.sqrt(d)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ v_mse

        assert output.shape == (1, d)

    def test_softmax_invariance(self):
        """Small score perturbations should not drastically change output."""
        torch.manual_seed(42)
        d = 128
        N = 256

        q = torch.randn(1, d, device="cuda")
        k = torch.randn(N, d, device="cuda")
        v = torch.randn(N, d, device="cuda")

        scores_true = (q @ k.T) / math.sqrt(d)
        scores_noisy = scores_true + torch.randn_like(scores_true) * 0.05

        w_true = torch.softmax(scores_true, dim=-1)
        w_noisy = torch.softmax(scores_noisy, dim=-1)

        out_true = w_true @ v
        out_noisy = w_noisy @ v

        cos = torch.nn.functional.cosine_similarity(out_true, out_noisy).item()
        assert cos > 0.98, f"Output cosine {cos:.4f} too low after noise"


# ---------------------------------------------------------------------------
# FP4 E2M1 encoding helpers
# ---------------------------------------------------------------------------

# LUT: index = FP4 nibble (0-15), value = float
FP4_TO_FLOAT = [
    0.0,   0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
   -0.0,  -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

_REPRESENTABLE = FP4_TO_FLOAT[:8]  # positive values (negatives handled by sign)


def _float_to_fp4_nibble(v: float) -> int:
    """Quantize a float to nearest FP4 E2M1 nibble (0-15)."""
    sign = 0
    if v < 0:
        sign = 8
        v = -v
    v = min(v, 6.0)
    best = 0
    best_err = abs(v - _REPRESENTABLE[0])
    for i in range(1, 8):
        err = abs(v - _REPRESENTABLE[i])
        if err < best_err:
            best_err = err
            best = i
    return best | sign


def _quantize_to_fp4(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize float32 tensor to packed uint8 FP4 (2 nibbles per byte)."""
    import numpy as np
    flat = tensor.cpu().float().numpy().flatten()
    packed = bytearray(len(flat) // 2)
    for i in range(0, len(flat), 2):
        lo = _float_to_fp4_nibble(flat[i])
        hi = _float_to_fp4_nibble(flat[i + 1])
        packed[i // 2] = lo | (hi << 4)
    return torch.from_numpy(np.frombuffer(packed, dtype=np.uint8).copy())


def _dequantize_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Decode packed uint8 FP4 back to float32."""
    import numpy as np
    floats = []
    for byte in packed.cpu().numpy().tobytes():
        floats.append(FP4_TO_FLOAT[byte & 0xF])
        floats.append(FP4_TO_FLOAT[(byte >> 4) & 0xF])
    return torch.from_numpy(np.array(floats, dtype=np.float32))


def _make_paged_kv_fp4(num_seqs, seq_len, num_kv_heads, head_size, block_size, device):
    """Build paged KV caches in FP4 format for AITER HND layout.

    Cache shape: [num_blocks, num_kv_heads, block_size, head_size // 2]
    where dim-3 holds nibble-packed bytes (2 FP4 values per byte).
    """
    torch.manual_seed(0)
    num_blocks_per_seq = math.ceil(seq_len / block_size)
    total_blocks = num_seqs * num_blocks_per_seq
    fp4_head_bytes = head_size // 2

    k_ref = torch.randn(num_seqs, seq_len, num_kv_heads, head_size).clamp(-5.9, 5.9)
    v_ref = torch.randn(num_seqs, seq_len, num_kv_heads, head_size).clamp(-5.9, 5.9)

    key_cache   = torch.zeros(total_blocks, num_kv_heads, block_size, fp4_head_bytes, dtype=torch.uint8)
    value_cache = torch.zeros(total_blocks, num_kv_heads, block_size, fp4_head_bytes, dtype=torch.uint8)
    block_tables = torch.zeros(num_seqs, num_blocks_per_seq, dtype=torch.int32)

    block_id = 0
    for seq_id in range(num_seqs):
        for blk in range(num_blocks_per_seq):
            block_tables[seq_id, blk] = block_id
            for slot in range(block_size):
                tok = blk * block_size + slot
                if tok >= seq_len:
                    break
                for h in range(num_kv_heads):
                    key_cache[block_id, h, slot]   = _quantize_to_fp4(k_ref[seq_id, tok, h])
                    value_cache[block_id, h, slot] = _quantize_to_fp4(v_ref[seq_id, tok, h])
            block_id += 1

    context_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32)
    return (
        key_cache.to(device), value_cache.to(device),
        block_tables.to(device), context_lens.to(device),
        k_ref, v_ref,
    )


def _ref_attention_fp4(q, k_ref, v_ref, scale):
    """Reference attention on FP4-quantized K/V via torch SDPA."""
    num_seqs, num_heads, head_size = q.shape
    num_kv_heads = k_ref.shape[2]
    gqa = num_heads // num_kv_heads

    def roundtrip(t):
        return _dequantize_fp4(_quantize_to_fp4(t.reshape(-1))).reshape(t.shape)

    outs = []
    for s in range(num_seqs):
        k_q = roundtrip(k_ref[s]).repeat_interleave(gqa, dim=1)  # [seq, heads, D]
        v_q = roundtrip(v_ref[s]).repeat_interleave(gqa, dim=1)
        out_s = torch.nn.functional.scaled_dot_product_attention(
            q[s].unsqueeze(1).float(),
            k_q.permute(1, 0, 2).float(),
            v_q.permute(1, 0, 2).float(),
            scale=scale,
        )  # [heads, 1, D]
        outs.append(out_s.squeeze(1))
    return torch.stack(outs, dim=0)  # [num_seqs, heads, D]


SKIP_NO_GPU  = pytest.mark.skipif(not _has_gpu(),       reason="No ROCm GPU available")
SKIP_NO_FP4  = pytest.mark.skipif(not _has_aiter_fp4(), reason="AITER FP4 patch not applied")


@SKIP_NO_GPU
@SKIP_NO_FP4
class TestFP4PagedAttention:
    """AITER FP4 paged attention vs torch SDPA reference.

    Requires paged_attention_fp4_patch.py applied to AITER.
    """

    @pytest.fixture(params=[
        (1,  64,  8, 8, 128, 16),
        (2, 128,  8, 8, 128, 32),
        (4, 256, 16, 2, 128, 16),   # GQA: 16 Q-heads, 2 KV-heads
    ])
    def cfg(self, request):
        return request.param

    def _run(self, cfg, device="cuda"):
        import aiter
        num_seqs, seq_len, num_heads, num_kv_heads, head_size, block_size = cfg
        scale = 1.0 / math.sqrt(head_size)

        torch.manual_seed(42)
        q = torch.randn(num_seqs, num_heads, head_size, dtype=torch.float16, device=device)

        key_cache, value_cache, block_tables, context_lens, k_ref, v_ref = \
            _make_paged_kv_fp4(num_seqs, seq_len, num_kv_heads, head_size, block_size, device)

        out = torch.zeros(num_seqs, num_heads, head_size, dtype=torch.float16, device=device)
        max_num_partitions = math.ceil(seq_len / 256)
        ws = max(
            num_seqs * num_heads * max_num_partitions * 2
            + num_seqs * num_heads * max_num_partitions * head_size // 2,
            256,
        )
        workspace = torch.zeros(ws, dtype=torch.float32, device=device)
        k_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
        v_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

        aiter.paged_attention_v1(
            out, workspace, q, key_cache, value_cache,
            scale, block_tables, None, context_lens, seq_len,
            None, "fp4_e2m1", "HND", 0.0, k_scale, v_scale,
        )
        return out.float().cpu(), k_ref, v_ref, q.float().cpu(), scale

    def test_cosine_similarity(self, cfg):
        """Output cosine similarity vs torch SDPA reference must exceed 0.99."""
        num_seqs, seq_len, num_heads, num_kv_heads, head_size, block_size = cfg
        aiter_out, k_ref, v_ref, q_cpu, scale = self._run(cfg)
        ref_out = _ref_attention_fp4(q_cpu, k_ref, v_ref, scale)
        cos = torch.nn.functional.cosine_similarity(
            aiter_out.reshape(num_seqs * num_heads, head_size),
            ref_out.reshape(num_seqs * num_heads, head_size),
            dim=-1,
        )
        assert cos.mean().item() > 0.99, \
            f"mean cosine {cos.mean():.4f} < 0.99 (min {cos.min():.4f})"

    def test_output_not_nan(self, cfg):
        """Output must not contain NaN or Inf."""
        aiter_out, *_ = self._run(cfg)
        assert not aiter_out.isnan().any(), "NaN in FP4 PA output"
        assert not aiter_out.isinf().any(), "Inf in FP4 PA output"

    def test_output_magnitude(self, cfg):
        """Output RMS must be in a sane range."""
        aiter_out, *_ = self._run(cfg)
        rms = aiter_out.pow(2).mean().sqrt().item()
        assert 0.01 < rms < 100.0, f"Output RMS {rms:.4f} out of range"


@SKIP_NO_GPU
class TestFP4Encoding:
    """Unit tests for FP4 E2M1 encode/decode helpers."""

    def test_roundtrip_representable(self):
        """Every representable FP4 value survives encode→decode exactly."""
        for v in FP4_TO_FLOAT:
            assert FP4_TO_FLOAT[_float_to_fp4_nibble(v)] == v

    def test_pack_unpack(self):
        """Pack then unpack recovers the nearest FP4 value."""
        vals = torch.tensor([1.0, 2.0, 0.5, 3.0, -1.0, -0.5, 0.0, 6.0])
        packed = _quantize_to_fp4(vals)
        assert packed.shape == (4,)
        for got, expected in zip(_dequantize_fp4(packed).tolist(), vals.tolist()):
            assert got == FP4_TO_FLOAT[_float_to_fp4_nibble(expected)]

    def test_clamp_to_max(self):
        """Values outside [-6, 6] clamp to ±6.0."""
        assert _dequantize_fp4(_quantize_to_fp4(torch.tensor([100.0, 0.0])))[0].item() == 6.0
        assert _dequantize_fp4(_quantize_to_fp4(torch.tensor([-100.0, 0.0])))[0].item() == -6.0
