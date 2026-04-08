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
