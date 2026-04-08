"""Tests for PolarQuant compression correctness."""

import pytest
import torch
from turboquant.core.codebook import make_codebook
from turboquant.core.rotation import make_wht_matrix
from turboquant.core.quantizer import PolarQuantConfig, polarquant_compress, polarquant_decompress


class TestPolarQuantCompress:
    """Test PolarQuant compression pipeline."""

    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        d = 128
        config = PolarQuantConfig(head_dim=d, bits=3, use_qjl=True)
        centroids, boundaries = make_codebook(d, 3)
        H = make_wht_matrix(d)
        x = torch.randn(64, d)
        return config, centroids, boundaries, H, x

    def test_indices_in_range(self, setup):
        config, centroids, boundaries, H, x = setup
        result = polarquant_compress(x, centroids, boundaries, config, H)
        assert result["indices"].min() >= 0
        assert result["indices"].max() < config.n_centroids

    def test_norms_positive(self, setup):
        config, centroids, boundaries, H, x = setup
        result = polarquant_compress(x, centroids, boundaries, config, H)
        assert (result["norms"] >= 0).all()

    def test_reconstruction_shape(self, setup):
        config, centroids, boundaries, H, x = setup
        result = polarquant_compress(x, centroids, boundaries, config, H)
        assert result["k_mse_rot"].shape == x.shape

    def test_signs_binary(self, setup):
        """QJL signs should be +1 or -1."""
        config, centroids, boundaries, H, x = setup
        result = polarquant_compress(x, centroids, boundaries, config, H)
        signs = result["signs"]
        assert ((signs == 1.0) | (signs == -1.0)).all()

    def test_cosine_similarity(self, setup):
        """Reconstructed vectors should have high cosine similarity with originals."""
        config, centroids, boundaries, H, x = setup
        result = polarquant_compress(x, centroids, boundaries, config, H)
        x_rot = x @ H
        recon = result["k_mse_rot"]
        cos = torch.nn.functional.cosine_similarity(x_rot, recon, dim=-1)
        assert cos.mean() > 0.95, f"Mean cosine sim {cos.mean():.4f} too low"

    def test_3d_input(self, setup):
        """Should handle [B, H, D] input."""
        config, centroids, boundaries, H, _ = setup
        torch.manual_seed(0)
        x = torch.randn(4, 8, 128)
        result = polarquant_compress(x, centroids, boundaries, config, H)
        assert result["k_mse_rot"].shape == (4, 8, 128)
        assert result["indices"].shape == (4, 8, 128)

    def test_no_qjl_mode(self):
        """V compression uses MSE-only (no QJL)."""
        torch.manual_seed(42)
        config = PolarQuantConfig(head_dim=128, bits=3, use_qjl=False)
        centroids, boundaries = make_codebook(128, 3)
        H = make_wht_matrix(128)
        x = torch.randn(32, 128)
        result = polarquant_compress(x, centroids, boundaries, config, H)
        assert "signs" not in result

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_compression_ratio(self, bits):
        config = PolarQuantConfig(head_dim=128, bits=bits)
        ratio = config.compression_ratio
        expected = {2: 5.7, 3: 4.6, 4: 3.8}
        # Allow some tolerance
        assert abs(ratio - expected[bits]) < 0.5, (
            f"bits={bits}: expected ~{expected[bits]}x, got {ratio:.1f}x"
        )


class TestRoundtrip:
    """Test compress → decompress roundtrip."""

    def test_decompress_returns_mse(self):
        torch.manual_seed(42)
        config = PolarQuantConfig(head_dim=128, bits=3)
        centroids, boundaries = make_codebook(128, 3)
        H = make_wht_matrix(128)
        x = torch.randn(32, 128)
        compressed = polarquant_compress(x, centroids, boundaries, config, H)
        decompressed = polarquant_decompress(compressed, centroids, config)
        assert torch.equal(decompressed, compressed["k_mse_rot"])
