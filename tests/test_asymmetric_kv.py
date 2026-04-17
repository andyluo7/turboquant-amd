"""Test asymmetric K/V compression."""
import torch
import pytest
from turboquant.core.quantizer import PolarQuantConfig
from turboquant.integration.vllm_backend import (
    TurboQuantFP8Config,
    TurboQuantFP8State,
)


class TestAsymmetricKV:
    """Test asymmetric K/V bit width support."""

    def test_symmetric_default(self):
        """Default: K and V use same bits."""
        config = PolarQuantConfig(bits=4)
        assert config.k_bits == 4
        assert config.effective_v_bits == 4
        assert not config.is_asymmetric

    def test_asymmetric_config(self):
        """Asymmetric: K=4, V=2."""
        config = PolarQuantConfig(bits=4, v_bits=2)
        assert config.k_bits == 4
        assert config.effective_v_bits == 2
        assert config.is_asymmetric

    def test_compression_ratios(self):
        """Asymmetric gives better combined compression."""
        sym = PolarQuantConfig(head_dim=128, bits=4)
        asym = PolarQuantConfig(head_dim=128, bits=4, v_bits=2)
        
        # Symmetric: K and V same ratio
        assert sym.k_compression_ratio == sym.v_compression_ratio
        
        # Asymmetric: V ratio is higher (more compression)
        assert asym.v_compression_ratio > asym.k_compression_ratio
        
        # Combined asymmetric is better than symmetric
        assert asym.combined_compression_ratio > sym.combined_compression_ratio

    def test_state_different_centroids(self):
        """K and V should have different centroid counts when asymmetric."""
        state = TurboQuantFP8State(
            head_size=128, layer_idx=5, device=torch.device("cpu"),
            k_bits=4, v_bits=2,
        )
        # K: 16 centroids (2^4), V: 4 centroids (2^2)
        assert len(state.k_centroids) == 16
        assert len(state.v_centroids) == 4
        assert len(state.k_boundaries) == 15
        assert len(state.v_boundaries) == 3

    def test_state_symmetric_shares_centroids(self):
        """Symmetric mode should share centroid tensors (no extra memory)."""
        state = TurboQuantFP8State(
            head_size=128, layer_idx=5, device=torch.device("cpu"),
            k_bits=4, v_bits=4,
        )
        assert state.k_centroids is state.v_centroids
        assert state.k_boundaries is state.v_boundaries

    def test_config_passes_bits_to_state(self):
        """TurboQuantFP8Config passes k_bits/v_bits to created states."""
        config = TurboQuantFP8Config(
            head_size=128, num_layers=40,
            k_bits=4, v_bits=2,
        )
        state = config.get_state(10, torch.device("cpu"))
        assert state.k_bits == 4
        assert state.v_bits == 2
        assert len(state.k_centroids) == 16
        assert len(state.v_centroids) == 4

    def test_compress_k_vs_v_different(self):
        """K and V compression should produce different results when asymmetric."""
        state = TurboQuantFP8State(
            head_size=128, layer_idx=5, device=torch.device("cpu"),
            k_bits=4, v_bits=2,
        )
        x = torch.randn(1, 4, 128)  # [T=1, H=4, D=128]
        
        fp8_k, scales_k = state.compress_to_fp8(x, is_value=False)
        fp8_v, scales_v = state.compress_to_fp8(x, is_value=True)
        
        # Different quantization → different output (V is coarser)
        assert fp8_k.shape == fp8_v.shape
        # V with 2 bits (4 centroids) should have less unique values
        # than K with 4 bits (16 centroids)
        k_unique = fp8_k.float().unique().numel()
        v_unique = fp8_v.float().unique().numel()
        assert v_unique <= k_unique  # V is coarser


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
