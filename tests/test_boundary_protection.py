"""Test boundary layer protection for TurboQuant."""
import torch
import pytest
from turboquant.core.quantizer import PolarQuantConfig
from turboquant.integration.vllm_backend import TurboQuantFP8Config


class TestBoundaryProtection:
    """Test that boundary layers skip TQ compression."""

    def test_config_should_compress_with_protection(self):
        """First/last 2 layers should NOT compress."""
        config = TurboQuantFP8Config(
            head_size=128,
            num_layers=40,
            protect_boundary_layers=True,
            num_protected_layers=2,
        )
        # Protected layers
        assert not config.should_compress(0)   # first
        assert not config.should_compress(1)   # second
        assert not config.should_compress(38)  # second-to-last
        assert not config.should_compress(39)  # last
        
        # Compressed layers
        assert config.should_compress(2)
        assert config.should_compress(20)
        assert config.should_compress(37)

    def test_config_should_compress_without_protection(self):
        """All layers compress when protection is disabled."""
        config = TurboQuantFP8Config(
            head_size=128,
            num_layers=40,
            protect_boundary_layers=False,
        )
        for i in range(40):
            assert config.should_compress(i)

    def test_config_should_compress_zero_layers(self):
        """When num_layers=0, all layers compress (can't determine boundaries)."""
        config = TurboQuantFP8Config(
            head_size=128,
            num_layers=0,
            protect_boundary_layers=True,
        )
        for i in range(40):
            assert config.should_compress(i)

    def test_get_state_returns_none_for_protected(self):
        """get_state returns None for protected boundary layers."""
        config = TurboQuantFP8Config(
            head_size=128,
            num_layers=40,
            protect_boundary_layers=True,
            num_protected_layers=2,
        )
        device = torch.device("cpu")
        
        # Protected layers return None
        assert config.get_state(0, device) is None
        assert config.get_state(1, device) is None
        assert config.get_state(39, device) is None
        
        # Compressed layers return a state object
        state = config.get_state(2, device)
        assert state is not None
        assert state.layer_idx == 2

    def test_custom_protection_count(self):
        """Custom number of protected layers."""
        config = TurboQuantFP8Config(
            head_size=128,
            num_layers=60,
            protect_boundary_layers=True,
            num_protected_layers=4,
        )
        # First 4 protected
        for i in range(4):
            assert not config.should_compress(i)
        assert config.should_compress(4)
        
        # Last 4 protected
        assert config.should_compress(55)
        for i in range(56, 60):
            assert not config.should_compress(i)

    def test_polarquant_config_should_compress(self):
        """PolarQuantConfig boundary protection."""
        config = PolarQuantConfig(
            head_dim=128,
            bits=4,
            protect_boundary_layers=True,
            num_protected_layers=2,
            num_layers=40,
        )
        assert not config.should_compress(0)
        assert not config.should_compress(1)
        assert config.should_compress(2)
        assert config.should_compress(37)
        assert not config.should_compress(38)
        assert not config.should_compress(39)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
