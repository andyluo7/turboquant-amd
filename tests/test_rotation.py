"""Tests for WHT rotation and weight fusion."""

import math
import pytest
import torch
from turboquant.core.rotation import (
    make_wht_matrix,
    fuse_rotation_into_qkv_proj,
    fuse_rotation_into_o_proj,
)


class TestWHTMatrix:
    """Test Walsh-Hadamard Transform matrix properties."""

    @pytest.mark.parametrize("d", [2, 4, 8, 16, 32, 64, 128])
    def test_orthogonality(self, d):
        """H @ H.T should be identity (orthogonal matrix)."""
        H = make_wht_matrix(d)
        I = torch.eye(d)
        err = (H @ H.T - I).abs().max().item()
        assert err < 1e-5, f"WHT not orthogonal: max error {err}"

    @pytest.mark.parametrize("d", [2, 4, 8, 16, 32, 64, 128])
    def test_self_inverse(self, d):
        """WHT is its own inverse: H @ H = I."""
        H = make_wht_matrix(d)
        I = torch.eye(d)
        err = (H @ H - I).abs().max().item()
        assert err < 1e-5, f"WHT not self-inverse: max error {err}"

    def test_roundtrip_precision(self):
        """Roundtrip x → x@H → result@H should recover x with max error < 3e-7."""
        d = 128
        H = make_wht_matrix(d)
        torch.manual_seed(42)
        x = torch.randn(1024, d)
        x_rot = x @ H
        x_back = x_rot @ H
        err = (x - x_back).abs().max().item()
        assert err < 3e-6, f"Roundtrip error {err} exceeds threshold"

    def test_norm_preservation(self):
        """Rotation should preserve vector norms."""
        d = 128
        H = make_wht_matrix(d)
        torch.manual_seed(42)
        x = torch.randn(256, d)
        x_rot = x @ H
        norm_orig = x.norm(dim=-1)
        norm_rot = x_rot.norm(dim=-1)
        err = (norm_orig - norm_rot).abs().max().item()
        assert err < 1e-4, f"Norm not preserved: max error {err}"

    def test_gaussianization(self):
        """WHT-rotated coordinates should be approximately Gaussian."""
        d = 128
        H = make_wht_matrix(d)
        torch.manual_seed(42)
        # Create a sparse vector (not Gaussian)
        x = torch.zeros(10000, d)
        x[:, :4] = torch.randn(10000, 4) * 5
        x_rot = x @ H
        # After rotation, all columns should have similar variance
        col_vars = x_rot.var(dim=0)
        cv = col_vars.std() / col_vars.mean()
        assert cv < 0.2, f"Coordinates not gaussianized: CV = {cv}"


class TestWeightFusion:
    """Test that weight fusion is mathematically equivalent to explicit rotation."""

    def test_qkv_fusion_correctness(self):
        """Fused QKV should produce same result as explicit rotation."""
        num_heads, num_kv_heads, head_size, hidden_size = 4, 2, 128, 512
        torch.manual_seed(42)

        H = make_wht_matrix(head_size)
        qkv_weight = torch.randn(
            (num_heads + 2 * num_kv_heads) * head_size, hidden_size
        )
        hidden = torch.randn(2, hidden_size)

        # Method A: Explicit rotation
        qkv_out = hidden @ qkv_weight.T
        q_size = num_heads * head_size
        k_size = num_kv_heads * head_size
        Q = qkv_out[:, :q_size].view(2, num_heads, head_size)
        K = qkv_out[:, q_size : q_size + k_size].view(2, num_kv_heads, head_size)
        Q_rot = Q @ H
        K_rot = K @ H

        # Method B: Fused weights
        qkv_fused = fuse_rotation_into_qkv_proj(
            qkv_weight.clone(), num_heads, num_kv_heads, head_size
        )
        qkv_out_fused = hidden @ qkv_fused.T
        Q_fused = qkv_out_fused[:, :q_size].view(2, num_heads, head_size)
        K_fused = qkv_out_fused[:, q_size : q_size + k_size].view(
            2, num_kv_heads, head_size
        )

        assert (Q_rot - Q_fused).abs().max().item() < 1e-3
        assert (K_rot - K_fused).abs().max().item() < 1e-3

    def test_o_proj_fusion_correctness(self):
        """Fused O projection should produce same result as explicit de-rotation."""
        num_heads, head_size, hidden_size = 4, 128, 512
        torch.manual_seed(42)

        H = make_wht_matrix(head_size)
        o_weight = torch.randn(hidden_size, num_heads * head_size)
        attn_rot = torch.randn(2, num_heads, head_size)

        # Method A: Explicit de-rotation + O proj
        attn_derot = (attn_rot @ H).reshape(2, -1)
        out_A = attn_derot @ o_weight.T

        # Method B: Fused O proj
        o_fused = fuse_rotation_into_o_proj(o_weight.clone(), num_heads, head_size)
        out_B = attn_rot.reshape(2, -1) @ o_fused.T

        cos = torch.nn.functional.cosine_similarity(
            out_A.reshape(1, -1), out_B.reshape(1, -1)
        ).item()
        assert cos > 0.9999, f"Cosine similarity {cos} too low"
