"""Tests for Lloyd-Max codebook generation."""

import math
import pytest
import torch
from turboquant.core.codebook import solve_codebook, make_codebook


class TestSolveCodebook:
    """Test codebook generation for various bit widths."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_correct_number_of_centroids(self, bits):
        centroids, boundaries = solve_codebook(128, bits)
        assert len(centroids) == 2**bits
        assert len(boundaries) == 2**bits - 1

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_centroids_sorted(self, bits):
        centroids, boundaries = solve_codebook(128, bits)
        for i in range(len(centroids) - 1):
            assert centroids[i] < centroids[i + 1], "Centroids must be sorted"

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_boundaries_sorted(self, bits):
        centroids, boundaries = solve_codebook(128, bits)
        for i in range(len(boundaries) - 1):
            assert boundaries[i] < boundaries[i + 1], "Boundaries must be sorted"

    def test_boundaries_between_centroids(self):
        """Each boundary should lie between adjacent centroids."""
        centroids, boundaries = solve_codebook(128, 3)
        for i, b in enumerate(boundaries):
            assert centroids[i] < b < centroids[i + 1]

    def test_symmetry(self):
        """Codebook should be approximately symmetric around 0."""
        centroids, _ = solve_codebook(128, 3)
        for i in range(len(centroids)):
            j = len(centroids) - 1 - i
            assert abs(centroids[i] + centroids[j]) < 1e-6

    def test_sigma_scaling(self):
        """Centroids should scale with sigma = 1/sqrt(d)."""
        c64, _ = solve_codebook(64, 2)
        c256, _ = solve_codebook(256, 2)
        # sigma_64 / sigma_256 = sqrt(256/64) = 2
        ratio = abs(c64[-1] / c256[-1])
        assert 1.8 < ratio < 2.2


class TestMakeCodebook:
    """Test GPU tensor creation."""

    def test_tensor_shapes(self):
        centroids, boundaries = make_codebook(128, 3)
        assert centroids.shape == (8,)
        assert boundaries.shape == (7,)

    def test_dtype(self):
        centroids, boundaries = make_codebook(128, 3, dtype=torch.float16)
        assert centroids.dtype == torch.float16
        assert boundaries.dtype == torch.float16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
    def test_device(self):
        centroids, boundaries = make_codebook(128, 3, device=torch.device("cuda"))
        assert centroids.is_cuda
        assert boundaries.is_cuda
