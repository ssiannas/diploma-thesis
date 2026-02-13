"""Validate that our metrics are mathematically correct.

This test suite ensures our metrics match expected values and reference
implementations.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcml.metrics.compression import CompressionCalculator
from pcml.metrics.curvature import CurvatureQualityCalculator
from pcml.metrics.quality import ColorQualityCalculator, GeometryQualityCalculator
from pcml.models.losses import ChamferDistance


class TestMetricsValidation:
    """Test suite for metrics validation."""

    @pytest.fixture
    def identical_point_clouds(self):
        """Create two identical point clouds."""
        np.random.seed(42)
        pc = np.random.randn(1000, 3).astype(np.float32)
        return pc, pc.copy()

    @pytest.fixture
    def slightly_noisy_clouds(self):
        """Create original and slightly noisy version."""
        np.random.seed(42)
        original = np.random.randn(1000, 3).astype(np.float32)
        noise = np.random.randn(1000, 3).astype(np.float32) * 0.01
        noisy = original + noise
        return original, noisy

    def test_identical_clouds_zero_error(self, identical_point_clouds):
        """Identical point clouds should have near-zero error."""
        pc1, pc2 = identical_point_clouds

        metrics = GeometryQualityCalculator.calculate_all(pc1, pc2)

        assert metrics.mse < 1e-10, f"MSE should be ~0, got {metrics.mse}"
        assert metrics.rmse < 1e-5, f"RMSE should be ~0, got {metrics.rmse}"
        assert metrics.d1 < 1e-5, f"D1 should be ~0, got {metrics.d1}"
        assert metrics.psnr > 100, f"PSNR should be very high, got {metrics.psnr}"

        print(
            f"[OK] Identical clouds: MSE={metrics.mse:.2e}, PSNR={metrics.psnr:.1f} dB"
        )

    def test_psnr_calculation_correctness(self, slightly_noisy_clouds):
        """Verify PSNR calculation is correct."""
        original, noisy = slightly_noisy_clouds

        metrics = GeometryQualityCalculator.calculate_all(original, noisy)

        bbox_diagonal = np.linalg.norm(original.max(axis=0) - original.min(axis=0))
        expected_psnr = 10 * np.log10(bbox_diagonal**2 / metrics.mse)

        assert np.abs(metrics.psnr - expected_psnr) < 0.01, (
            f"PSNR mismatch: calculated={metrics.psnr:.2f}, "
            f"expected={expected_psnr:.2f}"
        )

        print(f"[OK] PSNR calculation correct: {metrics.psnr:.2f} dB")

    def test_chamfer_distance_symmetry(self):
        """Chamfer distance should be symmetric."""
        np.random.seed(42)
        pc1 = np.random.randn(500, 3).astype(np.float32)
        pc2 = np.random.randn(500, 3).astype(np.float32) * 0.5

        pc1_torch = torch.from_numpy(pc1).unsqueeze(0)
        pc2_torch = torch.from_numpy(pc2).unsqueeze(0)

        chamfer = ChamferDistance()

        dist_12 = chamfer(pc1_torch, pc2_torch)
        dist_21 = chamfer(pc2_torch, pc1_torch)

        assert torch.allclose(
            dist_12, dist_21, rtol=1e-5
        ), f"Chamfer not symmetric: {dist_12:.6f} vs {dist_21:.6f}"

        print(f"[OK] Chamfer symmetry: {dist_12:.6f} == {dist_21:.6f}")

    def test_chamfer_distance_components(self):
        """Test Chamfer distance forward and backward components."""
        np.random.seed(42)
        pc1 = np.random.randn(500, 3).astype(np.float32)
        pc2 = np.random.randn(500, 3).astype(np.float32)

        pc1_torch = torch.from_numpy(pc1).unsqueeze(0)
        pc2_torch = torch.from_numpy(pc2).unsqueeze(0)

        chamfer = ChamferDistance()
        total, forward, backward = chamfer(pc1_torch, pc2_torch, return_components=True)

        assert torch.allclose(
            total, forward + backward
        ), "Total should equal forward + backward"

        print(
            f"[OK] Chamfer components: total={total:.6f}, "
            f"forward={forward:.6f}, backward={backward:.6f}"
        )

    def test_color_metrics_identical(self):
        """Identical colors should have zero error."""
        np.random.seed(42)
        colors = np.random.randint(0, 256, size=(1000, 3), dtype=np.uint8)

        metrics = ColorQualityCalculator.calculate_all(colors, colors)

        assert metrics.mse < 1e-10, f"Color MSE should be ~0, got {metrics.mse}"
        assert metrics.psnr > 100, f"Color PSNR should be very high, got {metrics.psnr}"

        print(
            f"[OK] Identical colors: MSE={metrics.mse:.2e}, PSNR={metrics.psnr:.1f} dB"
        )

    def test_compression_ratio_calculation(self):
        """Verify compression ratio calculation."""
        original_size = 1000000
        compressed_size = 100000
        num_points = 10000

        metrics = CompressionCalculator.calculate_from_sizes(
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size,
            num_points=num_points,
        )

        expected_ratio = compressed_size / original_size
        expected_bpp = (compressed_size * 8) / num_points

        assert np.abs(metrics.compression_ratio - expected_ratio) < 1e-6
        assert np.abs(metrics.bits_per_point - expected_bpp) < 1e-6
        assert np.abs(metrics.compression_gain - 10.0) < 1e-6

        print(
            f"[OK] Compression: ratio={metrics.compression_ratio:.2f}, "
            f"bpp={metrics.bits_per_point:.2f}"
        )

    def test_d1_d2_relationship(self, slightly_noisy_clouds):
        """D2 should be >= D1 (both directions)."""
        original, noisy = slightly_noisy_clouds

        metrics = GeometryQualityCalculator.calculate_all(original, noisy)

        assert (
            metrics.d2_sym >= metrics.d1
        ), f"D2 ({metrics.d2_sym}) should be >= D1 ({metrics.d1})"

        print(f"[OK] D1={metrics.d1:.6f}, D2={metrics.d2_sym:.6f}")

    def test_hausdorff_equals_d2(self, slightly_noisy_clouds):
        """Hausdorff distance should equal D2_symmetric."""
        original, noisy = slightly_noisy_clouds

        metrics = GeometryQualityCalculator.calculate_all(original, noisy)

        assert np.abs(metrics.hausdorff - metrics.d2_sym) < 1e-6, (
            f"Hausdorff ({metrics.hausdorff}) should equal " f"D2 ({metrics.d2_sym})"
        )

        print(f"[OK] Hausdorff == D2: {metrics.hausdorff:.6f}")


class TestCurvatureMetrics:
    """Test suite for curvature-stratified oversmoothing metrics."""

    @pytest.fixture
    def plane_and_noisy(self):
        """Flat plane (low curvature) with noisy version.

        Returns original, reconstructed, and curvature arrays.
        """
        np.random.seed(42)
        n = 2000
        # Create a flat grid
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        z = np.zeros(n)
        original = np.column_stack([x, y, z]).astype(np.float64)

        # Add noise: more noise at high-x (simulating edge region)
        noise = np.random.randn(n, 3) * 0.1
        # Make noise proportional to x coordinate
        noise *= (x / 100.0)[:, None]
        reconstructed = (original + noise).astype(np.float64)

        curvature = CurvatureQualityCalculator.compute_curvature(original, k=15)
        return original, reconstructed, curvature

    @pytest.fixture
    def sphere_cloud(self):
        """Point cloud on a sphere surface (uniform curvature).

        Creates original + noisy version where edge-like points get more noise.
        """
        np.random.seed(42)
        n = 1500
        # Points on a unit sphere
        phi = np.random.uniform(0, 2 * np.pi, n)
        theta = np.random.uniform(0, np.pi, n)
        r = 50.0
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        original = np.column_stack([x, y, z]).astype(np.float64)

        curvature = CurvatureQualityCalculator.compute_curvature(original, k=15)

        # Add curvature-dependent noise (simulate oversmoothing)
        noise_scale = 0.5 + 2.0 * (curvature / (curvature.max() + 1e-10))
        noise = np.random.randn(n, 3) * noise_scale[:, None]
        reconstructed = (original + noise).astype(np.float64)

        return original, reconstructed, curvature

    def test_kl_divergence_identical_distributions(self):
        """KL divergence of identical distributions should be ~0."""
        np.random.seed(42)
        curvature = np.random.exponential(0.1, 1000)
        kl = CurvatureQualityCalculator.compute_curvature_kl(curvature, curvature)
        assert kl < 0.01, f"KL of identical distributions should be ~0, got {kl}"

    def test_kl_divergence_different_distributions(self):
        """KL divergence of very different distributions should be large."""
        np.random.seed(42)
        orig = np.random.exponential(0.05, 1000)
        dec = np.random.exponential(0.5, 1000)
        kl = CurvatureQualityCalculator.compute_curvature_kl(orig, dec)
        assert kl > 0.1, f"KL of different distributions should be large, got {kl}"

    def test_kl_uses_independent_curvature(self, sphere_cloud):
        """KL in compute_stratified_psnr should use independently-computed curvature."""
        original, reconstructed, curvature = sphere_cloud

        # The KL value should be the same regardless of direction
        # (it always compares original vs independently-computed rec curvature)
        fwd = CurvatureQualityCalculator.compute_stratified_psnr(
            original,
            reconstructed,
            curvature,
            direction="forward",
            k=15,
        )
        rev = CurvatureQualityCalculator.compute_stratified_psnr(
            original,
            reconstructed,
            curvature,
            direction="reverse",
            k=15,
        )
        fwd_self = CurvatureQualityCalculator.compute_stratified_psnr(
            original,
            reconstructed,
            curvature,
            direction="forward_self",
            k=15,
        )

        # All three should compute KL from independently-computed rec curvature
        assert abs(fwd.curvature_kl - rev.curvature_kl) < 1e-6, (
            f"KL should be same across directions: "
            f"fwd={fwd.curvature_kl}, rev={rev.curvature_kl}"
        )
        assert abs(fwd.curvature_kl - fwd_self.curvature_kl) < 1e-6, (
            f"KL should be same across directions: "
            f"fwd={fwd.curvature_kl}, fwd_self={fwd_self.curvature_kl}"
        )

    def test_forward_self_direction(self, sphere_cloud):
        """forward_self direction should produce valid stratified metrics."""
        original, reconstructed, curvature = sphere_cloud

        result = CurvatureQualityCalculator.compute_stratified_psnr(
            original,
            reconstructed,
            curvature,
            direction="forward_self",
            k=15,
        )

        assert result.direction == "forward_self"
        assert np.isfinite(result.flat_psnr)
        assert np.isfinite(result.edge_psnr)
        assert np.isfinite(result.degradation)
        # With curvature-dependent noise, we expect positive degradation
        assert result.degradation > 0, (
            f"Expected positive degradation for curvature-dependent noise, "
            f"got {result.degradation}"
        )

    def test_forward_self_no_nan_psnr(self, plane_and_noisy):
        """forward_self should not produce NaN PSNR values."""
        original, reconstructed, curvature = plane_and_noisy

        result = CurvatureQualityCalculator.compute_stratified_psnr(
            original,
            reconstructed,
            curvature,
            direction="forward_self",
            k=15,
        )
        assert np.isfinite(result.flat_psnr)
        assert np.isfinite(result.medium_psnr)
        assert np.isfinite(result.edge_psnr)

    def test_nn_multiplicity_basic(self, sphere_cloud):
        """NN multiplicity should produce valid results."""
        original, reconstructed, curvature = sphere_cloud

        result = CurvatureQualityCalculator.compute_nn_multiplicity(
            original,
            reconstructed,
            curvature,
            n_strata=3,
        )

        # All strata should be present
        assert "flat" in result.per_stratum_mean
        assert "medium" in result.per_stratum_mean
        assert "edge" in result.per_stratum_mean

        # Mean multiplicity should be > 0
        assert result.overall_mean > 0
        assert result.overall_max >= 1

        # Per-stratum means should be positive
        for name in ["flat", "medium", "edge"]:
            assert (
                result.per_stratum_mean[name] > 0
            ), f"Stratum {name} has zero multiplicity"

    def test_nn_multiplicity_identical_clouds(self):
        """For identical clouds, multiplicity should be 1 everywhere."""
        np.random.seed(42)
        points = np.random.randn(500, 3)
        curvature = CurvatureQualityCalculator.compute_curvature(points, k=10)

        result = CurvatureQualityCalculator.compute_nn_multiplicity(
            points,
            points.copy(),
            curvature,
        )

        assert (
            result.overall_mean == 1.0
        ), f"Identical clouds should have multiplicity 1, got {result.overall_mean}"

    def test_psnr_vs_curvature_curve_basic(self, sphere_cloud):
        """Continuous PSNR-vs-curvature curve should produce valid results."""
        original, reconstructed, curvature = sphere_cloud

        result = CurvatureQualityCalculator.compute_psnr_vs_curvature_curve(
            original,
            reconstructed,
            curvature,
            n_bins=10,
            direction="reverse",
            k=15,
        )

        assert len(result.bin_centers) > 0
        assert len(result.psnrs) == len(result.bin_centers)
        assert len(result.counts) == len(result.bin_centers)
        assert result.direction == "reverse"

        # All bins should have points
        assert np.all(result.counts > 0)

        # With curvature-dependent noise, high-curvature bins should have
        # lower PSNR than low-curvature bins (monotonically decreasing trend)
        valid = np.isfinite(result.psnrs)
        if np.sum(valid) >= 2:
            first_valid = result.psnrs[valid][0]
            last_valid = result.psnrs[valid][-1]
            assert first_valid > last_valid, (
                f"Expected decreasing PSNR trend, got first={first_valid:.2f}, "
                f"last={last_valid:.2f}"
            )

    def test_psnr_vs_curvature_forward_self(self, sphere_cloud):
        """PSNR-vs-curvature curve should work with forward_self direction."""
        original, reconstructed, curvature = sphere_cloud

        result = CurvatureQualityCalculator.compute_psnr_vs_curvature_curve(
            original,
            reconstructed,
            curvature,
            n_bins=10,
            direction="forward_self",
            k=15,
        )

        assert result.direction == "forward_self"
        assert len(result.bin_centers) > 0
        assert np.any(np.isfinite(result.psnrs))

    def test_stratified_psnr_invalid_direction(self):
        """Invalid direction should raise ValueError."""
        np.random.seed(42)
        pts = np.random.randn(100, 3)
        curv = np.random.rand(100)

        with pytest.raises(ValueError, match="direction"):
            CurvatureQualityCalculator.compute_stratified_psnr(
                pts,
                pts,
                curv,
                direction="invalid",
            )

    def test_curvature_computation_shape(self):
        """Curvature computation should return correct shape and range."""
        np.random.seed(42)
        points = np.random.randn(200, 3)
        curvature = CurvatureQualityCalculator.compute_curvature(points, k=10)

        assert curvature.shape == (200,)
        assert np.all(curvature >= 0)
        assert np.all(curvature <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
