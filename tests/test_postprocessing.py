"""Tests for postprocessing dataset utilities.

Uses synthetic small clouds -- no GPU or real data required.
"""

import json

import numpy as np
import pytest

from postprocessing.dataset import compute_curvature, extract_patches

try:
    import MinkowskiEngine

    HAS_ME = True
except ImportError:
    HAS_ME = False


def _make_sphere(n: int = 2000, radius: float = 100.0) -> np.ndarray:
    """Generate points on a sphere surface."""
    rng = np.random.default_rng(42)
    phi = rng.uniform(0, 2 * np.pi, n)
    costheta = rng.uniform(-1, 1, n)
    theta = np.arccos(costheta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.column_stack([x, y, z]).astype(np.float32)


def _make_cube_cloud(n: int = 3000, scale: float = 200.0) -> np.ndarray:
    """Generate points uniformly in a cube."""
    rng = np.random.default_rng(123)
    return (rng.uniform(0, scale, (n, 3))).astype(np.float32)


class TestComputeCurvature:
    def test_shape_and_range(self):
        points = _make_sphere(1500)
        curvature = compute_curvature(points, k=20)
        assert curvature.shape == (1500,)
        assert curvature.dtype == np.float32
        assert np.all(curvature >= 0)
        assert np.all(curvature <= 1)

    def test_small_cloud(self):
        """Works on clouds smaller than k."""
        points = _make_sphere(10)
        curvature = compute_curvature(points, k=30)
        assert curvature.shape == (10,)
        assert np.all(np.isfinite(curvature))

    def test_vectorized_matches_loop(self):
        """Vectorized implementation matches per-point loop."""
        from scipy.spatial import cKDTree

        points = _make_sphere(500)
        k = 20
        k_actual = min(k, len(points) - 1)

        # Vectorized (current implementation)
        curvature_vec = compute_curvature(points, k=k)

        # Reference loop implementation
        tree = cKDTree(points)
        _, indices = tree.query(points, k=k_actual + 1)
        curvature_loop = np.zeros(len(points), dtype=np.float32)
        for i in range(len(points)):
            neighbors = points[indices[i]]
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered / len(neighbors)
            eigenvalues = np.linalg.eigvalsh(cov)
            total = eigenvalues.sum()
            if total > 0:
                curvature_loop[i] = eigenvalues[0] / total

        np.testing.assert_allclose(curvature_vec, curvature_loop, atol=1e-6)


class TestExtractPatches:
    def test_basic(self):
        points = _make_cube_cloud(3000, scale=200.0)
        patches = extract_patches(points, patch_size=64, stride=32, min_points=10)
        assert len(patches) > 0
        for idx_arr in patches:
            assert len(idx_arr) >= 10
            assert idx_arr.max() < 3000

    def test_min_points_filter(self):
        points = _make_cube_cloud(3000, scale=200.0)
        patches_low = extract_patches(points, patch_size=64, stride=32, min_points=5)
        patches_high = extract_patches(points, patch_size=64, stride=32, min_points=500)
        assert len(patches_low) >= len(patches_high)

    def test_no_duplicate_indices(self):
        points = _make_cube_cloud(1000, scale=100.0)
        patches = extract_patches(points, patch_size=64, stride=32, min_points=5)
        for idx_arr in patches:
            assert len(idx_arr) == len(np.unique(idx_arr))


@pytest.mark.skipif(not HAS_ME, reason="MinkowskiEngine not installed")
class TestMultiFrameDataset:
    """Tests for MultiFrameDataset with fake data in tmp_path."""

    def _create_fake_data(self, tmp_path, n_points=500, rates=("r2",)):
        """Create minimal manifest + .npy files for testing."""
        rng = np.random.default_rng(0)
        seq = "test_seq"
        frame = 100

        frame_dir = tmp_path / seq / str(frame)
        frame_dir.mkdir(parents=True)

        original = (rng.uniform(0, 200, (n_points, 3))).astype(np.float32)
        np.save(frame_dir / "original.npy", original)

        manifest = []
        for rate in rates:
            decoded = original + rng.normal(0, 2, original.shape).astype(np.float32)
            np.save(frame_dir / f"pcgcv2_{rate}.npy", decoded)
            manifest.append({"sequence": seq, "frame": frame, "rate": rate})

        with open(tmp_path / "manifest.json", "w") as f:
            json.dump(manifest, f)

        return manifest

    def test_inline_loading(self, tmp_path):
        """Dataset computes features inline when no precomputed files exist."""
        self._create_fake_data(tmp_path, n_points=500, rates=["r2"])

        from postprocessing.dataset import MultiFrameDataset

        ds = MultiFrameDataset(
            data_root=str(tmp_path),
            patch_size=128,
            stride=64,
            min_points=5,
        )
        assert len(ds) > 0
        coords, feats, disp, curv, rate_index = ds[0]
        assert coords.ndim == 2 and coords.shape[1] == 3
        assert feats.shape[1] == 4
        assert disp.shape[1] == 3
        assert curv.ndim == 1
        assert isinstance(rate_index, float)

    def test_precomputed_loading(self, tmp_path):
        """Dataset loads precomputed .npy files when available."""
        self._create_fake_data(tmp_path, n_points=500, rates=["r2"])

        frame_dir = tmp_path / "test_seq" / "100"
        decoded = np.load(frame_dir / "pcgcv2_r2.npy")
        original = np.load(frame_dir / "original.npy")

        # Create precomputed files
        from scipy.spatial import cKDTree

        tree = cKDTree(original)
        _, nn_idx = tree.query(decoded)
        displacement = (original[nn_idx] - decoded).astype(np.float32)
        curvature = compute_curvature(decoded, k=30)

        np.save(frame_dir / "displacement_r2.npy", displacement)
        np.save(frame_dir / "curvature_r2.npy", curvature)

        from postprocessing.dataset import MultiFrameDataset

        ds = MultiFrameDataset(
            data_root=str(tmp_path),
            patch_size=128,
            stride=64,
            min_points=5,
        )
        assert len(ds) > 0
        assert len(ds.decoded_clouds) == 1

        # Verify loaded values match what we saved
        np.testing.assert_array_equal(ds.displacements[0], displacement)
        np.testing.assert_array_equal(ds.curvatures[0], curvature)

    def test_len_and_getitem(self, tmp_path):
        """Verify __len__ and __getitem__ return correct shapes."""
        self._create_fake_data(tmp_path, n_points=1000, rates=["r2", "r3"])

        from postprocessing.dataset import MultiFrameDataset

        ds = MultiFrameDataset(
            data_root=str(tmp_path),
            patch_size=256,
            stride=128,
            min_points=5,
        )
        assert len(ds) == len(ds.patches)

        for i in range(min(3, len(ds))):
            coords, feats, disp, curv, rate_index = ds[i]
            n = coords.shape[0]
            assert feats.shape == (n, 4)
            assert disp.shape == (n, 3)
            assert curv.shape == (n,)
            assert isinstance(rate_index, float)
