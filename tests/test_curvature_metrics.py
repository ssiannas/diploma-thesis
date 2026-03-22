"""Unit tests for CurvatureQualityCalculator and D1 PSNR computation.

Tests cover:
  - D1 PSNR formula correctness (symmetric max definition)
  - curvature computation (flat plane vs curved surface)
  - compute_stratified_psnr: tercile split, rev_sq_dists shortcut, direction=reverse
  - Optimization equivalence: rev_sq_dists + dummy rec_curvature give identical results
"""

import numpy as np
from scipy.spatial import cKDTree

from pcml.metrics.curvature import CurvatureQualityCalculator

PEAK = 1023.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def flat_grid(n: int = 20, scale: float = 10.0) -> np.ndarray:
    """Regular grid on z=0 plane: perfectly flat, curvature → 0."""
    xs = np.linspace(0, scale, n)
    ys = np.linspace(0, scale, n)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n * n)])
    return pts.astype(np.float32)


def sphere_surface(n: int = 200, radius: float = 5.0) -> np.ndarray:
    """Roughly uniform points on a sphere surface: high curvature throughout."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.column_stack([x, y, z]).astype(np.float32)


def d1_psnr_reference(
    original: np.ndarray, reconstructed: np.ndarray, peak: float = PEAK
) -> float:
    """Reference D1 PSNR: symmetric max(forward MSE, backward MSE)."""
    ot = cKDTree(original)
    rt = cKDTree(reconstructed)
    fwd_mse = float(np.mean(ot.query(reconstructed)[0] ** 2))
    bwd_mse = float(np.mean(rt.query(original)[0] ** 2))
    mse = max(fwd_mse, bwd_mse)
    return float("inf") if mse == 0 else 10.0 * np.log10(peak**2 / mse)


# ---------------------------------------------------------------------------
# D1 PSNR
# ---------------------------------------------------------------------------


def test_d1_psnr_identical_cloud():
    """Identical clouds → infinite PSNR (zero MSE)."""
    pts = flat_grid()
    assert d1_psnr_reference(pts, pts) == float("inf")


def test_d1_psnr_known_displacement():
    """Shift entire cloud by known amount → verify PSNR formula."""
    pts = flat_grid(10)
    shift = 2.0
    shifted = pts + np.array([shift, 0, 0], dtype=np.float32)
    # Nearest-neighbour in a regular grid: forward and backward MSE both = shift²
    # (grid spacing = 10/9 ≈ 1.11, shift=2 means each point maps to next-nearest)
    # We just check the formula holds given actual MSE
    ot = cKDTree(pts)
    rt = cKDTree(shifted)
    fwd_mse = float(np.mean(ot.query(shifted)[0] ** 2))
    bwd_mse = float(np.mean(rt.query(pts)[0] ** 2))
    expected = 10.0 * np.log10(PEAK**2 / max(fwd_mse, bwd_mse))
    assert abs(d1_psnr_reference(pts, shifted) - expected) < 1e-6


def test_d1_psnr_symmetric():
    """D1 is symmetric: max(fwd, bwd) doesn't depend on argument order
    when both MSEs are the same (grid → grid with uniform shift)."""
    pts = flat_grid(10)
    shifted = pts.copy()
    shifted[:, 2] += 0.5
    psnr_ab = d1_psnr_reference(pts, shifted)
    psnr_ba = d1_psnr_reference(shifted, pts)
    assert abs(psnr_ab - psnr_ba) < 1e-4


def test_d1_psnr_decreases_with_distortion():
    """More distortion → lower PSNR."""
    pts = flat_grid(15)
    rng = np.random.default_rng(1)
    small = pts + rng.uniform(-0.1, 0.1, pts.shape).astype(np.float32)
    large = pts + rng.uniform(-1.0, 1.0, pts.shape).astype(np.float32)
    assert d1_psnr_reference(pts, small) > d1_psnr_reference(pts, large)


# ---------------------------------------------------------------------------
# Curvature computation
# ---------------------------------------------------------------------------


def test_curvature_flat_plane_is_low():
    """Flat grid → curvature near 0 for interior points (k=8 to keep fast)."""
    pts = flat_grid(20)  # 400 pts
    curv = CurvatureQualityCalculator.compute_curvature(pts, k=8)
    assert curv.shape == (len(pts),)
    assert curv.min() >= 0.0
    assert curv.max() <= 1.0
    # Interior points of a flat grid have λ_min ≈ 0 → curvature ≈ 0
    # Allow some boundary effects; median should be very low
    assert float(np.median(curv)) < 0.05


def test_curvature_sphere_is_high():
    """Sphere surface → curvature values spread across [0, 1], median > flat."""
    sphere = sphere_surface(300, radius=5.0)
    flat = flat_grid(15)  # 225 pts
    curv_sphere = CurvatureQualityCalculator.compute_curvature(sphere, k=10)
    curv_flat = CurvatureQualityCalculator.compute_curvature(flat, k=8)
    assert float(np.median(curv_sphere)) > float(np.median(curv_flat))


def test_curvature_range():
    """Curvature output is always in [0, 1]."""
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((500, 3)).astype(np.float32)
    curv = CurvatureQualityCalculator.compute_curvature(pts, k=10)
    assert np.all(curv >= 0.0)
    assert np.all(curv <= 1.0)


# ---------------------------------------------------------------------------
# compute_stratified_psnr: core correctness
# ---------------------------------------------------------------------------


def mixed_cloud(n_flat: int = 150, n_curved: int = 150) -> np.ndarray:
    """Mixed cloud with actual curvature variation: flat region + sphere region.
    Needed because tercile stratification on a pure flat plane collapses to 0 edge points.
    """
    flat = flat_grid(n=int(n_flat**0.5) + 1)[:n_flat]
    curved = sphere_surface(n=n_curved, radius=20.0)
    return np.vstack([flat, curved]).astype(np.float32)


def test_stratified_psnr_identical_cloud():
    """Identical reconstruction with curvature-varying cloud → edge and flat PSNR are inf."""
    pts = mixed_cloud(150, 150)
    curv = CurvatureQualityCalculator.compute_curvature(pts, k=10)
    m = CurvatureQualityCalculator.compute_stratified_psnr(
        pts, pts, curvature=curv, k=10, direction="reverse"
    )
    assert m.edge_psnr == float("inf")
    assert m.flat_psnr == float("inf")


def test_stratified_psnr_flat_plane_edge_is_nan():
    """A purely flat plane has curvature ≈ 0 everywhere → terciles collapse,
    edge/medium strata are empty → NaN PSNR. This is expected metric behavior."""
    pts = flat_grid(20)
    curv = CurvatureQualityCalculator.compute_curvature(pts, k=8)
    m = CurvatureQualityCalculator.compute_stratified_psnr(
        pts, pts, curvature=curv, k=8, direction="reverse"
    )
    # flat PSNR is valid (all points land there), edge is NaN (no edge points)
    assert m.flat_psnr == float("inf")
    assert np.isnan(m.edge_psnr)


def test_stratified_psnr_point_counts_sum_to_n():
    """Total point count across all strata must equal the original cloud size.

    Note: strata are NOT necessarily equal-sized — the split uses percentile
    thresholds so a bimodal curvature distribution can produce unequal groups.
    """
    pts = mixed_cloud(150, 150)
    curv = CurvatureQualityCalculator.compute_curvature(pts, k=10)
    m = CurvatureQualityCalculator.compute_stratified_psnr(
        pts, pts, curvature=curv, k=10, direction="reverse", n_strata=3
    )
    assert sum(m.point_counts.values()) == len(pts)
    assert set(m.point_counts.keys()) == {"flat", "medium", "edge"}


def test_stratified_psnr_edge_worse_than_flat_for_oversmoothing():
    """Construct a cloud where edge points are displaced more than flat ones.
    Edge PSNR should be lower than flat PSNR."""
    rng = np.random.default_rng(7)

    # Flat region: 300 points on z=0 grid
    flat = flat_grid(n=18)  # 324 pts
    # Curved region: points on a sphere (high curvature)
    curved = sphere_surface(n=200, radius=20.0)

    original = np.vstack([flat, curved]).astype(np.float32)
    curv = CurvatureQualityCalculator.compute_curvature(original, k=10)

    # "Oversmoothed" reconstruction: flat region preserved, curved displaced heavily
    rec = original.copy()
    n_flat = len(flat)
    rec[n_flat:] += rng.uniform(-3.0, 3.0, (len(curved), 3)).astype(np.float32)

    m = CurvatureQualityCalculator.compute_stratified_psnr(
        original, rec, curvature=curv, k=10, direction="reverse"
    )
    assert (
        m.edge_psnr < m.flat_psnr
    ), f"Expected edge_psnr ({m.edge_psnr:.2f}) < flat_psnr ({m.flat_psnr:.2f})"
    assert m.degradation > 0  # flat - edge > 0


# ---------------------------------------------------------------------------
# Optimization equivalence: rev_sq_dists + dummy rec_curvature
# ---------------------------------------------------------------------------


def test_rev_sq_dists_shortcut_matches_full_call():
    """Passing rev_sq_dists should give identical edge/flat PSNR as full call."""
    pts = mixed_cloud(150, 150)
    curv = CurvatureQualityCalculator.compute_curvature(pts, k=10)
    rng = np.random.default_rng(3)
    rec = pts + rng.uniform(-0.5, 0.5, pts.shape).astype(np.float32)

    m_full = CurvatureQualityCalculator.compute_stratified_psnr(
        pts,
        rec,
        curvature=curv,
        k=10,
        direction="reverse",
        rec_curvature=curv,
    )

    rec_tree = cKDTree(rec)
    bwd_sq = rec_tree.query(pts)[0] ** 2
    m_fast = CurvatureQualityCalculator.compute_stratified_psnr(
        pts,
        rec,
        curvature=curv,
        k=10,
        direction="reverse",
        rev_sq_dists=bwd_sq,
        rec_curvature=curv,
    )

    assert abs(m_fast.edge_psnr - m_full.edge_psnr) < 1e-6
    assert abs(m_fast.flat_psnr - m_full.flat_psnr) < 1e-6


def test_dummy_rec_curvature_does_not_affect_stratification():
    """Using original curvature as dummy rec_curvature does not change edge/flat PSNR.

    KL divergence is affected, but we don't use it. The stratification itself
    (edge/flat split) only depends on the original cloud curvature.
    """
    pts = mixed_cloud(150, 150)
    curv = CurvatureQualityCalculator.compute_curvature(pts, k=10)
    rng = np.random.default_rng(5)
    rec = pts + rng.uniform(-0.3, 0.3, pts.shape).astype(np.float32)

    rec_curv = CurvatureQualityCalculator.compute_curvature(rec, k=10)
    m_real = CurvatureQualityCalculator.compute_stratified_psnr(
        pts, rec, curvature=curv, k=10, direction="reverse", rec_curvature=rec_curv
    )

    m_dummy = CurvatureQualityCalculator.compute_stratified_psnr(
        pts, rec, curvature=curv, k=10, direction="reverse", rec_curvature=curv
    )

    # Stratification results must be identical; KL will differ (don't assert it)
    assert abs(m_real.edge_psnr - m_dummy.edge_psnr) < 1e-6
    assert abs(m_real.flat_psnr - m_dummy.flat_psnr) < 1e-6


def test_multicore_query_matches_single_core():
    """cKDTree.query(workers=-1) gives same distances as workers=1."""
    pts = flat_grid(20)
    rec = pts + 0.1
    tree = cKDTree(pts)
    d1 = tree.query(rec, workers=1)[0]
    d_multi = tree.query(rec, workers=-1)[0]
    np.testing.assert_allclose(d1, d_multi, rtol=0, atol=1e-6)
