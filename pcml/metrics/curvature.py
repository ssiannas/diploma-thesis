"""Curvature-stratified quality metrics for oversmoothing detection.

Learned point cloud codecs (e.g., PCGCv2) tend to produce oversmoothed
reconstructions at high-curvature regions (edges, corners). Standard PSNR
averages over all points and masks this effect.

This module provides:
- PCA-based curvature estimation via eigenvalue ratio
- Curvature histogram comparison (KL divergence)
- Stratified D1-PSNR: separate PSNR for flat / medium / edge regions

Key metric: `degradation = flat_psnr - edge_psnr`
    A large positive value indicates oversmoothing at edges.

References:
    Cignoni, P., et al. (1998). "Metro: Measuring Error on Simplified
        Surfaces." Computer Graphics Forum.
    Wang, J., et al. (2021). "Lossy Point Cloud Geometry Compression via
        End-to-End Learning." IEEE TCSVT. (PCGCv2)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import rel_entr

logger = logging.getLogger(__name__)


@dataclass
class CurvatureMetrics:
    """Curvature distribution statistics for a point cloud."""

    mean_curvature: float
    std_curvature: float
    histogram: np.ndarray  # 50-bin normalized histogram
    bin_edges: np.ndarray


@dataclass
class StratifiedQualityMetrics:
    """Quality metrics stratified by curvature region.

    Points are split into terciles by curvature value:
    - flat: bottom tercile (low curvature, planar regions)
    - medium: middle tercile
    - edge: top tercile (high curvature, edges/corners)

    The key oversmoothing indicator is `degradation = flat_psnr - edge_psnr`.
    """

    flat_psnr: float  # PSNR in low-curvature regions (bottom tercile)
    medium_psnr: float  # PSNR in mid-curvature regions
    edge_psnr: float  # PSNR in high-curvature regions (top tercile)
    degradation: float  # flat_psnr - edge_psnr (THE key metric)
    curvature_kl: float  # KL divergence of curvature distributions
    point_counts: dict = field(default_factory=dict)  # {flat: N, medium: N, edge: N}


class CurvatureQualityCalculator:
    """Compute curvature-stratified quality metrics for point clouds.

    Uses PCA-based curvature estimation (eigenvalue ratio of local
    covariance) and splits points into terciles to compute per-region PSNR.
    """

    @staticmethod
    def compute_curvature(points: np.ndarray, k: int = 30) -> np.ndarray:
        """Estimate surface curvature at each point via PCA eigenvalue ratio.

        For each point, the local neighborhood covariance is computed and
        curvature is defined as: lambda_min / (lambda_0 + lambda_1 + lambda_2),
        where lambda_min is the smallest eigenvalue. High values indicate
        high curvature (edges/corners); low values indicate flat regions.

        Args:
            points: (N, 3) point coordinates.
            k: Number of nearest neighbors for local PCA.

        Returns:
            (N,) curvature values in [0, 1].
        """
        if points.shape[1] != 3:
            raise ValueError(f"Points must have shape (N, 3), got {points.shape}")

        n = len(points)
        k = min(k, n - 1)  # Can't have more neighbors than points

        tree = cKDTree(points)
        _, indices = tree.query(points, k=k + 1)  # +1 because query includes self

        curvatures = np.zeros(n, dtype=np.float64)

        for i in range(n):
            neighbors = points[indices[i]]  # (k+1, 3)
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered / len(neighbors)

            eigenvalues = np.linalg.eigvalsh(cov)  # sorted ascending
            total = eigenvalues.sum()

            if total > 0:
                curvatures[i] = eigenvalues[0] / total
            else:
                curvatures[i] = 0.0

        return curvatures

    @staticmethod
    def compute_curvature_histogram(
        curvatures: np.ndarray, n_bins: int = 50
    ) -> CurvatureMetrics:
        """Compute curvature distribution statistics.

        Args:
            curvatures: (N,) curvature values.
            n_bins: Number of histogram bins.

        Returns:
            CurvatureMetrics with histogram and statistics.
        """
        histogram, bin_edges = np.histogram(curvatures, bins=n_bins, density=True)

        return CurvatureMetrics(
            mean_curvature=float(np.mean(curvatures)),
            std_curvature=float(np.std(curvatures)),
            histogram=histogram,
            bin_edges=bin_edges,
        )

    @staticmethod
    def compute_curvature_kl(
        orig_curvature: np.ndarray,
        dec_curvature: np.ndarray,
        n_bins: int = 50,
    ) -> float:
        """Compute KL divergence between curvature distributions.

        Measures how much the decoded curvature distribution diverges from
        the original. A large value indicates the codec changed the surface
        structure (e.g., smoothed edges).

        Args:
            orig_curvature: (N,) original curvature values.
            dec_curvature: (M,) decoded curvature values.
            n_bins: Number of histogram bins.

        Returns:
            KL divergence (non-negative, 0 = identical distributions).
        """
        # Use shared bin edges from combined range
        all_curv = np.concatenate([orig_curvature, dec_curvature])
        bin_edges = np.linspace(all_curv.min(), all_curv.max(), n_bins + 1)

        orig_hist, _ = np.histogram(orig_curvature, bins=bin_edges, density=True)
        dec_hist, _ = np.histogram(dec_curvature, bins=bin_edges, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        orig_hist = orig_hist + eps
        dec_hist = dec_hist + eps

        # Normalize to proper probability distributions
        orig_hist = orig_hist / orig_hist.sum()
        dec_hist = dec_hist / dec_hist.sum()

        kl = float(np.sum(rel_entr(orig_hist, dec_hist)))
        return kl

    @staticmethod
    def compute_stratified_psnr(
        original: np.ndarray,
        reconstructed: np.ndarray,
        curvature: np.ndarray,
        peak: float = 1023.0,
    ) -> StratifiedQualityMetrics:
        """Compute D1-PSNR separately for flat, medium, and edge regions.

        Points in the original cloud are split into terciles by curvature.
        For each region, the nearest-neighbor MSE from reconstructed to
        original is computed, then converted to PSNR.

        Args:
            original: (N, 3) original point coordinates.
            reconstructed: (M, 3) reconstructed point coordinates.
            curvature: (N,) curvature values for original points.
            peak: Peak value for PSNR (resolution - 1 for voxelized clouds).

        Returns:
            StratifiedQualityMetrics with per-region PSNR and degradation.
        """
        if original.shape[1] != 3 or reconstructed.shape[1] != 3:
            raise ValueError("Point clouds must have shape (N, 3)")
        if len(curvature) != len(original):
            raise ValueError(
                f"Curvature length ({len(curvature)}) must match "
                f"original points ({len(original)})"
            )

        # Split into terciles by curvature
        t1 = np.percentile(curvature, 33.33)
        t2 = np.percentile(curvature, 66.67)

        flat_mask = curvature <= t1
        medium_mask = (curvature > t1) & (curvature <= t2)
        edge_mask = curvature > t2

        # Build KD-tree on original for nearest-neighbor queries
        orig_tree = cKDTree(original)

        # For each reconstructed point, find nearest original and its region
        dists, nn_idx = orig_tree.query(reconstructed)
        sq_dists = dists**2

        # Assign each reconstructed point to the region of its nearest original
        rec_flat = sq_dists[flat_mask[nn_idx]]
        rec_medium = sq_dists[medium_mask[nn_idx]]
        rec_edge = sq_dists[edge_mask[nn_idx]]

        def _mse_to_psnr(sq_d: np.ndarray, peak_val: float) -> float:
            if len(sq_d) == 0:
                return float("nan")
            mse = float(np.mean(sq_d))
            if mse == 0:
                return float("inf")
            return 10.0 * np.log10(peak_val**2 / mse)

        flat_psnr = _mse_to_psnr(rec_flat, peak)
        medium_psnr = _mse_to_psnr(rec_medium, peak)
        edge_psnr = _mse_to_psnr(rec_edge, peak)

        degradation = flat_psnr - edge_psnr

        # Compute curvature KL divergence
        rec_curvature_at_nn = curvature[nn_idx]
        kl = CurvatureQualityCalculator.compute_curvature_kl(
            curvature, rec_curvature_at_nn
        )

        return StratifiedQualityMetrics(
            flat_psnr=flat_psnr,
            medium_psnr=medium_psnr,
            edge_psnr=edge_psnr,
            degradation=degradation,
            curvature_kl=kl,
            point_counts={
                "flat": int(rec_flat.size),
                "medium": int(rec_medium.size),
                "edge": int(rec_edge.size),
            },
        )

    @staticmethod
    def calculate_all(
        original: np.ndarray,
        reconstructed: np.ndarray,
        peak: float = 1023.0,
        k: int = 30,
    ) -> StratifiedQualityMetrics:
        """Compute all curvature-stratified quality metrics.

        Convenience method that runs curvature estimation and stratified
        PSNR computation in one call.

        Args:
            original: (N, 3) original point coordinates.
            reconstructed: (M, 3) reconstructed point coordinates.
            peak: Peak value for PSNR (default 1023 for 10-bit voxelized).
            k: Number of neighbors for curvature estimation.

        Returns:
            StratifiedQualityMetrics with all oversmoothing indicators.
        """
        logger.info(
            "Computing curvature for %d original points (k=%d)...", len(original), k
        )
        curvature = CurvatureQualityCalculator.compute_curvature(original, k=k)

        logger.info("Computing stratified PSNR (peak=%.1f)...", peak)
        return CurvatureQualityCalculator.compute_stratified_psnr(
            original, reconstructed, curvature, peak=peak
        )
