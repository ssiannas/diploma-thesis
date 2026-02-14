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
from scipy.stats import spearmanr

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
    direction: str = "forward"  # "forward", "reverse", or "forward_self"


@dataclass
class CorrelationResult:
    """Spearman correlation between per-point curvature and D1 error."""

    rho: float  # Spearman rank correlation coefficient
    p_value: float  # Two-sided p-value
    n_points: int  # Number of points used


@dataclass
class MultiplicityResult:
    """NN multiplicity diagnostic for reverse-direction metrics.

    When many original points map to the same reconstructed NN, it indicates
    the codec redistributed points away from that region.
    """

    per_stratum_mean: dict  # {stratum_name: mean_multiplicity}
    per_stratum_max: dict  # {stratum_name: max_multiplicity}
    overall_mean: float
    overall_max: int
    histogram: (
        np.ndarray
    )  # counts of multiplicity values (how many rec pts have mult=1, 2, ...)
    histogram_bin_edges: np.ndarray


@dataclass
class PSNRvsCurvatureResult:
    """Continuous PSNR-vs-curvature curve (fine-binned)."""

    bin_centers: np.ndarray  # (n_bins,) curvature bin centers
    psnrs: np.ndarray  # (n_bins,) PSNR per bin
    counts: np.ndarray  # (n_bins,) number of points per bin
    direction: str  # "forward_self" or "reverse"


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
    def compute_per_point_error(
        original: np.ndarray,
        reconstructed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-point nearest-neighbor distances.

        Args:
            original: (N, 3) original point coordinates.
            reconstructed: (M, 3) reconstructed point coordinates.

        Returns:
            (distances, nn_indices) where distances is (M,) NN distances
            from each reconstructed point to the original, and nn_indices
            is (M,) indices into the original cloud.
        """
        if original.shape[1] != 3 or reconstructed.shape[1] != 3:
            raise ValueError("Point clouds must have shape (N, 3)")

        orig_tree = cKDTree(original)
        dists, nn_idx = orig_tree.query(reconstructed)
        return dists, nn_idx

    @staticmethod
    def compute_assignment_accuracy(
        original: np.ndarray,
        reconstructed: np.ndarray,
        curvature: np.ndarray,
        k: int = 30,
        rec_curvature: Optional[np.ndarray] = None,
        fwd_nn_idx: Optional[np.ndarray] = None,
    ) -> float:
        """Check how often NN-based curvature assignment agrees with direct computation.

        For each reconstructed point, we compare:
        - "assigned" curvature tercile: inherited from the nearest original point
        - "direct" curvature tercile: computed from the reconstructed cloud itself

        When displacement is large (e.g. at low rate points), the NN assignment
        maps reconstructed points to the wrong curvature group, making the
        forward-direction stratified PSNR unreliable.

        Args:
            original: (N, 3) original point coordinates.
            reconstructed: (M, 3) reconstructed point coordinates.
            curvature: (N,) curvature values for original points.
            k: Number of neighbors for PCA curvature on the reconstructed cloud.
            rec_curvature: Pre-computed curvature on reconstructed cloud.
            fwd_nn_idx: Pre-computed forward NN indices (rec->orig).

        Returns:
            Agreement rate in [0, 1]. Values below ~0.8 indicate the forward-
            direction stratified PSNR is unreliable at this rate.
        """
        # Assigned tercile: via NN from reconstructed → original
        if fwd_nn_idx is not None:
            nn_idx = fwd_nn_idx
        else:
            _, nn_idx = CurvatureQualityCalculator.compute_per_point_error(
                original, reconstructed
            )
        assigned_curvature = curvature[nn_idx]

        # Direct tercile: compute curvature on the reconstructed cloud
        if rec_curvature is not None:
            direct_curvature = rec_curvature
        else:
            direct_curvature = CurvatureQualityCalculator.compute_curvature(
                reconstructed, k=k
            )

        # Bin both into terciles
        def _to_tercile(values: np.ndarray) -> np.ndarray:
            t1 = np.percentile(values, 33.33)
            t2 = np.percentile(values, 66.67)
            tercile = np.zeros(len(values), dtype=int)
            tercile[values > t1] = 1
            tercile[values > t2] = 2
            return tercile

        assigned_tercile = _to_tercile(assigned_curvature)
        direct_tercile = _to_tercile(direct_curvature)

        agreement = float(np.mean(assigned_tercile == direct_tercile))
        return agreement

    @staticmethod
    def compute_curvature_error_correlation(
        original: np.ndarray,
        reconstructed: np.ndarray,
        curvature: np.ndarray,
    ) -> CorrelationResult:
        """Compute Spearman rank correlation between curvature and D1 error.

        For each original point, compute the nearest-neighbor distance to
        the reconstructed cloud (reverse-direction D1 error), then correlate
        with the point's curvature.

        A positive rho means higher curvature correlates with higher error,
        i.e. oversmoothing. This avoids the arbitrary tercile split entirely.

        Args:
            original: (N, 3) original point coordinates.
            reconstructed: (M, 3) reconstructed point coordinates.
            curvature: (N,) curvature values for original points.

        Returns:
            CorrelationResult with Spearman rho, p-value, and point count.
        """
        # Reverse direction: for each original point, NN distance to reconstructed
        rec_tree = cKDTree(reconstructed)
        dists, _ = rec_tree.query(original)

        rho, p_value = spearmanr(curvature, dists)

        return CorrelationResult(
            rho=float(rho),
            p_value=float(p_value),
            n_points=len(original),
        )

    @staticmethod
    def _stratify_and_psnr(
        sq_dists: np.ndarray,
        strat_curvature: np.ndarray,
        peak: float,
        n_strata: int,
    ) -> tuple[list[float], list[str], dict]:
        """Core stratification: split sq_dists by curvature quantiles.

        Returns (strata_psnrs, stratum_names, point_counts).
        """

        def _mse_to_psnr(sq_d: np.ndarray, peak_val: float) -> float:
            if len(sq_d) == 0:
                return float("nan")
            mse = float(np.mean(sq_d))
            if mse == 0:
                return float("inf")
            return 10.0 * np.log10(peak_val**2 / mse)

        percentiles = np.linspace(0, 100, n_strata + 1)[1:-1]
        thresholds = [np.percentile(strat_curvature, p) for p in percentiles]

        masks = []
        for i in range(n_strata):
            if i == 0:
                masks.append(strat_curvature <= thresholds[0])
            elif i == n_strata - 1:
                masks.append(strat_curvature > thresholds[-1])
            else:
                masks.append(
                    (strat_curvature > thresholds[i - 1])
                    & (strat_curvature <= thresholds[i])
                )

        strata_sq_dists = [sq_dists[m] for m in masks]
        strata_psnrs = [_mse_to_psnr(sd, peak) for sd in strata_sq_dists]

        if n_strata == 3:
            stratum_names = ["flat", "medium", "edge"]
        elif n_strata == 5:
            stratum_names = ["very_flat", "flat", "medium", "edge", "sharp_edge"]
        else:
            stratum_names = [f"q{i+1}" for i in range(n_strata)]

        point_counts = {
            name: int(sd.size) for name, sd in zip(stratum_names, strata_sq_dists)
        }
        return strata_psnrs, stratum_names, point_counts

    @staticmethod
    def compute_stratified_psnr(
        original: np.ndarray,
        reconstructed: np.ndarray,
        curvature: np.ndarray,
        peak: float = 1023.0,
        n_strata: int = 3,
        direction: str = "forward",
        k: int = 30,
        rec_curvature: Optional[np.ndarray] = None,
        rev_sq_dists: Optional[np.ndarray] = None,
        fwd_dists: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> StratifiedQualityMetrics:
        """Compute D1-PSNR separately for curvature-based strata.

        Three directions are supported:
        - **forward** (rec->orig): for each reconstructed point, NN distance
          to original. The reconstructed point is assigned to the curvature
          group of its nearest original neighbor. This can be unreliable when
          displacement is large (NN-assignment scrambling).
        - **reverse** (orig->rec): for each original point, NN distance to
          reconstructed. The curvature group is known directly (no assignment
          needed). This fixes the r1 anomaly.
        - **forward_self**: for each reconstructed point, NN distance to
          original. Stratified by curvature computed on the reconstructed
          cloud itself. No NN correspondence needed for stratification.
          This is what a post-processing module would use.

        KL divergence always compares independently-computed curvature
        distributions (original vs reconstructed), not NN-inherited values.

        Args:
            original: (N, 3) original point coordinates.
            reconstructed: (M, 3) reconstructed point coordinates.
            curvature: (N,) curvature values for original points.
            peak: Peak value for PSNR (resolution - 1 for voxelized clouds).
            n_strata: Number of curvature strata (default 3 = terciles).
            direction: "forward", "reverse", or "forward_self".
            k: Number of neighbors for curvature estimation on reconstructed
               cloud (used by forward_self and for KL computation).
            rec_curvature: Pre-computed curvature on reconstructed cloud.
                Avoids redundant recomputation when calling multiple directions.
            rev_sq_dists: Pre-computed reverse squared distances (orig->rec).
                Only used when direction="reverse".
            fwd_dists: Pre-computed forward (dists, nn_idx) tuple (rec->orig).
                Only used when direction="forward" or "forward_self".

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
        if n_strata < 2:
            raise ValueError(f"n_strata must be >= 2, got {n_strata}")
        if direction not in ("forward", "reverse", "forward_self"):
            raise ValueError(
                f"direction must be 'forward', 'reverse', or 'forward_self', "
                f"got '{direction}'"
            )

        # Compute rec_curvature if not provided (needed for KL in all directions)
        if rec_curvature is None:
            rec_curvature = CurvatureQualityCalculator.compute_curvature(
                reconstructed, k=k
            )

        if direction == "forward_self":
            strat_curvature = rec_curvature

            if fwd_dists is not None:
                dists, _ = fwd_dists
            else:
                dists, _ = CurvatureQualityCalculator.compute_per_point_error(
                    original, reconstructed
                )
            sq_dists = dists**2
        elif direction == "forward":
            if fwd_dists is not None:
                dists, nn_idx = fwd_dists
            else:
                dists, nn_idx = CurvatureQualityCalculator.compute_per_point_error(
                    original, reconstructed
                )
            sq_dists = dists**2
            strat_curvature = curvature[nn_idx]
        else:
            # Reverse
            if rev_sq_dists is not None:
                sq_dists = rev_sq_dists
            else:
                rec_tree = cKDTree(reconstructed)
                dists, _ = rec_tree.query(original)
                sq_dists = dists**2
            strat_curvature = curvature

        strata_psnrs, stratum_names, point_counts = (
            CurvatureQualityCalculator._stratify_and_psnr(
                sq_dists, strat_curvature, peak, n_strata
            )
        )

        flat_psnr = strata_psnrs[0]
        medium_psnr = strata_psnrs[n_strata // 2]
        edge_psnr = strata_psnrs[-1]
        degradation = flat_psnr - edge_psnr

        kl = CurvatureQualityCalculator.compute_curvature_kl(curvature, rec_curvature)

        return StratifiedQualityMetrics(
            flat_psnr=flat_psnr,
            medium_psnr=medium_psnr,
            edge_psnr=edge_psnr,
            degradation=degradation,
            curvature_kl=kl,
            point_counts=point_counts,
            direction=direction,
        )

    @staticmethod
    def compute_nn_multiplicity(
        original: np.ndarray,
        reconstructed: np.ndarray,
        curvature: np.ndarray,
        n_strata: int = 3,
    ) -> MultiplicityResult:
        """Count how many original points share the same NN in the reconstructed cloud.

        In the reverse direction, each original point finds its NN in the
        reconstructed cloud. If the codec redistributed points (e.g., moved
        them away from edges), many original edge points will map to the same
        few reconstructed points, inflating multiplicity for edge strata.

        Args:
            original: (N, 3) original point coordinates.
            reconstructed: (M, 3) reconstructed point coordinates.
            curvature: (N,) curvature values for original points.
            n_strata: Number of curvature strata.

        Returns:
            MultiplicityResult with per-stratum and overall statistics.
        """
        rec_tree = cKDTree(reconstructed)
        _, nn_idx = rec_tree.query(original)

        # Count how many original points map to each reconstructed point
        multiplicity = np.bincount(nn_idx, minlength=len(reconstructed))

        # Build strata masks on original curvature
        percentiles = np.linspace(0, 100, n_strata + 1)[1:-1]
        thresholds = [np.percentile(curvature, p) for p in percentiles]

        masks = []
        for i in range(n_strata):
            if i == 0:
                masks.append(curvature <= thresholds[0])
            elif i == n_strata - 1:
                masks.append(curvature > thresholds[-1])
            else:
                masks.append(
                    (curvature > thresholds[i - 1]) & (curvature <= thresholds[i])
                )

        if n_strata == 3:
            stratum_names = ["flat", "medium", "edge"]
        elif n_strata == 5:
            stratum_names = ["very_flat", "flat", "medium", "edge", "sharp_edge"]
        else:
            stratum_names = [f"q{i+1}" for i in range(n_strata)]

        # For each stratum: get the NN indices, look up their multiplicity
        per_stratum_mean = {}
        per_stratum_max = {}
        for name, mask in zip(stratum_names, masks):
            stratum_nn_idx = nn_idx[mask]
            stratum_mult = multiplicity[stratum_nn_idx]
            per_stratum_mean[name] = (
                float(np.mean(stratum_mult)) if len(stratum_mult) > 0 else 0.0
            )
            per_stratum_max[name] = (
                int(np.max(stratum_mult)) if len(stratum_mult) > 0 else 0
            )

        # Overall multiplicity histogram (only for rec points that are actually used)
        used_mult = multiplicity[multiplicity > 0]
        max_mult = int(used_mult.max()) if len(used_mult) > 0 else 1
        hist, bin_edges = np.histogram(
            used_mult, bins=np.arange(0.5, max_mult + 1.5, 1.0)
        )

        return MultiplicityResult(
            per_stratum_mean=per_stratum_mean,
            per_stratum_max=per_stratum_max,
            overall_mean=float(np.mean(used_mult)) if len(used_mult) > 0 else 0.0,
            overall_max=max_mult,
            histogram=hist,
            histogram_bin_edges=bin_edges,
        )

    @staticmethod
    def compute_psnr_vs_curvature_curve(
        original: np.ndarray,
        reconstructed: np.ndarray,
        curvature: np.ndarray,
        peak: float = 1023.0,
        n_bins: int = 20,
        direction: str = "reverse",
        k: int = 30,
        rec_curvature: Optional[np.ndarray] = None,
        rev_sq_dists: Optional[np.ndarray] = None,
        fwd_dists: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> PSNRvsCurvatureResult:
        """Compute PSNR as a continuous function of curvature.

        Bins points into fine curvature bins (vigintiles by default) and
        computes PSNR per bin. If oversmoothing is real, PSNR should
        decrease monotonically with curvature.

        Args:
            original: (N, 3) original point coordinates.
            reconstructed: (M, 3) reconstructed point coordinates.
            curvature: (N,) curvature values for original points.
            peak: Peak value for PSNR.
            n_bins: Number of curvature bins (default 20 = vigintiles).
            direction: "reverse" (use original curvature) or "forward_self"
                      (compute and use reconstructed curvature).
            k: Number of neighbors for curvature on reconstructed cloud.
            rec_curvature: Pre-computed curvature on reconstructed cloud.
            rev_sq_dists: Pre-computed reverse squared distances.
            fwd_dists: Pre-computed forward (dists, nn_idx) tuple.

        Returns:
            PSNRvsCurvatureResult with bin centers, PSNRs, and counts.
        """
        if direction == "forward_self":
            if rec_curvature is None:
                rec_curvature = CurvatureQualityCalculator.compute_curvature(
                    reconstructed, k=k
                )
            strat_curvature = rec_curvature
            if fwd_dists is not None:
                dists, _ = fwd_dists
            else:
                dists, _ = CurvatureQualityCalculator.compute_per_point_error(
                    original, reconstructed
                )
            sq_dists = dists**2
        elif direction == "reverse":
            strat_curvature = curvature
            if rev_sq_dists is not None:
                sq_dists = rev_sq_dists
            else:
                rec_tree = cKDTree(reconstructed)
                dists, _ = rec_tree.query(original)
                sq_dists = dists**2
        else:
            raise ValueError(
                f"direction must be 'reverse' or 'forward_self', got '{direction}'"
            )

        # Use quantile-based bins to ensure roughly equal counts
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.array([np.percentile(strat_curvature, p) for p in percentiles])
        # Ensure unique edges (can happen when many points share same curvature)
        bin_edges = np.unique(bin_edges)
        actual_n_bins = len(bin_edges) - 1

        bin_centers = np.zeros(actual_n_bins)
        psnrs = np.zeros(actual_n_bins)
        counts = np.zeros(actual_n_bins, dtype=int)

        for i in range(actual_n_bins):
            if i == actual_n_bins - 1:
                mask = (strat_curvature >= bin_edges[i]) & (
                    strat_curvature <= bin_edges[i + 1]
                )
            else:
                mask = (strat_curvature >= bin_edges[i]) & (
                    strat_curvature < bin_edges[i + 1]
                )

            bin_sq_dists = sq_dists[mask]
            counts[i] = len(bin_sq_dists)
            bin_centers[i] = (bin_edges[i] + bin_edges[i + 1]) / 2.0

            if len(bin_sq_dists) == 0:
                psnrs[i] = float("nan")
            else:
                mse = float(np.mean(bin_sq_dists))
                if mse == 0:
                    psnrs[i] = float("inf")
                else:
                    psnrs[i] = 10.0 * np.log10(peak**2 / mse)

        return PSNRvsCurvatureResult(
            bin_centers=bin_centers,
            psnrs=psnrs,
            counts=counts,
            direction=direction,
        )

    @staticmethod
    def compute_random_stratified_psnr(
        original: np.ndarray,
        reconstructed: np.ndarray,
        curvature: np.ndarray,
        peak: float = 1023.0,
        n_strata: int = 3,
        n_iterations: int = 100,
        direction: str = "reverse",
        seed: int = 42,
        rev_sq_dists: Optional[np.ndarray] = None,
    ) -> dict:
        """Sanity check: stratified PSNR with random group assignments.

        Repeats the stratified PSNR computation with randomly shuffled
        curvature values. If the metric is valid, random stratification
        should yield ~0 degradation (all groups have similar error).

        Precomputes distances once and only re-stratifies in each iteration,
        avoiding redundant KD-tree builds and curvature recomputation.

        Args:
            original: (N, 3) original point coordinates.
            reconstructed: (M, 3) reconstructed point coordinates.
            curvature: (N,) curvature values for original points.
            peak: Peak value for PSNR.
            n_strata: Number of strata.
            n_iterations: Number of random trials.
            direction: Direction for stratified PSNR.
            seed: Random seed for reproducibility.
            rev_sq_dists: Pre-computed reverse squared distances (orig->rec).

        Returns:
            Dict with 'mean_degradation', 'std_degradation', 'n_iterations'.
        """
        # Precompute distances once
        if direction == "reverse":
            if rev_sq_dists is None:
                rec_tree = cKDTree(reconstructed)
                dists, _ = rec_tree.query(original)
                sq_dists = dists**2
            else:
                sq_dists = rev_sq_dists
        else:
            dists, _ = CurvatureQualityCalculator.compute_per_point_error(
                original, reconstructed
            )
            sq_dists = dists**2

        # Shuffle and re-stratify (no curvature/KL/tree recomputation)
        rng = np.random.RandomState(seed)
        degradations = np.empty(n_iterations)

        for i in range(n_iterations):
            shuffled = rng.permutation(curvature)
            strata_psnrs, _, _ = CurvatureQualityCalculator._stratify_and_psnr(
                sq_dists, shuffled, peak, n_strata
            )
            degradations[i] = strata_psnrs[0] - strata_psnrs[-1]

        return {
            "mean_degradation": float(np.mean(degradations)),
            "std_degradation": float(np.std(degradations)),
            "n_iterations": n_iterations,
        }

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
