"""Oversmoothing visualization for learned point cloud codecs.

Provides spatial error heatmaps, curvature-colored point clouds,
distribution comparisons, and multi-panel report figures for
diagnosing oversmoothing artifacts in geometry compression.
"""

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

from pcml.metrics.curvature import CurvatureQualityCalculator
from pcml.visualization.point_clouds import plot_point_cloud

matplotlib.use("Agg")


def _set_equal_aspect_3d(ax: Axes3D, geometry: np.ndarray) -> None:
    """Set equal aspect ratio on a 3D axes for a point cloud."""
    max_range = np.ptp(geometry, axis=0).max()
    mid = (geometry.max(axis=0) + geometry.min(axis=0)) * 0.5
    ax.set_xlim(mid[0] - max_range * 0.5, mid[0] + max_range * 0.5)
    ax.set_ylim(mid[1] - max_range * 0.5, mid[1] + max_range * 0.5)
    ax.set_zlim(mid[2] - max_range * 0.5, mid[2] + max_range * 0.5)


def plot_error_heatmap(
    original: np.ndarray,
    reconstructed: np.ndarray,
    ax: Optional[Axes3D] = None,
    cmap: str = "hot",
    size: int = 1,
    alpha: float = 0.8,
    elev: float = 30,
    azim: float = 45,
    title: str = "Error Heatmap",
    vmax_percentile: float = 95.0,
) -> Axes3D:
    """Color each reconstructed point by its NN distance to the original.

    Args:
        original: (N, 3) original point coordinates.
        reconstructed: (M, 3) reconstructed point coordinates.
        ax: Optional existing 3D axes.
        cmap: Colormap name.
        size: Point size.
        alpha: Point transparency.
        elev: Elevation angle.
        azim: Azimuth angle.
        title: Plot title.
        vmax_percentile: Percentile for colorbar upper limit (avoids
            outliers dominating the scale).

    Returns:
        The matplotlib 3D axes.
    """
    dists, _ = CurvatureQualityCalculator.compute_per_point_error(
        original, reconstructed
    )

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    vmax = np.percentile(dists, vmax_percentile) if len(dists) > 0 else 1.0
    norm = Normalize(vmin=0, vmax=vmax)
    colormap = plt.get_cmap(cmap)
    colors = colormap(norm(dists))[:, :3]

    ax.scatter(
        reconstructed[:, 0],
        reconstructed[:, 1],
        reconstructed[:, 2],
        c=colors,
        s=size,
        alpha=alpha,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, label="NN distance")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    _set_equal_aspect_3d(ax, reconstructed)

    return ax


def plot_curvature_heatmap(
    geometry: np.ndarray,
    curvature: np.ndarray,
    ax: Optional[Axes3D] = None,
    cmap: str = "viridis",
    size: int = 1,
    alpha: float = 0.8,
    elev: float = 30,
    azim: float = 45,
    title: str = "Curvature",
) -> Axes3D:
    """Color each point by its PCA curvature value.

    Args:
        geometry: (N, 3) point coordinates.
        curvature: (N,) curvature values.
        ax: Optional existing 3D axes.
        cmap: Colormap name.
        size: Point size.
        alpha: Point transparency.
        elev: Elevation angle.
        azim: Azimuth angle.
        title: Plot title.

    Returns:
        The matplotlib 3D axes.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    vmax = np.percentile(curvature, 99) if len(curvature) > 0 else 1.0
    norm = Normalize(vmin=0, vmax=vmax)
    colormap = plt.get_cmap(cmap)
    colors = colormap(norm(curvature))[:, :3]

    ax.scatter(
        geometry[:, 0],
        geometry[:, 1],
        geometry[:, 2],
        c=colors,
        s=size,
        alpha=alpha,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, label="Curvature")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    _set_equal_aspect_3d(ax, geometry)

    return ax


def plot_region_colored(
    geometry: np.ndarray,
    curvature: np.ndarray,
    ax: Optional[Axes3D] = None,
    size: int = 1,
    alpha: float = 0.8,
    elev: float = 30,
    azim: float = 45,
    title: str = "Curvature Regions",
) -> Axes3D:
    """Color points by curvature tercile: blue=flat, green=medium, red=edge.

    Args:
        geometry: (N, 3) point coordinates.
        curvature: (N,) curvature values.
        ax: Optional existing 3D axes.
        size: Point size.
        alpha: Point transparency.
        elev: Elevation angle.
        azim: Azimuth angle.
        title: Plot title.

    Returns:
        The matplotlib 3D axes.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    t1 = np.percentile(curvature, 33.33)
    t2 = np.percentile(curvature, 66.67)

    colors = np.zeros((len(geometry), 3))
    flat_mask = curvature <= t1
    medium_mask = (curvature > t1) & (curvature <= t2)
    edge_mask = curvature > t2

    colors[flat_mask] = [0.2, 0.4, 0.8]  # blue
    colors[medium_mask] = [0.2, 0.7, 0.3]  # green
    colors[edge_mask] = [0.9, 0.2, 0.2]  # red

    ax.scatter(
        geometry[:, 0],
        geometry[:, 1],
        geometry[:, 2],
        c=colors,
        s=size,
        alpha=alpha,
    )

    # Legend via proxy artists
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=[0.2, 0.4, 0.8],
            markersize=8,
            label="Flat",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=[0.2, 0.7, 0.3],
            markersize=8,
            label="Medium",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=[0.9, 0.2, 0.2],
            markersize=8,
            label="Edge",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    _set_equal_aspect_3d(ax, geometry)

    return ax


def plot_curvature_distributions(
    orig_curvature: np.ndarray,
    dec_curvature: np.ndarray,
    ax: Optional[plt.Axes] = None,
    n_bins: int = 50,
    title: str = "Curvature Distribution",
) -> plt.Axes:
    """Overlaid histograms of original vs decoded curvature distributions.

    Args:
        orig_curvature: (N,) original curvature values.
        dec_curvature: (M,) decoded curvature values.
        ax: Optional existing 2D axes.
        n_bins: Number of histogram bins.
        title: Plot title.

    Returns:
        The matplotlib 2D axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    vmax = max(np.percentile(orig_curvature, 99), np.percentile(dec_curvature, 99))
    bins = np.linspace(0, vmax, n_bins + 1)

    ax.hist(
        orig_curvature,
        bins=bins,
        alpha=0.6,
        density=True,
        label="Original",
        color="steelblue",
    )
    ax.hist(
        dec_curvature,
        bins=bins,
        alpha=0.6,
        density=True,
        label="Decoded",
        color="coral",
    )

    kl = CurvatureQualityCalculator.compute_curvature_kl(
        orig_curvature, dec_curvature, n_bins=n_bins
    )
    ax.annotate(
        f"KL div: {kl:.4f}",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8),
    )

    ax.set_xlabel("Curvature")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    return ax


def plot_stratified_psnr_bars(
    rate_labels: list[str],
    flat_psnrs: list[float],
    medium_psnrs: list[float],
    edge_psnrs: list[float],
    ax: Optional[plt.Axes] = None,
    title: str = "Stratified PSNR by Rate",
) -> plt.Axes:
    """Grouped bar chart: 3 bars per rate point (flat/medium/edge PSNR).

    Args:
        rate_labels: List of rate point labels (e.g. ["r1", "r3", "r7"]).
        flat_psnrs: PSNR values for flat regions at each rate.
        medium_psnrs: PSNR values for medium regions at each rate.
        edge_psnrs: PSNR values for edge regions at each rate.
        ax: Optional existing 2D axes.
        title: Plot title.

    Returns:
        The matplotlib 2D axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(rate_labels))
    width = 0.25

    ax.bar(x - width, flat_psnrs, width, label="Flat", color="steelblue")
    ax.bar(x, medium_psnrs, width, label="Medium", color="forestgreen")
    ax.bar(x + width, edge_psnrs, width, label="Edge", color="firebrick")

    ax.set_xlabel("Rate Point")
    ax.set_ylabel("D1-PSNR (dB)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(rate_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    return ax


def plot_degradation_vs_rate(
    rate_labels: list[str],
    degradations: list[float],
    bpps: list[float],
    ax: Optional[plt.Axes] = None,
    title: str = "Oversmoothing Degradation vs BPP",
) -> plt.Axes:
    """Line plot: degradation (flat_psnr - edge_psnr) vs BPP.

    Args:
        rate_labels: Rate point labels for annotation.
        degradations: Degradation values (flat_psnr - edge_psnr) per rate.
        bpps: Bits per point for each rate.
        ax: Optional existing 2D axes.
        title: Plot title.

    Returns:
        The matplotlib 2D axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(bpps, degradations, "o-", color="firebrick", linewidth=2, markersize=8)

    for label, bpp, deg in zip(rate_labels, bpps, degradations):
        ax.annotate(
            label, (bpp, deg), textcoords="offset points", xytext=(5, 8), fontsize=9
        )

    ax.set_xlabel("Bits Per Point (BPP)")
    ax.set_ylabel("Degradation (flat PSNR - edge PSNR) [dB]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    return ax


def plot_correlation_vs_rate(
    rate_labels: list[str],
    rhos: list[float],
    p_values: list[float],
    bpps: list[float],
    ax: Optional[plt.Axes] = None,
    title: str = "Spearman Correlation vs BPP",
) -> plt.Axes:
    """Line plot of Spearman rho (curvature-error correlation) vs BPP.

    Positive rho indicates oversmoothing (higher curvature = higher error).
    Points are annotated with rate labels and significance markers.

    Args:
        rate_labels: Rate point labels for annotation.
        rhos: Spearman rho at each rate.
        p_values: Two-sided p-values at each rate.
        bpps: Bits per point at each rate.
        ax: Optional existing 2D axes.
        title: Plot title.

    Returns:
        The matplotlib 2D axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(bpps, rhos, "o-", color="darkorange", linewidth=2, markersize=8)

    for label, bpp, rho, pval in zip(rate_labels, bpps, rhos, p_values):
        sig = (
            "***"
            if pval < 0.001
            else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        )
        ax.annotate(
            f"{label}{sig}",
            (bpp, rho),
            textcoords="offset points",
            xytext=(5, 8),
            fontsize=9,
        )

    ax.set_xlabel("Bits Per Point (BPP)")
    ax.set_ylabel("Spearman rho (curvature vs error)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    return ax


def plot_psnr_vs_curvature_curve(
    bin_centers: np.ndarray,
    psnrs: np.ndarray,
    counts: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "PSNR vs Curvature",
    show_counts: bool = True,
) -> plt.Axes:
    """Plot continuous PSNR-vs-curvature curve.

    Args:
        bin_centers: (n_bins,) curvature bin centers.
        psnrs: (n_bins,) PSNR per bin.
        counts: (n_bins,) point counts per bin.
        ax: Optional existing 2D axes.
        title: Plot title.
        show_counts: If True, show point counts as bar chart on twin axis.

    Returns:
        The matplotlib 2D axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out NaN/inf bins
    valid = np.isfinite(psnrs)
    ax.plot(
        bin_centers[valid],
        psnrs[valid],
        "o-",
        color="steelblue",
        linewidth=2,
        markersize=6,
        label="PSNR",
    )

    ax.set_xlabel("Curvature (surface variation)")
    ax.set_ylabel("D1-PSNR (dB)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    if show_counts and counts is not None:
        ax2 = ax.twinx()
        ax2.bar(
            bin_centers,
            counts,
            width=np.diff(bin_centers).mean() * 0.8 if len(bin_centers) > 1 else 0.01,
            alpha=0.15,
            color="gray",
            label="Count",
        )
        ax2.set_ylabel("Point count", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

    return ax


def plot_nn_multiplicity(
    per_stratum_mean: dict,
    ax: Optional[plt.Axes] = None,
    title: str = "NN Multiplicity by Curvature Stratum",
) -> plt.Axes:
    """Bar chart of mean NN multiplicity per curvature stratum.

    Higher multiplicity in edge strata indicates the codec redistributed
    points away from high-curvature regions.

    Args:
        per_stratum_mean: {stratum_name: mean_multiplicity}.
        ax: Optional existing 2D axes.
        title: Plot title.

    Returns:
        The matplotlib 2D axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    names = list(per_stratum_mean.keys())
    values = [per_stratum_mean[n] for n in names]
    colors = {
        "flat": "steelblue",
        "medium": "forestgreen",
        "edge": "firebrick",
        "very_flat": "lightblue",
        "sharp_edge": "darkred",
    }
    bar_colors = [colors.get(n, "gray") for n in names]

    ax.bar(names, values, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Curvature Stratum")
    ax.set_ylabel("Mean NN Multiplicity")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    for i, (name, val) in enumerate(zip(names, values)):
        ax.text(i, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    return ax


def create_oversmoothing_report(
    original: np.ndarray,
    reconstructed: np.ndarray,
    curvature: np.ndarray,
    metrics: dict,
    rate_label: str,
    output_path: Optional[str] = None,
    elev: float = 30,
    azim: float = 45,
) -> plt.Figure:
    """Multi-panel figure combining error, curvature, and distribution views.

    Layout (2x3):
        [error heatmap]  [curvature heatmap]  [region colored]
        [curv distrib]   [metrics summary]    [empty/spare]

    Args:
        original: (N, 3) original point coordinates.
        reconstructed: (M, 3) reconstructed point coordinates.
        curvature: (N,) curvature values for original points.
        metrics: Dict with keys: flat_psnr, medium_psnr, edge_psnr,
            degradation, curvature_kl, bpp, overall_psnr (optional).
        rate_label: Rate point label (e.g. "r3").
        output_path: If provided, saves figure to this path.
        elev: Elevation angle for 3D plots.
        azim: Azimuth angle for 3D plots.

    Returns:
        The matplotlib Figure.
    """
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Oversmoothing Report — {rate_label}", fontsize=16, y=0.98)

    # Panel 1: Error heatmap
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    plot_error_heatmap(
        original,
        reconstructed,
        ax=ax1,
        elev=elev,
        azim=azim,
        title="Error Heatmap",
        size=0.5,
    )

    # Panel 2: Curvature heatmap
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    plot_curvature_heatmap(
        original,
        curvature,
        ax=ax2,
        elev=elev,
        azim=azim,
        title="Curvature (Original)",
        size=0.5,
    )

    # Panel 3: Region-colored cloud
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    plot_region_colored(
        original,
        curvature,
        ax=ax3,
        elev=elev,
        azim=azim,
        title="Curvature Regions",
        size=0.5,
    )

    # Panel 4: Curvature distributions
    # Compute decoded curvature independently on the reconstructed cloud
    dec_curvature = CurvatureQualityCalculator.compute_curvature(reconstructed, k=30)

    ax4 = fig.add_subplot(2, 3, 4)
    plot_curvature_distributions(
        curvature, dec_curvature, ax=ax4, title="Curvature Distribution"
    )

    # Panel 5: Metrics text summary
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis("off")

    summary_lines = [
        f"Rate Point: {rate_label}",
        "",
        f"Flat PSNR:    {metrics.get('flat_psnr', 0):.2f} dB",
        f"Medium PSNR:  {metrics.get('medium_psnr', 0):.2f} dB",
        f"Edge PSNR:    {metrics.get('edge_psnr', 0):.2f} dB",
        "",
        f"Degradation:  {metrics.get('degradation', 0):.2f} dB",
        f"KL Divergence: {metrics.get('curvature_kl', 0):.4f}",
    ]
    if "bpp" in metrics:
        summary_lines.append(f"BPP:          {metrics['bpp']:.4f}")
    if "overall_psnr" in metrics:
        summary_lines.append(f"Overall PSNR: {metrics['overall_psnr']:.2f} dB")
    if "num_points_input" in metrics:
        summary_lines.append(f"Input pts:    {metrics['num_points_input']:,}")
    if "num_points_output" in metrics:
        summary_lines.append(f"Output pts:   {metrics['num_points_output']:,}")
    if "rev_degradation" in metrics:
        summary_lines.append("")
        summary_lines.append(f"Rev. Degrad:  {metrics['rev_degradation']:.2f} dB")
    if "assignment_accuracy" in metrics:
        summary_lines.append(f"Assign. Acc:  {metrics['assignment_accuracy']:.3f}")
    if "spearman_rho" in metrics:
        sig = ""
        pval = metrics.get("spearman_pvalue", 1.0)
        if pval < 0.001:
            sig = " ***"
        elif pval < 0.01:
            sig = " **"
        elif pval < 0.05:
            sig = " *"
        summary_lines.append(f"Spearman rho: {metrics['spearman_rho']:.4f}{sig}")

    text = "\n".join(summary_lines)
    ax5.text(
        0.1,
        0.9,
        text,
        transform=ax5.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.9),
    )

    # Panel 6: empty
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
