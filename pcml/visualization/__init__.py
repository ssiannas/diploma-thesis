"""Visualization utilities for point clouds and metrics."""

from pcml.visualization.metrics import (
    plot_compression_metrics,
    plot_quality_metrics,
    plot_rate_distortion,
)
from pcml.visualization.oversmoothing import (
    create_oversmoothing_report,
    plot_correlation_vs_rate,
    plot_curvature_distributions,
    plot_curvature_heatmap,
    plot_degradation_vs_rate,
    plot_error_heatmap,
    plot_nn_multiplicity,
    plot_psnr_vs_curvature_curve,
    plot_region_colored,
    plot_stratified_psnr_bars,
)
from pcml.visualization.point_clouds import (
    plot_point_cloud,
    plot_point_cloud_comparison,
    plot_point_cloud_grid,
)

__all__ = [
    "plot_point_cloud",
    "plot_point_cloud_comparison",
    "plot_point_cloud_grid",
    "plot_compression_metrics",
    "plot_quality_metrics",
    "plot_rate_distortion",
    "plot_error_heatmap",
    "plot_curvature_heatmap",
    "plot_region_colored",
    "plot_curvature_distributions",
    "plot_stratified_psnr_bars",
    "plot_degradation_vs_rate",
    "plot_correlation_vs_rate",
    "plot_psnr_vs_curvature_curve",
    "plot_nn_multiplicity",
    "create_oversmoothing_report",
]
