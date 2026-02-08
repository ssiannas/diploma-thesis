"""Metrics visualization functions."""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_compression_metrics(
    results: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot compression metrics comparison across models.

    Args:
        results: Dictionary mapping model names to metric dictionaries
                 e.g., {"model1": {"compression_ratio": 0.1, "bpp": 8.5}, ...}
        metrics: List of metric names to plot. If None, plots all available metrics
        figsize: Figure size

    Returns:
        Tuple of (figure, axes)
    """
    # Convert to DataFrame
    df = pd.DataFrame(results).T

    if metrics is None:
        metrics = df.columns.tolist()

    # Filter metrics
    df = df[metrics]

    # Plot
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        df[metric].plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(df)))
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Model")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_quality_metrics(
    results: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
    figsize: tuple[int, int] = (14, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot quality metrics comparison across models.

    Args:
        results: Dictionary mapping model names to metric dictionaries
                 e.g., {"model1": {"psnr": 35.2, "d1": 0.01}, ...}
        metrics: List of metric names to plot. If None, plots common quality metrics
        figsize: Figure size

    Returns:
        Tuple of (figure, axes)
    """
    # Default quality metrics
    if metrics is None:
        metrics = ["psnr", "mse", "rmse", "d1", "d2_symmetric", "hausdorff"]

    # Convert to DataFrame
    df = pd.DataFrame(results).T

    # Filter available metrics
    available_metrics = [m for m in metrics if m in df.columns]

    # Plot
    fig, axes = plt.subplots(1, len(available_metrics), figsize=figsize)
    if len(available_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        df[metric].plot(kind="bar", ax=ax, color=sns.color_palette("coolwarm", len(df)))
        ax.set_title(
            metric.upper() if len(metric) <= 4 else metric.replace("_", " ").title()
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_rate_distortion(
    results: dict[str, dict[str, float]],
    rate_metric: str = "bits_per_point",
    distortion_metric: str = "psnr",
    figsize: tuple[int, int] = (10, 8),
    log_scale: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot rate-distortion curve for compression models.

    Args:
        results: Dictionary mapping model names to metric dictionaries
        rate_metric: Metric to use for rate (x-axis), e.g., "bits_per_point"
        distortion_metric: Metric to use for distortion (y-axis), e.g., "psnr"
        figsize: Figure size
        log_scale: Whether to use log scale for rate axis

    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data
    model_names = []
    rates = []
    distortions = []

    for model_name, metrics in results.items():
        if rate_metric in metrics and distortion_metric in metrics:
            model_names.append(model_name)
            rates.append(metrics[rate_metric])
            distortions.append(metrics[distortion_metric])

    # Plot points
    colors = sns.color_palette("husl", len(model_names))
    for name, rate, dist, color in zip(model_names, rates, distortions, colors):
        ax.scatter(rate, dist, s=100, label=name, color=color, alpha=0.7)

    # Connect points if they form a curve
    if len(rates) > 1:
        sorted_indices = np.argsort(rates)
        sorted_rates = [rates[i] for i in sorted_indices]
        sorted_distortions = [distortions[i] for i in sorted_indices]
        ax.plot(sorted_rates, sorted_distortions, "k--", alpha=0.3, linewidth=1)

    # Labels and formatting
    ax.set_xlabel(rate_metric.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(
        (
            distortion_metric.upper()
            if len(distortion_metric) <= 4
            else distortion_metric.replace("_", " ").title()
        ),
        fontsize=12,
    )
    ax.set_title("Rate-Distortion Curve", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    if log_scale:
        ax.set_xscale("log")

    plt.tight_layout()
    return fig, ax
