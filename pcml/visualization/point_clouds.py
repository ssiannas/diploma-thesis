"""Point cloud visualization functions."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_point_cloud(
    geometry: np.ndarray,
    colors: Optional[np.ndarray] = None,
    title: str = "Point Cloud",
    size: int = 1,
    alpha: float = 0.8,
    elev: float = 30,
    azim: float = 45,
    ax: Optional[Axes3D] = None,
) -> Axes3D:
    """Plot a single point cloud.

    Args:
        geometry: Point cloud coordinates (N, 3)
        colors: Optional point colors (N, 3), values in [0, 255] or [0, 1]
        title: Plot title
        size: Point size
        alpha: Point transparency
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
        ax: Optional existing axes to plot on

    Returns:
        The matplotlib 3D axes object
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Normalize colors if needed
    if colors is not None:
        if colors.max() > 1.0:
            colors = colors / 255.0

    # Plot
    ax.scatter(
        geometry[:, 0],
        geometry[:, 1],
        geometry[:, 2],
        c=colors if colors is not None else "blue",
        s=size,
        alpha=alpha,
    )

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Set view angle
    ax.view_init(elev=elev, azim=azim)

    # Equal aspect ratio
    max_range = np.array(
        [
            geometry[:, 0].max() - geometry[:, 0].min(),
            geometry[:, 1].max() - geometry[:, 1].min(),
            geometry[:, 2].max() - geometry[:, 2].min(),
        ]
    ).max()
    mid_x = (geometry[:, 0].max() + geometry[:, 0].min()) * 0.5
    mid_y = (geometry[:, 1].max() + geometry[:, 1].min()) * 0.5
    mid_z = (geometry[:, 2].max() + geometry[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)

    return ax


def plot_point_cloud_comparison(
    original_geometry: np.ndarray,
    reconstructed_geometry: np.ndarray,
    original_colors: Optional[np.ndarray] = None,
    reconstructed_colors: Optional[np.ndarray] = None,
    title: str = "Point Cloud Comparison",
    size: int = 1,
    alpha: float = 0.8,
    elev: float = 30,
    azim: float = 45,
) -> tuple[plt.Figure, tuple[Axes3D, Axes3D]]:
    """Plot original and reconstructed point clouds side by side.

    Args:
        original_geometry: Original point cloud (N, 3)
        reconstructed_geometry: Reconstructed point cloud (M, 3)
        original_colors: Optional original colors (N, 3)
        reconstructed_colors: Optional reconstructed colors (M, 3)
        title: Overall plot title
        size: Point size
        alpha: Point transparency
        elev: Elevation angle
        azim: Azimuth angle

    Returns:
        Tuple of (figure, (ax1, ax2))
    """
    fig = plt.figure(figsize=(20, 8))

    # Original
    ax1 = fig.add_subplot(121, projection="3d")
    plot_point_cloud(
        original_geometry,
        original_colors,
        title="Original",
        size=size,
        alpha=alpha,
        elev=elev,
        azim=azim,
        ax=ax1,
    )

    # Reconstructed
    ax2 = fig.add_subplot(122, projection="3d")
    plot_point_cloud(
        reconstructed_geometry,
        reconstructed_colors,
        title="Reconstructed",
        size=size,
        alpha=alpha,
        elev=elev,
        azim=azim,
        ax=ax2,
    )

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    return fig, (ax1, ax2)


def plot_point_cloud_grid(
    geometries: list[np.ndarray],
    colors_list: Optional[list[Optional[np.ndarray]]] = None,
    titles: Optional[list[str]] = None,
    ncols: int = 3,
    size: int = 1,
    alpha: float = 0.8,
    elev: float = 30,
    azim: float = 45,
    figsize: tuple[int, int] = (15, 10),
) -> tuple[plt.Figure, list[Axes3D]]:
    """Plot multiple point clouds in a grid layout.

    Args:
        geometries: List of point clouds to plot
        colors_list: Optional list of color arrays
        titles: Optional list of titles for each subplot
        ncols: Number of columns in the grid
        size: Point size
        alpha: Point transparency
        elev: Elevation angle
        azim: Azimuth angle
        figsize: Figure size

    Returns:
        Tuple of (figure, list of axes)
    """
    n_clouds = len(geometries)
    nrows = (n_clouds + ncols - 1) // ncols

    fig = plt.figure(figsize=figsize)
    axes = []

    if colors_list is None:
        colors_list = [None] * n_clouds
    if titles is None:
        titles = [f"Cloud {i+1}" for i in range(n_clouds)]

    for i, (geom, colors, title) in enumerate(zip(geometries, colors_list, titles)):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        plot_point_cloud(
            geom,
            colors,
            title=title,
            size=size,
            alpha=alpha,
            elev=elev,
            azim=azim,
            ax=ax,
        )
        axes.append(ax)

    plt.tight_layout()
    return fig, axes
