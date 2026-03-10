"""Interactive point cloud viewer with WASD + mouse navigation.

Supports PLY and NPY files. Can load up to 3 clouds simultaneously
(e.g. original, decoded, refined) in different colours.
Optionally colours points by curvature to highlight edges.

Usage:
    # Single cloud
    python scripts/analysis/view_pointcloud.py --clouds file.ply

    # Three clouds (original / decoded / refined)
    python scripts/analysis/view_pointcloud.py \
        --clouds datasets/8iVFB/longdress/Ply/longdress_vox10_1300.ply \
                 results/oversmoothing/decoded_clouds/longdress_vox10_1300/pcgcv2_r2.npy
                 results/oversmoothing/decoded_clouds/.../refined_..._r2.npy
        --labels Original Decoded Refined

    # Colour by curvature (single cloud only)
    python scripts/analysis/view_pointcloud.py --clouds file.ply --curvature

Controls:
    W / S          move forward / backward
    A / D          strafe left / right
    Q / E          move up / down
    Mouse drag     rotate
    Right drag     pan
    Scroll         zoom
    R              reset view
    H              print this help
    ESC / Q key    quit  (use ESC in viewer window)
"""

import argparse
import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Colours per cloud
# ---------------------------------------------------------------------------
COLORS = [
    [0.18, 0.80, 0.44],  # green  → original
    [0.91, 0.30, 0.24],  # red    → decoded
    [0.20, 0.60, 0.86],  # blue   → refined
]

STEP = 20.0  # camera translation step (voxels)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_cloud(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".ply":
        data = PlyData.read(path)["vertex"]
        return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    elif p.suffix == ".npy":
        return np.load(path).astype(np.float64)
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")


def compute_curvature(pts: np.ndarray, k: int = 30) -> np.ndarray:
    print("Computing curvature...")
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)
    curv = np.zeros(len(pts))
    for i in range(len(pts)):
        nb = pts[idx[i, 1:]]
        nb = nb - nb.mean(axis=0)
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(nb.T @ nb)))
        s = eigvals.sum()
        curv[i] = eigvals[0] / s if s > 1e-10 else 0.0
    return curv


def curvature_colormap(curv: np.ndarray) -> np.ndarray:
    """Map curvature to blue→red heatmap."""
    c = (curv - curv.min()) / (curv.max() - curv.min() + 1e-10)
    colors = np.zeros((len(c), 3))
    colors[:, 0] = c  # red channel
    colors[:, 2] = 1.0 - c  # blue channel
    return colors


# ---------------------------------------------------------------------------
# Build open3d geometries
# ---------------------------------------------------------------------------


def make_pcd(pts: np.ndarray, color=None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if color is not None:
        if color.ndim == 1:
            pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts), 1)))
        else:
            pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


def move_camera(vis, dx=0.0, dy=0.0, dz=0.0):
    """Translate camera in its local frame."""
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    extr = np.array(params.extrinsic)

    # Camera axes from extrinsic (row vectors = right, up, forward in world)
    right = extr[0, :3]
    up = extr[1, :3]
    forward = extr[2, :3]

    # Shift the translation part
    extr[:3, 3] += right * dx - up * dy + forward * dz

    params.extrinsic = extr
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    vis.update_renderer()


# ---------------------------------------------------------------------------
# Capture: project clouds to 2D from current camera and save figure
# ---------------------------------------------------------------------------


def project_points(
    pts: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray
) -> tuple:
    """Project Nx3 world points through camera. Returns (u, v, mask)."""
    # World → camera
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    cam = (R @ pts.T).T + t  # (N, 3) in camera space

    # Keep points in front of camera
    mask = cam[:, 2] > 0
    cam = cam[mask]

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    u = fx * cam[:, 0] / cam[:, 2] + cx
    v = fy * cam[:, 1] / cam[:, 2] + cy
    return u, v, mask


def save_capture(vis, all_pts: list, labels: list, out_dir: str = "results/vis"):
    """Project all clouds from current view and save side-by-side figure."""
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    extr = np.array(params.extrinsic)
    intr = np.array(params.intrinsic.intrinsic_matrix)

    n = len(all_pts)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, pts, label, color in zip(axes, all_pts, labels, COLORS):
        u, v, _ = project_points(pts, extr, intr)
        ax.scatter(u, -v, s=0.3, c=[color], alpha=0.5, linewidths=0)
        ax.set_title(f"{label}  ({len(u):,} pts)", fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%H%M%S")
    out = f"{out_dir}/capture_{ts}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"Captured: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--clouds", nargs="+", required=True, help="PLY or NPY files (up to 3)"
    )
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument(
        "--curvature",
        action="store_true",
        help="Colour first cloud by curvature (single cloud only)",
    )
    p.add_argument("--curv_k", type=int, default=30)
    p.add_argument("--point_size", type=float, default=1.5)
    return p.parse_args()


def main():
    args = parse_args()

    if len(args.clouds) > 3:
        print("Warning: max 3 clouds supported, ignoring the rest.")
        args.clouds = args.clouds[:3]

    default_labels = ["Original", "Decoded", "Refined"]
    labels = args.labels or default_labels[: len(args.clouds)]

    print("Loading clouds...")
    pcds = []
    all_pts = []
    for i, path in enumerate(args.clouds):
        pts = load_cloud(path)
        print(f"  {labels[i]}: {len(pts):,} points")
        all_pts.append(pts)

        if args.curvature and i == 0:
            curv = compute_curvature(pts, k=args.curv_k)
            color = curvature_colormap(curv)
        else:
            color = np.array(COLORS[i % len(COLORS)])

        pcds.append(make_pcd(pts, color))

    # -----------------------------------------------------------------------
    # Viewer setup
    # -----------------------------------------------------------------------
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="Point Cloud Viewer  |  WASD=move  Mouse=rotate  Scroll=zoom  R=reset  ESC=quit",  # noqa: E501
        width=1400,
        height=900,
    )

    for pcd in pcds:
        vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.05, 0.05, 0.05])  # dark background

    # Print legend
    print("\nLegend:")
    color_names = ["green", "red", "blue"]
    for i, label in enumerate(labels):
        print(f"  {color_names[i % 3]:6s}  {label}")
    print("\nControls: WASD=move  Q/E=up/down  R=reset  P=capture  ESC=quit")
    print("Mouse: left-drag=rotate  right-drag=pan  scroll=zoom\n")

    # -----------------------------------------------------------------------
    # Key callbacks
    # -----------------------------------------------------------------------
    vis.register_key_callback(ord("W"), lambda v: move_camera(v, dz=-STEP))
    vis.register_key_callback(ord("S"), lambda v: move_camera(v, dz=+STEP))
    vis.register_key_callback(ord("A"), lambda v: move_camera(v, dx=-STEP))
    vis.register_key_callback(ord("D"), lambda v: move_camera(v, dx=+STEP))
    vis.register_key_callback(ord("Q"), lambda v: move_camera(v, dy=-STEP))
    vis.register_key_callback(ord("E"), lambda v: move_camera(v, dy=+STEP))

    def reset_view(vis):
        vis.reset_view_point(True)

    vis.register_key_callback(ord("R"), reset_view)

    def print_help(vis):
        print(__doc__)

    vis.register_key_callback(ord("H"), print_help)

    def capture(vis):
        save_capture(vis, all_pts, labels)

    vis.register_key_callback(ord("P"), capture)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
