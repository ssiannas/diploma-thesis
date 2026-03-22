"""Headless 3D projection screenshot matching the view_pointcloud.py save_capture style.

Crops a region around the peak curvature centre (or manual centre), then
projects all three clouds from az/el camera angle and saves a side-by-side
matplotlib figure on dark background — identical to pressing P in the viewer.

Usage:
    python scripts/analysis/screenshot_3d_crop.py \
        --original  datasets/8iVFB_unicorn_test/soldier_vox10_0690.ply \
        --decoded   results/oversmoothing/decoded_clouds/soldier_vox10_0690/pcgcv2_r2.npy \
        --refined   results/oversmoothing/decoded_clouds/soldier_vox10_0690/refined_nocurv_ep24_r2.npy \
        --radius 80 --az 10 --el 5 \
        --output results/vis/soldier_3d_nocurv_az10_el5_r2.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from tqdm import tqdm

COLORS = [
    [0.05, 0.55, 0.20],  # dark green -> original
    [0.75, 0.10, 0.08],  # dark red   -> decoded
    [0.05, 0.35, 0.72],  # dark blue  -> refined
]
LABELS = ["Original", "Compressed (PCGCv2, R2)", "Post-processed (Ours)"]


def load(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".ply":
        data = PlyData.read(str(p))["vertex"]
        return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    return np.load(path).astype(np.float64)


def compute_curvature(pts: np.ndarray, k: int = 30) -> np.ndarray:
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)
    curv = np.zeros(len(pts))
    for i in tqdm(range(len(pts)), desc="Curvature", leave=False):
        nb = pts[idx[i, 1:]]
        nb = nb - nb.mean(0)
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(nb.T @ nb)))
        s = eigvals.sum()
        curv[i] = eigvals[0] / s if s > 1e-10 else 0.0
    return curv


def find_peak_centre(pts: np.ndarray, curv: np.ndarray, radius: float) -> np.ndarray:
    top_idx = np.argsort(curv)[-20:]
    tree = cKDTree(pts)
    best, best_score = pts[top_idx[-1]], -1.0
    for i in top_idx:
        nb = tree.query_ball_point(pts[i], radius)
        if len(nb) < 50:
            continue
        score = curv[nb].mean()
        if score > best_score:
            best_score, best = score, pts[i]
    return best


def crop(pts: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    mask = np.abs(pts - centre).max(axis=1) <= radius
    return pts[mask]


def build_camera(centre: np.ndarray, extent: float, az_deg: float, el_deg: float):
    """Return (extrinsic 4x4, intrinsic 3x3) for a pinhole camera."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    dist = extent * 1.5
    eye = centre + dist * np.array(
        [
            np.cos(el) * np.sin(az),
            np.sin(el),
            np.cos(el) * np.cos(az),
        ]
    )
    forward = centre - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    R = np.stack([right, -up, forward], axis=0)
    t = -R @ eye
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    W, H = 800, 600
    f = 0.8 * max(W, H)
    intrinsic = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    return extrinsic, intrinsic


def project(pts: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    cam = (R @ pts.T).T + t
    mask = cam[:, 2] > 0
    cam = cam[mask]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u = fx * cam[:, 0] / cam[:, 2] + cx
    v = fy * cam[:, 1] / cam[:, 2] + cy
    return u, v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True)
    p.add_argument("--decoded", required=True)
    p.add_argument("--refined", required=True)
    p.add_argument("--radius", type=float, default=80)
    p.add_argument("--az", type=float, default=10.0)
    p.add_argument("--el", type=float, default=5.0)
    p.add_argument("--centre", type=float, nargs=3, default=None)
    p.add_argument("--point_size", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--labels", nargs=3, default=LABELS)
    p.add_argument("--output", default="results/vis/screenshot_3d_crop.png")
    args = p.parse_args()

    print("Loading clouds...")
    orig = load(args.original)
    dec = load(args.decoded)
    ref = load(args.refined)

    if args.centre is not None:
        centre = np.array(args.centre)
    else:
        print("Computing curvature to find peak region...")
        curv = compute_curvature(orig)
        centre = find_peak_centre(orig, curv, args.radius)
        print(f"Peak centre: {centre.astype(int).tolist()}")

    clouds = [crop(c, centre, args.radius) for c in [orig, dec, ref]]
    for pts, lbl in zip(clouds, args.labels):
        print(f"  {lbl}: {len(pts):,} pts")

    extent = args.radius * 2
    extrinsic, intrinsic = build_camera(centre, extent, args.az, args.el)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d0d0d")
    for ax, pts, color, label in zip(axes, clouds, COLORS, args.labels):
        u, v = project(pts, extrinsic, intrinsic)
        ax.set_facecolor("#0d0d0d")
        ax.scatter(u, -v, s=args.point_size, c=[color], alpha=args.alpha, linewidths=0)
        ax.set_title(f"{label}\n({len(u):,} pts)", fontsize=10, color="white")
        ax.set_aspect("equal")
        ax.axis("off")

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
