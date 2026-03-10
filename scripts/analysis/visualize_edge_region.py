"""Visualize a high-curvature edge region: original vs decoded vs refined.

Finds the highest-curvature region in the original cloud, crops a cubic
neighbourhood, and renders three side-by-side 2D projections.

Usage:
    python scripts/analysis/visualize_edge_region.py \
        --original  datasets/8iVFB/longdress/Ply/longdress_vox10_1300.ply \
        --decoded   results/oversmoothing/decoded_clouds/.../pcgcv2_r2.npy
        --refined   results/oversmoothing/decoded_clouds/.../refined_..._r2.npy
        --radius 80 \
        --output  results/vis/edge_longdress_r2.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from tqdm import tqdm

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_ply(path: str) -> np.ndarray:
    """Return (N, 3) float32 xyz from a PLY file."""
    data = PlyData.read(path)["vertex"]
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)


def load_npy(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------


def compute_curvature(pts: np.ndarray, k: int = 30) -> np.ndarray:
    """PCA surface variation: lambda_min / (lambda_1 + lambda_2 + lambda_3)."""
    tree = cKDTree(pts)
    curv = np.zeros(len(pts), dtype=np.float32)
    _, idx = tree.query(pts, k=k + 1)
    for i in tqdm(range(len(pts)), desc="Curvature", leave=False):
        nb = pts[idx[i, 1:]]
        nb -= nb.mean(axis=0)
        eigvals = np.linalg.eigvalsh(nb.T @ nb)
        eigvals = np.sort(np.abs(eigvals))
        s = eigvals.sum()
        curv[i] = eigvals[0] / s if s > 1e-10 else 0.0
    return curv


# ---------------------------------------------------------------------------
# Region finding
# ---------------------------------------------------------------------------


def find_peak_curvature_region(
    pts: np.ndarray, curv: np.ndarray, radius: float, top_k: int = 10
) -> np.ndarray:
    """Return centre of the cubic patch with highest mean curvature.

    Samples top_k high-curvature candidate seeds and picks the one whose
    radius-neighbourhood has the highest mean curvature.
    """
    top_idx = np.argsort(curv)[-top_k:]
    tree = cKDTree(pts)
    best_centre = pts[top_idx[-1]]
    best_score = -1.0
    for i in top_idx:
        centre = pts[i]
        nb_idx = tree.query_ball_point(centre, radius)
        if len(nb_idx) < 50:
            continue
        score = curv[nb_idx].mean()
        if score > best_score:
            best_score = score
            best_centre = centre
    return best_centre


def crop(pts: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    mask = np.all(np.abs(pts - centre) <= radius, axis=1)
    return pts[mask]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

# Axis pairs for three projections (indices into xyz)
PROJECTIONS = [
    (0, 2, "X", "Z", "Front view"),
    (0, 1, "X", "Y", "Top view"),
    (1, 2, "Y", "Z", "Side view"),
]

CLOUDS = [
    ("Original", "#2ecc71"),
    ("Decoded", "#e74c3c"),
    ("Refined", "#3498db"),
]


def render(ax, pts: np.ndarray, xi: int, yi: int, color: str, title: str):
    ax.scatter(pts[:, xi], pts[:, yi], s=0.8, c=color, alpha=0.6, linewidths=0)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.set_xlabel(PROJECTIONS[0][2], fontsize=7)
    ax.set_ylabel(PROJECTIONS[0][3], fontsize=7)


def make_figure(
    crops: list, labels: list, colors: list, projection: tuple, output: str
):
    xi, yi, xl, yl, proj_name = projection
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Edge region — {proj_name}", fontsize=11, y=1.01)

    for ax, pts, (label, color) in zip(axes, crops, zip(labels, colors)):
        ax.scatter(pts[:, xi], pts[:, yi], s=0.8, c=color, alpha=0.6, linewidths=0)
        ax.set_title(f"{label}  ({len(pts):,} pts)", fontsize=9, pad=4)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)

    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True, help="Original PLY path")
    p.add_argument("--decoded", required=True, help="Decoded NPY path")
    p.add_argument("--refined", required=True, help="Refined NPY path")
    p.add_argument(
        "--radius",
        type=float,
        default=80,
        help="Crop half-width in voxels (default 80)",
    )
    p.add_argument("--curv_k", type=int, default=30)
    p.add_argument(
        "--output",
        default="results/vis/edge_region.png",
        help="Output PNG path (stem used for all projections)",
    )
    p.add_argument(
        "--projection",
        choices=["front", "top", "side", "all"],
        default="all",
        help="Which projection(s) to render",
    )
    p.add_argument(
        "--centre",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Manual crop centre (voxel coords). Auto-detected if omitted.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading clouds...")
    orig = load_ply(args.original)
    decoded = load_npy(args.decoded)
    refined = load_npy(args.refined)

    # Align decoded/refined to original coordinate system if needed
    # (PCGCv2 decoded npy are already in voxel coords matching the PLY)

    if args.centre is not None:
        centre = np.array(args.centre, dtype=np.float32)
        print(f"Using manual centre: {centre}")
    else:
        print("Computing curvature on original cloud...")
        curv = compute_curvature(orig, k=args.curv_k)
        centre = find_peak_curvature_region(orig, curv, radius=args.radius)
        print(f"Peak curvature centre: {centre.astype(int)}")

    print(f"Cropping radius={args.radius} voxels...")
    crops = [crop(c, centre, args.radius) for c in [orig, decoded, refined]]
    labels = ["Original", "Decoded", "Refined"]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    for c, l in zip(crops, labels):
        print(f"  {l}: {len(c):,} points")

    stem = Path(args.output).with_suffix("")
    proj_map = {"front": 0, "top": 1, "side": 2}

    if args.projection == "all":
        for proj in PROJECTIONS:
            suffix = proj[4].split()[0].lower()
            out = f"{stem}_{suffix}.png"
            make_figure(crops, labels, colors, proj, out)
    else:
        proj = PROJECTIONS[proj_map[args.projection]]
        make_figure(crops, labels, colors, proj, args.output)


if __name__ == "__main__":
    main()
