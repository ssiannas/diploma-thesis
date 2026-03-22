"""Headless 3D point cloud screenshot using Open3D offscreen rendering.

Usage:
    python scripts/analysis/screenshot_3d.py \
        --original  datasets/8iVFB_unicorn_test/soldier_vox10_0690.ply \
        --decoded   results/oversmoothing/decoded_clouds/soldier_vox10_0690/pcgcv2_r2.npy \
        --refined   results/oversmoothing/decoded_clouds/soldier_vox10_0690/refined_nocurv_ep24_r2.npy \
        --az 10 --el 5 \
        --output results/vis/soldier_3d_nocurv_az10_el5_r2.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData

COLORS = [
    [0.18, 0.80, 0.44],  # green  -> original
    [0.91, 0.30, 0.24],  # red    -> decoded
    [0.20, 0.60, 0.86],  # blue   -> refined
]


def load(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".ply":
        data = PlyData.read(str(p))["vertex"]
        return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    return np.load(path).astype(np.float32)


def make_pcd(pts: np.ndarray, color: list) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.paint_uniform_color(color)
    return pcd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True)
    p.add_argument("--decoded", required=True)
    p.add_argument("--refined", required=True)
    p.add_argument("--az", type=float, default=10.0, help="Azimuth degrees")
    p.add_argument("--el", type=float, default=5.0, help="Elevation degrees")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=700)
    p.add_argument("--point_size", type=float, default=2.0)
    p.add_argument("--output", default="results/vis/screenshot_3d.png")
    args = p.parse_args()

    clouds = [load(args.original), load(args.decoded), load(args.refined)]
    labels = ["Original", "Decoded", "Refined"]

    render = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    render.scene.set_background([0.05, 0.05, 0.05, 1.0])

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = args.point_size

    for pts, color, label in zip(clouds, COLORS, labels):
        pcd = make_pcd(pts, color)
        m = o3d.visualization.rendering.MaterialRecord()
        m.shader = "defaultUnlit"
        m.point_size = args.point_size
        render.scene.add_geometry(label, pcd, m)

    # Set camera from azimuth + elevation
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        np.array(clouds[0].min(0), dtype=np.float64),
        np.array(clouds[0].max(0), dtype=np.float64),
    )
    center = bbox.get_center()
    extent = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())

    az = np.radians(args.az)
    el = np.radians(args.el)
    eye = center + extent * 0.9 * np.array(
        [
            np.cos(el) * np.sin(az),
            np.sin(el),
            np.cos(el) * np.cos(az),
        ]
    )
    render.setup_camera(60.0, center, eye, [0, 1, 0])

    img = render.render_to_image()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_image(args.output, img)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
