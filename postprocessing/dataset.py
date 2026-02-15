"""Dataset for training displacement-based post-processing on decoded point clouds.

Loads cached decoded/original .npy pairs, computes per-point displacement targets
via nearest-neighbor assignment, and extracts fixed-size voxel patches.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None

logger = logging.getLogger(__name__)

SEQUENCES = [
    "longdress_vox10_1300",
    "loot_vox10_1200",
    "redandblack_vox10_1550",
    "soldier_vox10_0690",
]

COORD_SCALE = 1023.0  # 10-bit voxel grid


def compute_curvature(points: np.ndarray, k: int = 30) -> np.ndarray:
    """PCA-based curvature: lambda_min / sum(lambdas) per point (vectorized)."""
    n = len(points)
    k = min(k, n - 1)
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k + 1)

    neighbors = points[indices]  # (N, k+1, 3)
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)  # (N, k+1, 3)
    cov = np.einsum("nki,nkj->nij", centered, centered) / (k + 1)  # (N, 3, 3)
    eigenvalues = np.linalg.eigvalsh(cov)  # (N, 3), sorted ascending
    total = eigenvalues.sum(axis=1).clip(min=1e-10)
    curvatures = (eigenvalues[:, 0] / total).astype(np.float32)
    return curvatures


def extract_patches(
    coords: np.ndarray,
    patch_size: int = 64,
    stride: int = 32,
    min_points: int = 100,
) -> List[np.ndarray]:
    """Extract cubic patches from a point cloud. Returns list of index arrays."""
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    patches = []
    for x in range(int(mins[0]), int(maxs[0]) + 1, stride):
        for y in range(int(mins[1]), int(maxs[1]) + 1, stride):
            for z in range(int(mins[2]), int(maxs[2]) + 1, stride):
                mask = (
                    (coords[:, 0] >= x)
                    & (coords[:, 0] < x + patch_size)
                    & (coords[:, 1] >= y)
                    & (coords[:, 1] < y + patch_size)
                    & (coords[:, 2] >= z)
                    & (coords[:, 2] < z + patch_size)
                )
                indices = np.where(mask)[0]
                if len(indices) >= min_points:
                    patches.append(indices)
    return patches


class PatchPairDataset(Dataset):
    """Dataset of (decoded patch, displacement target) pairs.

    For each decoded point, the displacement target is:
        original[nn_idx] - decoded  (where nn_idx is NN in original cloud)

    Features per point: (x/1023, y/1023, z/1023, curvature)
    """

    def __init__(
        self,
        data_root: str,
        sequences: List[str],
        rate: str = "r7",
        patch_size: int = 64,
        stride: int = 32,
        min_points: int = 100,
        curvature_k: int = 30,
    ):
        self.data_root = Path(data_root)
        self.rate = rate
        self.patch_size = patch_size
        self.patches: List[Tuple[int, np.ndarray]] = []  # (cloud_idx, indices)

        self.decoded_clouds = []
        self.displacements = []
        self.curvatures = []

        for seq in sequences:
            seq_dir = self.data_root / seq
            original = np.load(seq_dir / "original.npy").astype(np.float32)
            decoded = np.load(seq_dir / f"pcgcv2_{rate}.npy").astype(np.float32)

            # NN displacement target
            tree = cKDTree(original)
            _, nn_idx = tree.query(decoded)
            displacement = original[nn_idx] - decoded

            # Curvature on decoded cloud (forward_self signal)
            logger.info(f"Computing curvature for {seq} ({len(decoded)} points)...")
            curvature = compute_curvature(decoded, k=curvature_k)

            cloud_idx = len(self.decoded_clouds)
            self.decoded_clouds.append(decoded)
            self.displacements.append(displacement.astype(np.float32))
            self.curvatures.append(curvature)

            # Extract patches
            cloud_patches = extract_patches(decoded, patch_size, stride, min_points)
            for patch_indices in cloud_patches:
                self.patches.append((cloud_idx, patch_indices))

            logger.info(f"  {seq}: {len(decoded)} pts, {len(cloud_patches)} patches")

        logger.info(f"Total patches: {len(self.patches)}")

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        cloud_idx, point_indices = self.patches[idx]

        coords = self.decoded_clouds[cloud_idx][point_indices]
        curvature = self.curvatures[cloud_idx][point_indices]
        displacement = self.displacements[cloud_idx][point_indices]

        # Features: normalized coords + curvature
        features = np.column_stack(
            [
                coords / COORD_SCALE,
                curvature[:, np.newaxis],
            ]
        ).astype(np.float32)

        # Integer coordinates for sparse tensor
        coords_int = np.floor(coords).astype(np.int32)

        return coords_int, features, displacement, curvature

    @staticmethod
    def collate_fn(batch):
        """Collate using ME sparse_collate, also batching displacement and curvature."""
        coords_list, feats_list, disp_list, curv_list = zip(*batch)

        coords_batch, feats_batch = ME.utils.sparse_collate(
            [torch.from_numpy(c) for c in coords_list],
            [torch.from_numpy(f) for f in feats_list],
        )
        disp_batch = torch.from_numpy(np.concatenate(disp_list, axis=0))
        curv_batch = torch.from_numpy(np.concatenate(curv_list, axis=0))

        return coords_batch, feats_batch, disp_batch, curv_batch


class MultiFrameDataset(Dataset):
    """Dataset for multi-frame, multi-rate training data.

    Reads the directory layout produced by generate_training_data.py:
        data_root/
            manifest.json
            <sequence>/<frame>/original.npy
            <sequence>/<frame>/pcgcv2_<rate>.npy

    Each item is a patch from a decoded cloud with displacement targets.
    Supports filtering by sequence (for train/val split) and rate.
    """

    def __init__(
        self,
        data_root: str,
        sequences: Optional[List[str]] = None,
        rates: Optional[List[str]] = None,
        patch_size: int = 64,
        stride: int = 32,
        min_points: int = 100,
        curvature_k: int = 30,
        max_clouds: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.patches: List[Tuple[int, np.ndarray]] = []

        self.decoded_clouds: List[np.ndarray] = []
        self.displacements: List[np.ndarray] = []
        self.curvatures: List[np.ndarray] = []

        # Load manifest and filter
        manifest_path = self.data_root / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        if sequences is not None:
            manifest = [e for e in manifest if e["sequence"] in sequences]
        if rates is not None:
            manifest = [e for e in manifest if e["rate"] in rates]
        if max_clouds is not None:
            manifest = manifest[:max_clouds]

        logger.info(
            f"MultiFrameDataset: {len(manifest)} clouds "
            f"(sequences={sequences}, rates={rates})"
        )

        # Deduplicate originals: cache by (sequence, frame)
        original_cache: Dict[Tuple[str, int], np.ndarray] = {}

        for entry in tqdm(manifest, desc="Loading clouds"):
            seq, frame, rate = entry["sequence"], entry["frame"], entry["rate"]
            frame_dir = self.data_root / seq / str(frame)

            decoded_path = frame_dir / f"pcgcv2_{rate}.npy"
            if not decoded_path.exists():
                logger.warning(f"Missing decoded: {decoded_path}")
                continue

            decoded = np.load(decoded_path).astype(np.float32)

            # Try precomputed displacement and curvature
            disp_path = frame_dir / f"displacement_{rate}.npy"
            curv_path = frame_dir / f"curvature_{rate}.npy"

            if disp_path.exists() and curv_path.exists():
                displacement = np.load(disp_path)
                curvature = np.load(curv_path)
            else:
                # Fallback: compute inline
                cache_key = (seq, frame)
                if cache_key not in original_cache:
                    original_path = frame_dir / "original.npy"
                    original_cache[cache_key] = np.load(original_path).astype(
                        np.float32
                    )
                original = original_cache[cache_key]

                tree = cKDTree(original)
                _, nn_idx = tree.query(decoded)
                displacement = (original[nn_idx] - decoded).astype(np.float32)
                curvature = compute_curvature(decoded, k=curvature_k)

            cloud_idx = len(self.decoded_clouds)
            self.decoded_clouds.append(decoded)
            self.displacements.append(displacement)
            self.curvatures.append(curvature)

            # Extract patches
            cloud_patches = extract_patches(decoded, patch_size, stride, min_points)
            for patch_indices in cloud_patches:
                self.patches.append((cloud_idx, patch_indices))

        # Clear original cache to free memory
        original_cache.clear()

        logger.info(
            f"MultiFrameDataset ready: {len(self.decoded_clouds)} clouds, "
            f"{len(self.patches)} patches"
        )

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        cloud_idx, point_indices = self.patches[idx]

        coords = self.decoded_clouds[cloud_idx][point_indices]
        curvature = self.curvatures[cloud_idx][point_indices]
        displacement = self.displacements[cloud_idx][point_indices]

        features = np.column_stack(
            [
                coords / COORD_SCALE,
                curvature[:, np.newaxis],
            ]
        ).astype(np.float32)

        coords_int = np.floor(coords).astype(np.int32)

        return coords_int, features, displacement, curvature

    @staticmethod
    def collate_fn(batch):
        """Collate using ME sparse_collate, also batching displacement and curvature."""
        coords_list, feats_list, disp_list, curv_list = zip(*batch)

        coords_batch, feats_batch = ME.utils.sparse_collate(
            [torch.from_numpy(c) for c in coords_list],
            [torch.from_numpy(f) for f in feats_list],
        )
        disp_batch = torch.from_numpy(np.concatenate(disp_list, axis=0))
        curv_batch = torch.from_numpy(np.concatenate(curv_list, axis=0))

        return coords_batch, feats_batch, disp_batch, curv_batch
