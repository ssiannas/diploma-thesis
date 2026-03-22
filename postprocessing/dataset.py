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
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None

logger = logging.getLogger(__name__)

# 26-connectivity offsets for voxel dilation (all neighbors except self)
_OFFSETS_26 = np.array(
    [
        [dx, dy, dz]
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if dx or dy or dz
    ],
    dtype=np.int32,
)  # (26, 3)


def _dilate_and_label(
    decoded_int: np.ndarray,  # (N, 3) int32, decoded voxels after aug
    orig_int: np.ndarray,  # (M, 3) int32, original voxels (GT) after aug
) -> tuple:
    """Dilate decoded voxels by 1, label each candidate by GT occupancy.

    Returns:
        all_coords: (N+K, 3) int32 -- decoded voxels then unique empty neighbors
        features:   (N+K, 4) float32 -- [x/S, y/S, z/S, is_occupied]
        labels:     (N+K,)   float32 -- 1 if coord in GT set, else 0
    """
    # All 26-neighbors of decoded voxels
    all_neighbors = (decoded_int[:, None, :] + _OFFSETS_26[None, :, :]).reshape(-1, 3)

    occupied_set = {tuple(c) for c in decoded_int.tolist()}
    # Empty candidates: neighbors not already occupied
    empty = np.array(
        [c for c in all_neighbors.tolist() if tuple(c) not in occupied_set],
        dtype=np.int32,
    )
    if len(empty) > 0:
        empty = np.unique(empty, axis=0)
        all_coords = np.vstack([decoded_int, empty])
    else:
        all_coords = decoded_int

    n_dec = len(decoded_int)

    gt_set = {tuple(c) for c in orig_int.tolist()}
    labels = np.array(
        [float(tuple(c) in gt_set) for c in all_coords.tolist()], dtype=np.float32
    )

    is_occ = np.zeros(len(all_coords), dtype=np.float32)
    is_occ[:n_dec] = 1.0
    features = np.column_stack([all_coords.astype(np.float32) / COORD_SCALE, is_occ])

    return all_coords, features, labels


SEQUENCES = [
    "longdress_vox10_1300",
    "loot_vox10_1200",
    "redandblack_vox10_1550",
    "soldier_vox10_0690",
]

COORD_SCALE = 1023.0  # 10-bit voxel grid

# BPP values from PCGCv2 checkpoint names (bits per point)
RATE_BPP = {
    "r1": 0.025,
    "r2": 0.05,
    "r3": 0.10,
    "r4": 0.15,
    "r5": 0.25,
    "r6": 0.30,
    "r7": 0.40,
}


def bpp_to_log(bpp: float) -> float:
    """Convert raw BPP to log-space for model conditioning.

    log(bpp) maps [0.025, 0.40] to [-3.69, -0.92], giving better spread
    across rates and natural handling of unseen values.
    """
    return float(np.log(bpp))


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
    """Extract cubic patches from a point cloud. Returns list of index arrays.

    Uses spatial binning for O(N) instead of O(N * V) complexity.
    """
    from collections import defaultdict

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    # Bin points into grid cells of size `stride`
    cell_idx = np.floor((coords - mins) / stride).astype(np.int32)
    cell_points: Dict[tuple, List[int]] = defaultdict(list)
    for i, c in enumerate(cell_idx):
        cell_points[(c[0], c[1], c[2])].append(i)

    # How many cells each patch spans per dimension
    cells_per_patch = int(np.ceil(patch_size / stride))
    n_cells = np.floor((maxs - mins) / stride).astype(int) + 1

    patches = []
    for cx in range(n_cells[0]):
        for cy in range(n_cells[1]):
            for cz in range(n_cells[2]):
                # Gather point indices from all cells this patch covers
                candidates = []
                for dx in range(cells_per_patch):
                    for dy in range(cells_per_patch):
                        for dz in range(cells_per_patch):
                            key = (cx + dx, cy + dy, cz + dz)
                            if key in cell_points:
                                candidates.extend(cell_points[key])
                if len(candidates) < min_points:
                    continue
                # Exact bounds check for non-aligned edges
                origin = mins + np.array([cx, cy, cz]) * stride
                pts = coords[candidates]
                mask = np.all((pts >= origin) & (pts < origin + patch_size), axis=1)
                valid = np.array(candidates)[mask]
                if len(valid) >= min_points:
                    patches.append(valid)
    return patches


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
        augment: bool = False,
        frame_stride: int = 1,
        use_chamfer: bool = False,
        chamfer_padding: int = 10,
        rate_jitter: float = 0.0,
        use_occupancy: bool = False,
    ):
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.rate_jitter = rate_jitter
        self.augment = augment
        self.use_occupancy = use_occupancy
        # Occupancy mode requires original clouds (same as chamfer)
        self.use_chamfer = use_chamfer or use_occupancy
        self.chamfer_padding = 2 if use_occupancy else chamfer_padding
        self.patches: List[Tuple[int, np.ndarray]] = []
        self.patch_rates: List[str] = []  # rate label per patch

        self._epoch: int = 0
        self.decoded_clouds: List[np.ndarray] = []
        self.original_clouds: List[np.ndarray] = []  # only populated when use_chamfer
        self.displacements: List[np.ndarray] = []
        self.curvatures: List[np.ndarray] = []
        self.cloud_metadata: List[Tuple[str, int]] = []  # (sequence, frame) per cloud
        self.patch_origins: List[Tuple[int, int, int]] = []  # grid origin per patch

        # Load manifest and filter
        manifest_path = self.data_root / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        if sequences is not None:
            manifest = [e for e in manifest if e["sequence"] in sequences]
        if rates is not None:
            manifest = [e for e in manifest if e["rate"] in rates]
        # Subsample frames to reduce temporal redundancy
        if frame_stride > 1:
            from collections import defaultdict

            groups = defaultdict(list)
            for e in manifest:
                groups[(e["sequence"], e["rate"])].append(e)
            subsampled = []
            for key, entries in groups.items():
                entries.sort(key=lambda e: e["frame"])
                subsampled.extend(entries[::frame_stride])
            manifest = subsampled

        if max_clouds is not None:
            # Apply limit per rate to avoid truncating to a single rate
            from collections import defaultdict as _dd

            rate_groups = _dd(list)
            for e in manifest:
                rate_groups[e["rate"]].append(e)
            per_rate = max(1, max_clouds // max(len(rate_groups), 1))
            manifest = []
            for entries in rate_groups.values():
                manifest.extend(entries[:per_rate])

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

            # Load original cloud (needed for chamfer mode or inline displacement)
            cache_key = (seq, frame)
            if cache_key not in original_cache:
                original_path = frame_dir / "original.npy"
                if original_path.exists():
                    original_cache[cache_key] = np.load(original_path).astype(
                        np.float32
                    )

            # Curvature
            curv_path = frame_dir / f"curvature_{rate}.npy"
            if curv_path.exists():
                curvature = np.load(curv_path)
            else:
                curvature = compute_curvature(decoded, k=curvature_k)

            if self.use_chamfer:
                # No displacement needed -- store dummy, original stored separately
                displacement = np.zeros((len(decoded), 3), dtype=np.float32)
            else:
                disp_path = frame_dir / f"displacement_{rate}.npy"
                if disp_path.exists():
                    displacement = np.load(disp_path)
                elif cache_key in original_cache:
                    original = original_cache[cache_key]
                    tree = cKDTree(original)
                    _, nn_idx = tree.query(decoded)
                    displacement = (original[nn_idx] - decoded).astype(np.float32)
                else:
                    logger.warning(f"No displacement source for {frame_dir}")
                    continue

            cloud_idx = len(self.decoded_clouds)
            self.decoded_clouds.append(decoded)
            self.displacements.append(displacement)
            self.curvatures.append(curvature)
            self.cloud_metadata.append((seq, frame))

            if self.use_chamfer and cache_key in original_cache:
                self.original_clouds.append(original_cache[cache_key])
            elif self.use_chamfer:
                logger.warning(f"No original for chamfer: {frame_dir}")
                continue

            # Extract patches
            cloud_patches = extract_patches(decoded, patch_size, stride, min_points)
            cloud_mins = decoded.min(axis=0)
            for patch_indices in cloud_patches:
                self.patches.append((cloud_idx, patch_indices))
                self.patch_rates.append(rate)
                # Compute grid origin: quantized min of patch points
                patch_min = decoded[patch_indices].min(axis=0)
                grid_origin = np.floor((patch_min - cloud_mins) / stride).astype(int)
                self.patch_origins.append(tuple(grid_origin))

        # Clear original cache to free memory (keep originals in self if chamfer)
        original_cache.clear()

        logger.info(
            f"MultiFrameDataset ready: {len(self.decoded_clouds)} clouds, "
            f"{len(self.patches)} patches"
        )

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic augmentation seeding."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        cloud_idx, point_indices = self.patches[idx]

        coords = self.decoded_clouds[cloud_idx][point_indices].copy()
        curvature = self.curvatures[cloud_idx][point_indices]
        displacement = self.displacements[cloud_idx][point_indices].copy()

        # Crop original points BEFORE augmentation (coords are in original space)
        orig_crop = None
        if self.use_chamfer:
            orig_cloud = self.original_clouds[cloud_idx]
            pad = self.chamfer_padding
            mins = coords.min(axis=0) - pad
            maxs = coords.max(axis=0) + pad
            mask = np.all((orig_cloud >= mins) & (orig_cloud <= maxs), axis=1)
            orig_crop = orig_cloud[mask].astype(np.float32)

        # Deterministic rigid augmentation: seeded by spatial identity (not rate)
        if self.augment:
            seq, frame = self.cloud_metadata[cloud_idx]
            origin = self.patch_origins[idx]
            seed = hash((seq, frame, origin, self._epoch)) & 0xFFFFFFFF
            rng = np.random.RandomState(seed)
            axes = rng.permutation(3)
            signs = rng.choice([-1, 1], size=3).astype(np.float32)
            coords = coords[:, axes] * signs
            displacement = displacement[:, axes] * signs
            if orig_crop is not None:
                orig_crop = orig_crop[:, axes] * signs

        features = np.column_stack(
            [
                coords / COORD_SCALE,
                curvature[:, np.newaxis],
            ]
        ).astype(np.float32)

        coords_int = np.floor(coords).astype(np.int32)

        log_bpp = bpp_to_log(RATE_BPP[self.patch_rates[idx]])
        if self.rate_jitter > 0 and self.augment:
            log_bpp += np.random.normal(0, self.rate_jitter)
        rate_index = log_bpp

        if self.use_occupancy:
            decoded_int = np.floor(coords).astype(np.int32)
            orig_int = np.floor(orig_crop).astype(np.int32)
            all_coords_int, occ_feats, labels = _dilate_and_label(decoded_int, orig_int)
            dummy_curv = np.zeros(len(all_coords_int), dtype=np.float32)
            return all_coords_int, occ_feats, labels, dummy_curv, rate_index

        if self.use_chamfer:
            return coords_int, features, orig_crop, curvature, rate_index

        return coords_int, features, displacement, curvature, rate_index

    @staticmethod
    def collate_fn(batch):
        """Collate using ME sparse_collate, also batching displacement and curvature."""
        coords_list, feats_list, disp_list, curv_list, rate_list = zip(*batch)

        coords_batch, feats_batch = ME.utils.sparse_collate(
            [torch.from_numpy(c) for c in coords_list],
            [torch.from_numpy(f) for f in feats_list],
        )
        disp_batch = torch.from_numpy(np.concatenate(disp_list, axis=0))
        curv_batch = torch.from_numpy(np.concatenate(curv_list, axis=0))
        rate_batch = torch.tensor(rate_list, dtype=torch.float32)

        return coords_batch, feats_batch, disp_batch, curv_batch, rate_batch

    @staticmethod
    def collate_fn_occupancy(batch):
        """Collate for occupancy mode.

        Labels are packed as an extra feature channel before ME.sparse_collate
        so that ME's coordinate reordering keeps labels aligned with features.
        Returns (coords, feats, labels, rates) -- no curvature.
        """
        coords_list, feats_list, labels_list, _, rate_list = zip(*batch)

        # Pack labels as last feature so ME sorts them consistently with feats
        feats_with_label = [
            np.column_stack([f, lbl[:, np.newaxis]])
            for f, lbl in zip(feats_list, labels_list)
        ]
        coords_batch, feats_label_batch = ME.utils.sparse_collate(
            [torch.from_numpy(c) for c in coords_list],
            [torch.from_numpy(f) for f in feats_with_label],
        )
        feats_batch = feats_label_batch[:, :-1]  # (N_total, 4)
        labels_batch = feats_label_batch[:, -1]  # (N_total,)
        rate_batch = torch.tensor(rate_list, dtype=torch.float32)
        return coords_batch, feats_batch, labels_batch, rate_batch

    @staticmethod
    def collate_fn_chamfer(batch):
        """Collate for chamfer mode: orig_coords kept as list (variable size)."""
        coords_list, feats_list, orig_list, curv_list, rate_list = zip(*batch)

        coords_batch, feats_batch = ME.utils.sparse_collate(
            [torch.from_numpy(c) for c in coords_list],
            [torch.from_numpy(f) for f in feats_list],
        )
        # orig_coords are variable size -- keep as list of tensors
        orig_batch = [torch.from_numpy(o) for o in orig_list]
        curv_batch = torch.from_numpy(np.concatenate(curv_list, axis=0))
        rate_batch = torch.tensor(rate_list, dtype=torch.float32)

        return coords_batch, feats_batch, orig_batch, curv_batch, rate_batch


class RateBalancedBatchSampler(Sampler):
    """Yields mixed-rate batches with equal patches per rate.

    Each batch has batch_size // num_rates patches from each rate.
    The larger rate group is undersampled to match the smaller group per epoch.
    """

    def __init__(
        self, patch_rates: List[str], batch_size: int, drop_last: bool = False
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.rate_indices: Dict[str, List[int]] = {}
        for idx, rate in enumerate(patch_rates):
            self.rate_indices.setdefault(rate, []).append(idx)

        self.rates = sorted(self.rate_indices.keys())
        self.num_rates = len(self.rates)
        self.per_rate = batch_size // self.num_rates

        # Undersample to smallest rate group
        self.epoch_size = min(len(v) for v in self.rate_indices.values())
        self._n_batches = self.epoch_size // self.per_rate
        if not drop_last and self.epoch_size % self.per_rate != 0:
            self._n_batches += 1

    def __iter__(self):
        # Shuffle and truncate each rate to epoch_size
        rate_pools = {}
        for rate in self.rates:
            indices = self.rate_indices[rate].copy()
            np.random.shuffle(indices)
            rate_pools[rate] = indices[: self.epoch_size]

        # Zip rates together into mixed batches
        for i in range(0, self.epoch_size, self.per_rate):
            batch = []
            for rate in self.rates:
                batch.extend(rate_pools[rate][i : i + self.per_rate])
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch

    def __len__(self):
        return self._n_batches


class RateStratifiedBatchSampler(Sampler):
    """Yields batches where all patches share the same compression rate.

    Each epoch: shuffle rates, then for each rate shuffle its patches and
    yield fixed-size batches. Ensures no gradient conflict across rates.
    """

    def __init__(
        self, patch_rates: List[str], batch_size: int, drop_last: bool = False
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group patch indices by rate
        self.rate_indices: Dict[str, List[int]] = {}
        for idx, rate in enumerate(patch_rates):
            self.rate_indices.setdefault(rate, []).append(idx)

        # Total batches for __len__
        self._n_batches = 0
        for indices in self.rate_indices.values():
            n = (
                len(indices) // batch_size
                if drop_last
                else (len(indices) + batch_size - 1) // batch_size
            )
            self._n_batches += n

    def __iter__(self):
        rates = list(self.rate_indices.keys())
        np.random.shuffle(rates)

        for rate in rates:
            indices = self.rate_indices[rate].copy()
            np.random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self):
        return self._n_batches
