"""Point cloud data loaders for PLY and other formats.

This module provides efficient data loading utilities for various point cloud formats,
with support for the JPEG Pleno 8iVFB dynamic sequences.

References:
    8i Voxelized Full Bodies: d'Eon et al., "8i Voxelized Full Bodies - A Voxelized
        Point Cloud Dataset," ISO/IEC JTC1/SC29 2017.
    Open3D: Zhou et al., "Open3D: A Modern Library for 3D Data Processing,"
        arXiv:1801.09847, 2018.
    PLY format: Turk, "The PLY Polygon File Format," 1994.

See .docs/BIBLIOGRAPHY.md for full citations.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class PointCloudData:
    """Container for point cloud data with metadata."""

    geometry: np.ndarray  # (N, 3) float32 - X, Y, Z coordinates
    colors: Optional[np.ndarray] = None  # (N, 3) uint8 or float32 - R, G, B values
    normals: Optional[np.ndarray] = None  # (N, 3) float32 - Normal vectors
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata

    def __post_init__(self) -> None:
        """Validate data consistency."""
        if self.geometry.ndim != 2 or self.geometry.shape[1] != 3:
            raise ValueError(f"Geometry must be (N, 3), got {self.geometry.shape}")

        if self.colors is not None:
            if self.geometry.shape[0] != self.colors.shape[0]:
                raise ValueError(
                    f"Geometry and colors must have same point count: "
                    f"{self.geometry.shape[0]} vs {self.colors.shape[0]}"
                )

        if self.normals is not None:
            if self.geometry.shape[0] != self.normals.shape[0]:
                raise ValueError(
                    f"Geometry and normals must have same point count: "
                    f"{self.geometry.shape[0]} vs {self.normals.shape[0]}"
                )

    @property
    def num_points(self) -> int:
        """Return number of points."""
        return self.geometry.shape[0]

    @property
    def has_colors(self) -> bool:
        """Check if color data is available."""
        return self.colors is not None

    @property
    def has_normals(self) -> bool:
        """Check if normal data is available."""
        return self.normals is not None

    def to_open3d_pcd(self) -> o3d.geometry.PointCloud:
        """Convert to Open3D point cloud format."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.geometry)

        if self.has_colors:
            # Normalize colors to [0, 1] if they're uint8
            colors = self.colors.astype(np.float32)
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        if self.has_normals:
            pcd.normals = o3d.utility.Vector3dVector(self.normals)

        return pcd


class PLYPointCloudLoader:
    """Loader for PLY point cloud files."""

    def __init__(self, verbose: bool = False) -> None:
        """
        Initialize PLY loader.

        Args:
            verbose: Enable verbose logging.
        """
        self.verbose = verbose

    def load(self, filepath: Union[str, Path]) -> PointCloudData:
        """
        Load a point cloud from a PLY file.

        Args:
            filepath: Path to the PLY file.

        Returns:
            PointCloudData object containing geometry, colors, and metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the PLY file is malformed or unsupported.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"PLY file not found: {filepath}")

        if self.verbose:
            logger.info(f"Loading PLY file: {filepath}")

        try:
            # Load using Open3D for robustness
            pcd = o3d.io.read_point_cloud(str(filepath))

            # Extract geometry
            geometry = np.asarray(pcd.points, dtype=np.float32)

            # Extract colors if available
            colors: Optional[np.ndarray] = None
            if len(pcd.colors) > 0:
                colors = np.asarray(pcd.colors, dtype=np.float32)
                # Convert to [0, 255] range if in [0, 1]
                if colors.max() <= 1.0:
                    colors = (colors * 255).astype(np.uint8)
                else:
                    colors = colors.astype(np.uint8)

            # Extract normals if available
            normals: Optional[np.ndarray] = None
            if len(pcd.normals) > 0:
                normals = np.asarray(pcd.normals, dtype=np.float32)

            # Parse metadata from comment lines (specific to 8iVFB format)
            metadata = self._extract_ply_metadata(filepath)

            if self.verbose:
                logger.info(
                    f"  Points: {geometry.shape[0]:,} | "
                    f"Has colors: {colors is not None} | "
                    f"Has normals: {normals is not None}"
                )

            return PointCloudData(
                geometry=geometry, colors=colors, normals=normals, metadata=metadata
            )

        except Exception as e:
            raise ValueError(f"Failed to load PLY file {filepath}: {str(e)}")

    def _extract_ply_metadata(self, filepath: Path) -> Dict[str, Any]:
        """
        Extract metadata from PLY header comments (8iVFB specific).

        Args:
            filepath: Path to the PLY file.

        Returns:
            Dictionary containing extracted metadata.
        """
        metadata: Dict[str, Any] = {"filename": filepath.name}

        try:
            with open(filepath, "r") as f:
                for line in f:
                    if line.startswith("comment"):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            key = parts[1]
                            value_str = " ".join(parts[2:])
                            try:
                                metadata[key] = float(value_str)
                            except ValueError:
                                metadata[key] = value_str
                    elif line.startswith("end_header"):
                        break
        except Exception as e:
            logger.warning(f"Could not extract metadata from {filepath}: {str(e)}")

        return metadata


class JPEGPleno8iVFBSequence:
    """Handler for JPEG Pleno 8iVFB dynamic sequences."""

    SEQUENCE_NAMES = ["longdress", "loot", "redandblack", "soldier"]
    FRAMES_PER_SEQUENCE = 300

    def __init__(self, root_dir: Union[str, Path], sequence_name: str) -> None:
        """
        Initialize sequence handler.

        Args:
            root_dir: Root directory of JPEG Pleno dataset.
            sequence_name: Name of the sequence (one of SEQUENCE_NAMES).

        Raises:
            ValueError: If sequence_name is invalid.
            FileNotFoundError: If sequence directory not found.
        """
        if sequence_name not in self.SEQUENCE_NAMES:
            raise ValueError(
                f"Invalid sequence name: {sequence_name}. "
                f"Must be one of {self.SEQUENCE_NAMES}"
            )

        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.sequence_dir = self.root_dir / sequence_name / "Ply"

        if not self.sequence_dir.exists():
            raise FileNotFoundError(
                f"Sequence directory not found: {self.sequence_dir}"
            )

        self.loader = PLYPointCloudLoader(verbose=False)
        self._frame_list: Optional[List[Path]] = None

    @property
    def frame_paths(self) -> List[Path]:
        """Get sorted list of frame file paths."""
        if self._frame_list is None:
            ply_files = sorted(self.sequence_dir.glob("*.ply"))
            if not ply_files:
                raise FileNotFoundError(f"No PLY files found in {self.sequence_dir}")
            self._frame_list = ply_files
        return self._frame_list

    def get_frame(self, frame_idx: int) -> PointCloudData:
        """
        Load a specific frame from the sequence.

        Args:
            frame_idx: Frame index (0-based).

        Returns:
            PointCloudData for the requested frame.

        Raises:
            IndexError: If frame_idx is out of bounds.
        """
        if frame_idx < 0 or frame_idx >= len(self.frame_paths):
            raise IndexError(
                f"Frame index out of bounds: {frame_idx} "
                f"(sequence has {len(self.frame_paths)} frames)"
            )

        return self.loader.load(self.frame_paths[frame_idx])

    def get_frames(
        self, start: int = 0, end: Optional[int] = None, step: int = 1
    ) -> List[PointCloudData]:
        """
        Load a range of frames from the sequence.

        Args:
            start: Starting frame index.
            end: Ending frame index (exclusive). If None, use sequence length.
            step: Frame sampling step.

        Returns:
            List of PointCloudData objects.
        """
        if end is None:
            end = len(self.frame_paths)

        frames = []
        for idx in range(start, min(end, len(self.frame_paths)), step):
            frames.append(self.get_frame(idx))
        return frames

    def num_frames(self) -> int:
        """Return total number of frames in sequence."""
        return len(self.frame_paths)


class PointCloudDataset(Dataset):
    """
    PyTorch Dataset for point cloud sequences.

    Supports loading individual frames or frame pairs for temporal compression tasks.
    """

    def __init__(
        self,
        sequence: JPEGPleno8iVFBSequence,
        frame_indices: Optional[List[int]] = None,
        temporal_pairs: bool = False,
        normalize: bool = False,
        num_points: Optional[int] = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            sequence: JPEGPleno8iVFBSequence object.
            frame_indices: Specific frame indices to use. If None, use all frames.
            temporal_pairs: If True, return (frame_t, frame_t+1) pairs.
            normalize: Normalize point coordinates to [-1, 1].
            num_points: Target number of points (via random sampling/padding).

        Raises:
            ValueError: If temporal_pairs is True but fewer than 2 frames available.
        """
        self.sequence = sequence
        self.temporal_pairs = temporal_pairs
        self.normalize = normalize
        self.num_points = num_points

        if frame_indices is None:
            self.frame_indices = list(range(self.sequence.num_frames()))
        else:
            self.frame_indices = frame_indices

        if temporal_pairs and len(self.frame_indices) < 2:
            raise ValueError(
                "Need at least 2 frames for temporal pairs. "
                f"Got {len(self.frame_indices)}"
            )

        self._frame_cache: Dict[int, PointCloudData] = {}

    def __len__(self) -> int:
        """Return dataset length."""
        if self.temporal_pairs:
            return len(self.frame_indices) - 1
        return len(self.frame_indices)

    def __getitem__(
        self, idx: int
    ) -> Union[PointCloudData, Tuple[PointCloudData, PointCloudData]]:
        """
        Get a point cloud or frame pair.

        Args:
            idx: Index in the dataset.

        Returns:
            Single PointCloudData or tuple of (current_frame, next_frame).
        """
        if self.temporal_pairs:
            frame_a = self._load_and_process_frame(self.frame_indices[idx])
            frame_b = self._load_and_process_frame(self.frame_indices[idx + 1])
            return (frame_a, frame_b)
        else:
            return self._load_and_process_frame(self.frame_indices[idx])

    def _load_and_process_frame(self, frame_idx: int) -> PointCloudData:
        """
        Load and process a frame with optional normalization and sampling.

        Args:
            frame_idx: Index of the frame to load.

        Returns:
            Processed PointCloudData.
        """
        # Check cache
        if frame_idx in self._frame_cache:
            return self._frame_cache[frame_idx]

        # Load frame
        frame = self.sequence.get_frame(frame_idx)

        # Normalize coordinates
        if self.normalize:
            # Compute bounding box
            min_coords = frame.geometry.min(axis=0)
            max_coords = frame.geometry.max(axis=0)
            center = (min_coords + max_coords) / 2
            scale = (max_coords - min_coords).max() / 2
            scale = max(scale, 1e-6)  # Avoid division by zero

            normalized_geometry = (frame.geometry - center) / scale
            frame = PointCloudData(
                geometry=normalized_geometry,
                colors=frame.colors,
                normals=frame.normals,
                metadata=frame.metadata,
            )

        # Resample points
        if self.num_points is not None and frame.num_points != self.num_points:
            frame = self._resample_points(frame, self.num_points)

        # Cache
        self._frame_cache[frame_idx] = frame

        return frame

    @staticmethod
    def _resample_points(
        point_cloud: PointCloudData, target_num_points: int
    ) -> PointCloudData:
        """
        Resample point cloud to target number of points.

        Uses random sampling if target is smaller, or random duplication if larger.

        Args:
            point_cloud: Input point cloud.
            target_num_points: Target number of points.

        Returns:
            Resampled PointCloudData.
        """
        current_num_points = point_cloud.num_points

        if current_num_points == target_num_points:
            return point_cloud

        indices = np.random.choice(
            current_num_points, size=target_num_points, replace=True
        )

        geometry = point_cloud.geometry[indices]
        colors = point_cloud.colors[indices] if point_cloud.has_colors else None
        normals = point_cloud.normals[indices] if point_cloud.has_normals else None

        return PointCloudData(
            geometry=geometry,
            colors=colors,
            normals=normals,
            metadata=point_cloud.metadata,
        )

    def clear_cache(self) -> None:
        """Clear the frame cache to free memory."""
        self._frame_cache.clear()
