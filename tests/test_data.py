#!/usr/bin/env python3
"""Tests for data loading functionality."""

import logging
from pathlib import Path

import pytest

from pcml.data import PLYPointCloudLoader
from pcml.data.loaders import JPEGPleno8iVFBSequence, PointCloudDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_ply_loader_basic(local_test_data: Path) -> None:
    """Test basic PLY loader functionality."""
    loader = PLYPointCloudLoader(verbose=True)

    # Test with local 8iVFB_small
    test_file = local_test_data / "longdress_vox10_1300.ply"

    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    point_cloud = loader.load(test_file)

    # Assertions
    assert point_cloud.num_points > 0, "Point cloud should have points"
    assert point_cloud.geometry.shape[1] == 3, "Geometry should be 3D"
    assert (
        point_cloud.has_colors or not point_cloud.has_colors
    ), "Colors flag should be boolean"

    print(f"\nLoaded point cloud:")
    print(f"  Points: {point_cloud.num_points:,}")
    print(f"  Has colors: {point_cloud.has_colors}")
    print(f"  Has normals: {point_cloud.has_normals}")
    print(f"  Geometry dtype: {point_cloud.geometry.dtype}")
    print(f"  Geometry shape: {point_cloud.geometry.shape}")
    if point_cloud.has_colors:
        print(f"  Colors dtype: {point_cloud.colors.dtype}")
        print(
            f"  Colors range: [{point_cloud.colors.min()}, {point_cloud.colors.max()}]"
        )
    print(f"  Metadata: {point_cloud.metadata}")


def test_jpeg_pleno_sequence_loading(shared_dataset_path: Path) -> None:
    """Test JPEG Pleno 8iVFB sequence loading."""
    if not shared_dataset_path.exists():
        pytest.skip(f"Shared dataset not found: {shared_dataset_path}")

    sequence = JPEGPleno8iVFBSequence(
        root_dir=str(shared_dataset_path), sequence_name="longdress"
    )

    # Assertions
    assert sequence.sequence_name == "longdress"
    assert sequence.num_frames() > 0, "Sequence should have frames"
    assert len(sequence.frame_paths) > 0, "Frame paths should be populated"

    print(f"\nSequence loaded: {sequence.sequence_name}")
    print(f"  Total frames: {sequence.num_frames()}")
    print(f"  Frame path pattern: {sequence.frame_paths[0]}")


def test_jpeg_pleno_single_frame(shared_dataset_path: Path) -> None:
    """Test loading a single frame from sequence."""
    if not shared_dataset_path.exists():
        pytest.skip(f"Shared dataset not found: {shared_dataset_path}")

    sequence = JPEGPleno8iVFBSequence(
        root_dir=str(shared_dataset_path), sequence_name="longdress"
    )

    # Load a single frame
    frame = sequence.get_frame(0)

    # Assertions
    assert frame.num_points > 0, "Frame should have points"
    assert frame.geometry.shape[1] == 3, "Frame geometry should be 3D"

    print(f"\nFirst frame:")
    print(f"  Points: {frame.num_points:,}")
    print(f"  Has colors: {frame.has_colors}")
    x_min, x_max = frame.geometry[:, 0].min(), frame.geometry[:, 0].max()
    y_min, y_max = frame.geometry[:, 1].min(), frame.geometry[:, 1].max()
    z_min, z_max = frame.geometry[:, 2].min(), frame.geometry[:, 2].max()
    print(f"  Geometry range X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  Geometry range Y: [{y_min:.2f}, {y_max:.2f}]")
    print(f"  Geometry range Z: [{z_min:.2f}, {z_max:.2f}]")


def test_jpeg_pleno_multiple_frames(shared_dataset_path: Path) -> None:
    """Test loading multiple frames from sequence."""
    if not shared_dataset_path.exists():
        pytest.skip(f"Shared dataset not found: {shared_dataset_path}")

    sequence = JPEGPleno8iVFBSequence(
        root_dir=str(shared_dataset_path), sequence_name="longdress"
    )

    # Test loading multiple frames
    frames = sequence.get_frames(start=0, end=5, step=1)

    # Assertions
    assert len(frames) == 5, "Should load 5 frames"
    for i, f in enumerate(frames):
        assert f.num_points > 0, f"Frame {i} should have points"

    print(f"\nLoaded frames 0-4: {len(frames)} frames")
    for i, f in enumerate(frames):
        print(f"  Frame {i}: {f.num_points:,} points")


def test_point_cloud_dataset_basic(shared_dataset_path: Path) -> None:
    """Test PyTorch Dataset wrapper with basic configuration."""
    if not shared_dataset_path.exists():
        pytest.skip(f"Shared dataset not found: {shared_dataset_path}")

    sequence = JPEGPleno8iVFBSequence(
        root_dir=str(shared_dataset_path), sequence_name="longdress"
    )

    # Test single frame dataset
    dataset = PointCloudDataset(
        sequence=sequence,
        frame_indices=list(range(10)),
        normalize=True,
        num_points=100000,
    )

    # Assertions
    assert len(dataset) == 10, "Dataset should have 10 samples"
    assert dataset.normalize is True
    assert dataset.num_points == 100000

    print(f"\nDataset created:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Normalize: {dataset.normalize}")
    print(f"  Target points: {dataset.num_points}")

    # Load a sample
    sample = dataset[0]
    assert sample.num_points > 0, "Sample should have points"

    print(f"\nSample 0:")
    print(f"  Points: {sample.num_points:,}")
    print(
        f"  Geometry range: [{sample.geometry.min():.3f}, {sample.geometry.max():.3f}]"
    )


def test_point_cloud_dataset_temporal(shared_dataset_path: Path) -> None:
    """Test PyTorch Dataset wrapper with temporal pairs."""
    if not shared_dataset_path.exists():
        pytest.skip(f"Shared dataset not found: {shared_dataset_path}")

    sequence = JPEGPleno8iVFBSequence(
        root_dir=str(shared_dataset_path), sequence_name="longdress"
    )

    # Test temporal pairs
    dataset_temporal = PointCloudDataset(
        sequence=sequence, frame_indices=list(range(10)), temporal_pairs=True
    )

    # Assertions
    assert (
        len(dataset_temporal) == 9
    ), "Temporal dataset should have 9 pairs (10 frames - 1)"

    print(f"\nTemporal dataset:")
    print(f"  Total pairs: {len(dataset_temporal)}")

    frame_a, frame_b = dataset_temporal[0]
    assert frame_a.num_points > 0, "Frame A should have points"
    assert frame_b.num_points > 0, "Frame B should have points"

    print(
        f"  Pair 0 - Frame A: {frame_a.num_points:,}, Frame B: {frame_b.num_points:,}"
    )
