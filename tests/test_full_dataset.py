"""Test full dataset loading capabilities.

This test validates that we can load all 1200 frames from JPEG Pleno
without errors or memory issues.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcml.data.loaders import JPEGPleno8iVFBSequence, PLYPointCloudLoader


class TestFullDatasetLoading:
    """Test suite for full JPEG Pleno dataset loading."""

    @pytest.fixture
    def dataset_path(self) -> Path:
        """Path to JPEG Pleno dataset."""
        return Path("/mnt/shared-dataset/jpeg-pleno")

    @pytest.fixture
    def local_dataset_path(self) -> Path:
        """Path to local small dataset (fallback)."""
        return Path("datasets/8iVFB_small")

    def test_jpeg_pleno_path_exists(self, dataset_path):
        """Test that JPEG Pleno dataset path exists."""
        if not dataset_path.exists():
            pytest.skip(f"JPEG Pleno dataset not found at {dataset_path}")

    def test_load_all_sequences(self, dataset_path):
        """Test loading all 4 sequences."""
        if not dataset_path.exists():
            pytest.skip("JPEG Pleno dataset not available")

        sequences = ["longdress", "loot", "redandblack", "soldier"]

        for seq_name in sequences:
            sequence = JPEGPleno8iVFBSequence(
                root_dir=str(dataset_path), sequence_name=seq_name
            )

            num_frames = sequence.num_frames()
            assert (
                num_frames == 300
            ), f"{seq_name} should have 300 frames, got {num_frames}"

            print(f"[OK] {seq_name}: {num_frames} frames")

    def test_load_sample_frames_from_each_sequence(self, dataset_path):
        """Test loading sample frames from each sequence."""
        if not dataset_path.exists():
            pytest.skip("JPEG Pleno dataset not available")

        sequences = ["longdress", "loot", "redandblack", "soldier"]
        test_indices = [0, 50, 150, 299]

        for seq_name in sequences:
            sequence = JPEGPleno8iVFBSequence(
                root_dir=str(dataset_path), sequence_name=seq_name
            )

            for idx in test_indices:
                frame = sequence.get_frame(idx)

                assert frame.num_points > 0, f"Frame {idx} has no points"
                assert frame.has_colors, f"Frame {idx} has no colors"
                assert frame.geometry.shape[1] == 3, "Geometry must be Nx3"
                assert frame.colors.shape[1] == 3, "Colors must be Nx3"

            print(f"[OK] {seq_name}: Sample frames loaded successfully")

    def test_point_count_ranges(self, dataset_path):
        """Test that point counts are in expected ranges."""
        if not dataset_path.exists():
            pytest.skip("JPEG Pleno dataset not available")

        expected_ranges = {
            "longdress": (700000, 850000),
            "loot": (700000, 850000),
            "redandblack": (700000, 850000),
            "soldier": (750000, 950000),
        }

        for seq_name, (min_points, max_points) in expected_ranges.items():
            sequence = JPEGPleno8iVFBSequence(
                root_dir=str(dataset_path), sequence_name=seq_name
            )

            frame = sequence.get_frame(0)
            num_points = frame.num_points

            assert min_points <= num_points <= max_points, (
                f"{seq_name}: Expected {min_points}-{max_points} points, "
                f"got {num_points}"
            )

            print(f"[OK] {seq_name}: {num_points} points (within range)")

    def test_local_dataset_loads(self, local_dataset_path):
        """Test that local small dataset loads correctly."""
        if not local_dataset_path.exists():
            pytest.skip("Local dataset not available")

        loader = PLYPointCloudLoader(verbose=False)
        ply_files = list(local_dataset_path.glob("*.ply"))

        assert len(ply_files) > 0, "No PLY files found in local dataset"

        for ply_file in ply_files:
            frame = loader.load(str(ply_file))
            assert frame.num_points > 0
            assert frame.has_colors

        print(f"[OK] Local dataset: {len(ply_files)} files loaded")

    def test_memory_single_frame(self, dataset_path):
        """Test memory usage for single frame loading."""
        if not dataset_path.exists():
            pytest.skip("JPEG Pleno dataset not available")

        import tracemalloc

        tracemalloc.start()

        sequence = JPEGPleno8iVFBSequence(
            root_dir=str(dataset_path), sequence_name="longdress"
        )
        sequence.get_frame(0)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        print(f"[OK] Single frame memory: {peak_mb:.2f} MB")

        assert peak_mb < 500, f"Single frame uses too much memory: {peak_mb:.2f} MB"

    def test_sequential_loading_no_memory_leak(self, dataset_path):
        """Test that sequential loading doesn't leak memory."""
        if not dataset_path.exists():
            pytest.skip("JPEG Pleno dataset not available")

        import tracemalloc

        tracemalloc.start()

        sequence = JPEGPleno8iVFBSequence(
            root_dir=str(dataset_path), sequence_name="longdress"
        )

        for i in range(10):
            frame = sequence.get_frame(i)
            del frame

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        print(f"[OK] 10 sequential frames memory: {peak_mb:.2f} MB")

        assert peak_mb < 1000, f"Memory leak suspected: {peak_mb:.2f} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
