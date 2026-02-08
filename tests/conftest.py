"""
Pytest configuration and shared fixtures for PCML tests.
"""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def datasets_dir(project_root: Path) -> Path:
    """Return the datasets directory."""
    return project_root / "datasets"


@pytest.fixture
def local_test_data(datasets_dir: Path) -> Path:
    """Return path to local test data (8iVFB_small)."""
    return datasets_dir / "8iVFB_small"


@pytest.fixture
def shared_dataset_path() -> Path:
    """Return path to shared JPEG Pleno dataset."""
    return Path("/mnt/shared-dataset/jpeg-pleno")


@pytest.fixture
def sample_point_cloud() -> np.ndarray:
    """Create a sample point cloud for testing."""
    np.random.seed(42)
    return np.random.randn(1000, 3).astype(np.float32)


@pytest.fixture
def sample_colors() -> np.ndarray:
    """Create sample color data for testing."""
    np.random.seed(42)
    return np.random.randint(0, 256, (1000, 3), dtype=np.uint8)


@pytest.fixture
def sample_point_cloud_with_noise(sample_point_cloud: np.ndarray) -> np.ndarray:
    """Create a noisy version of sample point cloud."""
    np.random.seed(43)
    return sample_point_cloud + np.random.randn(1000, 3).astype(np.float32) * 0.1


@pytest.fixture
def sample_colors_with_noise(sample_colors: np.ndarray) -> np.ndarray:
    """Create a noisy version of sample colors."""
    np.random.seed(43)
    noisy = sample_colors.astype(np.int16) + np.random.randint(-10, 10, (1000, 3))
    return np.clip(noisy, 0, 255).astype(np.uint8)
