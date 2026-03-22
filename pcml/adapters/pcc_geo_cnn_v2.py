"""Adapter for pcc_geo_cnn_v2 framework (Quach et al., 2020)."""

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseAdapter, CompressionResult


@dataclass
class PCCGeoCNNConfig:
    """Configuration for pcc_geo_cnn_v2."""

    model_name: str = "c3"  # c1-c6 (low to high quality)
    resolution: int = 1024
    octree_level: int = 3
    model_config: str = "c2"  # c1, c2, c3, c3p, etc.
    conda_env: str = "pcc_geo_cnn"


class PCCGeoCNNv2Adapter(BaseAdapter):
    """Adapter for pcc_geo_cnn_v2 learned compression framework."""

    def __init__(
        self,
        model_name: str = "c3",
        resolution: int = 1024,
        octree_level: int = 3,
        framework_dir: Optional[str] = None,
    ):
        super().__init__(name=f"pcc_geo_cnn_v2_{model_name}")

        # Extract model config from model name (e.g., "c1" -> "c1", "c3p-a0.5" -> "c3p")
        model_config = model_name.split("-")[0]

        self.config = PCCGeoCNNConfig(
            model_name=model_name,
            resolution=resolution,
            octree_level=octree_level,
            model_config=model_config,
        )

        if framework_dir is None:
            project_root = Path(__file__).parent.parent.parent
            framework_dir = project_root / "frameworks" / "pcc_geo_cnn_v2"

        self.framework_dir = Path(framework_dir)
        base_model_dir = self.framework_dir / "models" / model_name

        # Models have subdirectories for different lambda values (e.g., 1.00e-04)
        # Use the first available subdirectory
        if base_model_dir.exists():
            lambda_dirs = sorted([d for d in base_model_dir.iterdir() if d.is_dir()])
            if lambda_dirs:
                self.models_dir = lambda_dirs[0]  # Use first lambda value
            else:
                self.models_dir = base_model_dir
        else:
            raise FileNotFoundError(
                f"Model directory not found: {base_model_dir}\n"
                f"Download models from: "
                f"https://drive.google.com/file/d/"
                f"18uHmr0ZpgFLeL9Y5TUFTsQkRfz4XpQdJ/view"
            )

        self.src_dir = self.framework_dir / "src"

    def _get_conda_python(self) -> str:
        """Get path to conda environment python."""
        conda_base = Path.home() / "miniconda3"
        if not conda_base.exists():
            conda_base = Path.home() / "anaconda3"

        python_path = conda_base / "envs" / self.config.conda_env / "bin" / "python"

        if not python_path.exists():
            raise FileNotFoundError(
                f"Conda environment '{self.config.conda_env}' not found.\n"
                f"Run: ./scripts/install_pcc_geo_cnn_v2_deps.sh"
            )

        return str(python_path)

    def compress(
        self, geometry: np.ndarray, colors: Optional[np.ndarray] = None
    ) -> tuple[bytes, CompressionResult]:
        """Compress point cloud using pcc_geo_cnn_v2."""
        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write input PLY
            input_ply = tmpdir / "input.ply"
            self._write_ply(input_ply, geometry, colors)

            # Output paths
            output_bin = tmpdir / "compressed.bin.gz"

            # Run compression
            compress_script = self.src_dir / "compress_octree.py"
            python_path = self._get_conda_python()

            cmd = [
                python_path,
                str(compress_script),
                "--input_files",
                str(input_ply),
                "--output_files",
                str(output_bin),
                "--checkpoint_dir",
                str(self.models_dir),
                "--model_config",
                self.config.model_config,
                "--resolution",
                str(self.config.resolution),
                "--octree_level",
                str(self.config.octree_level),
                "--read_batch_size",
                "1",
                "--opt_metrics",
                "d1_mse",
                "--data_format",
                "channels_last",  # Required for CPU compatibility
            ]

            # Force CPU execution (3D convolutions have GPU incompatibility)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(self.src_dir), env=env
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Compression failed:\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )

            # Read compressed data
            with open(output_bin, "rb") as f:
                compressed_data = f.read()

            compression_time = time.time() - start_time

            result = CompressionResult(
                compressed_size_bytes=len(compressed_data),
                compression_time_seconds=compression_time,
                decompression_time_seconds=0.0,  # Will be updated after decompress
                metadata={
                    "model": self.config.model_name,
                    "resolution": self.config.resolution,
                    "octree_level": self.config.octree_level,
                    "num_points_input": len(geometry),
                },
            )

            return compressed_data, result

    def decompress(self, compressed_data: bytes) -> tuple[np.ndarray, np.ndarray]:
        """Decompress point cloud using pcc_geo_cnn_v2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write compressed data
            input_bin = tmpdir / "compressed.bin.gz"
            with open(input_bin, "wb") as f:
                f.write(compressed_data)

            # Output PLY
            output_ply = tmpdir / "decompressed.ply"

            # Run decompression
            decompress_script = self.src_dir / "decompress_octree.py"
            python_path = self._get_conda_python()

            cmd = [
                python_path,
                str(decompress_script),
                "--input_files",
                str(input_bin),
                "--output_files",
                str(output_ply),
                "--checkpoint_dir",
                str(self.models_dir),
                "--model_config",
                self.config.model_config,
                "--data_format",
                "channels_last",  # Required for CPU compatibility
            ]

            # Force CPU execution (3D convolutions have GPU incompatibility)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(self.src_dir), env=env
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Decompression failed:\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )

            # Read decompressed point cloud
            geometry, colors = self._read_ply(output_ply)

            return geometry, colors

    def _write_ply(
        self, path: Path, geometry: np.ndarray, colors: Optional[np.ndarray] = None
    ):
        """Write PLY file."""
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(geometry)}",
            "property float x",
            "property float y",
            "property float z",
        ]

        if colors is not None:
            header.extend(
                ["property uchar red", "property uchar green", "property uchar blue"]
            )

        header.append("end_header")

        with open(path, "w") as f:
            f.write("\n".join(header) + "\n")
            for i in range(len(geometry)):
                line = f"{geometry[i, 0]:.6f} {geometry[i, 1]:.6f} {geometry[i, 2]:.6f}"
                if colors is not None:
                    line += (
                        f" {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}"
                    )
                f.write(line + "\n")

    def _read_ply(self, path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Read PLY file (ASCII or binary)."""
        from plyfile import PlyData

        plydata = PlyData.read(str(path))
        vertices = plydata["vertex"]

        # Extract geometry
        geometry = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(
            np.float32
        )

        # Extract colors if present
        colors = None
        if "red" in vertices and "green" in vertices and "blue" in vertices:
            colors = np.vstack(
                [vertices["red"], vertices["green"], vertices["blue"]]
            ).T.astype(np.uint8)

        return geometry, colors
