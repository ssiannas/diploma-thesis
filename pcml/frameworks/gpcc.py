"""G-PCC (MPEG TMC13) adapter."""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from pcml.data.loaders import PLYPointCloudLoader, PointCloudData
from pcml.frameworks.base import BaseAdapter, CompressionResult


class GPCCAdapter(BaseAdapter):
    """Adapter for MPEG G-PCC (tmc3) codec."""

    def __init__(
        self,
        tmc3_path: str = "frameworks/mpeg-pcc-tmc13/build/tmc3/tmc3",
        config_path: Optional[str] = None,
    ):
        super().__init__("G-PCC")
        self.tmc3_path = Path(tmc3_path)
        self.config_path = Path(config_path) if config_path else None

        if not self.tmc3_path.exists():
            raise FileNotFoundError(f"G-PCC binary not found: {self.tmc3_path}")

    def compress(
        self, geometry: np.ndarray, colors: Optional[np.ndarray] = None
    ) -> tuple[bytes, CompressionResult]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_ply = tmpdir / "input.ply"
            output_bin = tmpdir / "compressed.bin"

            self._save_ply(input_ply, geometry, colors)

            cmd = [
                str(self.tmc3_path),
                "--mode=0",
                f"--uncompressedDataPath={input_ply}",
                f"--compressedStreamPath={output_bin}",
            ]

            if self.config_path:
                cmd.append(f"--config={self.config_path}")

            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            compression_time = time.time() - start_time

            if result.returncode != 0:
                raise RuntimeError(f"G-PCC compression failed: {result.stderr}")

            compressed_data = output_bin.read_bytes()
            compressed_size = len(compressed_data)

            return compressed_data, CompressionResult(
                compressed_size_bytes=compressed_size,
                compression_time_seconds=compression_time,
                decompression_time_seconds=0.0,
            )

    def decompress(
        self, compressed_data: bytes
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_bin = tmpdir / "compressed.bin"
            output_ply = tmpdir / "decompressed.ply"

            input_bin.write_bytes(compressed_data)

            cmd = [
                str(self.tmc3_path),
                "--mode=1",
                f"--compressedStreamPath={input_bin}",
                f"--reconstructedDataPath={output_ply}",
            ]

            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            decompression_time = time.time() - start_time

            if result.returncode != 0:
                raise RuntimeError(f"G-PCC decompression failed: {result.stderr}")

            loader = PLYPointCloudLoader(verbose=False)
            pc = loader.load(str(output_ply))

            return pc.geometry, pc.colors

    def _save_ply(
        self, path: Path, geometry: np.ndarray, colors: Optional[np.ndarray] = None
    ):
        """Save point cloud as PLY file."""
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(geometry)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")

            if colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")

            f.write("end_header\n")

            for i in range(len(geometry)):
                x, y, z = geometry[i]
                if colors is not None:
                    r, g, b = colors[i].astype(int)
                    f.write(f"{x} {y} {z} {r} {g} {b}\n")
                else:
                    f.write(f"{x} {y} {z}\n")
