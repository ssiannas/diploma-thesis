"""G-PCC (MPEG TMC13) adapter.

Supports both lossless and lossy geometry compression. For lossy mode,
Trisoup (surface triangulation) is used with ``trisoup_node_size_log2``
controlling the rate: larger values = coarser surface = lower bitrate.

Usage:
    # Lossless (default)
    adapter = GPCCAdapter()
    compressed, result = adapter.compress(geometry)

    # Lossy at various Trisoup node sizes
    for ns in [2, 3, 4, 5, 6]:
        adapter = GPCCAdapter(trisoup_node_size_log2=ns)
        result, decoded = adapter.compress_and_decompress(geometry)
"""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from pcml.data.loaders import PLYPointCloudLoader, PointCloudData
from pcml.frameworks.base import BaseAdapter, CompressionResult


class GPCCAdapter(BaseAdapter):
    """Adapter for MPEG G-PCC (tmc3) codec.

    Args:
        tmc3_path: Path to the tmc3 binary.
        config_path: Optional path to a G-PCC config file.
        trisoup_node_size_log2: Trisoup node size (log2). Values >= 2 enable
            lossy Trisoup mode: higher = coarser = lower bitrate. 0 = lossless
            octree (default).
        coding_scale: (Deprecated) Geometry coding scale. Use
            trisoup_node_size_log2 instead for lossy mode.
    """

    def __init__(
        self,
        tmc3_path: str = "frameworks/mpeg-pcc-tmc13/build/tmc3/tmc3",
        config_path: Optional[str] = None,
        trisoup_node_size_log2: int = 0,
        coding_scale: float = 1.0,
    ):
        if trisoup_node_size_log2 >= 2:
            name = f"G-PCC_ts{trisoup_node_size_log2}"
        elif coding_scale != 1.0:
            name = f"G-PCC_qs{coding_scale}"
        else:
            name = "G-PCC"
        super().__init__(name)
        self.tmc3_path = Path(tmc3_path)
        self.config_path = Path(config_path) if config_path else None
        self.trisoup_node_size_log2 = trisoup_node_size_log2
        self.coding_scale = coding_scale

        if not self.tmc3_path.exists():
            raise FileNotFoundError(f"G-PCC binary not found: {self.tmc3_path}")

    def compress(
        self, geometry: np.ndarray, colors: Optional[np.ndarray] = None
    ) -> tuple[bytes, CompressionResult]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_ply = tmpdir / "input.ply"
            output_bin = tmpdir / "compressed.bin"
            recon_ply = tmpdir / "recon.ply"

            self._save_ply(input_ply, geometry, colors)

            cmd = [
                str(self.tmc3_path),
                "--mode=0",
                f"--uncompressedDataPath={input_ply}",
                f"--compressedStreamPath={output_bin}",
                f"--reconstructedDataPath={recon_ply}",
            ]

            if self.config_path:
                cmd.append(f"--config={self.config_path}")

            if self.trisoup_node_size_log2 >= 2:
                cmd.append(f"--trisoupNodeSizeLog2={self.trisoup_node_size_log2}")
            elif self.coding_scale != 1.0:
                cmd.append(f"--codingScale={self.coding_scale}")

            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            compression_time = time.time() - start_time

            if result.returncode != 0:
                raise RuntimeError(f"G-PCC compression failed: {result.stderr}")

            compressed_data = output_bin.read_bytes()
            compressed_size = len(compressed_data)

            # Read the reconstructed cloud produced during encoding
            decoded_geometry = None
            if recon_ply.exists():
                loader = PLYPointCloudLoader(verbose=False)
                pc = loader.load(str(recon_ply))
                decoded_geometry = pc.geometry

            self._last_decoded_geometry = decoded_geometry

            return compressed_data, CompressionResult(
                compressed_size_bytes=compressed_size,
                compression_time_seconds=compression_time,
                decompression_time_seconds=0.0,
                metadata={
                    "trisoup_node_size_log2": self.trisoup_node_size_log2,
                    "coding_scale": self.coding_scale,
                    "num_points_input": len(geometry),
                    "num_points_output": (
                        len(decoded_geometry) if decoded_geometry is not None else 0
                    ),
                },
            )

    def decompress(
        self, compressed_data: bytes
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        # Fast path: return cached decode from compress()
        if (
            hasattr(self, "_last_decoded_geometry")
            and self._last_decoded_geometry is not None
        ):
            geom = self._last_decoded_geometry
            self._last_decoded_geometry = None
            return geom, None

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

    def compress_and_decompress(
        self, geometry: np.ndarray, colors: Optional[np.ndarray] = None
    ) -> tuple[CompressionResult, np.ndarray]:
        """Compress and decompress in one call.

        Args:
            geometry: (N, 3) point coordinates.
            colors: Optional (N, 3) RGB colors [0-255].

        Returns:
            (compression_result, decoded_geometry)
        """
        compressed_data, result = self.compress(geometry, colors)
        decoded_geometry, _ = self.decompress(compressed_data)
        return result, decoded_geometry

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
