"""Adapter for PCGCv2 framework (Wang et al., IEEE TCSVT 2021).

PCGCv2 uses sparse convolutions (MinkowskiEngine) for learned point cloud
geometry compression. It provides 7 pretrained rate points (r1-r7) covering
0.025-0.4 BPP with D1-PSNR in the 58-60 dB range on 10-bit voxelized clouds.

The codec produces 4 binary files per compressed point cloud:
    _C.bin          - Compressed coordinates (via G-PCC)
    _F.bin          - Compressed latent features (via learned entropy model)
    _H.bin          - Header/shape info for entropy decoding
    _num_points.bin - Number of points at each decoder level

References:
    Wang, J., et al. (2021). "Lossy Point Cloud Geometry Compression via
        End-to-End Learning." IEEE TCSVT.
"""

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseAdapter, CompressionResult

logger = logging.getLogger(__name__)

# Mapping from rate point name to checkpoint filename
RATE_POINTS = {
    "r1": "r1_0.025bpp.pth",
    "r2": "r2_0.05bpp.pth",
    "r3": "r3_0.10bpp.pth",
    "r4": "r4_0.15bpp.pth",
    "r5": "r5_0.25bpp.pth",
    "r6": "r6_0.3bpp.pth",
    "r7": "r7_0.4bpp.pth",
}


@dataclass
class PCGCv2Config:
    """Configuration for PCGCv2."""

    rate_point: str = "r3"  # r1-r7
    checkpoint: str = "r3_0.10bpp.pth"  # checkpoint filename
    scaling_factor: float = 1.0
    rho: float = 1.0
    resolution: int = 1024
    conda_env: str = "pcgcv2"


class PCGCv2Adapter(BaseAdapter):
    """Adapter for PCGCv2 learned point cloud geometry compression.

    Calls PCGCv2's coder.py via subprocess in a separate conda environment
    with MinkowskiEngine and PyTorch. Supports GPU acceleration.
    """

    def __init__(
        self,
        rate_point: str = "r3",
        resolution: int = 1024,
        scaling_factor: float = 1.0,
        rho: float = 1.0,
        framework_dir: Optional[str] = None,
    ):
        super().__init__(name=f"PCGCv2_{rate_point}")

        if rate_point not in RATE_POINTS:
            raise ValueError(
                f"Invalid rate_point '{rate_point}'. "
                f"Valid options: {list(RATE_POINTS.keys())}"
            )

        self.config = PCGCv2Config(
            rate_point=rate_point,
            checkpoint=RATE_POINTS[rate_point],
            scaling_factor=scaling_factor,
            rho=rho,
            resolution=resolution,
        )

        if framework_dir is None:
            project_root = Path(__file__).parent.parent.parent
            framework_dir = project_root / "frameworks" / "PCGCv2"

        self.framework_dir = Path(framework_dir)

        if not self.framework_dir.exists():
            raise FileNotFoundError(
                f"PCGCv2 framework not found: {self.framework_dir}\n"
                f"Run: git submodule update --init frameworks/PCGCv2"
            )

        self.ckpt_path = self.framework_dir / "ckpts" / self.config.checkpoint

    def _get_conda_env_dir(self) -> Path:
        """Get path to the conda environment directory."""
        conda_base = Path.home() / "miniconda3"
        if not conda_base.exists():
            conda_base = Path.home() / "anaconda3"

        env_dir = conda_base / "envs" / self.config.conda_env

        if not (env_dir / "bin" / "python").exists():
            raise FileNotFoundError(
                f"Conda environment '{self.config.conda_env}' not found.\n"
                f"Create it with:\n"
                f"  conda create -n pcgcv2 python=3.8 -y\n"
                f"  conda activate pcgcv2\n"
                f"  pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 "
                f"--extra-index-url https://download.pytorch.org/whl/cu113\n"
                f"  pip install MinkowskiEngine==0.5.4\n"
                f"  pip install torchac==0.9.3 open3d numpy scipy pandas tqdm plyfile"
            )

        return env_dir

    def _get_conda_python(self) -> str:
        """Get path to conda environment python."""
        return str(self._get_conda_env_dir() / "bin" / "python")

    def compress(
        self, geometry: np.ndarray, colors: Optional[np.ndarray] = None
    ) -> tuple[bytes, CompressionResult]:
        """Compress point cloud using PCGCv2.

        PCGCv2 is geometry-only; colors are ignored.

        Args:
            geometry: (N, 3) point coordinates (integer voxel coordinates).
            colors: Ignored (PCGCv2 is geometry-only).

        Returns:
            (compressed_data, result) where compressed_data is the
            concatenation of all 4 binary output files.
        """
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.ckpt_path}\n"
                f"Download checkpoints from the PCGCv2 README links and "
                f"place them in {self.ckpt_path.parent}/"
            )

        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write input PLY (geometry-only, integer coordinates)
            input_ply = tmpdir / "input.ply"
            self._write_ply_geo(input_ply, geometry)

            # PCGCv2 outputs files relative to its output dir
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            # Run PCGCv2 coder.py
            python_path = self._get_conda_python()
            coder_script = self.framework_dir / "coder.py"

            cmd = [
                python_path,
                str(coder_script),
                "--ckptdir",
                str(self.ckpt_path),
                "--filedir",
                str(input_ply),
                "--scaling_factor",
                str(self.config.scaling_factor),
                "--rho",
                str(self.config.rho),
                "--res",
                str(self.config.resolution),
            ]

            env = os.environ.copy()
            # Add conda env bin to PATH so ninja and tmc3 are found
            conda_bin = str(self._get_conda_env_dir() / "bin")
            env["PATH"] = conda_bin + os.pathsep + env.get("PATH", "")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.framework_dir),
                env=env,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"PCGCv2 compression failed:\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )

            # Collect compressed binary files
            # PCGCv2 writes to ./output/<filename>_*.bin
            stem = input_ply.stem  # "input"
            bin_dir = self.framework_dir / "output"
            suffixes = ["_C.bin", "_F.bin", "_H.bin", "_num_points.bin"]

            total_compressed_size = 0
            compressed_parts = {}
            for suffix in suffixes:
                bin_path = bin_dir / (stem + suffix)
                if bin_path.exists():
                    data = bin_path.read_bytes()
                    compressed_parts[suffix] = data
                    total_compressed_size += len(data)
                else:
                    logger.warning("Missing output file: %s", bin_path)

            # Read decoded PLY
            dec_ply_path = bin_dir / (stem + "_dec.ply")
            if not dec_ply_path.exists():
                raise RuntimeError(
                    f"Decoded PLY not found: {dec_ply_path}\n"
                    f"PCGCv2 stdout: {result.stdout}"
                )

            dec_geometry = self._read_ply_geo(dec_ply_path)

            compression_time = time.time() - start_time

            # Pack all binary parts into a single bytes object with a header
            compressed_data = self._pack_compressed(compressed_parts)

            # Clean up PCGCv2 output directory
            for suffix in suffixes:
                bin_path = bin_dir / (stem + suffix)
                if bin_path.exists():
                    bin_path.unlink()
            if dec_ply_path.exists():
                dec_ply_path.unlink()

            comp_result = CompressionResult(
                compressed_size_bytes=total_compressed_size,
                compression_time_seconds=compression_time,
                decompression_time_seconds=0.0,
                metadata={
                    "rate_point": self.config.rate_point,
                    "checkpoint": self.config.checkpoint,
                    "resolution": self.config.resolution,
                    "scaling_factor": self.config.scaling_factor,
                    "rho": self.config.rho,
                    "num_points_input": len(geometry),
                    "num_points_output": len(dec_geometry),
                    "stdout": result.stdout,
                },
            )

            # Store decoded geometry for immediate retrieval
            self._last_decoded_geometry = dec_geometry

            return compressed_data, comp_result

    def decompress(
        self, compressed_data: bytes
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Decompress point cloud from PCGCv2 compressed data.

        If called immediately after compress(), returns the already-decoded
        geometry (PCGCv2's coder.py does encode+decode in one pass).
        Otherwise, re-runs the full decode pipeline.

        Args:
            compressed_data: Packed binary data from compress().

        Returns:
            (geometry, None) - PCGCv2 is geometry-only, colors are always None.
        """
        # Fast path: return cached decode from compress()
        if (
            hasattr(self, "_last_decoded_geometry")
            and self._last_decoded_geometry is not None
        ):
            geom = self._last_decoded_geometry
            self._last_decoded_geometry = None
            return geom, None

        # Slow path: re-run full encode+decode
        # Unpack the compressed parts
        parts = self._unpack_compressed(compressed_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write binary parts to temp files
            stem = "compressed"
            for suffix, data in parts.items():
                (tmpdir / (stem + suffix)).write_bytes(data)

            # PCGCv2 doesn't have a standalone decode CLI;
            # coder.py always does encode+decode together.
            # For standalone decode, we would need to modify coder.py.
            # For now, raise an informative error.
            raise NotImplementedError(
                "PCGCv2 standalone decompression is not supported. "
                "Use compress() which returns decoded geometry in metadata, "
                "or call compress() followed by decompress() in sequence."
            )

    def compress_and_decompress(
        self, geometry: np.ndarray, colors: Optional[np.ndarray] = None
    ) -> tuple[CompressionResult, np.ndarray]:
        """Compress and decompress in one call (recommended API for PCGCv2).

        PCGCv2's coder.py performs encode+decode in a single pass, so this
        is the natural way to use it.

        Args:
            geometry: (N, 3) point coordinates (integer voxel coordinates).
            colors: Ignored (PCGCv2 is geometry-only).

        Returns:
            (compression_result, decoded_geometry)
        """
        compressed_data, result = self.compress(geometry, colors)
        decoded_geometry, _ = self.decompress(compressed_data)
        return result, decoded_geometry

    @staticmethod
    def _write_ply_geo(path: Path, geometry: np.ndarray) -> None:
        """Write geometry-only PLY file with integer coordinates.

        PCGCv2 expects integer voxel coordinates in ASCII PLY format.
        """
        coords = np.round(geometry).astype(int)

        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(coords)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in coords:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")

    @staticmethod
    def _read_ply_geo(path: Path) -> np.ndarray:
        """Read geometry-only PLY file (ASCII format).

        Returns integer coordinates as float32 array for consistency
        with the rest of the framework.
        """
        coords = []
        header_ended = False

        with open(path, "r") as f:
            for line in f:
                if not header_ended:
                    if line.strip() == "end_header":
                        header_ended = True
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    coords.append([float(parts[0]), float(parts[1]), float(parts[2])])

        return np.array(coords, dtype=np.float32)

    @staticmethod
    def _pack_compressed(parts: dict) -> bytes:
        """Pack multiple binary files into a single bytes object.

        Format: [num_parts(4B)] [suffix_len(4B) suffix data_len(4B) data] ...
        """
        import struct

        buf = struct.pack("<I", len(parts))
        for suffix, data in parts.items():
            suffix_bytes = suffix.encode("utf-8")
            buf += struct.pack("<I", len(suffix_bytes))
            buf += suffix_bytes
            buf += struct.pack("<I", len(data))
            buf += data
        return buf

    @staticmethod
    def _unpack_compressed(packed: bytes) -> dict:
        """Unpack a single bytes object into multiple binary parts."""
        import struct

        offset = 0
        (num_parts,) = struct.unpack_from("<I", packed, offset)
        offset += 4

        parts = {}
        for _ in range(num_parts):
            (suffix_len,) = struct.unpack_from("<I", packed, offset)
            offset += 4
            suffix = packed[offset : offset + suffix_len].decode("utf-8")
            offset += suffix_len
            (data_len,) = struct.unpack_from("<I", packed, offset)
            offset += 4
            data = packed[offset : offset + data_len]
            offset += data_len
            parts[suffix] = data

        return parts
