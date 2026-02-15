"""Generate training data for post-processing model.

Decodes 8iVFB frames through PCGCv2 at multiple rate points and saves
original/decoded .npy pairs for training the displacement U-Net.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate pcgcv2
    python scripts/setup/generate_training_data.py \
        --input_root datasets/8iVFB \
        --output_root datasets/pcgcv2_decoded \
        --rates r2 r3 r4
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pcml.adapters.pcgcv2 import PCGCv2Adapter
from pcml.data.loaders import PLYPointCloudLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 8iVFB sequence info: (dir_name, filename_prefix, start_frame)
SEQUENCES = {
    "longdress": ("longdress", "longdress_vox10", 1051),
    "loot": ("loot", "loot_vox10", 1000),
    "redandblack": ("redandblack", "redandblack_vox10", 1450),
    "soldier": ("soldier", "soldier_vox10", 536),
}

TOTAL_FRAMES = 300
FRAME_STEP = 3  # every 3rd frame -> 100 per sequence


def get_frame_indices():
    """Return 0-based indices: 0, 3, 6, ..., 297 -> 100 frames."""
    return list(range(0, TOTAL_FRAMES, FRAME_STEP))


def parse_args():
    p = argparse.ArgumentParser(description="Generate PCGCv2 training data")
    p.add_argument(
        "--input_root",
        type=str,
        default="datasets/8iVFB",
        help="Root directory of 8iVFB dataset",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="datasets/pcgcv2_decoded",
        help="Output directory for generated data",
    )
    p.add_argument(
        "--rates",
        nargs="+",
        default=["r2", "r3", "r4"],
        help="Rate points to generate (e.g., r2 r3 r4)",
    )
    p.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Sequences to process (default: all)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without running codec",
    )
    return p.parse_args()


def build_work_items(args, input_root, output_root, frame_indices, seq_names):
    """Build list of (seq_name, frame_num, ply_path, frame_dir, rate) work items."""
    items = []
    for seq_name in seq_names:
        if seq_name not in SEQUENCES:
            logger.error(f"Unknown sequence: {seq_name}")
            continue

        dir_name, prefix, start_frame = SEQUENCES[seq_name]
        ply_dir = input_root / dir_name / "Ply"

        if not ply_dir.exists():
            logger.error(f"PLY directory not found: {ply_dir}")
            continue

        for idx in frame_indices:
            frame_num = start_frame + idx
            ply_path = ply_dir / f"{prefix}_{frame_num:04d}.ply"
            frame_dir = output_root / seq_name / str(frame_num)

            for rate in args.rates:
                items.append((seq_name, frame_num, ply_path, frame_dir, rate))
    return items


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    loader = PLYPointCloudLoader()
    frame_indices = get_frame_indices()
    seq_names = args.sequences or list(SEQUENCES.keys())

    work_items = build_work_items(
        args, input_root, output_root, frame_indices, seq_names
    )

    manifest = []
    skipped = 0
    failed = 0

    logger.info(
        f"Generating data: {len(seq_names)} sequences x "
        f"{len(frame_indices)} frames x {len(args.rates)} rates = "
        f"{len(work_items)} codec runs"
    )

    pbar = tqdm(
        work_items, desc="Codec runs", unit="run", dynamic_ncols=True, smoothing=0.1
    )

    for seq_name, frame_num, ply_path, frame_dir, rate in pbar:
        pbar.set_postfix_str(
            f"{seq_name}/{frame_num}/{rate} skip={skipped} fail={failed}",
            refresh=False,
        )

        if not ply_path.exists():
            pbar.write(f"WARNING: Frame not found: {ply_path}")
            continue

        frame_dir.mkdir(parents=True, exist_ok=True)

        # Save original (once per frame)
        original_path = frame_dir / "original.npy"
        decoded_path = frame_dir / f"pcgcv2_{rate}.npy"

        if decoded_path.exists():
            skipped += 1
            manifest.append(
                {
                    "sequence": seq_name,
                    "frame": frame_num,
                    "rate": rate,
                }
            )
            continue

        if args.dry_run:
            manifest.append(
                {
                    "sequence": seq_name,
                    "frame": frame_num,
                    "rate": rate,
                }
            )
            continue

        # Load/save original geometry
        if original_path.exists():
            geometry = np.load(original_path)
        else:
            pc_data = loader.load(ply_path)
            geometry = pc_data.geometry
            np.save(original_path, geometry)

        try:
            adapter = PCGCv2Adapter(rate_point=rate)
            _, decoded = adapter.compress_and_decompress(geometry)
            np.save(decoded_path, decoded)

            manifest.append(
                {
                    "sequence": seq_name,
                    "frame": frame_num,
                    "rate": rate,
                }
            )
        except Exception as e:
            failed += 1
            pbar.write(f"FAILED {seq_name}/{frame_num}/{rate}: {e}")

    pbar.close()

    # Write manifest
    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"Done. {len(manifest)} entries ({skipped} skipped, {failed} failed). "
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
