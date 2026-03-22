"""MVUB cross-dataset evaluation: decode with PCGCv2, run nocurv ep24, compute metrics.

Usage:
    conda activate pcgcv2
    python scripts/evaluation/eval_mvub.py \
        --mvub_root "/media/ssiannas/Shared Driv/mvub" \
        --checkpoint models/postprocessing/good_candidates/film_v10_nocurv_ep24_best.pt \
        --output_root datasets/mvub_decoded \
        --rates r2 r4 r6 \
        --n_frames 10
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "postprocessing"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEQUENCES = {"david": "david10", "ricardo": "ricardo10"}
PEAK = 1023.0


def d1_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    orig_tree = cKDTree(original)
    rec_tree = cKDTree(reconstructed)
    fwd_mse = float(np.mean(orig_tree.query(reconstructed)[0] ** 2))
    bwd_mse = float(np.mean(rec_tree.query(original)[0] ** 2))
    mse = max(fwd_mse, bwd_mse)
    return float("inf") if mse == 0 else 10.0 * np.log10(PEAK**2 / mse)


def load_model(ckpt_path: Path, device: torch.device):
    from postprocessing.model import FiLMHeadSparseUNetV2

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    no_curv = not cfg.get("data", {}).get("use_curvature", True)
    in_ch = 3 if no_curv else 4
    model = FiLMHeadSparseUNetV2(
        in_channels=in_ch,
        film_embed_dim=cfg.get("model", {}).get("film_embed_dim", 64),
        rate_repr=cfg.get("model", {}).get("film_rate_repr", "bpp"),
    ).to(device)
    sd = ckpt.get("ema_state_dict", ckpt["model_state_dict"])
    model.load_state_dict(sd)
    model.eval()
    logger.info(
        f"Loaded checkpoint ep={ckpt.get('epoch')} | {'nocurv' if no_curv else 'curv'}"
    )
    return model


@torch.no_grad()
def run_inference(
    decoded: np.ndarray, model, actual_bpp: float, device: torch.device
) -> np.ndarray:
    import MinkowskiEngine as ME

    from postprocessing.dataset import COORD_SCALE, bpp_to_log

    coords = np.floor(decoded).astype(np.int32)
    feats = coords.astype(np.float32) / COORD_SCALE
    rate_tensor = torch.tensor(
        [bpp_to_log(actual_bpp)], dtype=torch.float32, device=device
    )
    sin = ME.SparseTensor(
        features=torch.from_numpy(feats).float(),
        coordinates=torch.from_numpy(
            np.column_stack([np.zeros(len(coords), dtype=np.int32), coords])
        ).int(),
        device=device,
    )
    disp = model(sin, rate_tensor)
    coords_out = disp.C[:, 1:].cpu().numpy().astype(np.float32)
    return coords_out + disp.F.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mvub_root", default="/media/ssiannas/Shared Driv/mvub")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_root", default="datasets/mvub_decoded")
    p.add_argument("--rates", nargs="+", default=["r2", "r4", "r6"])
    p.add_argument(
        "--n_frames", type=int, default=10, help="Frames per sequence (evenly spaced)"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_csv", default="results/metrics/mvub_metrics.csv")
    p.add_argument("--tag", default="nocurv_ep24", help="Tag for refined output files")
    p.add_argument(
        "--use_actual_bpp",
        action="store_true",
        default=True,
        help="Use actual codec BPP from saved bpp_{rate}.txt files instead of nominal",
    )
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    model = load_model(Path(args.checkpoint), device)

    from pcml.adapters.pcgcv2 import PCGCv2Adapter
    from pcml.data.loaders import PLYPointCloudLoader
    from pcml.metrics.curvature import CurvatureQualityCalculator

    # adapter is instantiated per-rate inside the loop (rate_point set at construction)

    rows = []
    fieldnames = [
        "sequence",
        "frame",
        "rate",
        "n_original",
        "n_decoded",
        "n_refined",
        "dec_d1",
        "ref_d1",
        "d1_delta",
        "dec_edge",
        "ref_edge",
        "edge_delta",
        "dec_flat",
        "ref_flat",
        "flat_delta",
    ]

    for seq_name, seq_dir_name in SEQUENCES.items():
        ply_dir = Path(args.mvub_root) / seq_dir_name / seq_dir_name / "ply"
        if not ply_dir.exists():
            logger.warning("Missing: %s", ply_dir)
            continue

        all_plys = sorted(ply_dir.glob("*.ply"))
        if not all_plys:
            logger.warning("No PLYs in %s", ply_dir)
            continue

        # Evenly sample n_frames
        indices = np.linspace(0, len(all_plys) - 1, args.n_frames, dtype=int)
        selected = [all_plys[i] for i in indices]

        for ply_path in tqdm(selected, desc=seq_name):
            frame_name = ply_path.stem
            out_dir = output_root / seq_name / frame_name
            out_dir.mkdir(parents=True, exist_ok=True)

            # Load original
            orig_path = out_dir / "original.npy"
            if orig_path.exists():
                original = np.load(orig_path).astype(np.float32)
            else:
                pcd = PLYPointCloudLoader().load(str(ply_path))
                original = pcd.geometry.astype(np.float32)
                np.save(orig_path, original)

            # Curvature (cached)
            curv_path = out_dir / "curvature_k30.npy"
            if curv_path.exists():
                curvature = np.load(curv_path)
            else:
                curvature = CurvatureQualityCalculator.compute_curvature(original, k=30)
                np.save(curv_path, curvature)

            for rate in args.rates:
                dec_path = out_dir / f"pcgcv2_{rate}.npy"
                bpp_path = out_dir / f"bpp_{rate}.txt"
                ref_path = out_dir / f"refined_{args.tag}_{rate}.npy"

                # Decode if not cached (also saves actual BPP)
                if not dec_path.exists():
                    try:
                        adapter = PCGCv2Adapter(rate_point=rate)
                        result, decoded = adapter.compress_and_decompress(
                            original.astype(np.int32)
                        )
                        decoded = decoded.astype(np.float32)
                        np.save(dec_path, decoded)
                        actual_bpp = result.compressed_size_bytes * 8 / len(original)
                        bpp_path.write_text(str(actual_bpp))
                    except Exception as e:
                        logger.error("PCGCv2 failed %s %s: %s", frame_name, rate, e)
                        continue
                else:
                    decoded = np.load(dec_path).astype(np.float32)

                # Resolve actual BPP for rate embedding
                from postprocessing.dataset import RATE_BPP

                if args.use_actual_bpp and bpp_path.exists():
                    actual_bpp = float(bpp_path.read_text())
                else:
                    actual_bpp = RATE_BPP[rate]
                    if args.use_actual_bpp:
                        logger.warning(
                            "No bpp file for %s %s, using nominal %.4f",
                            frame_name,
                            rate,
                            actual_bpp,
                        )

                # Inference if not cached
                if not ref_path.exists():
                    refined = run_inference(decoded, model, actual_bpp, device)
                    np.save(ref_path, refined)
                else:
                    refined = np.load(ref_path).astype(np.float32)

                # Metrics
                dec_d1 = d1_psnr(original, decoded)
                ref_d1 = d1_psnr(original, refined)

                dec_m = CurvatureQualityCalculator.compute_stratified_psnr(
                    original,
                    decoded,
                    curvature=curvature,
                    k=30,
                    direction="reverse",
                    rec_curvature=curvature,
                )
                ref_m = CurvatureQualityCalculator.compute_stratified_psnr(
                    original,
                    refined,
                    curvature=curvature,
                    k=30,
                    direction="reverse",
                    rec_curvature=curvature,
                )

                rows.append(
                    {
                        "sequence": seq_name,
                        "frame": frame_name,
                        "rate": rate,
                        "n_original": len(original),
                        "n_decoded": len(decoded),
                        "n_refined": len(refined),
                        "dec_d1": round(dec_d1, 4),
                        "ref_d1": round(ref_d1, 4),
                        "d1_delta": round(ref_d1 - dec_d1, 4),
                        "dec_edge": round(dec_m.edge_psnr, 4),
                        "ref_edge": round(ref_m.edge_psnr, 4),
                        "edge_delta": round(ref_m.edge_psnr - dec_m.edge_psnr, 4),
                        "dec_flat": round(dec_m.flat_psnr, 4),
                        "ref_flat": round(ref_m.flat_psnr, 4),
                        "flat_delta": round(ref_m.flat_psnr - dec_m.flat_psnr, 4),
                    }
                )

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    import collections

    print(f"\n{'='*70}")
    print(
        f"{'Sequence':<12} {'Rate':>5} {'N':>4}  {'D1 Δ':>7}  {'Edge Δ':>7}  {'Flat Δ':>7}"
    )
    print(f"{'-'*70}")
    rate_agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        rate_agg[r["rate"]][r["sequence"]].append(r)
    for rate in args.rates:
        for seq in SEQUENCES:
            seq_rows = rate_agg[rate].get(seq, [])
            if not seq_rows:
                continue
            print(
                f"{seq:<12} {rate:>5} {len(seq_rows):>4}  "
                f"{np.mean([x['d1_delta'] for x in seq_rows]):>+7.3f}  "
                f"{np.mean([x['edge_delta'] for x in seq_rows]):>+7.3f}  "
                f"{np.mean([x['flat_delta'] for x in seq_rows]):>+7.3f}"
            )
    print(f"{'='*70}")
    logger.info("Wrote %d rows to %s", len(rows), out_csv)


if __name__ == "__main__":
    main()
