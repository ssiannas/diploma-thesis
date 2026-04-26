"""Streaming evaluation: inference + metrics without saving refined .npy files.

Reads existing decoded .npy files from pcgcv2_decoded, runs model inference
in memory, computes D1 + stratified PSNR, saves only the results CSV.

Usage:
    python scripts/evaluation/eval_streaming.py \
        --checkpoint models/postprocessing/good_candidates/film_v18_varweights_best.pt \
        --rates r2 r4 r6 \
        --tag v18_varweights \
        --output results/metrics/v18_varweights_all.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_ROOT = Path("datasets/pcgcv2_decoded")
SEQUENCES = ["longdress", "loot", "redandblack", "soldier"]
PEAK = 1023.0


def load_model(ckpt_path: Path, device: torch.device):
    from postprocessing.model import FiLMHeadSparseUNetV2, FiLMHeadSparseUNetV3

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    no_curv = not cfg.get("data", {}).get("use_curvature", True)
    in_ch = 3 if no_curv else 4
    model_type = cfg.get("model", {}).get("model_type", "film_head_v2")
    mc = cfg.get("model", {})
    if model_type == "film_head_v3":
        model = FiLMHeadSparseUNetV3(
            in_channels=in_ch,
            max_displacement=mc.get("max_displacement", 2.0),
            film_embed_dim=mc.get("film_embed_dim", 64),
            rate_repr=mc.get("film_rate_repr", "bpp"),
        ).to(device)
    else:
        model = FiLMHeadSparseUNetV2(
            in_channels=in_ch,
            film_embed_dim=mc.get("film_embed_dim", 64),
            rate_repr=mc.get("film_rate_repr", "bpp"),
        ).to(device)
    sd = ckpt.get("ema_state_dict", ckpt["model_state_dict"])
    model.load_state_dict(sd)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    logger.info(
        f"Loaded epoch {epoch} | {'nocurv' if no_curv else 'curv'} | rates={cfg.get('data',{}).get('rates')}"
    )
    return model


@torch.no_grad()
def run_inference(
    decoded: np.ndarray,
    rate: str,
    model,
    device: torch.device,
    conf_threshold: float = 0.0,
) -> np.ndarray:
    from postprocessing.dataset import COORD_SCALE, RATE_BPP, bpp_to_log

    coords = np.floor(decoded).astype(np.int32)
    feats = coords.astype(np.float32) / COORD_SCALE
    rate_tensor = torch.tensor(
        [bpp_to_log(RATE_BPP[rate])], dtype=torch.float32, device=device
    )
    sin = ME.SparseTensor(
        features=torch.from_numpy(feats).float(),
        coordinates=torch.from_numpy(
            np.column_stack([np.zeros(len(coords), dtype=np.int32), coords])
        ).int(),
        device=device,
    )
    is_v3 = hasattr(model, "n_classes")
    if is_v3 and conf_threshold > 0.0:
        pred, logits = model(sin, rate_tensor, return_logits=True)
        # Suppress moves where any axis has max_prob < threshold
        max_prob = (
            torch.softmax(logits, dim=-1).max(dim=-1).values.min(dim=-1).values
        )  # (N,)
        disp = pred.F.clone()
        disp[max_prob < conf_threshold] = 0.0
        disp = disp.cpu().numpy()
    else:
        pred = model(sin, rate_tensor)
        disp = pred.F.cpu().numpy()
    coords_out = pred.C[:, 1:].cpu().numpy().astype(np.float32)
    return coords_out + disp


def d1_psnr(orig_tree: cKDTree, rec: np.ndarray, peak: float = PEAK) -> float:
    rec_tree = cKDTree(rec)
    fwd_d, _ = orig_tree.query(rec, workers=-1)
    rev_d, _ = rec_tree.query(next(iter([orig_tree.data])), workers=-1)
    # rebuild since orig_tree.data is not exposed; query original from rec_tree
    return 10 * np.log10(peak**2 / max(np.mean(fwd_d**2), np.mean(rev_d**2)))


def d1_psnr_v2(
    original: np.ndarray, decoded: np.ndarray, orig_tree: cKDTree, peak: float = PEAK
) -> float:
    dec_tree = cKDTree(decoded)
    fwd_d, _ = orig_tree.query(decoded, workers=-1)
    rev_d, _ = dec_tree.query(original, workers=-1)
    return 10 * np.log10(peak**2 / max(np.mean(fwd_d**2), np.mean(rev_d**2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rates", nargs="+", default=["r2", "r4", "r6"])
    p.add_argument("--tag", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--curvature_k", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--sequences", nargs="+", default=None, help="Subset of sequences to evaluate"
    )
    p.add_argument(
        "--max_frames", type=int, default=None, help="Max frames per sequence"
    )
    p.add_argument("--data_root", default="datasets/pcgcv2_decoded")
    p.add_argument("--no_d1", action="store_true", help="Skip D1 PSNR (faster)")
    p.add_argument(
        "--conf_threshold",
        type=float,
        default=0.0,
        help="V3 only: suppress moves where min-axis max_prob < threshold",
    )
    args = p.parse_args()

    if args.output is None:
        args.output = f"results/metrics/{args.tag}_all.csv"
    DATA_ROOT = Path(args.data_root)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)

    from pcml.metrics.curvature import CurvatureQualityCalculator

    calc = CurvatureQualityCalculator()

    # Collect all frame dirs
    seqs = args.sequences if args.sequences else SEQUENCES
    all_frames = []
    for seq in seqs:
        seq_dir = DATA_ROOT / seq
        if not seq_dir.exists():
            continue
        frame_dirs = sorted(d for d in seq_dir.iterdir() if d.is_dir())
        if args.max_frames:
            frame_dirs = frame_dirs[: args.max_frames]
        for frame_dir in frame_dirs:
            all_frames.append((seq, frame_dir))
    logger.info(f"Found {len(all_frames)} frames | rates={args.rates}")

    rows = []
    for seq, frame_dir in tqdm(all_frames, desc="Frames"):
        original_path = frame_dir / "original.npy"
        if not original_path.exists():
            continue
        original = np.load(original_path).astype(np.float32)

        curv_path = frame_dir / f"curvature_k{args.curvature_k}.npy"
        if not curv_path.exists():
            logger.warning(f"Missing curvature: {curv_path} -- skipping frame")
            continue
        curvature = np.load(curv_path)

        orig_tree = cKDTree(original) if not args.no_d1 else None

        for rate in args.rates:
            decoded_path = frame_dir / f"pcgcv2_{rate}.npy"
            if not decoded_path.exists():
                continue
            decoded = np.load(decoded_path).astype(np.float32)

            dec_m = calc.compute_stratified_psnr(
                original,
                decoded,
                curvature=curvature,
                k=args.curvature_k,
                direction="reverse",
            )
            refined = run_inference(
                decoded, rate, model, device, conf_threshold=args.conf_threshold
            )
            ref_m = calc.compute_stratified_psnr(
                original,
                refined,
                curvature=curvature,
                k=args.curvature_k,
                direction="reverse",
            )

            row = {
                "sequence": seq,
                "frame": frame_dir.name,
                "rate": rate,
                "n_original": len(original),
                "n_decoded": len(decoded),
                "n_refined": len(refined),
                "dec_edge": dec_m.edge_psnr,
                "ref_edge": ref_m.edge_psnr,
                "edge_delta": ref_m.edge_psnr - dec_m.edge_psnr,
                "dec_flat": dec_m.flat_psnr,
                "ref_flat": ref_m.flat_psnr,
                "flat_delta": ref_m.flat_psnr - dec_m.flat_psnr,
            }
            if not args.no_d1:
                dec_d1 = d1_psnr_v2(original, decoded, orig_tree)
                ref_d1 = d1_psnr_v2(original, refined, orig_tree)
                row.update(
                    {"dec_d1": dec_d1, "ref_d1": ref_d1, "d1_delta": ref_d1 - dec_d1}
                )
            rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Saved {len(df)} rows to {out}")

    # Summary
    cols = (
        ["d1_delta", "edge_delta", "flat_delta"]
        if not args.no_d1
        else ["edge_delta", "flat_delta"]
    )
    avg = df.groupby("rate")[cols].mean()
    header = f"{'Rate':<6}  " + (
        "  ".join(f"{c.replace('_delta','').upper()+' Δ':>8}" for c in cols)
    )
    print("\n" + "=" * 60)
    print(header)
    print("-" * 60)
    for rate, row in avg.iterrows():
        vals = "  ".join(f"{row[c]:>+8.3f}" for c in cols)
        print(f"{rate:<6}  {vals}")
    print("=" * 60)


if __name__ == "__main__":
    main()
