"""Full dataset evaluation: inference + stratified PSNR over all frames.

Usage:
    python scripts/evaluation/eval_full_dataset.py \
        --checkpoint models/postprocessing/good_candidates/film_v10_nocurv_ep24_best.pt \
        --rates r2 r4 r6 \
        --tag nocurv_ep24 \
        --workers 4
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
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


def compute_curvature(pts: np.ndarray, k: int = 30) -> np.ndarray:
    from postprocessing.dataset import compute_curvature as _compute_curvature

    return _compute_curvature(pts, k=k)


def run_inference(
    decoded_npy: np.ndarray, model, rate: str, device: torch.device
) -> np.ndarray:
    import MinkowskiEngine as ME

    from postprocessing.dataset import COORD_SCALE, RATE_BPP, bpp_to_log

    coords = np.floor(decoded_npy).astype(np.int32)
    feats = coords.astype(np.float32) / COORD_SCALE
    rate_scalar = bpp_to_log(RATE_BPP[rate])
    rate_tensor = torch.tensor([rate_scalar], dtype=torch.float32, device=device)
    sin = ME.SparseTensor(
        features=torch.from_numpy(feats).float(),
        coordinates=torch.from_numpy(
            np.column_stack([np.zeros(len(coords), dtype=np.int32), coords])
        ).int(),
        device=device,
    )
    with torch.no_grad():
        disp = model(sin, rate_tensor)
    disp_np = disp.F.cpu().numpy()
    coords_out = disp.C[:, 1:].cpu().numpy().astype(np.float32)
    refined = coords_out + disp_np
    return refined


def eval_frame(
    original: np.ndarray, refined: np.ndarray, curvature: np.ndarray, k: int = 30
) -> dict:
    from pcml.metrics.curvature import CurvatureQualityCalculator

    calc = CurvatureQualityCalculator()
    ref_m = calc.compute_stratified_psnr(
        original, refined, curvature=curvature, k=k, direction="reverse"
    )
    dec_m_key = "_dec"  # placeholder — caller computes decoded stats separately
    return {"edge": ref_m.edge_psnr, "flat": ref_m.flat_psnr, "degr": ref_m.degradation}


def load_model(ckpt_path: Path, device: torch.device):
    from postprocessing.model import FiLMHeadSparseUNetV2

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    no_curv = not cfg.get("data", {}).get("use_curvature", True)
    in_ch = 3 if no_curv else 4
    film_embed_dim = cfg.get("model", {}).get("film_embed_dim", 64)
    film_rate_repr = cfg.get("model", {}).get("film_rate_repr", "bpp")
    model = FiLMHeadSparseUNetV2(
        in_channels=in_ch,
        film_embed_dim=film_embed_dim,
        rate_repr=film_rate_repr,
    ).to(device)
    sd = ckpt.get("ema_state_dict", ckpt["model_state_dict"])
    model.load_state_dict(sd)
    model.eval()
    logger.info(
        f"Loaded epoch {ckpt.get('epoch')} | {'nocurv' if no_curv else 'curv'}"
        f" | rates={cfg.get('data', {}).get('rates')}"
    )
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rates", nargs="+", default=["r2", "r4", "r6"])
    p.add_argument("--tag", required=True)
    p.add_argument("--curvature_k", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force_inference", action="store_true")
    p.add_argument(
        "--infer_only",
        action="store_true",
        help="Save refined NPY files only, skip metric computation",
    )
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)

    from pcml.metrics.curvature import CurvatureQualityCalculator

    calc = CurvatureQualityCalculator()

    # Collect all (seq, frame_dir) pairs
    all_frames = []
    for seq in SEQUENCES:
        seq_dir = DATA_ROOT / seq
        for frame_dir in sorted(seq_dir.iterdir()):
            if frame_dir.is_dir():
                all_frames.append((seq, frame_dir))
    logger.info(
        f"Total frames: {len(all_frames)} | rates: {args.rates} | tag: {args.tag}"
    )

    # Per-rate accumulators: {rate: {seq: {edge:[], flat:[], edge_dec:[], flat_dec:[]}}}
    from collections import defaultdict

    stats = {
        rate: {
            seq: {"edge": [], "flat": [], "edge_dec": [], "flat_dec": []}
            for seq in SEQUENCES
        }
        for rate in args.rates
    }

    for seq, frame_dir in tqdm(all_frames, desc="Frames"):
        original = np.load(frame_dir / "original.npy").astype(np.float32)

        # Compute/load curvature (skipped in infer_only mode)
        curvature = None
        if not args.infer_only:
            curv_path = frame_dir / f"curvature_k{args.curvature_k}.npy"
            if curv_path.exists():
                curvature = np.load(curv_path)
            else:
                curvature = compute_curvature(original, k=args.curvature_k)
                np.save(curv_path, curvature)

        for rate in args.rates:
            decoded_path = frame_dir / f"pcgcv2_{rate}.npy"
            if not decoded_path.exists():
                continue

            decoded = np.load(decoded_path).astype(np.float32)
            refined_path = frame_dir / f"refined_{args.tag}_{rate}.npy"

            # Run inference if needed
            if args.force_inference or not refined_path.exists():
                refined = run_inference(decoded, model, rate, device)
                np.save(refined_path, refined)
            else:
                refined = np.load(refined_path).astype(np.float32)

            if args.infer_only:
                continue

            # Eval refined
            ref_m = calc.compute_stratified_psnr(
                original,
                refined,
                curvature=curvature,
                k=args.curvature_k,
                direction="reverse",
            )
            # Eval decoded
            dec_m = calc.compute_stratified_psnr(
                original,
                decoded,
                curvature=curvature,
                k=args.curvature_k,
                direction="reverse",
            )

            stats[rate][seq]["edge"].append(ref_m.edge_psnr - dec_m.edge_psnr)
            stats[rate][seq]["flat"].append(ref_m.flat_psnr - dec_m.flat_psnr)

    # Print results
    W = 94
    print("\n" + "=" * W)
    print(
        f"{'Sequence':<20}  {'Rate':>5}  {'Frames':>7}  {'Edge dB':>8}  {'Flat dB':>8}"
    )
    print("-" * W)

    rate_totals = {rate: {"edge": [], "flat": []} for rate in args.rates}
    for rate in args.rates:
        for seq in SEQUENCES:
            s = stats[rate][seq]
            n = len(s["edge"])
            if n == 0:
                continue
            e = np.mean(s["edge"])
            f = np.mean(s["flat"])
            rate_totals[rate]["edge"].append(e)
            rate_totals[rate]["flat"].append(f)
            print(f"{seq:<20}  {rate:>5}  {n:>7}  {e:>+8.2f}  {f:>+8.2f}")

    print("-" * W)
    for rate in args.rates:
        e = np.mean(rate_totals[rate]["edge"])
        f = np.mean(rate_totals[rate]["flat"])
        print(f"{'AVERAGE':<20}  {rate:>5}  {'':>7}  {e:>+8.2f}  {f:>+8.2f}")
    print("=" * W)


if __name__ == "__main__":
    main()
