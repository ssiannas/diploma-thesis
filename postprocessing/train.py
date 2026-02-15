"""Training entrypoint for displacement post-processing model.

Plain PyTorch training loop. Runs in pcgcv2 conda env.

Usage (legacy single-rate, single-frame):
    python postprocessing/train.py \
        --data_root results/oversmoothing/decoded_clouds \
        --rate r7 \
        --val_sequence redandblack_vox10_1550 \
        --epochs 50

Usage (multi-frame, multi-rate):
    python postprocessing/train.py \
        --data_root datasets/pcgcv2_decoded \
        --rates r2 r3 r4 \
        --val_sequence redandblack \
        --epochs 50 --batch_size 4
"""

import argparse
import logging
import time
from pathlib import Path

import MinkowskiEngine as ME
import torch
from dataset import SEQUENCES, MultiFrameDataset, PatchPairDataset
from losses import curvature_weighted_l1_loss
from model import SparseUNet
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Sequences available in multi-frame layout
MULTI_FRAME_SEQUENCES = ["longdress", "loot", "redandblack", "soldier"]


def parse_args():
    p = argparse.ArgumentParser(description="Train displacement post-processor")
    p.add_argument(
        "--data_root",
        type=str,
        default="results/oversmoothing/decoded_clouds",
    )
    # Legacy single-rate
    p.add_argument(
        "--rate", type=str, default=None, help="Single rate (legacy mode, e.g. r7)"
    )
    # Multi-rate
    p.add_argument(
        "--rates",
        nargs="+",
        default=None,
        help="Rate points for multi-frame mode (e.g., r2 r3 r4)",
    )
    p.add_argument(
        "--val_sequence",
        type=str,
        default="redandblack",
        help="Held-out sequence for validation",
    )
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--min_points", type=int, default=100)
    p.add_argument("--curvature_k", type=int, default=30)
    p.add_argument("--max_displacement", type=float, default=5.0)
    p.add_argument(
        "--alpha", type=float, default=10.0, help="Curvature weighting strength"
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_decay_epoch", type=int, default=20)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--save_dir", type=str, default="models/postprocessing")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--max_clouds",
        type=int,
        default=None,
        help="Limit number of clouds to load (for debugging)",
    )
    return p.parse_args()


def _use_multi_frame(args) -> bool:
    """Determine whether to use MultiFrameDataset based on args."""
    if args.rates is not None:
        return True
    # Check if manifest.json exists in data_root (multi-frame layout)
    return (Path(args.data_root) / "manifest.json").exists()


def _build_datasets(args):
    """Build train/val datasets, choosing legacy or multi-frame mode."""
    if _use_multi_frame(args):
        rates = args.rates or ["r2", "r3", "r4"]
        rate_label = "_".join(rates)

        train_seqs = [s for s in MULTI_FRAME_SEQUENCES if s != args.val_sequence]
        val_seqs = [args.val_sequence]

        logger.info(f"Multi-frame mode: rates={rates}")
        logger.info(f"Train sequences: {train_seqs}")
        logger.info(f"Val sequence: {val_seqs}")

        logger.info("Loading training data...")
        train_ds = MultiFrameDataset(
            data_root=args.data_root,
            sequences=train_seqs,
            rates=rates,
            patch_size=args.patch_size,
            stride=args.stride,
            min_points=args.min_points,
            curvature_k=args.curvature_k,
            max_clouds=args.max_clouds,
        )
        logger.info("Loading validation data...")
        val_ds = MultiFrameDataset(
            data_root=args.data_root,
            sequences=val_seqs,
            rates=rates,
            patch_size=args.patch_size,
            stride=args.stride,
            min_points=args.min_points,
            curvature_k=args.curvature_k,
        )
        collate_fn = MultiFrameDataset.collate_fn
        return train_ds, val_ds, collate_fn, rate_label

    # Legacy single-rate mode
    rate = args.rate or "r7"
    train_seqs = [s for s in SEQUENCES if s != args.val_sequence]
    val_seqs = [args.val_sequence]

    logger.info(f"Legacy mode: rate={rate}")
    logger.info(f"Train: {train_seqs}")
    logger.info(f"Val: {val_seqs}")

    logger.info("Loading training data...")
    train_ds = PatchPairDataset(
        data_root=args.data_root,
        sequences=train_seqs,
        rate=rate,
        patch_size=args.patch_size,
        stride=args.stride,
        min_points=args.min_points,
        curvature_k=args.curvature_k,
    )
    logger.info("Loading validation data...")
    val_ds = PatchPairDataset(
        data_root=args.data_root,
        sequences=val_seqs,
        rate=rate,
        patch_size=args.patch_size,
        stride=args.stride,
        min_points=args.min_points,
        curvature_k=args.curvature_k,
    )
    collate_fn = PatchPairDataset.collate_fn
    return train_ds, val_ds, collate_fn, rate


def train_one_epoch(model, loader, optimizer, device, alpha):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for coords, feats, displacements, curvature in tqdm(
        loader, desc="Train", leave=False
    ):
        sin = ME.SparseTensor(
            features=feats.float(),
            coordinates=coords.int(),
            device=device,
        )
        gt_disp = displacements.float().to(device)
        curv = curvature.float().to(device)

        pred = model(sin)
        loss = curvature_weighted_l1_loss(pred, gt_disp, curv, alpha=alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device, alpha):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for coords, feats, displacements, curvature in tqdm(
        loader, desc="Val", leave=False
    ):
        sin = ME.SparseTensor(
            features=feats.float(),
            coordinates=coords.int(),
            device=device,
        )
        gt_disp = displacements.float().to(device)
        curv = curvature.float().to(device)

        pred = model(sin)
        loss = curvature_weighted_l1_loss(pred, gt_disp, curv, alpha=alpha)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Curvature weighting alpha={args.alpha}")

    train_dataset, val_dataset, collate_fn, rate_label = _build_datasets(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Model
    model = SparseUNet(in_channels=4, max_displacement=args.max_displacement).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_epoch, gamma=0.5
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.alpha)
        val_loss = validate(model, val_loader, device, args.alpha)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"lr={lr:.1e} | {elapsed:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = save_dir / f"best_{rate_label}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                ckpt_path,
            )
            logger.info(f"  Saved best model (val_loss={val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            ckpt_path = save_dir / f"epoch_{epoch}_{rate_label}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                ckpt_path,
            )

    logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
