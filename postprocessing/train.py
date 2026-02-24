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

import os

os.environ.setdefault("OMP_NUM_THREADS", "20")

import argparse
import logging
import math
import time
from pathlib import Path

import MinkowskiEngine as ME
import torch
from dataset import SEQUENCES, MultiFrameDataset, PatchPairDataset
from losses import (
    KendallUncertaintyWeights,
    chamfer_loss,
    curvature_weighted_l1_loss,
    dynamic_chamfer_loss,
    gated_displacement_loss,
    laplacian_loss,
    min_of_k_displacement_loss,
    stratified_displacement_loss,
    stratified_loss,
)
from model import GatedSparseUNet, SparseUNet, ThresholdGatedUNet
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
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
    p.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Keep every Nth frame per (sequence, rate) to reduce temporal redundancy",
    )
    p.add_argument(
        "--no_augment", action="store_true", help="Disable data augmentation"
    )
    p.add_argument(
        "--model_type",
        type=str,
        default="unet",
        choices=["unet", "gated", "threshold"],
        help="Model architecture",
    )
    p.add_argument(
        "--loss_type",
        type=str,
        default="displacement",
        choices=[
            "displacement",
            "chamfer",
            "laplacian",
            "stratified",
            "stratified_disp",
            "gated",
            "threshold_gated",
            "cd",
        ],
        help="Loss function type",
    )
    p.add_argument(
        "--lambda_gate",
        type=float,
        default=1.0,
        help="Weight for gate BCE loss in gated mode",
    )
    p.add_argument(
        "--mag_floor",
        type=float,
        default=0.1,
        help="Magnitude weight floor for zero-GT points (0=nonzero only)",
    )
    p.add_argument(
        "--chamfer_weight",
        type=float,
        default=0.1,
        help="Weight for Chamfer regularizer when using laplacian loss",
    )
    p.add_argument(
        "--laplacian_k",
        type=int,
        default=8,
        help="Number of neighbors for Laplacian computation",
    )
    p.add_argument(
        "--smooth_l1_beta",
        type=float,
        default=0.0,
        help="Smooth L1 beta (0 = standard L1)",
    )
    p.add_argument(
        "--lambda_shrink",
        type=float,
        default=0.0,
        help="L2 shrinkage penalty weight for zero-GT predictions (0 = disabled)",
    )
    p.add_argument(
        "--shrink_gamma",
        type=float,
        default=0.0,
        help="Curvature gating for shrinkage (0=binary, >0=soft)",
    )
    p.add_argument(
        "--edge_threshold",
        type=float,
        default=0.0,
        help="Curvature threshold for edge masking (0=disabled)",
    )
    p.add_argument(
        "--displacement_k",
        type=int,
        default=1,
        help="NN candidates for min-of-K loss (1=standard)",
    )
    p.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file (logs to both file and stderr)",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (loads model + optimizer state)",
    )
    p.add_argument(
        "--chamfer_padding",
        type=int,
        default=10,
        help="Padding (voxels) around patch bbox when cropping originals for CD loss",
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
    use_chamfer = args.loss_type == "cd"

    if _use_multi_frame(args):
        rates = args.rates or ["r2", "r3", "r4"]
        rate_label = "_".join(rates)

        train_seqs = [s for s in MULTI_FRAME_SEQUENCES if s != args.val_sequence]
        val_seqs = [args.val_sequence]

        logger.info(f"Multi-frame mode: rates={rates}")
        logger.info(f"Train sequences: {train_seqs}")
        logger.info(f"Val sequence: {val_seqs}")
        if use_chamfer:
            logger.info(f"Chamfer mode: padding={args.chamfer_padding}")

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
            augment=not args.no_augment,
            frame_stride=args.frame_stride,
            displacement_k=args.displacement_k,
            use_chamfer=use_chamfer,
            chamfer_padding=args.chamfer_padding,
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
            max_clouds=args.max_clouds,
            augment=False,
            frame_stride=args.frame_stride,
            displacement_k=args.displacement_k,
            use_chamfer=use_chamfer,
            chamfer_padding=args.chamfer_padding,
        )
        if use_chamfer:
            collate_fn = MultiFrameDataset.collate_fn_chamfer
        else:
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


def _compute_loss(
    pred,
    sin,
    gt_disp,
    curv,
    loss_type,
    alpha,
    smooth_l1_beta,
    chamfer_weight,
    laplacian_k,
    kendall=None,
    beta=5.0,
    lambda_shrink=0.0,
    shrink_gamma=0.0,
    edge_threshold=0.0,
    gate=None,
    lambda_gate=1.0,
    mag_floor=0.1,
):
    """Dispatch to the appropriate loss function.

    Returns (loss, extras_dict) where extras_dict has per-component losses for logging.
    gt_disp is (N, 3) for K=1 or (N, K, 3) for K>1 (min-of-K).
    """
    if loss_type in ("gated", "threshold_gated"):
        return gated_displacement_loss(
            pred,
            gate,
            gt_disp,
            smooth_l1_beta=smooth_l1_beta,
            lambda_gate=lambda_gate,
        )

    # Min-of-K: gt_disp has shape (N, K, 3) with K > 1
    if gt_disp.ndim == 3 and gt_disp.shape[1] > 1:
        return (
            min_of_k_displacement_loss(
                pred,
                gt_disp,
                curv,
                alpha=alpha,
                smooth_l1_beta=smooth_l1_beta,
                lambda_shrink=lambda_shrink,
            ),
            {},
        )

    if loss_type == "stratified_disp":
        total, edge_l, flat_l = stratified_displacement_loss(
            pred,
            gt_disp,
            curv,
            kendall,
            smooth_l1_beta=smooth_l1_beta,
            beta=beta,
        )
        return total, {"edge_loss": edge_l.item(), "flat_loss": flat_l.item()}
    if loss_type == "stratified":
        total, edge_l, flat_l = stratified_loss(
            pred,
            sin,
            gt_disp,
            curv,
            kendall,
            alpha=alpha,
            k=laplacian_k,
            beta=beta,
        )
        return total, {"edge_loss": edge_l.item(), "flat_loss": flat_l.item()}
    if loss_type == "laplacian":
        lap = laplacian_loss(pred, sin, gt_disp, curv, alpha=alpha, k=laplacian_k)
        if chamfer_weight > 0:
            lap = lap + chamfer_weight * chamfer_loss(
                pred, sin, gt_disp, curv, alpha=alpha
            )
        return lap, {}
    elif loss_type == "chamfer":
        return chamfer_loss(pred, sin, gt_disp, curv, alpha=alpha), {}
    else:
        return (
            curvature_weighted_l1_loss(
                pred,
                gt_disp,
                curv,
                alpha=alpha,
                mag_floor=mag_floor,
                smooth_l1_beta=smooth_l1_beta,
                lambda_shrink=lambda_shrink,
                shrink_gamma=shrink_gamma,
                edge_threshold=edge_threshold,
            ),
            {},
        )


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    alpha,
    smooth_l1_beta=0.0,
    loss_type="displacement",
    max_grad_norm=1.0,
    chamfer_weight=0.1,
    laplacian_k=8,
    kendall=None,
    beta=5.0,
    lambda_shrink=0.0,
    shrink_gamma=0.0,
    edge_threshold=0.0,
    lambda_gate=1.0,
    mag_floor=0.1,
    use_chamfer=False,
):
    is_gated = loss_type in ("gated", "threshold_gated")
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_pred_abs = 0.0
    total_gt_abs = 0.0
    total_near_zero = 0
    total_points = 0
    total_grad_norm = 0.0
    total_extras = {}

    for batch in tqdm(loader, desc="Train", leave=False):
        coords, feats = batch[0], batch[1]
        sin = ME.SparseTensor(
            features=feats.float(),
            coordinates=coords.int(),
            device=device,
        )

        output = model(sin)
        if is_gated:
            pred, gate = output
        else:
            pred, gate = output, None

        if use_chamfer:
            orig_patches = batch[2]  # list of tensors, variable size
            loss, extras = dynamic_chamfer_loss(pred, orig_patches)
        else:
            gt_disp = batch[2].float().to(device)
            curv = batch[3].float().to(device)
            loss, extras = _compute_loss(
                pred,
                sin,
                gt_disp,
                curv,
                loss_type,
                alpha,
                smooth_l1_beta,
                chamfer_weight,
                laplacian_k,
                kendall=kendall,
                beta=beta,
                lambda_shrink=lambda_shrink,
                shrink_gamma=shrink_gamma,
                edge_threshold=edge_threshold,
                gate=gate,
                lambda_gate=lambda_gate,
                mag_floor=mag_floor,
            )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Diagnostics
        with torch.no_grad():
            pred_mag = pred.F.norm(dim=-1)
            n_pts = pred_mag.shape[0]
            total_pred_abs += pred_mag.sum().item()
            total_near_zero += (pred_mag < 0.01).sum().item()
            total_points += n_pts
            total_grad_norm += grad_norm.item()
            if not use_chamfer:
                gt_mag = batch[2].float().to(device).norm(dim=-1)
                total_gt_abs += gt_mag.sum().item()

        total_loss += loss.item()
        for k_extra, v in extras.items():
            total_extras[k_extra] = total_extras.get(k_extra, 0.0) + v
        n_batches += 1

    n = max(n_batches, 1)
    stats = {
        "loss": total_loss / n,
        "pred_mean_abs": total_pred_abs / max(total_points, 1),
        "pred_near_zero_pct": 100.0 * total_near_zero / max(total_points, 1),
        "grad_norm": total_grad_norm / n,
    }
    if not use_chamfer:
        stats["gt_mean_abs"] = total_gt_abs / max(total_points, 1)
    for k_extra, v in total_extras.items():
        stats[k_extra] = v / n
    return stats


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    alpha,
    smooth_l1_beta=0.0,
    loss_type="displacement",
    chamfer_weight=0.1,
    laplacian_k=8,
    kendall=None,
    beta=5.0,
    lambda_shrink=0.0,
    shrink_gamma=0.0,
    edge_threshold=0.0,
    lambda_gate=1.0,
    mag_floor=0.1,
    use_chamfer=False,
):
    is_gated = loss_type in ("gated", "threshold_gated")
    model.eval()
    total_loss = 0.0
    n_batches = 0
    total_pred_abs = 0.0
    total_gt_abs = 0.0
    total_near_zero = 0
    total_points = 0
    total_extras = {}

    for batch in tqdm(loader, desc="Val", leave=False):
        coords, feats = batch[0], batch[1]
        sin = ME.SparseTensor(
            features=feats.float(),
            coordinates=coords.int(),
            device=device,
        )

        output = model(sin)
        if is_gated:
            pred, gate = output
        else:
            pred, gate = output, None

        if use_chamfer:
            orig_patches = batch[2]
            loss, extras = dynamic_chamfer_loss(pred, orig_patches)
        else:
            gt_disp = batch[2].float().to(device)
            curv = batch[3].float().to(device)
            loss, extras = _compute_loss(
                pred,
                sin,
                gt_disp,
                curv,
                loss_type,
                alpha,
                smooth_l1_beta,
                chamfer_weight,
                laplacian_k,
                kendall=kendall,
                beta=beta,
                lambda_shrink=lambda_shrink,
                shrink_gamma=shrink_gamma,
                edge_threshold=edge_threshold,
                gate=gate,
                lambda_gate=lambda_gate,
                mag_floor=mag_floor,
            )

        pred_mag = pred.F.norm(dim=-1)
        n_pts = pred_mag.shape[0]
        total_pred_abs += pred_mag.sum().item()
        total_near_zero += (pred_mag < 0.01).sum().item()
        total_points += n_pts
        if not use_chamfer:
            gt_mag = batch[2].float().to(device).norm(dim=-1)
            total_gt_abs += gt_mag.sum().item()

        total_loss += loss.item()
        for k_extra, v in extras.items():
            total_extras[k_extra] = total_extras.get(k_extra, 0.0) + v
        n_batches += 1

    n = max(n_batches, 1)
    stats = {
        "loss": total_loss / n,
        "pred_mean_abs": total_pred_abs / max(total_points, 1),
        "pred_near_zero_pct": 100.0 * total_near_zero / max(total_points, 1),
    }
    if not use_chamfer:
        stats["gt_mean_abs"] = total_gt_abs / max(total_points, 1)
    for k_extra, v in total_extras.items():
        stats[k_extra] = v / n
    return stats


def _setup_logging(log_file: str = None):
    """Configure logging to stderr + optional file."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # stderr handler
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        logger.info(f"Logging to {log_file}")


def main():
    args = parse_args()
    _setup_logging(args.log_file)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Args: {vars(args)}")
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
    if args.model_type == "gated":
        model = GatedSparseUNet(
            in_channels=4, max_displacement=args.max_displacement
        ).to(device)
    elif args.model_type == "threshold":
        model = ThresholdGatedUNet(
            in_channels=4, max_displacement=args.max_displacement
        ).to(device)
    else:
        model = SparseUNet(in_channels=4, max_displacement=args.max_displacement).to(
            device
        )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.model_type} | Parameters: {n_params:,}")

    # Kendall uncertainty weights for stratified loss
    kendall = None
    if args.loss_type in ("stratified", "stratified_disp"):
        kendall = KendallUncertaintyWeights(n_tasks=2).to(device)
        logger.info(
            "Using Kendall learned uncertainty weighting (2 tasks: edge + flat)"
        )

    # AdamW with decoupled weight decay (exclude norm params)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bn" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    param_groups = [
        {"params": decay_params, "weight_decay": 1e-2},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if kendall is not None:
        param_groups.append({"params": kendall.parameters(), "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        prev_args = ckpt.get("args", {})
        prev_rates = prev_args.get("rates", [])
        curr_rates = args.rates or [args.rate or "r7"]
        same_config = (
            set(prev_rates) == set(curr_rates) and prev_args.get("lr") == args.lr
        )
        if same_config:
            # Same rates + LR: full resume (optimizer, epoch, best_val)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            logger.info(
                f"Resumed from {args.resume} (epoch {ckpt['epoch']}, "
                f"val_loss={ckpt.get('val_loss', '?')})"
            )
        else:
            # Different config (curriculum): model weights only, fresh optimizer
            logger.info(
                f"Loaded model weights from {args.resume} (epoch {ckpt['epoch']}). "
                f"Fresh optimizer for new config: rates={curr_rates}, lr={args.lr}"
            )

    # Linear warmup then cosine annealing
    warmup_epochs = args.warmup_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup from lr/10 to lr
            return 0.1 + 0.9 * epoch / warmup_epochs
        # Cosine decay to lr/100
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    n_train = len(train_dataset)
    n_val = len(val_dataset)
    n_tc = len(train_dataset.decoded_clouds)
    n_vc = len(val_dataset.decoded_clouds)
    logger.info(
        f"Train: {n_train} patches from {n_tc} clouds | "
        f"Val: {n_val} patches from {n_vc} clouds"
    )
    if args.edge_threshold > 0:
        # Log edge mask selectivity
        import numpy as np

        all_curv = np.concatenate(train_dataset.curvatures)
        edge_pct = 100.0 * (all_curv > args.edge_threshold).mean()
        logger.info(
            f"Edge mask: {edge_pct:.1f}% of points above "
            f"kappa={args.edge_threshold}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Beta annealing for stratified loss: 1 -> 10 over training
        beta = 1.0 + 9.0 * (epoch - 1) / max(args.epochs - 1, 1)

        use_chamfer = args.loss_type == "cd"

        t_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.alpha,
            args.smooth_l1_beta,
            args.loss_type,
            chamfer_weight=args.chamfer_weight,
            laplacian_k=args.laplacian_k,
            kendall=kendall,
            beta=beta,
            lambda_shrink=args.lambda_shrink,
            shrink_gamma=args.shrink_gamma,
            edge_threshold=args.edge_threshold,
            lambda_gate=args.lambda_gate,
            mag_floor=args.mag_floor,
            use_chamfer=use_chamfer,
        )
        v_stats = validate(
            model,
            val_loader,
            device,
            args.alpha,
            args.smooth_l1_beta,
            args.loss_type,
            chamfer_weight=args.chamfer_weight,
            laplacian_k=args.laplacian_k,
            kendall=kendall,
            beta=beta,
            lambda_shrink=args.lambda_shrink,
            shrink_gamma=args.shrink_gamma,
            edge_threshold=args.edge_threshold,
            lambda_gate=args.lambda_gate,
            mag_floor=args.mag_floor,
            use_chamfer=use_chamfer,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        gap = v_stats["loss"] - t_stats["loss"]
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={t_stats['loss']:.4f} val={v_stats['loss']:.4f} gap={gap:.4f} | "
            f"lr={lr:.1e} | {elapsed:.1f}s"
        )
        t, v = t_stats, v_stats
        if use_chamfer:
            logger.info(
                f"  pred_abs: train={t['pred_mean_abs']:.4f} "
                f"val={v['pred_mean_abs']:.4f} | "
                f"cd_fwd: train={t.get('cd_fwd', 0):.4f} "
                f"val={v.get('cd_fwd', 0):.4f} | "
                f"cd_rev: train={t.get('cd_rev', 0):.4f} "
                f"val={v.get('cd_rev', 0):.4f}"
            )
        else:
            logger.info(
                f"  pred_abs: train={t['pred_mean_abs']:.4f} "
                f"val={v['pred_mean_abs']:.4f} | "
                f"gt_abs: train={t['gt_mean_abs']:.4f} "
                f"val={v['gt_mean_abs']:.4f}"
            )
        logger.info(
            f"  near_zero%%: train={t['pred_near_zero_pct']:.1f} "
            f"val={v['pred_near_zero_pct']:.1f} | "
            f"grad_norm={t['grad_norm']:.4f}"
        )
        # Stratified loss diagnostics
        if "edge_loss" in t:
            kw = kendall.weights() if kendall else [1.0, 1.0]
            logger.info(
                f"  edge: t={t['edge_loss']:.4f} "
                f"v={v.get('edge_loss', 0):.4f} | "
                f"flat: t={t['flat_loss']:.4f} "
                f"v={v.get('flat_loss', 0):.4f} | "
                f"beta={beta:.1f} | "
                f"kw=[{kw[0]:.3f}, {kw[1]:.3f}]"
            )
        # Gated loss diagnostics
        if "l_gate" in t:
            logger.info(
                f"  l_disp: t={t['l_disp']:.4f} v={v.get('l_disp', 0):.4f} | "
                f"l_gate: t={t['l_gate']:.4f} v={v.get('l_gate', 0):.4f}"
            )
            logger.info(
                f"  gate: t={t['gate_mean']:.3f} v={v.get('gate_mean', 0):.3f} | "
                f"gate_nz: t={t['gate_nonzero_mean']:.3f} "
                f"v={v.get('gate_nonzero_mean', 0):.3f} | "
                f"gate_z: t={t['gate_zero_mean']:.3f} "
                f"v={v.get('gate_zero_mean', 0):.3f}"
            )

        # Save best model
        val_loss = v_stats["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = save_dir / f"best_{rate_label}.pt"
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }
            if kendall is not None:
                ckpt_data["kendall_state_dict"] = kendall.state_dict()
            torch.save(ckpt_data, ckpt_path)
            logger.info(f"  * New best (val_loss={val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % 3 == 0:
            ckpt_path = save_dir / f"epoch_{epoch}_{rate_label}.pt"
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }
            if kendall is not None:
                ckpt_data["kendall_state_dict"] = kendall.state_dict()
            torch.save(ckpt_data, ckpt_path)

    logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
