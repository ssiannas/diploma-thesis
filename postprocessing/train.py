"""Training entrypoint for displacement post-processing model.

Plain PyTorch training loop. Runs in pcgcv2 conda env.

Usage:
    python postprocessing/train.py --config configs/cd_r2.yaml
    python postprocessing/train.py --config configs/cd_r2.yaml \
        --overrides max_clouds=3 lr=1e-4
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "20")

import argparse
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

import MinkowskiEngine as ME
import torch
import yaml
from config import TrainConfig, apply_overrides, config_to_dict, load_config
from dataset import (
    MultiFrameDataset,
    RateBalancedBatchSampler,
    RateStratifiedBatchSampler,
)
from logging_utils import setup_logging
from losses import (
    KendallUncertaintyWeights,
    LossContext,
    LossFunction,
    VoxelLaplacian,
    cls_ce_loss,
    focal_bce_loss,
    get_loss_fn,
)
from model import build_model, init_cls_bias
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

MULTI_FRAME_SEQUENCES = ["longdress", "loot", "redandblack", "soldier"]
EMA_DECAY = 0.999


def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module) -> None:
    """Update EMA model weights: ema = decay * ema + (1 - decay) * model."""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(EMA_DECAY).add_(p.data, alpha=1.0 - EMA_DECAY)


def parse_args():
    p = argparse.ArgumentParser(description="Train displacement post-processor")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="key=value overrides (e.g. lr=1e-4 data.rates=[r2,r3])",
    )
    return p.parse_args()


def _build_datasets(cfg: TrainConfig):
    """Build train/val datasets using MultiFrameDataset."""
    dc = cfg.data
    lc = cfg.loss
    use_occupancy = cfg.model.model_type == "occupancy"
    if use_occupancy:
        loss_fn = None
        use_chamfer = False
    else:
        loss_fn = get_loss_fn(lc.loss_type)
        use_chamfer = loss_fn.needs_chamfer
    rates = dc.rates
    rate_label = "_".join(rates)

    if dc.overfit:
        train_seqs = [dc.val_sequence]
        val_seqs = [dc.val_sequence]
    else:
        train_seqs = [s for s in MULTI_FRAME_SEQUENCES if s != dc.val_sequence]
        val_seqs = [dc.val_sequence]

    logger.info(f"Rates: {rates}")
    logger.info(f"Train sequences: {train_seqs}")
    logger.info(f"Val sequence: {val_seqs}")
    if use_chamfer:
        logger.info(f"Chamfer mode: padding={lc.chamfer_padding}")

    logger.info("Loading training data...")
    train_ds = MultiFrameDataset(
        data_root=dc.data_root,
        sequences=train_seqs,
        rates=rates,
        patch_size=dc.patch_size,
        stride=dc.stride,
        min_points=dc.min_points,
        curvature_k=dc.curvature_k,
        max_clouds=dc.max_clouds,
        augment=dc.augment,
        frame_stride=dc.frame_stride,
        use_chamfer=use_chamfer,
        chamfer_padding=lc.chamfer_padding if not use_occupancy else 0,
        rate_jitter=dc.rate_jitter,
        use_occupancy=use_occupancy,
    )
    logger.info("Loading validation data...")
    val_ds = MultiFrameDataset(
        data_root=dc.data_root,
        sequences=val_seqs,
        rates=rates,
        patch_size=dc.patch_size,
        stride=dc.stride,
        min_points=dc.min_points,
        curvature_k=dc.curvature_k,
        max_clouds=dc.max_clouds,
        augment=False,
        frame_stride=dc.frame_stride,
        use_chamfer=use_chamfer,
        chamfer_padding=lc.chamfer_padding if not use_occupancy else 0,
        use_occupancy=use_occupancy,
    )
    if use_occupancy:
        collate_fn = MultiFrameDataset.collate_fn_occupancy
    elif use_chamfer:
        collate_fn = MultiFrameDataset.collate_fn_chamfer
    else:
        collate_fn = MultiFrameDataset.collate_fn

    # Batch sampling strategy for multi-rate
    train_sampler = None
    val_sampler = None
    is_film = cfg.model.model_type in (
        "film",
        "film_head",
        "film_head_v2",
        "film_head_v3",
        "occupancy",
    )
    if len(rates) > 1 and train_ds.patch_rates:
        if is_film:
            # FiLM: mixed-rate batches with balanced rate representation
            train_sampler = RateBalancedBatchSampler(
                train_ds.patch_rates, cfg.batch_size
            )
            val_sampler = RateBalancedBatchSampler(val_ds.patch_rates, cfg.batch_size)
            logger.info(
                f"Rate-balanced batching (FiLM): "
                f"{len(train_sampler)} train batches, {len(val_sampler)} val batches"
            )
        elif cfg.rate_stratified:
            # Non-FiLM: single-rate batches
            train_sampler = RateStratifiedBatchSampler(
                train_ds.patch_rates, cfg.batch_size
            )
            val_sampler = RateStratifiedBatchSampler(val_ds.patch_rates, cfg.batch_size)
            logger.info(
                f"Rate-stratified batching: "
                f"{len(train_sampler)} train batches, {len(val_sampler)} val batches"
            )

    return train_ds, val_ds, collate_fn, rate_label, train_sampler, val_sampler


def _rate_to_label(rate_val: float, rate_repr: str) -> str:
    """Convert a numeric rate value back to a rate label (e.g. 'r2') for logging."""
    if rate_repr in ("bpp", "fourier"):
        from dataset import RATE_BPP, bpp_to_log

        # rate_val is in log-space; compare in log-space
        best_key = min(RATE_BPP, key=lambda k: abs(bpp_to_log(RATE_BPP[k]) - rate_val))
        return best_key
    return f"r{int(round(rate_val * 7))}"


def train_one_epoch_occupancy(
    model,
    loader,
    optimizer,
    device,
    cfg: TrainConfig,
    focal_gamma: float = 2.0,
    max_grad_norm: float = 1.0,
    ema_model=None,
):
    model.train()
    total_loss = 0.0
    total_correct = total_tp = total_fp = total_fn = total_points = 0
    total_grad_norm = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        coords, feats, labels, rates = batch[0], batch[1], batch[2], batch[3]
        rates = rates.to(device)

        # Pack labels into feats so ME reordering keeps them aligned
        feats_with_label = torch.cat(
            [feats.float(), labels.unsqueeze(1).float()], dim=1
        )
        sin_label = ME.SparseTensor(
            features=feats_with_label, coordinates=coords.int(), device=device
        )
        sin = ME.SparseTensor(
            features=sin_label.F[:, :-1],
            coordinate_map_key=sin_label.coordinate_map_key,
            coordinate_manager=sin_label.coordinate_manager,
        )
        labels_aligned = sin_label.F[:, -1]

        pred = model(sin, rates)
        logits = pred.F.squeeze(-1)
        loss = focal_bce_loss(logits, labels_aligned, gamma=focal_gamma)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if ema_model is not None:
            update_ema(ema_model, model)

        with torch.no_grad():
            pred_occ = torch.sigmoid(logits) > 0.5
            gt_occ = labels_aligned > 0.5
            total_correct += (pred_occ == gt_occ).sum().item()
            total_tp += (pred_occ & gt_occ).sum().item()
            total_fp += (pred_occ & ~gt_occ).sum().item()
            total_fn += (~pred_occ & gt_occ).sum().item()
            total_points += logits.shape[0]
            total_grad_norm += grad_norm.item()

        total_loss += loss.item()

    n = max(len(loader), 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "loss": total_loss / n,
        "acc": 100.0 * total_correct / max(total_points, 1),
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * f1,
        "grad_norm": total_grad_norm / n,
    }


@torch.no_grad()
def validate_occupancy(
    model, loader, device, cfg: TrainConfig, focal_gamma: float = 2.0, ema_model=None
):
    eval_model = ema_model if ema_model is not None else model
    eval_model.eval()
    total_loss = 0.0
    total_correct = total_tp = total_fp = total_fn = total_points = 0
    # Per-rate breakdown
    rate_stats = defaultdict(
        lambda: {"correct": 0, "tp": 0, "fp": 0, "fn": 0, "total": 0}
    )

    for batch in tqdm(loader, desc="Val", leave=False):
        coords, feats, labels, rates = batch[0], batch[1], batch[2], batch[3]
        rates = rates.to(device)

        feats_with_label = torch.cat(
            [feats.float(), labels.unsqueeze(1).float()], dim=1
        )
        sin_label = ME.SparseTensor(
            features=feats_with_label, coordinates=coords.int(), device=device
        )
        sin = ME.SparseTensor(
            features=sin_label.F[:, :-1],
            coordinate_map_key=sin_label.coordinate_map_key,
            coordinate_manager=sin_label.coordinate_manager,
        )
        labels_aligned = sin_label.F[:, -1]

        pred = eval_model(sin, rates)
        logits = pred.F.squeeze(-1)
        loss = focal_bce_loss(logits, labels_aligned, gamma=focal_gamma)

        pred_occ = torch.sigmoid(logits) > 0.5
        gt_occ = labels_aligned > 0.5
        total_correct += (pred_occ == gt_occ).sum().item()
        total_tp += (pred_occ & gt_occ).sum().item()
        total_fp += (pred_occ & ~gt_occ).sum().item()
        total_fn += (~pred_occ & gt_occ).sum().item()
        total_points += logits.shape[0]
        total_loss += loss.item()

        # Per-rate: each point's rate comes from its batch element
        batch_idx = pred.C[:, 0].long()
        for bi in range(rates.shape[0]):
            r_key = _rate_to_label(rates[bi].item(), cfg.model.film_rate_repr)
            mask = batch_idx == bi
            rs = rate_stats[r_key]
            rs["correct"] += (pred_occ[mask] == gt_occ[mask]).sum().item()
            rs["tp"] += (pred_occ[mask] & gt_occ[mask]).sum().item()
            rs["fp"] += (pred_occ[mask] & ~gt_occ[mask]).sum().item()
            rs["fn"] += (~pred_occ[mask] & gt_occ[mask]).sum().item()
            rs["total"] += mask.sum().item()

    n = max(len(loader), 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    stats = {
        "loss": total_loss / n,
        "acc": 100.0 * total_correct / max(total_points, 1),
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * f1,
    }
    for r_key, rs in rate_stats.items():
        if rs["total"] > 0:
            r_prec = rs["tp"] / max(rs["tp"] + rs["fp"], 1)
            r_rec = rs["tp"] / max(rs["tp"] + rs["fn"], 1)
            r_f1 = 2 * r_prec * r_rec / max(r_prec + r_rec, 1e-8)
            stats[f"{r_key}_f1"] = 100.0 * r_f1
            stats[f"{r_key}_recall"] = 100.0 * r_rec
    return stats


def _log_epoch_occupancy(epoch, total_epochs, t, v, lr, elapsed):
    gap = v["loss"] - t["loss"]
    logger.info(
        f"Epoch {epoch:3d}/{total_epochs} | "
        f"train={t['loss']:.4f} val={v['loss']:.4f} gap={gap:.4f} | "
        f"lr={lr:.1e} | {elapsed:.1f}s"
    )
    logger.info(
        f"  train: acc={t['acc']:.1f}% prec={t['precision']:.1f}% "
        f"rec={t['recall']:.1f}% f1={t['f1']:.1f}% | grad={t['grad_norm']:.4f}"
    )
    logger.info(
        f"  val:   acc={v['acc']:.1f}% prec={v['precision']:.1f}% "
        f"rec={v['recall']:.1f}% f1={v['f1']:.1f}%"
    )
    rate_f1_keys = sorted(k for k in v if k.endswith("_f1"))
    if rate_f1_keys:
        parts = [
            f"{k.split('_f1')[0]}: f1={v[k]:.1f}% rec={v.get(k.replace('_f1','_recall'), 0):.1f}%"
            for k in rate_f1_keys
        ]
        logger.info(f"  per_rate: {' | '.join(parts)}")


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    cfg: TrainConfig,
    loss_fn: LossFunction,
    max_grad_norm=1.0,
    kendall=None,
    lap_op=None,
    kendall_rate=None,
    ema_model=None,
):
    lc = cfg.loss
    use_chamfer = loss_fn.needs_chamfer
    no_curvature = not cfg.data.use_curvature
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_pred_abs = 0.0
    total_gt_abs = 0.0
    total_near_zero = 0
    total_points = 0
    total_grad_norm = 0.0
    total_extras = {}

    is_film = hasattr(model, "film_head") or hasattr(model, "film_gen")
    is_v3 = hasattr(model, "n_classes")
    use_ce = is_v3 and lc.ce_weight > 0
    # Per-rate tracking
    rate_stats = defaultdict(lambda: {"pred_abs": 0.0, "points": 0, "batches": 0})
    total_argmax_correct = 0
    total_argmax_points = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        coords, feats = batch[0], batch[1]
        if no_curvature:
            feats = feats[:, :3]
        sin = ME.SparseTensor(
            features=feats.float(),
            coordinates=coords.int(),
            device=device,
        )

        is_cls_focal = lc.loss_type == "cls_focal"
        if is_film:
            rates = batch[4].to(device)  # (B,) tensor
            if use_ce or is_cls_focal:
                pred, logits = model(sin, rates, return_logits=True)
            else:
                pred = model(sin, rates)
        else:
            pred = model(sin)

        use_kendall_rates = cfg.loss.use_kendall_rates and kendall_rate is not None
        use_dynamic_rates = cfg.loss.use_dynamic_rates
        use_per_patch = use_kendall_rates or use_dynamic_rates
        patch_weights = None
        if is_film and lc.rate_weights and not use_per_patch:
            rate_weight_map = dict(zip(cfg.data.rates, lc.rate_weights))
            patch_weights = torch.tensor(
                [
                    rate_weight_map.get(
                        _rate_to_label(r.item(), cfg.model.film_rate_repr), 1.0
                    )
                    for r in rates
                ],
                device=device,
            )

        ctx = LossContext(
            curvature=batch[3].float().to(device),
            original_patches=batch[2] if use_chamfer else None,
            gt_displacement=(
                logits
                if is_cls_focal
                else (None if use_chamfer else batch[2].float().to(device))
            ),
            input_sparse=None if use_chamfer else sin,
            kendall=kendall,
            lap_op=lap_op,
            patch_weights=patch_weights,
            return_per_patch=use_per_patch,
        )

        loss, extras = loss_fn(pred, lc, ctx)

        if use_kendall_rates or use_dynamic_rates:
            patch_cd_map = extras.pop("patch_cd_by_batch", {})
            rate_tensors = []
            for rate_name in cfg.data.rates:
                tensors = [
                    v
                    for b_idx, v in patch_cd_map.items()
                    if _rate_to_label(rates[b_idx].item(), cfg.model.film_rate_repr)
                    == rate_name
                ]
                if tensors:
                    rate_tensors.append(torch.stack(tensors).mean())
                else:
                    rate_tensors.append(
                        torch.zeros(1, device=device, requires_grad=True).squeeze()
                    )

            if use_kendall_rates:
                loss = kendall_rate(*rate_tensors)
                for i, r in enumerate(cfg.data.rates):
                    extras[f"kw_{r}"] = kendall_rate.weights()[i]
            else:
                rate_stack = torch.stack(rate_tensors)
                weights = rate_stack.detach().clamp(min=1e-6)
                weights = (weights / weights.mean()).clamp(0.1, 10.0)
                loss = (weights * rate_stack).sum()
                for i, r in enumerate(cfg.data.rates):
                    extras[f"dw_{r}"] = weights[i].item()

        if lc.cd_loss_scale != 1.0:
            loss = loss * lc.cd_loss_scale

        if use_ce:
            ce = cls_ce_loss(
                logits,
                pred.C[:, 1:],
                batch[2],
                pred.C[:, 0].long(),
                num_classes=model.n_classes,
                focal_gamma=lc.ce_focal_gamma,
            )
            loss = loss + lc.ce_weight * ce
            extras["ce_loss"] = ce.item()

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if ema_model is not None:
            update_ema(ema_model, model)

        # Diagnostics
        with torch.no_grad():
            if use_ce:
                argmax_disp = logits.detach().argmax(dim=-1) - model.n_classes // 2
                # Reuse the GT from CE (cheap since we just computed it above)
                ce_batch_idx = pred.C[:, 0].long()
                ce_decoded = pred.C[:, 1:]
                max_offset = model.n_classes // 2
                gt_disp = torch.zeros(
                    ce_decoded.shape[0], 3, dtype=torch.long, device=device
                )
                for i, b in enumerate(ce_batch_idx.unique()):
                    mask = ce_batch_idx == b
                    dec_b = ce_decoded[mask].float()
                    orig_b = batch[2][i].to(device)
                    if orig_b.shape[0] > 0:
                        tree = cKDTree(orig_b.cpu().numpy())
                        nn_idx = torch.from_numpy(
                            tree.query(dec_b.cpu().numpy(), k=1, workers=1)[1]
                        ).to(device)
                        disp_b = (
                            (orig_b[nn_idx] - dec_b)
                            .round()
                            .long()
                            .clamp(-max_offset, max_offset)
                        )
                        gt_disp[mask] = disp_b
                correct = (argmax_disp == gt_disp).float()
                total_argmax_correct += correct.sum().item()
                total_argmax_points += correct.numel()

            pred_mag = pred.F.norm(dim=-1)
            n_pts = pred_mag.shape[0]
            total_pred_abs += pred_mag.sum().item()
            total_near_zero += (pred_mag < 0.01).sum().item()
            total_points += n_pts
            total_grad_norm += grad_norm.item()
            if not use_chamfer:
                gt_mag = batch[2].float().to(device).norm(dim=-1)
                total_gt_abs += gt_mag.sum().item()
            # Per-rate diagnostics
            if is_film:
                batch_indices = pred.C[:, 0].long()
                for bi in range(rates.shape[0]):
                    r_key = _rate_to_label(rates[bi].item(), cfg.model.film_rate_repr)
                    mask = batch_indices == bi
                    rs = rate_stats[r_key]
                    rs["pred_abs"] += pred_mag[mask].sum().item()
                    rs["points"] += mask.sum().item()
                    rs["batches"] += 1

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
    if total_argmax_points > 0:
        stats["argmax_acc"] = 100.0 * total_argmax_correct / total_argmax_points
    for k_extra, v in total_extras.items():
        stats[k_extra] = v / n
    for r_key, rs in rate_stats.items():
        stats[f"{r_key}_pred_abs"] = rs["pred_abs"] / max(rs["points"], 1)
    return stats


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    cfg: TrainConfig,
    loss_fn: LossFunction,
    kendall=None,
    lap_op=None,
    kendall_rate=None,
    ema_model=None,
):
    lc = cfg.loss
    use_chamfer = loss_fn.needs_chamfer
    no_curvature = not cfg.data.use_curvature
    # Use EMA model for evaluation when available
    eval_model = ema_model if ema_model is not None else model
    eval_model.eval()
    total_loss = 0.0
    n_batches = 0
    total_pred_abs = 0.0
    total_gt_abs = 0.0
    total_near_zero = 0
    total_points = 0
    total_extras = {}

    is_film = hasattr(eval_model, "film_head") or hasattr(eval_model, "film_gen")
    is_v3 = hasattr(eval_model, "n_classes")
    rate_stats = defaultdict(lambda: {"pred_abs": 0.0, "points": 0})
    dwa_rate_losses: dict = defaultdict(float)
    dwa_rate_n: dict = defaultdict(int)
    total_argmax_correct = 0
    total_argmax_points = 0
    total_argmax_nonzero_correct = 0
    total_argmax_nonzero_points = 0
    total_max_prob = 0.0
    rate_nz_stats = defaultdict(lambda: {"correct": 0.0, "total": 0})
    total_pred_nonzero_correct = 0.0
    total_pred_nonzero_points = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        coords, feats = batch[0], batch[1]
        if no_curvature:
            feats = feats[:, :3]
        sin = ME.SparseTensor(
            features=feats.float(),
            coordinates=coords.int(),
            device=device,
        )

        is_cls_focal = lc.loss_type == "cls_focal"
        if is_film:
            rates = batch[4].to(device)  # (B,) tensor
            if is_v3:
                pred, logits = eval_model(sin, rates, return_logits=True)
            else:
                pred = eval_model(sin, rates)
        else:
            pred = eval_model(sin)

        use_kendall_rates = cfg.loss.use_kendall_rates and kendall_rate is not None
        use_dynamic_rates = cfg.loss.use_dynamic_rates
        use_dwa_collect = getattr(cfg.loss, "use_dwa", False) and is_film
        use_per_patch = use_kendall_rates or use_dynamic_rates or use_dwa_collect
        patch_weights = None
        if is_film and lc.rate_weights and not use_per_patch:
            rate_weight_map = dict(zip(cfg.data.rates, lc.rate_weights))
            patch_weights = torch.tensor(
                [
                    rate_weight_map.get(
                        _rate_to_label(r.item(), cfg.model.film_rate_repr), 1.0
                    )
                    for r in rates
                ],
                device=device,
            )

        ctx = LossContext(
            curvature=batch[3].float().to(device),
            original_patches=batch[2] if use_chamfer else None,
            gt_displacement=(
                logits
                if is_cls_focal
                else (None if use_chamfer else batch[2].float().to(device))
            ),
            input_sparse=None if use_chamfer else sin,
            kendall=kendall,
            lap_op=lap_op,
            patch_weights=patch_weights,
            return_per_patch=use_per_patch,
        )

        loss, extras = loss_fn(pred, lc, ctx)

        if use_kendall_rates or use_dynamic_rates:
            patch_cd_map = extras.pop("patch_cd_by_batch", {})
            rate_tensors = []
            for rate_name in cfg.data.rates:
                tensors = [
                    v
                    for b_idx, v in patch_cd_map.items()
                    if _rate_to_label(rates[b_idx].item(), cfg.model.film_rate_repr)
                    == rate_name
                ]
                if tensors:
                    rate_tensors.append(torch.stack(tensors).mean())
                else:
                    rate_tensors.append(torch.zeros(1, device=device).squeeze())

            if use_kendall_rates:
                rate_stack = torch.stack(rate_tensors)
                rate_stack = rate_stack / rate_stack.detach().mean().clamp(min=1e-8)
                loss = kendall_rate(*rate_stack.unbind())
                for i, r in enumerate(cfg.data.rates):
                    extras[f"kw_{r}"] = kendall_rate.weights()[i]
            else:
                rate_stack = torch.stack(rate_tensors)
                weights = rate_stack.detach().clamp(min=1e-6)
                weights = (weights / weights.mean()).clamp(0.1, 10.0)
                loss = (weights * rate_stack).sum()
                for i, r in enumerate(cfg.data.rates):
                    extras[f"dw_{r}"] = weights[i].item()
        elif use_dwa_collect:
            patch_cd_map = extras.pop("patch_cd_by_batch", {})
            for rate_name in cfg.data.rates:
                tensors = [
                    v
                    for b_idx, v in patch_cd_map.items()
                    if _rate_to_label(rates[b_idx].item(), cfg.model.film_rate_repr)
                    == rate_name
                ]
                if tensors:
                    dwa_rate_losses[rate_name] += torch.stack(tensors).mean().item()
                    dwa_rate_n[rate_name] += 1

        if lc.cd_loss_scale != 1.0:
            loss = loss * lc.cd_loss_scale

        pred_mag = pred.F.norm(dim=-1)
        n_pts = pred_mag.shape[0]
        total_pred_abs += pred_mag.sum().item()
        total_near_zero += (pred_mag < 0.01).sum().item()
        total_points += n_pts
        # Per-rate diagnostics
        if is_film:
            batch_indices = pred.C[:, 0].long()
            for bi in range(rates.shape[0]):
                r_key = _rate_to_label(rates[bi].item(), cfg.model.film_rate_repr)
                mask = batch_indices == bi
                rs = rate_stats[r_key]
                rs["pred_abs"] += pred_mag[mask].sum().item()
                rs["points"] += mask.sum().item()
        if not use_chamfer:
            gt_mag = batch[2].float().to(device).norm(dim=-1)
            total_gt_abs += gt_mag.sum().item()
        # Argmax accuracy + max_prob for V3 (pred.F is already argmax in eval mode)
        if is_v3 and use_chamfer:
            val_batch_idx = pred.C[:, 0].long()
            val_decoded = pred.C[:, 1:]
            max_offset = eval_model.n_classes // 2
            gt_disp = torch.zeros(
                val_decoded.shape[0], 3, dtype=torch.long, device=device
            )
            for i, b in enumerate(val_batch_idx.unique()):
                mask = val_batch_idx == b
                dec_b = val_decoded[mask].float()
                orig_b = batch[2][i].to(device)
                if orig_b.shape[0] > 0:
                    tree = cKDTree(orig_b.cpu().numpy())
                    nn_idx = torch.from_numpy(
                        tree.query(dec_b.cpu().numpy(), k=1, workers=1)[1]
                    ).to(device)
                    disp_b = (
                        (orig_b[nn_idx] - dec_b)
                        .round()
                        .long()
                        .clamp(-max_offset, max_offset)
                    )
                    gt_disp[mask] = disp_b
            correct = (pred.F.round().long() == gt_disp).float()
            total_argmax_correct += correct.sum().item()
            total_argmax_points += correct.numel()
            # Accuracy only on points where GT is non-zero in any dimension
            nonzero_mask = (gt_disp != 0).any(dim=-1)
            if nonzero_mask.sum() > 0:
                correct_nz = correct[nonzero_mask]
                total_argmax_nonzero_correct += correct_nz.sum().item()
                total_argmax_nonzero_points += correct_nz.numel()
            # Per-rate acc_nonzero
            for b in val_batch_idx.unique():
                mask_b = val_batch_idx == b
                r_key = _rate_to_label(rates[b.item()].item(), cfg.model.film_rate_repr)
                nz_b = nonzero_mask[mask_b]
                if nz_b.sum() > 0:
                    rate_nz_stats[r_key]["correct"] += (
                        correct[mask_b][nz_b].sum().item()
                    )
                    rate_nz_stats[r_key]["total"] += correct[mask_b][nz_b].numel()
            # Non-zero prediction precision: when model predicts non-zero, is GT also non-zero?
            pred_nz_mask = (pred.F.round().long() != 0).any(dim=-1)
            total_pred_nonzero_correct += (
                nonzero_mask[pred_nz_mask].float().sum().item()
            )
            total_pred_nonzero_points += pred_nz_mask.sum().item()
            # max_prob: mean of per-point peak softmax probability (distribution sharpness)
            probs = torch.softmax(logits, dim=-1)
            total_max_prob += probs.max(dim=-1).values.mean().item()

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
    if total_argmax_points > 0:
        stats["argmax_acc"] = 100.0 * total_argmax_correct / total_argmax_points
        stats["max_prob"] = total_max_prob / n
    if total_argmax_nonzero_points > 0:
        stats["argmax_acc_nonzero"] = (
            100.0 * total_argmax_nonzero_correct / total_argmax_nonzero_points
        )
    for r_key, rs in rate_nz_stats.items():
        if rs["total"] > 0:
            stats[f"{r_key}_acc_nonzero"] = 100.0 * rs["correct"] / rs["total"]
    if total_pred_nonzero_points > 0:
        stats["nonzero_precision"] = (
            100.0 * total_pred_nonzero_correct / total_pred_nonzero_points
        )
    for k_extra, v in total_extras.items():
        stats[k_extra] = v / n
    for r_key, rs in rate_stats.items():
        stats[f"{r_key}_pred_abs"] = rs["pred_abs"] / max(rs["points"], 1)
    for r in cfg.data.rates:
        if dwa_rate_n[r] > 0:
            stats[f"dwa_{r}_loss"] = dwa_rate_losses[r] / dwa_rate_n[r]
    return stats


def _log_epoch(epoch, total_epochs, t, v, lr, elapsed, use_chamfer):
    """Log epoch stats. Automatically logs extras based on which keys are present."""
    gap = v["loss"] - t["loss"]
    logger.info(
        f"Epoch {epoch:3d}/{total_epochs} | "
        f"train={t['loss']:.4f} val={v['loss']:.4f} gap={gap:.4f} | "
        f"lr={lr:.1e} | {elapsed:.1f}s"
    )

    # Base stats
    pred_line = f"  pred_abs: t={t['pred_mean_abs']:.4f} v={v['pred_mean_abs']:.4f}"
    if use_chamfer:
        pred_line += (
            f" | cd_fwd: t={t.get('cd_fwd', 0):.4f} v={v.get('cd_fwd', 0):.4f}"
            f" | cd_rev: t={t.get('cd_rev', 0):.4f} v={v.get('cd_rev', 0):.4f}"
        )
    else:
        pred_line += f" | gt_abs: t={t['gt_mean_abs']:.4f} v={v['gt_mean_abs']:.4f}"
    logger.info(pred_line)
    nz_t = t["pred_near_zero_pct"]
    nz_v = v["pred_near_zero_pct"]
    logger.info(
        f"  near_zero%: t={nz_t:.1f} v={nz_v:.1f} | grad_norm={t['grad_norm']:.4f}"
    )
    if "ce_loss" in t:
        ce_t = t["ce_loss"]
        logger.info(f"  ce_loss: t={ce_t:.4f}")
    if "argmax_acc" in t or "argmax_acc" in v:
        acc_t = t.get("argmax_acc", float("nan"))
        acc_v = v.get("argmax_acc", float("nan"))
        max_prob_v = v.get("max_prob", float("nan"))
        acc_nz_v = v.get("argmax_acc_nonzero", float("nan"))
        logger.info(
            f"  argmax_acc%: t={acc_t:.2f} v={acc_v:.2f} | max_prob(v)={max_prob_v:.3f} | acc_nonzero(v)={acc_nz_v:.2f}"
        )
    nz_rate_keys = sorted(
        k for k in v if k.endswith("_acc_nonzero") and k != "argmax_acc_nonzero"
    )
    if nz_rate_keys or "nonzero_precision" in v:
        parts = [f"{k.split('_acc')[0]}={v[k]:.1f}%" for k in nz_rate_keys]
        prec = v.get("nonzero_precision", float("nan"))
        logger.info(
            f"  acc_nonzero/rate: {' | '.join(parts)} | nz_precision={prec:.1f}%"
        )

    # Per-rate pred_abs (auto-detected)
    rate_keys = sorted(k for k in t if k.endswith("_pred_abs") and k != "pred_mean_abs")
    if rate_keys:
        parts = [
            f"{k.split('_pred')[0]}: t={t[k]:.4f} v={v.get(k, 0):.4f}"
            for k in rate_keys
        ]
        logger.info(f"  per_rate_abs: {' | '.join(parts)}")

    # Kendall rate weights (auto-detected by kw_r* keys)
    kw_rate_keys = sorted(k for k in t if k.startswith("kw_r"))
    if kw_rate_keys:
        parts = [f"{k}: {t[k]:.4f}" for k in kw_rate_keys]
        logger.info(f"  kendall_rate_weights: {' | '.join(parts)}")

    # Dynamic rate weights (auto-detected by dw_r* keys)
    dw_rate_keys = sorted(k for k in t if k.startswith("dw_r"))
    if dw_rate_keys:
        parts = [f"{k}: {t[k]:.4f}" for k in dw_rate_keys]
        logger.info(f"  dynamic_rate_weights: {' | '.join(parts)}")

    # Loss-specific extras (auto-detected from stats keys)
    extra_groups = [
        # (key_to_check, format_fn)
        (
            "vlap",
            lambda: (
                f"  vlap: t={t['vlap']:.4f} v={v.get('vlap', 0):.4f}"
                f" | kw: cd={t['kw_cd']:.3f} vlap={t['kw_vlap']:.3f}"
            ),
        ),
        ("lap", lambda: f"  lap: t={t['lap']:.4f} v={v.get('lap', 0):.4f}"),
        (
            "flat_penalty",
            lambda: (
                f"  flat_penalty: t={t['flat_penalty']:.4f}"
                f" v={v.get('flat_penalty', 0):.4f}"
                f" | edge_pct: t={t['edge_pct']:.1f}"
                f" v={v.get('edge_pct', 0):.1f}"
            ),
        ),
        (
            "edge_loss",
            lambda: (
                f"  edge: t={t['edge_loss']:.4f}"
                f" v={v.get('edge_loss', 0):.4f}"
                f" | flat: t={t['flat_loss']:.4f}"
                f" v={v.get('flat_loss', 0):.4f}"
            ),
        ),
    ]
    for key, fmt_fn in extra_groups:
        if key in t:
            for line in fmt_fn().split("\n"):
                logger.info(line)


def _apply_dwa_weights(cfg, prev_losses: dict, curr_losses: dict) -> None:
    """Update cfg.loss.rate_weights in-place using DWA (Liu et al., CVPR 2019).

    w_i ∝ exp(L_i(t-1) / L_i(t-2) / T)  -- tasks converging fast get downweighted.
    An optional per-rate floor prevents any task from being starved.
    """
    rates = cfg.data.rates
    T = cfg.loss.dwa_temp
    floors = cfg.loss.dwa_floor if cfg.loss.dwa_floor else [0.0] * len(rates)
    r = [
        curr_losses.get(rate, 1.0) / max(prev_losses.get(rate, 1.0), 1e-8)
        for rate in rates
    ]
    raw = [math.exp(ri / T) for ri in r]
    total = sum(raw)
    n = len(rates)
    cfg.loss.rate_weights = [
        max(fl, raw_i / total * n) for fl, raw_i in zip(floors, raw)
    ]


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.overrides:
        apply_overrides(cfg, args.overrides)

    # Resolve relative paths from project root (parent of postprocessing/)
    project_root = Path(__file__).resolve().parent.parent
    for attr in ("data_root",):
        val = getattr(cfg.data, attr)
        if not Path(val).is_absolute():
            setattr(cfg.data, attr, str(project_root / val))
    if not Path(cfg.save_dir).is_absolute():
        cfg.save_dir = str(project_root / cfg.save_dir)
    if cfg.pretrained_backbone and not Path(cfg.pretrained_backbone).is_absolute():
        cfg.pretrained_backbone = str(project_root / cfg.pretrained_backbone)

    if cfg.num_workers < 0:
        cfg.num_workers = os.cpu_count() or 8

    setup_logging(cfg.log_file)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Config:\n%s", yaml.dump(config_to_dict(cfg), default_flow_style=False).rstrip()
    )

    train_dataset, val_dataset, collate_fn, rate_label, train_sampler, val_sampler = (
        _build_datasets(cfg)
    )

    if train_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
        )

    # Model
    use_occupancy = cfg.model.model_type == "occupancy"
    in_ch = 3 if not cfg.data.use_curvature else 4
    if use_occupancy:
        in_ch = 4  # [x_norm, y_norm, z_norm, is_occupied] -- always 4 for occupancy
    mc = cfg.model
    model = build_model(
        mc.model_type,
        in_channels=in_ch,
        max_displacement=mc.max_displacement,
        film_embed_dim=mc.film_embed_dim,
        rate_repr=mc.film_rate_repr,
        quantize_output=mc.quantize_output,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {mc.model_type} | Parameters: {n_params:,} trainable")

    # Pretrained backbone: for occupancy models, skip conv_in and conv_out
    # (channel semantics change; backbone ResBlocks are compatible)
    if cfg.pretrained_backbone:
        bb_ckpt = torch.load(cfg.pretrained_backbone, map_location=device)
        bb_sd = bb_ckpt.get("ema_state_dict", bb_ckpt["model_state_dict"])
        # Skip conv_in if input channel count differs (e.g. nocurv 3-ch vs curvature 4-ch checkpoint)
        # Skip conv_out if output count differs (e.g. occupancy 1-ch vs displacement 3-ch)
        # Skip layers only when shapes are incompatible, not unconditionally
        skip_prefixes = []
        model_params = dict(model.named_parameters())
        for prefix in ("conv_in", "conv_out"):
            sample_key = next((k for k in bb_sd if k.startswith(prefix)), None)
            if sample_key is not None and sample_key in model_params:
                if bb_sd[sample_key].shape != model_params[sample_key].shape:
                    skip_prefixes.append(prefix)
        if skip_prefixes:
            bb_sd = {
                k: v
                for k, v in bb_sd.items()
                if not any(k.startswith(p) for p in skip_prefixes)
            }
        missing, unexpected = model.load_state_dict(bb_sd, strict=False)
        logger.info(
            f"Loaded pretrained backbone from {cfg.pretrained_backbone} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if not use_occupancy:
            # Displacement models: freeze backbone as before
            if mc.model_type in ("film_head_v2", "film_head_v3"):
                trainable_prefixes = (
                    "dec_block1.",
                    "conv_out.",
                    "cls_head.",
                    "rate_encoder.",
                    "film_head.",
                )
            else:
                trainable_prefixes = ("film_gen.", "film_proj.")
            n_frozen = 0
            for name, param in model.named_parameters():
                if not any(name.startswith(p) for p in trainable_prefixes):
                    param.requires_grad = False
                    n_frozen += 1
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Frozen {n_frozen} params; {n_trainable:,} trainable")

    # EMA model: fresh instance with copied weights (deepcopy fails on ME C++ ops)
    ema_model = build_model(
        mc.model_type,
        in_channels=in_ch,
        max_displacement=mc.max_displacement,
        film_embed_dim=mc.film_embed_dim,
        rate_repr=mc.film_rate_repr,
        quantize_output=mc.quantize_output,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)
    logger.info(f"EMA model initialized (decay={EMA_DECAY})")

    # Bias init for classification model: prevents zero-collapse
    if mc.model_type == "film_head_v3" and cfg.loss.loss_type == "cls_focal":
        p_zero = getattr(cfg.loss, "cls_p_zero", 0.70)
        init_cls_bias(model, p_zero)
        init_cls_bias(ema_model, p_zero)
        logger.info(f"Initialized cls_head zero-class bias (p_zero={p_zero:.2f})")

    # Kendall uncertainty weights + VoxelLaplacian (derived from loss function)
    loss_fn = None if use_occupancy else get_loss_fn(cfg.loss.loss_type)
    kendall = None
    lap_op = None
    kendall_rate = None
    if loss_fn is not None and loss_fn.needs_kendall:
        kendall = KendallUncertaintyWeights(n_tasks=2).to(device)
        logger.info("Using Kendall learned uncertainty weighting (2 tasks)")
    if loss_fn is not None and loss_fn.needs_lap_op:
        lap_op = VoxelLaplacian().to(device)
        logger.info("Using fixed VoxelLaplacian conv")
    if not use_occupancy and cfg.loss.use_kendall_rates:
        n_rates = len(cfg.data.rates)
        kendall_rate = KendallUncertaintyWeights(n_tasks=n_rates).to(device)
        # Initialize from static prior if provided: log_var = log(1 / (2*w))
        # This encodes domain knowledge (r2 richest signal) as a starting point
        # that Kendall can refine, rather than starting from equal weights.
        if cfg.loss.rate_weights and len(cfg.loss.rate_weights) == n_rates:
            init_log_vars = [math.log(1.0 / (2.0 * w)) for w in cfg.loss.rate_weights]
            with torch.no_grad():
                kendall_rate.log_vars.copy_(torch.tensor(init_log_vars))
            init_strs = [f"{v:.3f}" for v in init_log_vars]
            logger.info(
                f"Kendall log_vars initialized from prior "
                f"{cfg.loss.rate_weights}: {init_strs}"
            )
        logger.info(
            f"Using Kendall learned rate weighting ({n_rates} rates: {cfg.data.rates})"
        )

    # AdamW with decoupled weight decay (exclude norm params)
    # Support differential LR: encoder at encoder_lr_scale * lr
    enc_prefixes = ("conv_in", "bn_in", "enc_block", "down", "bn_down", "bottleneck")
    enc_decay, enc_nodecay = [], []
    dec_decay, dec_nodecay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_encoder = any(name.startswith(p) for p in enc_prefixes)
        is_norm = "bn" in name or "norm" in name
        if is_encoder:
            (enc_nodecay if is_norm else enc_decay).append(param)
        else:
            (dec_nodecay if is_norm else dec_decay).append(param)

    enc_lr = cfg.lr * cfg.encoder_lr_scale
    param_groups = [
        {"params": enc_decay, "lr": enc_lr, "weight_decay": cfg.weight_decay},
        {"params": enc_nodecay, "lr": enc_lr, "weight_decay": 0.0},
        {"params": dec_decay, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": dec_nodecay, "lr": cfg.lr, "weight_decay": 0.0},
    ]
    if kendall is not None:
        param_groups.append(
            {"params": kendall.parameters(), "lr": cfg.lr, "weight_decay": 0.0}
        )
    if kendall_rate is not None:
        param_groups.append(
            {"params": kendall_rate.parameters(), "lr": cfg.lr, "weight_decay": 0.0}
        )
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr)
    if cfg.encoder_lr_scale != 1.0:
        logger.info(f"Differential LR: encoder={enc_lr:.1e}, decoder={cfg.lr:.1e}")

    # Linear warmup then cosine annealing
    warmup_epochs = cfg.warmup_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(cfg.epochs - warmup_epochs, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float("inf")
    best_val_f1 = 0.0  # for occupancy: save on best F1, not loss
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        # Determine previous config for smart resume
        prev_cfg = ckpt.get("config", {})
        if not prev_cfg:
            # Backward compat: old checkpoints have "args" dict
            prev_args = ckpt.get("args", {})
            prev_rates = prev_args.get("rates", [])
            prev_lr = prev_args.get("lr")
        else:
            prev_rates = prev_cfg.get("data", {}).get("rates", [])
            prev_lr = prev_cfg.get("lr")

        prev_enc_scale = prev_cfg.get("encoder_lr_scale", 1.0) if prev_cfg else 1.0
        same_config = (
            set(prev_rates) == set(cfg.data.rates)
            and prev_lr == cfg.lr
            and prev_enc_scale == cfg.encoder_lr_scale
            and prev_cfg.get("batch_size", cfg.batch_size) == cfg.batch_size
        )
        if same_config:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            else:
                # Fast-forward scheduler to match resumed epoch
                for _ in range(ckpt["epoch"]):
                    scheduler.step()
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            logger.info(
                f"Resumed from {cfg.resume} (epoch {ckpt['epoch']}, "
                f"val_loss={ckpt.get('val_loss', '?')})"
            )
        else:
            logger.info(
                f"Loaded model weights from {cfg.resume} (epoch {ckpt['epoch']}). "
                f"Fresh optimizer for new config: rates={cfg.data.rates}, lr={cfg.lr}"
            )

    n_train = len(train_dataset)
    n_val = len(val_dataset)
    n_tc = len(train_dataset.decoded_clouds)
    n_vc = len(val_dataset.decoded_clouds)
    logger.info(
        f"Train: {n_train} patches from {n_tc} clouds | "
        f"Val: {n_val} patches from {n_vc} clouds"
    )
    use_chamfer = loss_fn.needs_chamfer if loss_fn is not None else False
    focal_gamma = cfg.loss.focal_gamma if use_occupancy else 2.0

    dwa_history: list = []  # keeps last 2 epochs of per-rate val losses for DWA

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)

        if use_occupancy:
            t_stats = train_one_epoch_occupancy(
                model,
                train_loader,
                optimizer,
                device,
                cfg,
                focal_gamma=focal_gamma,
                ema_model=ema_model,
            )
            v_stats = validate_occupancy(
                model,
                val_loader,
                device,
                cfg,
                focal_gamma=focal_gamma,
                ema_model=ema_model,
            )
        else:
            t_stats = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                cfg,
                loss_fn,
                kendall=kendall,
                lap_op=lap_op,
                kendall_rate=kendall_rate,
                ema_model=ema_model,
            )
            v_stats = validate(
                model,
                val_loader,
                device,
                cfg,
                loss_fn,
                kendall=kendall,
                lap_op=lap_op,
                kendall_rate=kendall_rate,
                ema_model=ema_model,
            )
            if cfg.loss.use_dwa:
                epoch_losses = {
                    r: v_stats.get(f"dwa_{r}_loss", 1.0) for r in cfg.data.rates
                }
                dwa_history.append(epoch_losses)
                if len(dwa_history) >= 2:
                    _apply_dwa_weights(cfg, dwa_history[-2], dwa_history[-1])
                    logger.info(
                        f"  DWA weights: { {r: f'{w:.3f}' for r, w in zip(cfg.data.rates, cfg.loss.rate_weights)} }"
                    )
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        if use_occupancy:
            _log_epoch_occupancy(epoch, cfg.epochs, t_stats, v_stats, lr, elapsed)
        else:
            _log_epoch(epoch, cfg.epochs, t_stats, v_stats, lr, elapsed, use_chamfer)

        # Save best model (skip warmup epochs -- low LR inflates val artifact)
        val_loss = v_stats["loss"]
        val_f1 = v_stats.get("f1", 0.0)
        is_best = epoch > cfg.warmup_epochs and (
            (use_occupancy and val_f1 > best_val_f1)
            or (not use_occupancy and val_loss < best_val_loss)
        )
        if is_best:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            ckpt_path = save_dir / f"best_{rate_label}.pt"
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "config": config_to_dict(cfg),
            }
            if kendall is not None:
                ckpt_data["kendall_state_dict"] = kendall.state_dict()
            if kendall_rate is not None:
                ckpt_data["kendall_rate_state_dict"] = kendall_rate.state_dict()
            torch.save(ckpt_data, ckpt_path)
            if use_occupancy:
                logger.info(
                    f"  * New best (val_f1={val_f1:.1f}% val_loss={val_loss:.4f})"
                )
            else:
                logger.info(f"  * New best (val_loss={val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % 3 == 0:
            ckpt_path = save_dir / f"epoch_{epoch}_{rate_label}.pt"
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "config": config_to_dict(cfg),
            }
            if kendall is not None:
                ckpt_data["kendall_state_dict"] = kendall.state_dict()
            if kendall_rate is not None:
                ckpt_data["kendall_rate_state_dict"] = kendall_rate.state_dict()
            torch.save(ckpt_data, ckpt_path)

    logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
