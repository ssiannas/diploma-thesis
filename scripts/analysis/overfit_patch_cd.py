"""Single-patch overfitting test for dynamic Chamfer Distance loss.

Finds the highest-curvature patch from a decoded cloud, then overfits the
SparseUNet on that single patch using CD loss against the actual original cloud.

Supports CD-only and composite (CD + Laplacian + Shrinkage) loss modes.
Reports edge vs flat stratified PSNR.

Usage:
    python scripts/analysis/overfit_patch_cd.py --rate r2 --epochs 500
    python scripts/analysis/overfit_patch_cd.py --rate r2 --epochs 500 --loss composite
"""

import argparse
import sys
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "postprocessing")
)

from dataset import compute_curvature, extract_patches
from model import SparseUNet

COORD_SCALE = 1023.0


def find_edgiest_patch(
    decoded: np.ndarray,
    curvature: np.ndarray,
    patch_size: int = 64,
    stride: int = 32,
    min_points: int = 100,
    top_k: int = 5,
):
    """Find patches with highest mean curvature (most edge geometry)."""
    patches = extract_patches(decoded, patch_size, stride, min_points)
    print(f"Total patches: {len(patches)}")

    scored = []
    for idx_arr in patches:
        mean_curv = curvature[idx_arr].mean()
        n_pts = len(idx_arr)
        scored.append((mean_curv, n_pts, idx_arr))
    scored.sort(key=lambda x: -x[0])

    print(f"\nTop {top_k} edgiest patches:")
    for i, (mc, np_, _) in enumerate(scored[:top_k]):
        print(f"  #{i}: mean_curv={mc:.4f}, n_points={np_}")

    return scored[0][2]


def extract_original_crop(original, decoded_patch, padding=10):
    """Crop original cloud to bounding box of decoded patch + padding."""
    mins = decoded_patch.min(axis=0) - padding
    maxs = decoded_patch.max(axis=0) + padding
    mask = np.all((original >= mins) & (original <= maxs), axis=1)
    return original[mask].astype(np.float32)


def compute_psnr(refined, original, peak=1023.0):
    """Point-to-point PSNR: for each refined point, distance to nearest original."""
    tree = cKDTree(original)
    dists, _ = tree.query(refined)
    mse = (dists**2).mean()
    if mse < 1e-10:
        return 999.0
    return 10 * np.log10(peak**2 / mse)


def stratified_psnr(refined, original, curvature, peak=1023.0):
    """PSNR stratified by curvature terciles (flat/mid/edge)."""
    tree = cKDTree(original)
    dists, _ = tree.query(refined)
    sq_err = dists**2

    t1, t2 = np.percentile(curvature, [33.3, 66.7])
    strata = {
        "flat": curvature <= t1,
        "mid": (curvature > t1) & (curvature <= t2),
        "edge": curvature > t2,
        "all": np.ones(len(curvature), bool),
    }

    results = {}
    for name, mask in strata.items():
        if mask.sum() == 0:
            results[name] = 0.0
            continue
        mse = sq_err[mask].mean()
        results[name] = 10 * np.log10(peak**2 / mse) if mse > 1e-10 else 999.0
    return results


def compute_laplacian(points, k=8):
    """Discrete Laplacian: point - mean(k neighbors). Returns (N, 3)."""
    dists = torch.cdist(points.unsqueeze(0), points.unsqueeze(0)).squeeze(0)
    _, idx = dists.topk(k + 1, dim=1, largest=False)
    idx = idx[:, 1:]  # exclude self
    neighbors = points[idx]  # (N, k, 3)
    return points - neighbors.mean(dim=1)  # (N, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rate", default="r2")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--padding", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--loss",
        default="cd",
        choices=["cd", "composite"],
        help="cd=CD only, composite=CD+Laplacian+Shrinkage",
    )
    p.add_argument("--lambda_lap", type=float, default=0.1, help="Laplacian weight")
    p.add_argument("--lambda_shrink", type=float, default=0.01, help="Shrinkage weight")
    p.add_argument("--laplacian_k", type=int, default=8)
    p.add_argument(
        "--data_dir",
        default="results/oversmoothing/decoded_clouds/redandblack_vox10_1550",
    )
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    # Load clouds
    print("Loading clouds...")
    original = np.load(data_dir / "original.npy").astype(np.float32)
    decoded = np.load(data_dir / f"pcgcv2_{args.rate}.npy").astype(np.float32)
    print(f"Original: {original.shape}, Decoded: {decoded.shape}")

    # Curvature on decoded cloud
    print("Computing curvature...")
    curvature = compute_curvature(decoded, k=30)

    # Find edgiest patch
    patch_indices = find_edgiest_patch(decoded, curvature, args.patch_size, args.stride)
    n_pts = len(patch_indices)
    print(f"\nSelected patch: {n_pts} points")

    # Extract patch data
    patch_coords = decoded[patch_indices]
    patch_curv = curvature[patch_indices]
    orig_crop = extract_original_crop(original, patch_coords, args.padding)
    print(f"Original crop: {orig_crop.shape[0]} points (padding={args.padding})")

    # Baseline metrics
    psnr_before = stratified_psnr(patch_coords, orig_crop, patch_curv)
    tree_eval = cKDTree(orig_crop)
    dists_before, _ = tree_eval.query(patch_coords)

    print(
        f"\nBaseline PSNR: all={psnr_before['all']:.2f}, "
        f"flat={psnr_before['flat']:.2f}, edge={psnr_before['edge']:.2f}"
    )
    print(f"Baseline mean_dist: {dists_before.mean():.4f}")

    # Curvature stats for this patch
    t1, t2 = np.percentile(patch_curv, [33.3, 66.7])
    n_flat = (patch_curv <= t1).sum()
    n_edge = (patch_curv > t2).sum()
    print(
        f"Curvature: flat={n_flat}, edge={n_edge}, " f"thresholds=[{t1:.4f}, {t2:.4f}]"
    )

    # Prepare tensors
    coords_int = np.floor(patch_coords).astype(np.int32)
    features = np.column_stack(
        [
            patch_coords / COORD_SCALE,
            patch_curv[:, np.newaxis],
        ]
    ).astype(np.float32)

    batch_col = np.zeros((n_pts, 1), dtype=np.int32)
    coords_me = np.hstack([batch_col, coords_int])

    coords_t = torch.from_numpy(coords_me).int()
    feats_t = torch.from_numpy(features).float()
    orig_t = torch.from_numpy(orig_crop).float().to(device)
    decoded_t = torch.from_numpy(patch_coords).float().to(device)

    # Precompute original Laplacian for composite loss
    if args.loss == "composite":
        # NN displacement to get original positions for this patch
        tree_orig = cKDTree(orig_crop)
        _, nn_idx = tree_orig.query(patch_coords)
        orig_matched = orig_crop[nn_idx]  # (N, 3) -- original points matched to decoded
        orig_matched_t = torch.from_numpy(orig_matched).float().to(device)
        lap_original = compute_laplacian(orig_matched_t, k=args.laplacian_k)

    # Model
    model = SparseUNet(in_channels=4, max_displacement=5.0).to(device)
    n_params = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(f"\nModel: SparseUNet, {n_params:,} params")
    print(f"Loss: {args.loss}")
    if args.loss == "composite":
        print(f"  lambda_lap={args.lambda_lap}, lambda_shrink={args.lambda_shrink}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # Header
    header = (
        f"{'Ep':>5} | {'Loss':>9} | {'CD_f':>8} | {'CD_r':>8} | "
        f"{'pred':>6} | {'all':>7} | {'flat':>7} | {'edge':>7} | "
        f"{'d_all':>6} | {'d_edge':>6}"
    )
    if args.loss == "composite":
        header = (
            f"{'Ep':>5} | {'Loss':>9} | {'CD':>8} | {'Lap':>8} | {'Shr':>8} | "
            f"{'pred':>6} | {'all':>7} | {'flat':>7} | {'edge':>7} | "
            f"{'d_all':>6} | {'d_edge':>6}"
        )
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in tqdm(range(1, args.epochs + 1), desc="Overfit"):
        model.train()

        sin = ME.SparseTensor(
            features=feats_t.clone(),
            coordinates=coords_t.clone(),
            device=device,
        )

        pred = model(sin)
        offsets = pred.F
        decoded_pts = pred.C[:, 1:].float()
        refined = decoded_pts + offsets

        # CD loss
        dists_sq = torch.cdist(refined, orig_t).pow(2)
        cd_fwd = dists_sq.min(dim=1).values.mean()
        cd_rev = dists_sq.min(dim=0).values.mean()
        cd_loss = cd_fwd + cd_rev

        if args.loss == "composite":
            # Laplacian preservation
            lap_refined = compute_laplacian(refined, k=args.laplacian_k)
            lap_loss = torch.abs(lap_refined - lap_original).mean()

            # Shrinkage (penalize large displacements)
            shrink_loss = offsets.pow(2).sum(dim=-1).mean()

            loss = (
                cd_loss + args.lambda_lap * lap_loss + args.lambda_shrink * shrink_loss
            )
        else:
            loss = cd_loss
            lap_loss = torch.tensor(0.0)
            shrink_loss = torch.tensor(0.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            with torch.no_grad():
                pred_abs = offsets.norm(dim=-1).mean().item()
                refined_np = refined.detach().cpu().numpy()
                psnr = stratified_psnr(refined_np, orig_crop, patch_curv)
                d_all = psnr["all"] - psnr_before["all"]
                d_edge = psnr["edge"] - psnr_before["edge"]

            if args.loss == "composite":
                tqdm.write(
                    f"{epoch:5d} | {loss.item():9.4f} | {cd_loss.item():8.4f} | "
                    f"{lap_loss.item():8.4f} | {shrink_loss.item():8.4f} | "
                    f"{pred_abs:6.3f} | {psnr['all']:7.2f} | {psnr['flat']:7.2f} | "
                    f"{psnr['edge']:7.2f} | {d_all:+6.2f} | {d_edge:+6.2f}"
                )
            else:
                tqdm.write(
                    f"{epoch:5d} | {loss.item():9.4f} | {cd_fwd.item():8.4f} | "
                    f"{cd_rev.item():8.4f} | "
                    f"{pred_abs:6.3f} | {psnr['all']:7.2f} | {psnr['flat']:7.2f} | "
                    f"{psnr['edge']:7.2f} | {d_all:+6.2f} | {d_edge:+6.2f}"
                )

    # Final evaluation
    model.eval()
    with torch.no_grad():
        sin = ME.SparseTensor(
            features=feats_t.clone(),
            coordinates=coords_t.clone(),
            device=device,
        )
        pred = model(sin)
        refined_final = (pred.C[:, 1:].float() + pred.F).cpu().numpy()

    psnr_after = stratified_psnr(refined_final, orig_crop, patch_curv)
    dists_after, _ = tree_eval.query(refined_final)

    print(f"\n{'='*60}")
    print(f"RESULTS (single-patch overfit, {args.rate}, loss={args.loss})")
    print(f"{'='*60}")
    print(f"{'':15s} {'All':>8} {'Flat':>8} {'Edge':>8}")
    print(
        f"Before PSNR:    {psnr_before['all']:8.2f} {psnr_before['flat']:8.2f} "
        f"{psnr_before['edge']:8.2f}"
    )
    print(
        f"After PSNR:     {psnr_after['all']:8.2f} {psnr_after['flat']:8.2f} "
        f"{psnr_after['edge']:8.2f}"
    )
    d = {k: psnr_after[k] - psnr_before[k] for k in ["all", "flat", "edge"]}
    print(f"Delta:          {d['all']:+8.2f} {d['flat']:+8.2f} {d['edge']:+8.2f}")
    print(
        f"\nmean_dist: before={dists_before.mean():.4f}, after={dists_after.mean():.4f}"
    )

    offsets_np = pred.F.cpu().numpy()
    print(
        f"pred: mean_abs={np.abs(offsets_np).mean():.4f}, "
        f"max_abs={np.abs(offsets_np).max():.4f}, "
        f"near_zero={100*(np.linalg.norm(offsets_np, axis=1) < 0.01).mean():.1f}%"
    )

    if d["edge"] > 0:
        print("\n>>> Edge PSNR improved. CD loss fixes edge geometry.")
    else:
        print("\n>>> Edge PSNR did not improve.")


if __name__ == "__main__":
    main()
