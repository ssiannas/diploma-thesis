"""Diagnose NN displacement target quality at edges vs flat regions.

Key questions:
1. Are edge displacement directions consistent among neighbors? (noise)
2. How many edge points get "pulled to wrong side" by their NN?
3. What's the displacement variance at edges vs flat?
"""

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def compute_curvature(points, k=30):
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1)
    idx = idx[:, 1:]
    neighbors = points[idx]
    centered = neighbors - points[:, None, :]
    cov = np.einsum("nki,nkj->nij", centered, centered) / k
    eigvals = np.linalg.eigvalsh(cov)
    return eigvals[:, 0] / (eigvals.sum(axis=1) + 1e-12)


def analyze_cloud(original_path, decoded_path, rate_label):
    original = np.load(original_path).astype(np.float64)
    decoded = np.load(decoded_path).astype(np.float64)

    print(f"\n{'='*60}")
    print(f"Analyzing {rate_label}: {decoded.shape[0]} points")
    print(f"{'='*60}")

    # NN displacement targets (what we train on)
    tree_orig = cKDTree(original)
    dd, ii = tree_orig.query(decoded, k=1)
    displacement = original[ii] - decoded
    disp_mag = np.linalg.norm(displacement, axis=1)

    # Curvature on original cloud, transferred to decoded via NN
    print("Computing curvature...")
    curv_orig = compute_curvature(original, k=30)
    curv_at_decoded = curv_orig[ii]  # curvature of matched original point

    # Stratify by curvature terciles
    t1, t2 = np.percentile(curv_at_decoded, [33.3, 66.7])
    flat_mask = curv_at_decoded <= t1
    edge_mask = curv_at_decoded > t2
    mid_mask = ~flat_mask & ~edge_mask

    print(f"\nCurvature thresholds: flat<={t1:.4f}, edge>{t2:.4f}")
    print(
        f"Counts: flat={flat_mask.sum()}, mid={mid_mask.sum()}, edge={edge_mask.sum()}"
    )

    # 1. Displacement magnitude stats
    print(f"\n--- Displacement magnitude ---")
    for name, mask in [("flat", flat_mask), ("mid", mid_mask), ("edge", edge_mask)]:
        mags = disp_mag[mask]
        zero_pct = 100 * (mags < 1e-6).mean()
        print(
            f"  {name:4s}: mean={mags.mean():.4f}, median={np.median(mags):.4f}, "
            f"std={mags.std():.4f}, zero%={zero_pct:.1f}%, max={mags.max():.3f}"
        )

    # 2. NN distance (how far is the match?)
    print(f"\n--- NN distance (decoded->original) ---")
    for name, mask in [("flat", flat_mask), ("mid", mid_mask), ("edge", edge_mask)]:
        dists = dd[mask]
        print(
            f"  {name:4s}: mean={dists.mean():.4f}, median={np.median(dists):.4f}, "
            f"std={dists.std():.4f}"
        )

    # 3. Displacement direction consistency among spatial neighbors
    # For each decoded point, check if its displacement direction agrees
    # with its k nearest decoded neighbors' displacement directions
    print(f"\n--- Displacement direction consistency (k=8 decoded neighbors) ---")
    tree_dec = cKDTree(decoded)
    _, dec_nn_idx = tree_dec.query(decoded, k=9)  # self + 8 neighbors
    dec_nn_idx = dec_nn_idx[:, 1:]  # exclude self

    # Cosine similarity of displacement with neighbors' displacement
    # Only for points with nonzero displacement
    nonzero = disp_mag > 0.01
    disp_normed = np.zeros_like(displacement)
    disp_normed[nonzero] = displacement[nonzero] / disp_mag[nonzero, None]

    # Mean cosine similarity with neighbors
    cos_sims = np.zeros(len(decoded))
    for i in tqdm(range(0, len(decoded), 10000), desc="Direction consistency"):
        batch_end = min(i + 10000, len(decoded))
        batch_idx = dec_nn_idx[i:batch_end]  # (batch, 8)
        batch_disp = disp_normed[i:batch_end]  # (batch, 3)
        neigh_disp = disp_normed[batch_idx]  # (batch, 8, 3)
        cos = np.einsum("bi,bki->bk", batch_disp, neigh_disp)  # (batch, 8)
        cos_sims[i:batch_end] = cos.mean(axis=1)

    for name, mask in [("flat", flat_mask), ("mid", mid_mask), ("edge", edge_mask)]:
        valid = mask & nonzero
        if valid.sum() > 0:
            cs = cos_sims[valid]
            print(
                f"  {name:4s}: mean_cos={cs.mean():.3f}, "
                f"negative%={100*(cs < 0).mean():.1f}%, "
                f"high(>0.8)%={100*(cs > 0.8).mean():.1f}%"
            )

    # 4. NN uniqueness: how many decoded points share the same original NN?
    print(f"\n--- NN multiplicity (decoded points sharing same original NN) ---")
    unique, counts = np.unique(ii, return_counts=True)
    multi = counts[counts > 1]
    print(f"  Unique NNs: {len(unique)}/{len(original)} original points matched")
    print(f"  Mean multiplicity: {counts.mean():.2f}, max: {counts.max()}")
    print(
        f"  Points with multiplicity>1: {(counts > 1).sum()}"
        f" ({100*(counts>1).mean():.1f}%)"
    )

    # Multiplicity by curvature stratum of the original point
    curv_of_matched = curv_orig[unique]
    for name, lo, hi in [("flat", 0, t1), ("mid", t1, t2), ("edge", t2, 1.0)]:
        stratum_mask = (curv_of_matched > lo) & (curv_of_matched <= hi)
        if name == "flat":
            stratum_mask = curv_of_matched <= t1
        sc = counts[stratum_mask]
        if len(sc) > 0:
            print(
                f"  {name:4s}: mean_mult={sc.mean():.2f},"
                f" mult>1: {100*(sc>1).mean():.1f}%"
            )

    # 5. "Oracle" test: if we apply the NN displacement, does PSNR improve?
    # Compare: decoded vs original, refined(decoded+disp) vs original
    refined = decoded + displacement
    # Per-point squared error
    tree_orig2 = cKDTree(original)
    dd_dec, _ = tree_orig2.query(decoded, k=1)
    dd_ref, _ = tree_orig2.query(refined, k=1)

    peak = 1023.0
    for name, mask in [
        ("flat", flat_mask),
        ("mid", mid_mask),
        ("edge", edge_mask),
        ("all", np.ones(len(decoded), dtype=bool)),
    ]:
        mse_dec = (dd_dec[mask] ** 2).mean()
        mse_ref = (dd_ref[mask] ** 2).mean()
        psnr_dec = 10 * np.log10(peak**2 / mse_dec) if mse_dec > 0 else 999
        psnr_ref = 10 * np.log10(peak**2 / mse_ref) if mse_ref > 0 else 999
        delta = psnr_ref - psnr_dec
        print(
            f"\n  Oracle PSNR ({name:4s}): decoded={psnr_dec:.2f}, "
            f"refined={psnr_ref:.2f}, delta={delta:+.2f} dB"
        )


if __name__ == "__main__":
    import sys

    base = "results/oversmoothing/decoded_clouds/redandblack_vox10_1550"
    orig = f"{base}/original.npy"

    rates = sys.argv[1:] if len(sys.argv) > 1 else ["r2", "r3"]
    for rate in rates:
        analyze_cloud(orig, f"{base}/pcgcv2_{rate}.npy", rate)
