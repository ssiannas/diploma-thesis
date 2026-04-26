# Summary

## Curvature-Stratified Evaluation and Post-Processing Correction of Geometric Oversmoothing in Learned Point Cloud Codecs

Learned point cloud codecs, such as PCGCv2, achieve competitive
rate-distortion performance against traditional standards like G-PCC, but
introduce a systematic geometric artifact known as **oversmoothing**.
This artifact concentrates at high-curvature surface regions — edges, ridges,
and fine structures — where rare, high-complexity occupancy patterns incur the
highest coding cost.
Standard quality metrics such as point-to-point PSNR (D1) average uniformly
over all surface points, masking this spatially non-uniform degradation
entirely.
A codec can report a high aggregate PSNR while systematically erasing all
geometric detail at surface boundaries.

This thesis makes two principal contributions to address this problem.

**Curvature-stratified PSNR metric.**
We introduce an evaluation framework that partitions the original point cloud
by local surface curvature and computes quality separately for edge
(high-curvature) and flat (low-curvature) regions.
Curvature is estimated from the eigenvalue structure of the local covariance
matrix; stratification uses percentile thresholds that adapt to each
dataset's curvature distribution.
Quality is measured in the reverse direction — from the original cloud to the
reconstructed cloud — which captures how well the reconstruction covers the
original surface, and correctly identifies which regions are most degraded.
Applied to PCGCv2 across the 8i Voxelized Full Bodies (8iVFB) dataset at
seven operating rates, the metric reveals a consistent edge-region quality
deficit of 2–4.5 dB relative to flat regions.
Counterintuitively, this gap **widens with increasing rate**: the codec
allocates additional bits preferentially toward flat surfaces, deepening the
quality imbalance rather than resolving it.
The same metric applied to G-PCC reveals a qualitatively different, non-monotone
artifact profile, confirming that the monotonically growing gap is a structural
property of the learned occupancy-coding paradigm.

**Rate-conditioned sparse U-Net post-processor.**
We propose a lightweight post-processing network that corrects geometric
distortions in decoded point clouds without modifying the underlying codec.
The architecture is a three-level sparse U-Net operating directly on the
decoded occupancy grid, providing the spatial awareness needed to locate and
correct edge-region errors.
A Feature-wise Linear Modulation (FiLM) head conditions displacement
predictions on the log-scaled bits-per-point value, enabling a single
2.5M-parameter model to serve multiple rate-distortion operating points.
Training uses a dynamic Chamfer Distance loss computed between decoded and
original cloud crops per patch.
This loss sidesteps the **zero-inflation failure mode** that defeats standard
displacement regression losses: at the lowest evaluated rate (r2), 60% of
decoded points already coincide with original voxels and require zero
correction, causing naive L1/L2 regression to collapse to a trivial
zero-prediction solution.

The final model achieves an average edge PSNR improvement of **+2.06 dB** at
the lowest evaluated rate (r2, ~0.05 bpp), **+0.58 dB** at medium rate (r4,
~0.15 bpp), and **+0.27 dB** at high rate (r6, ~0.32 bpp), averaged across
all four 8iVFB sequences and 400 frames (Bjøntegaard Delta PSNR: +0.97 dB on
the edge metric over r2–r6).
Improvements are consistent across all four sequences, three of which were
not seen during training.
An ablation study confirms that removing curvature as an explicit input feature
improves results at all rates, reflecting the observation that decoded-cloud
curvature is most unreliable precisely where corrections are most needed.

**Generalisation.**
Zero-shot transfer experiments to an unseen codec (Unicorn) and an unseen
dataset (MVUB) demonstrate that at extreme compression rates, geometric
oversmoothing is a codec-agnostic failure mode: the post-processor achieves
+1.09 to +1.36 dB edge PSNR improvement on Unicorn at its two lowest rates
without any retraining, while at intermediate Unicorn rates (where distortion
patterns differ from PCGCv2's) the transfer is neutral to negative.
This establishes both the cross-codec generality of the oversmoothing problem
and the practical scope of a codec-specific corrector.

---

## Presentation Structure (10–15 min)

### Slide 1 — Title (0:00–0:30)
- Title, author, institution, date.

---

### Slide 2 — Motivation: Why This Matters (0:30–2:00) · ~1.5 min
- Point clouds are everywhere: volumetric video, LiDAR, cultural heritage.
- Compression is required; learned codecs (PCGCv2) now rival traditional standards.
- **Hook**: show the teaser figure — original vs. decoded soldier. The decoded cloud looks plausible, but fine surface detail is gone.

> *Goal: set up the visual intuition before any math.*

---

### Slide 3 — The Problem: Standard Metrics Are Blind (2:00–4:00) · ~2 min
- D1 PSNR averages over every point equally — flat and edge alike.
- A codec can score high D1 while systematically erasing all edges.
- **Key insight**: degradation is spatially non-uniform; we need to measure *where it hurts*.
- Quick curvature heatmap on the original cloud to build visual intuition.

> *This slide motivates both contributions simultaneously.*

---

### Slide 4 — Contribution 1: Curvature-Stratified Metric (4:00–6:30) · ~2.5 min
- Partition points into edge / flat strata using PCA-based curvature + percentile thresholds.
- Measure PSNR separately per stratum (reverse direction: original → reconstructed).
- **Result**: PCGCv2 edge PSNR lags flat PSNR by **2–4.5 dB** — and the gap *widens* with rate.
- Show the oversmoothing-gap-vs-bpp plot. One sentence on the cause: the entropy model penalises rare occupancy patterns, so edges receive fewer bits at every rate.
- Contrast with G-PCC (non-monotone profile) to show this is a learned-codec signature.

> *One plot, one number, one causal sentence.*

---

### Slide 5 — Contribution 2: The Post-Processing Network (6:30–10:00) · ~3.5 min

**Architecture (half slide)**
- Sparse 3-level U-Net on the decoded occupancy grid → predicts per-point 3D displacement.
- FiLM head: log(bpp) scalar → scale + shift feature maps for each rate.
- One model, 2.5 M params, covers r2 / r4 / r6 simultaneously.

**Loss design (half slide)**
- Naive L1/L2 collapses: 60% of points need zero correction at r2 → model predicts zero everywhere.
- Dynamic Chamfer Distance: match decoded-cloud patches to original patches — no per-point targets needed.
- Show the zero-inflation table (fraction of zero displacements vs. rate) as the motivation.

> *The loss design is as important as the architecture — give it equal time.*

---

### Slide 6 — Results (10:00–13:00) · ~3 min
- Main numbers: edge PSNR Δ averaged over 4 sequences and 400 frames.
  - **r2: +2.06 dB, r4: +0.58 dB, r6: +0.27 dB.** BD-PSNR +0.97 dB on the edge metric.
- Show the per-rate improvement bar chart.
- Two highlighted ablation findings:
  - Removing curvature as an input *improves* results (decoded curvature is unreliable noise at low rates).
  - Zero-shot transfer to Unicorn: +1.1–1.4 dB at extreme rates — cross-codec generalisation.

> *Don't enumerate every ablation. Pick the two most surprising findings.*

---

### Slide 7 — Conclusion (13:00–15:00) · ~2 min
- Two contributions: a diagnostic metric + a post-processing corrector.
- Oversmoothing is a **structural property** of learned occupancy codecs, not a simple low-rate artifact — it grows with rate.
- The correction generalises beyond the training codec at extreme compression.
- **Future directions** (one bullet each): multi-codec training, occupancy-branch for point insertion, temporal coherence for video sequences.
- Leave the audience with an open question: *as learned codecs improve, does the stratification gap eventually close — or is it inherent to entropy-coded occupancy grids?*
