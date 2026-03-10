#!/bin/bash
# Evaluate film_v12_dynweight (best checkpoint) on all 4 sequences x r2/r4/r6
# Usage: bash scripts/evaluation/eval_v12_dynweight.sh
set -e

source ~/miniconda3/etc/profile.d/conda.sh && conda activate pcgcv2

CKPT="models/postprocessing/film_v12_dynweight/best_r2_r4_r6.pt"
DATA_ROOT="results/oversmoothing/decoded_clouds"
TAG="v12_dynweight_best"

SEQUENCES=(
    "longdress_vox10_1300"
    "loot_vox10_1200"
    "redandblack_vox10_1550"
    "soldier_vox10_0690"
)
RATES=(r2 r4 r6)

echo "=== Inference: ${CKPT} ==="
for SEQ in "${SEQUENCES[@]}"; do
    for RATE in "${RATES[@]}"; do
        INPUT="${DATA_ROOT}/${SEQ}/pcgcv2_${RATE}.npy"
        OUTPUT="${DATA_ROOT}/${SEQ}/refined_${TAG}_${RATE}.npy"
        echo "  ${SEQ} ${RATE}"
        python postprocessing/inference.py \
            --input "$INPUT" \
            --checkpoint "$CKPT" \
            --output "$OUTPUT" \
            --rate "$RATE"
    done
done

echo ""
echo "=== Evaluation (parallel) ==="
python scripts/evaluation/evaluate_postprocessing.py \
    --rates r2 r4 r6 \
    --refined_tag "$TAG" \
    --data_root "$DATA_ROOT" \
    --workers 8
