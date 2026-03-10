#!/bin/bash
# Evaluate v13 finetune vs v10 ep24 on all 4 sequences x r2/r4/r6
# Usage: bash scripts/evaluation/eval_v13_finetune.sh
set -e

source ~/miniconda3/etc/profile.d/conda.sh && conda activate pcgcv2

CKPT_V13="models/postprocessing/film_v13_finetune/best_r4_r6.pt"
CKPT_V10="models/postprocessing/good_candidates/film_v10_head_ep24.pt"
DATA_ROOT="results/oversmoothing/decoded_clouds"
TAG_V13="v13_finetune"
TAG_V10="v10_head_epoch_24"

SEQUENCES=(
    "longdress_vox10_1300"
    "loot_vox10_1200"
    "redandblack_vox10_1550"
    "soldier_vox10_0690"
)

echo "=== Inference: v13 (r2/r4/r6 -- r2 is OOD) ==="
for SEQ in "${SEQUENCES[@]}"; do
    for RATE in r2 r4 r6; do
        INPUT="${DATA_ROOT}/${SEQ}/pcgcv2_${RATE}.npy"
        OUTPUT="${DATA_ROOT}/${SEQ}/refined_${TAG_V13}_${RATE}.npy"
        echo "  ${SEQ} ${RATE}"
        python postprocessing/inference.py \
            --input "$INPUT" \
            --checkpoint "$CKPT_V13" \
            --output "$OUTPUT" \
            --rate "$RATE"
    done
done

echo ""
echo "=== Inference: v10 (skip if exists) ==="
for SEQ in "${SEQUENCES[@]}"; do
    for RATE in r2 r4 r6; do
        INPUT="${DATA_ROOT}/${SEQ}/pcgcv2_${RATE}.npy"
        OUTPUT="${DATA_ROOT}/${SEQ}/refined_${TAG_V10}_${RATE}.npy"
        if [ ! -f "$OUTPUT" ]; then
            echo "  ${SEQ} ${RATE}"
            python postprocessing/inference.py \
                --input "$INPUT" \
                --checkpoint "$CKPT_V10" \
                --output "$OUTPUT" \
                --rate "$RATE"
        fi
    done
done

echo ""
echo "=== Evaluation: v10 (parallel) ==="
python scripts/evaluation/evaluate_postprocessing.py \
    --rates r2 r4 r6 \
    --refined_tag "$TAG_V10" \
    --data_root "$DATA_ROOT" \
    --workers 8

echo ""
echo "=== Evaluation: v13 (parallel) ==="
python scripts/evaluation/evaluate_postprocessing.py \
    --rates r2 r4 r6 \
    --refined_tag "$TAG_V13" \
    --data_root "$DATA_ROOT" \
    --workers 8
