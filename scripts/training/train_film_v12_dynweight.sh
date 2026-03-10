#!/bin/bash
# FiLM v12: dynamic detached rate weighting
# Validation: dw_r2 > dw_r4 > dw_r6 throughout training
# Compare final PSNR against v10 (static [1.0, 0.3, 0.05])
#
# Usage: bash scripts/training/train_film_v12_dynweight.sh
set -e

source ~/miniconda3/etc/profile.d/conda.sh && conda activate pcgcv2
ulimit -n 65536

python postprocessing/train.py \
    --config configs/cd_film_v12_dynweight.yaml \
    --overrides \
        save_dir=models/postprocessing/film_v12_dynweight \
        log_file=logs/film_v12_dynweight.log
