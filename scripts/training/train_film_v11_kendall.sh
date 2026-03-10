#!/bin/bash
# FiLM v11: Kendall learned rate weighting
# Validation: sigma values should converge r2 < r4 < r6 (matching displacement density)
# Compare final PSNR against v10 (static [1.0, 0.3, 0.05])
#
# Usage: bash scripts/training/train_film_v11_kendall.sh
set -e

source ~/miniconda3/etc/profile.d/conda.sh && conda activate pcgcv2
ulimit -n 65536

python postprocessing/train.py \
    --config configs/cd_film_v11_kendall.yaml \
    --overrides \
        save_dir=models/postprocessing/film_v11_kendall \
        log_file=logs/film_v11_kendall.log
