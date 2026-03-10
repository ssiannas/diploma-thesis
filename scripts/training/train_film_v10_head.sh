#!/bin/bash
# FiLM v10: per-channel head FiLM + rate weighting
# Validation experiment: cd_fwd must decrease (was flat in v9 full FiLM)
#
# Usage: bash scripts/training/train_film_v10_head.sh
set -e

source ~/miniconda3/etc/profile.d/conda.sh && conda activate pcgcv2
ulimit -n 65536  # prevent "received 0 items of ancdata" with 24 DataLoader workers

python postprocessing/train.py \
    --config configs/cd_film_v10_head.yaml \
    --overrides \
        save_dir=models/postprocessing/film_v10_head \
        log_file=logs/film_v10_head.log
