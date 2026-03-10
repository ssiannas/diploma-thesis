#!/bin/bash
# FiLM v9: Full FiLM + curvature input + log(bpp) rate conditioning
# r2+r4+r6 joint training with CD loss, no rate jitter
#
# Usage: bash scripts/training/train_film_v9_curv.sh
set -e

source ~/miniconda3/etc/profile.d/conda.sh && conda activate pcgcv2
ulimit -n 65536  # prevent "received 0 items of ancdata" with 24 DataLoader workers

python postprocessing/train.py \
    --config configs/cd_film_v9_curv.yaml \
    --overrides \
        save_dir=models/postprocessing/film_v9_curv \
        log_file=logs/film_v9_curv.log
