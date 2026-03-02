#!/bin/bash
# Scaled CD training with temporal diversity (stride=12)
# Validated: 3 clouds stride=12 matches 20 clouds stride=1 on redandblack r2
#
# Usage: bash scripts/training/train_cd_scaled.sh
set -e

source ~/miniconda3/etc/profile.d/conda.sh && conda activate pcgcv2

python postprocessing/train.py \
    --data_root datasets/pcgcv2_decoded \
    --rates r2 \
    --val_sequence redandblack \
    --epochs 30 \
    --batch_size 16 \
    --num_workers 8 \
    --max_clouds 50 \
    --frame_stride 12 \
    --loss_type cd \
    --chamfer_padding 0 \
    --lr 1e-4 \
    --warmup_epochs 3 \
    --log_file logs/train_cd_50cloud_s12_r2.log
