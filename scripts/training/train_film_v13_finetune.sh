#!/bin/bash
# Fine-tune v10 ep24 backbone on r4+r6 with frozen encoder.
# Keeps dec_block1, conv_out, rate_encoder, film_head trainable (~0.5M params).
# Usage: bash scripts/training/train_film_v13_finetune.sh
set -e

source ~/miniconda3/etc/profile.d/conda.sh && conda activate pcgcv2

SESSION="v13"
LOG="logs/film_v13_finetune.log"
mkdir -p logs models/postprocessing/film_v13_finetune

tmux new-session -d -s "$SESSION" 2>/dev/null || tmux new-window -t "$SESSION"
tmux send-keys -t "$SESSION" "
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pcgcv2
python postprocessing/train.py \
    --config configs/cd_film_v13_finetune.yaml \
    --log_file $LOG \
    2>&1 | tee -a $LOG
" Enter

echo "Training v13 in tmux session '$SESSION'. Monitor with: tmux attach -t $SESSION"
echo "Log: $LOG"
