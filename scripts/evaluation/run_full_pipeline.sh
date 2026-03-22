#!/usr/bin/env bash
# Full evaluation pipeline: inference → metrics → merge
# Run from project root.  All output logged to logs/pipeline_*.log
# Expected runtime: ~35 min inference + ~8 min metrics (4 parallel)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcgcv2

CKPT="models/postprocessing/good_candidates/film_v10_nocurv_ep24_best.pt"
TAG="nocurv_ep24"
RATES="r2 r4 r6"
LOG_DIR="logs"
METRICS_DIR="results/metrics"

mkdir -p "$LOG_DIR" "$METRICS_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Step 1: Inference — all sequences, one GPU process, skips already-done frames
# ---------------------------------------------------------------------------
log "=== STEP 1: INFERENCE (all sequences, skip cached) ==="

python scripts/evaluation/eval_full_dataset.py \
    --checkpoint "$CKPT" \
    --rates $RATES \
    --tag "$TAG" \
    --infer_only \
    --device cuda \
    2>&1 | tee "$LOG_DIR/infer_all.log"

log "Inference DONE. Verifying counts..."
for seq in longdress loot redandblack soldier; do
    for rate in r2 r4 r6; do
        count=$(find "datasets/pcgcv2_decoded/$seq/" -name "refined_${TAG}_${rate}.npy" | wc -l)
        echo "  $seq $rate: $count/100"
    done
done

# ---------------------------------------------------------------------------
# Step 2: Metrics — 4 sequences in parallel (CPU-bound, no GPU)
# ---------------------------------------------------------------------------
log "=== STEP 2: METRICS (4 parallel processes) ==="

for seq in longdress loot redandblack soldier; do
    log "Launching metrics: $seq"
    python scripts/evaluation/compute_all_metrics.py \
        --tag "$TAG" \
        --rates $RATES \
        --sequences "$seq" \
        --curvature_k 30 \
        --output "$METRICS_DIR/${TAG}_${seq}.csv" \
        > "$LOG_DIR/metrics_${seq}.log" 2>&1 &
done

log "Waiting for all 4 metrics processes..."
wait
log "Metrics DONE"

# Check for failures
for seq in longdress loot redandblack soldier; do
    if grep -q "ERROR\|Traceback" "$LOG_DIR/metrics_${seq}.log" 2>/dev/null; then
        log "WARNING: possible error in $seq metrics — check $LOG_DIR/metrics_${seq}.log"
    fi
done

# ---------------------------------------------------------------------------
# Step 3: Merge all CSVs → single file + print aggregate table
# ---------------------------------------------------------------------------
log "=== STEP 3: MERGE ==="

python scripts/evaluation/compute_all_metrics.py merge \
    --inputs \
        "$METRICS_DIR/${TAG}_longdress.csv" \
        "$METRICS_DIR/${TAG}_loot.csv" \
        "$METRICS_DIR/${TAG}_redandblack.csv" \
        "$METRICS_DIR/${TAG}_soldier.csv" \
    --output "$METRICS_DIR/${TAG}_all.csv" \
    2>&1 | tee "$LOG_DIR/merge.log"

log "=== PIPELINE COMPLETE ==="
log "Final CSV : $METRICS_DIR/${TAG}_all.csv"
log "All logs  : $LOG_DIR/"
