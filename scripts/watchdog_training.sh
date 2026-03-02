#!/bin/bash
# Watchdog: monitors training log for anomalies and kills training if detected.
# Usage: bash scripts/watchdog_training.sh <log_file> [near_zero_threshold]
#
# Checks every 20 minutes. Kills the train.py process if:
#   - val near_zero% exceeds threshold (zero collapse)
#   - val pred_abs exceeds 2.0 (divergence / exploding predictions)
#   - val_loss increases for 5 consecutive epochs (stalled / diverging)

LOG_FILE="${1:?Usage: $0 <log_file> [threshold]}"
NZ_THRESHOLD="${2:-15.0}"  # default 15%
PRED_ABS_THRESHOLD=2.0
CONSEC_INCREASE_LIMIT=5
TIME=1200

echo "Watchdog started: monitoring $LOG_FILE"
echo "Kill conditions:"
echo "  - val near_zero% > $NZ_THRESHOLD"
echo "  - val pred_abs > $PRED_ABS_THRESHOLD"
echo "  - val_loss increasing $CONSEC_INCREASE_LIMIT consecutive epochs"
echo "Checking every $((TIME/60)) minutes..."

PREV_VAL_LOSS=""
CONSEC_INCREASES=0

while true; do
    sleep $TIME

    if [ ! -f "$LOG_FILE" ]; then
        echo "$(date +%H:%M:%S) Log file not found, waiting..."
        continue
    fi

    # Check if training is still running
    if ! pgrep -f "postprocessing/train.py" > /dev/null; then
        echo "$(date +%H:%M:%S) Training process not found. Exiting watchdog."
        exit 0
    fi

    # Get the last near_zero line
    LAST_NZ_LINE=$(grep "near_zero%%" "$LOG_FILE" | tail -1)
    if [ -z "$LAST_NZ_LINE" ]; then
        continue
    fi

    # Extract val near_zero%
    VAL_NZ=$(echo "$LAST_NZ_LINE" | grep -oP 'val=\K[0-9.]+')

    # Extract val pred_abs from the pred_abs line
    LAST_PRED_LINE=$(grep "pred_abs:" "$LOG_FILE" | tail -1)
    VAL_PRED_ABS=$(echo "$LAST_PRED_LINE" | grep -oP 'val=\K[0-9.]+')

    # Extract val_loss from epoch line
    LAST_EPOCH_LINE=$(grep "Epoch.*train=.*val=" "$LOG_FILE" | tail -1)
    VAL_LOSS=$(echo "$LAST_EPOCH_LINE" | grep -oP 'val=\K[0-9.]+')
    EPOCH=$(echo "$LAST_EPOCH_LINE" | grep -oP 'Epoch\s+\K[0-9]+')

    echo "$(date +%H:%M:%S) Epoch $EPOCH | val_loss=$VAL_LOSS | pred_abs=$VAL_PRED_ABS | near_zero=$VAL_NZ%"

    # Check 1: zero collapse
    if [ -n "$VAL_NZ" ]; then
        NZ_BAD=$(awk "BEGIN {print ($VAL_NZ > $NZ_THRESHOLD) ? 1 : 0}")
        if [ "$NZ_BAD" -eq 1 ]; then
            echo "ALERT: val near_zero%=$VAL_NZ exceeds $NZ_THRESHOLD -- zero collapse!"
            pkill -f "postprocessing/train.py"
            echo "Training killed."
            exit 1
        fi
    fi

    # Check 2: exploding predictions
    if [ -n "$VAL_PRED_ABS" ]; then
        PRED_BAD=$(awk "BEGIN {print ($VAL_PRED_ABS > $PRED_ABS_THRESHOLD) ? 1 : 0}")
        if [ "$PRED_BAD" -eq 1 ]; then
            echo "ALERT: val pred_abs=$VAL_PRED_ABS exceeds $PRED_ABS_THRESHOLD -- divergence!"
            pkill -f "postprocessing/train.py"
            echo "Training killed."
            exit 1
        fi
    fi

    # Check 3: consecutive val_loss increases
    if [ -n "$VAL_LOSS" ] && [ -n "$PREV_VAL_LOSS" ]; then
        INCREASED=$(awk "BEGIN {print ($VAL_LOSS > $PREV_VAL_LOSS) ? 1 : 0}")
        if [ "$INCREASED" -eq 1 ]; then
            CONSEC_INCREASES=$((CONSEC_INCREASES + 1))
            if [ "$CONSEC_INCREASES" -ge "$CONSEC_INCREASE_LIMIT" ]; then
                echo "ALERT: val_loss increased $CONSEC_INCREASES consecutive checks -- stalled!"
                pkill -f "postprocessing/train.py"
                echo "Training killed."
                exit 1
            fi
        else
            CONSEC_INCREASES=0
        fi
    fi
    PREV_VAL_LOSS="$VAL_LOSS"
done
