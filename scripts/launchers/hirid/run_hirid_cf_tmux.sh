#!/usr/bin/env bash
set -euo pipefail

# -------- Config --------
WORKDIR="/home/gaga/tamarw1/trajectory-modeling"
PYTHON_BIN="python"
SCRIPT_PATH="$WORKDIR/scripts/analysis/hirid/hirid_circulatory_failure_analysis.py"
BASE_DIR="/home/gaga/data/physionet/hirid/circulatory_failure"
OUTPUT_DIR="$WORKDIR/results/hirid"

LOG_DIR="$WORKDIR/logs/outs"
TS="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="hirid_cf_${TS}"
LOG_FILE="$LOG_DIR/hirid_circulatory_failure_${TS}.log"

# Script args
EVAL_PROFILE="full"   # quick or full
LOOKBACK_HOURS=12
N_JOBS=4
TRAIN_N_JOBS=2
TRAIN_BACKEND="threads"
# ------------------------

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKDIR"

CMD="$PYTHON_BIN $SCRIPT_PATH \
  --base-dir $BASE_DIR \
  --output-dir $OUTPUT_DIR \
  --lookback-hours $LOOKBACK_HOURS \
  --eval-profile $EVAL_PROFILE \
  --n-jobs $N_JOBS \
  --train-n-jobs $TRAIN_N_JOBS \
  --train-backend $TRAIN_BACKEND"

tmux new-session -d -s "$SESSION_NAME" "cd '$WORKDIR' && $CMD > '$LOG_FILE' 2>&1"

echo "Started."
echo "tmux session : $SESSION_NAME"
echo "log file     : $LOG_FILE"
echo
echo "Monitor logs : tail -f '$LOG_FILE'"
echo "Attach tmux  : tmux attach -t '$SESSION_NAME'"
echo "List sessions: tmux ls"
echo "Stop job     : tmux kill-session -t '$SESSION_NAME'"
