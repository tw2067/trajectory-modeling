#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="python"
SCRIPT_PATH="$WORKDIR/scripts/analysis/hirid/hirid_circulatory_failure_analysis.py"
BASE_DIR="/home/gaga/data/physionet/hirid/circulatory_failure"
OUTPUT_DIR="$WORKDIR/results/hirid"

LOG_DIR="$WORKDIR/logs/outs"
LOG_FILE="$LOG_DIR/hirid_circulatory_failure.log"

# Script args
EVAL_PROFILE="full"         # quick or full
LOOKBACK_HOURS=12
N_JOBS=4                     # summary-stat parallelism
TRAIN_N_JOBS=2               # training repeat parallelism
TRAIN_BACKEND="threads"     # threads or processes

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKDIR"

nohup "$PYTHON_BIN" "$SCRIPT_PATH" \
  --base-dir "$BASE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --lookback-hours "$LOOKBACK_HOURS" \
  --eval-profile "$EVAL_PROFILE" \
  --n-jobs "$N_JOBS" \
  --train-n-jobs "$TRAIN_N_JOBS" \
  --train-backend "$TRAIN_BACKEND" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "Started HiRiD circulatory analysis"
echo "PID      : $PID"
echo "Log file : $LOG_FILE"
echo "Watch    : tail -f $LOG_FILE"
echo "Stop     : kill $PID"
