#!/bin/bash
# Launch Eicu Ventilator Analysis via nohup
# Usage: ./scripts/run_eicu_ventilator_nohup.sh [--train-n-jobs N] [--train-backend threads|processes]

cd /home/gaga/tamarw1/trajectory-modeling || exit 1

TRAIN_N_JOBS=${TRAIN_N_JOBS:-1}
TRAIN_BACKEND=${TRAIN_BACKEND:-threading}
LOOKBACK_WINDOW=${LOOKBACK_WINDOW:-3}
LOOKBACK_UNIT=${LOOKBACK_UNIT:-days}
PARALLEL_AXIS=${PARALLEL_AXIS:-repeat}
OOF_SCOPE=${OOF_SCOPE:-representative-k}
OOF_K=${OOF_K:-10}
PLOT_K=${PLOT_K:-10}
SAVE_OOF=${SAVE_OOF:-0}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-n-jobs)
            TRAIN_N_JOBS="$2"
            shift 2
            ;;
        --train-backend)
            TRAIN_BACKEND="$2"
            shift 2
            ;;
        --lookback-window)
            LOOKBACK_WINDOW="$2"
            shift 2
            ;;
        --lookback-unit)
            LOOKBACK_UNIT="$2"
            shift 2
            ;;
        --parallel-axis)
            PARALLEL_AXIS="$2"
            shift 2
            ;;
        --oof-scope)
            OOF_SCOPE="$2"
            shift 2
            ;;
        --oof-k)
            OOF_K="$2"
            shift 2
            ;;
        --plot-k)
            PLOT_K="$2"
            shift 2
            ;;
        --save-oof)
            SAVE_OOF=1
            shift
            ;;
        --no-save-oof)
            SAVE_OOF=0
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting Eicu Ventilator Analysis (nohup)"
echo "  Train N-Jobs: $TRAIN_N_JOBS"
echo "  Train Backend: $TRAIN_BACKEND"
echo "  Lookback Window: $LOOKBACK_WINDOW"
echo "  Lookback Unit: $LOOKBACK_UNIT"
echo "  Parallel Axis: $PARALLEL_AXIS"
echo "  OOF Scope: $OOF_SCOPE"
echo "  OOF K: $OOF_K"
echo "  Plot K: $PLOT_K"
echo "  Save OOF: $SAVE_OOF"
echo "  Start time: $(date)"
echo ""

mkdir -p logs/outs logs/errs

OOF_ARGS=(--oof-scope "$OOF_SCOPE" --oof-k "$OOF_K")
if [[ "$SAVE_OOF" == "1" ]]; then
    OOF_ARGS+=(--save-oof)
else
    OOF_ARGS+=(--no-save-oof)
fi

nohup python scripts/analysis/eicu/eicu_ventilator_analysis.py \
    --train-n-jobs "$TRAIN_N_JOBS" \
    --train-backend "$TRAIN_BACKEND" \
    --lookback-window "$LOOKBACK_WINDOW" \
    --lookback-unit "$LOOKBACK_UNIT" \
    --parallel-axis "$PARALLEL_AXIS" \
    --plot-k "$PLOT_K" \
    "${OOF_ARGS[@]}" \
    > logs/outs/eicu_ventilator_analysis.out 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo $PID > .eicu_ventilator_analysis.pid

echo "Output: logs/outs/eicu_ventilator_analysis.out"
echo "To monitor: tail -f logs/outs/eicu_ventilator_analysis.out"
echo "To stop:    kill $PID"
