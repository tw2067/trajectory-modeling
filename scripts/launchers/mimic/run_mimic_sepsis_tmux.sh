#!/bin/bash
# Launch MIMIC Sepsis Analysis via tmux
# Usage: ./scripts/run_mimic_sepsis_tmux.sh [--train-n-jobs N] [--train-backend threading|processes] [--parallel-axis repeat|feature-set] [--save-oof|--no-save-oof]

SESSION_NAME="mimic_sepsis_analysis"
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

if [[ "$SAVE_OOF" == "1" ]]; then
    OOF_FLAG="--save-oof"
else
    OOF_FLAG="--no-save-oof"
fi

cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Kill existing session if present
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Create new session
tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50

# Send command
tmux send-keys -t "$SESSION_NAME" \
    "cd /home/gaga/tamarw1/trajectory-modeling && python scripts/analysis/mimic/mimic_sepsis_analysis.py --train-n-jobs $TRAIN_N_JOBS --train-backend $TRAIN_BACKEND --lookback-window $LOOKBACK_WINDOW --lookback-unit $LOOKBACK_UNIT --parallel-axis $PARALLEL_AXIS --plot-k $PLOT_K --oof-scope $OOF_SCOPE --oof-k $OOF_K $OOF_FLAG" \
    Enter

echo "✓ tmux session created: $SESSION_NAME"
echo "  Train N-Jobs: $TRAIN_N_JOBS"
echo "  Train Backend: $TRAIN_BACKEND"
echo "  Lookback Window: $LOOKBACK_WINDOW"
echo "  Lookback Unit: $LOOKBACK_UNIT"
echo "  Parallel Axis: $PARALLEL_AXIS"
echo "  OOF Scope: $OOF_SCOPE"
echo "  OOF K: $OOF_K"
echo "  Plot K: $PLOT_K"
echo "  Save OOF: $SAVE_OOF"
echo ""
echo "Attach: tmux attach -t $SESSION_NAME"
echo "List sessions: tmux list-sessions"
echo "Kill session: tmux kill-session -t $SESSION_NAME"
