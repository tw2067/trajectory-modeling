#!/usr/bin/env bash
#SBATCH --job-name=hirid_aki_trajs
#SBATCH --output=logs/outs/hirid_aki_%A.out
#SBATCH --error=logs/errs/hirid_aki_%A.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

#SBATCH --gres=gpu:1
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-shamir


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"
mkdir -p "$LOG_ROOT/outs" "$LOG_ROOT/errs"
cd "$REPO_ROOT" || exit 1

# Run HiRiD AKI trajectory computation
echo "Starting HiRiD AKI TRAJS run at $(date)"
echo ""

python "$REPO_ROOT/scripts/traj_scripts/hirid/hirid_aki_trajs.py" \
    --input "$DATA_ROOT/hirid/aki/creatinine_timeseries.csv" \
    --output "$DATA_ROOT/hirid/aki/aki_trajectory_probs_bayes.csv" \
    --window-days 7.0 \
    --flat-thr 0.1 \
    --decline-thr 0.3 \
    --n-batches 10

EXIT_CODE=$?

echo ""
echo "HiRiD AKI TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE
