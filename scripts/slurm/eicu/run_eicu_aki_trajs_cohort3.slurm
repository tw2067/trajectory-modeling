#!/usr/bin/env bash
#SBATCH --job-name=eicu_aki_cohort3
#SBATCH --output=logs/outs/eicu_aki_cohort3_%j.out
#SBATCH --error=logs/errs/eicu_aki_cohort3_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=cpu-killable

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }

RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"
mkdir -p "$LOG_ROOT/outs" "$LOG_ROOT/errs" "$RESULTS_ROOT/eicu/aki"
cd "$REPO_ROOT" || exit 1

# Activate conda environment
source /home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh
conda activate pymc_env

# Run only sub-cohort 3/8 (0-indexed cohort 2)
TOTAL_COHORTS=8
COHORT_INDEX=2

echo "Starting eICU AKI TRAJs for sub-cohort 3/8 at $(date)"
echo ""

python "$REPO_ROOT/scripts/traj_scripts/eicu/aki_trajs.py" \
    --input "$RESULTS_ROOT/eicu/aki/creatinine_timeseries.csv" \
    --pred-dataset "$RESULTS_ROOT/eicu/aki/aki_prediction_dataset.csv" \
    --output "$RESULTS_ROOT/eicu/aki/aki_trajectory_probs.csv" \
    --window-days 7.0 \
    --flat-thr 0.1 \
    --decline-thr 0.3 \
    --nonlinear-gap 0.5 \
    --n-batches 10 \
    --cohort-splits $TOTAL_COHORTS \
    --cohort-index $COHORT_INDEX

EXIT_CODE=$?

echo ""
echo "Sub-cohort 3/8 completed at $(date) with exit code $EXIT_CODE"

exit $EXIT_CODE
