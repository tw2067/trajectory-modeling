#!/usr/bin/env bash
#SBATCH --job-name=eicu_aki_trajs
#SBATCH --output=logs/outs/eicu_aki_%A_%a.out
#SBATCH --error=logs/errs/eicu_aki_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

#SBATCH --partition=cpu-killable
#SBATCH --array=0-39


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"
mkdir -p "$LOG_ROOT/outs" "$LOG_ROOT/errs" "$RESULTS_ROOT/eicu/aki"
cd "$REPO_ROOT" || exit 1

# Activate conda environment
source /home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh
conda activate pymc_env

# Cohort splitting
COHORT_SPLITS=40
COHORT_INDEX=${SLURM_ARRAY_TASK_ID}

# Run eICU AKI trajectory computation
echo "Starting eICU AKI TRAJS run at $(date)"
echo "Cohort: 4,062 patients, 37,766 samples"
echo ""

python "$REPO_ROOT/scripts/trajectory/eicu/aki_trajs.py" \
    --input "$DATA_ROOT/eicu/aki/creatinine_timeseries.csv" \
    --pred-dataset "$DATA_ROOT/eicu/aki/aki_prediction_dataset.csv" \
    --output "$DATA_ROOT/eicu/aki/aki_trajectory_probs.csv" \
    --probs-output "$RESULTS_ROOT/eicu/aki/aki_trajectory_probs_bayes.csv" \
    --window-days 7.0 \
    --flat-thr 0.1 \
    --decline-thr 0.3 \
    --nonlinear-gap 0.5 \
    --n-batches 20 \
    --cohort-splits ${COHORT_SPLITS} \
    --cohort-index ${COHORT_INDEX}

EXIT_CODE=$?

echo ""
echo "eICU AKI TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Attempt to clean PyTensor cache for this job (best-effort)
if [ -f "$REPO_ROOT/scripts/slurm/cleanup_pytensor_cache.sh" ]; then
    bash "$REPO_ROOT/scripts/slurm/cleanup_pytensor_cache.sh" || true
fi

# Exit with the same code as the Python script
exit $EXIT_CODE
