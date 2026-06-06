#!/usr/bin/env bash
#SBATCH --job-name=mimic_circ_gpu_test
#SBATCH --output=logs/outs/%x_%j.out
#SBATCH --error=logs/errs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:1

# Small GPU smoke test for MIMIC circulatory-failure Bayesian trajectories
# Run: sbatch scripts/slurm/mimic/run_mimic_circulatory_failure_bayes_gpu_test.slurm

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"
mkdir -p "$LOG_ROOT/outs" "$LOG_ROOT/errs"
cd "$REPO_ROOT" || exit 1

source /home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh
conda activate pymc_env

echo "=========================================="
echo "MIMIC GPU smoke test"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Partition: killable"
echo "Window: 12.0h"
echo "Batches: 2"
echo "Cohort splits: 20"
echo "Cohort index: 0"
echo "Sampler: numpyro"
echo "df-basis: 4"
echo "=========================================="

python scripts/traj_scripts/mimic/mimic_circulatory_failure_bayes_trajectories.py \
    --cohort-index 0 \
    --cohort-splits 20 \
    --window-hours 12.0 \
    --n-batches 2 \
    --df-basis 4 \
    --sampler numpyro

EXIT_CODE=$?

echo "✓ GPU smoke test complete"

# cleanup pytensor cache (best-effort)
if [ -f "$REPO_ROOT/scripts/slurm/cleanup_pytensor_cache.sh" ]; then
    bash "$REPO_ROOT/scripts/slurm/cleanup_pytensor_cache.sh" || true
fi

exit $EXIT_CODE
