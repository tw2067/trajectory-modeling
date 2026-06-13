#!/usr/bin/env bash
#SBATCH --job-name=eicu_circ_fail_bayes_traj
#SBATCH --output=logs/outs/%x_%j.out
#SBATCH --error=logs/errs/%x_%j.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu-killable

# eICU Circulatory Failure Bayesian Trajectory Generation
# Run: sbatch scripts/slurm/eicu/run_eicu_circulatory_failure_bayes_trajectories.slurm

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }

COHORT_INDEX=${SLURM_ARRAY_TASK_ID}
WINDOW_HOURS=12.0
N_BATCHES=8
DF_BASIS=${DF_BASIS:-4}
SAMPLER=nutpie

DATA_ROOT="${DATA_ROOT:-${TRAJ_DATA_ROOT:-/home/gaga/data/physionet}}"
export TRAJ_DATA_ROOT="$DATA_ROOT"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"
mkdir -p "$LOG_ROOT/outs" "$LOG_ROOT/errs"
cd "$REPO_ROOT" || exit 1

# Activate conda environment
source "${CONDA_SH:-/home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh}"
conda activate pymc_env

echo "=========================================="
echo "eICU Bayesian Trajectory Generation"
echo "=========================================="
echo "Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Cohort Index: ${COHORT_INDEX}"
echo "N Patients: ${N_PATIENTS}"
echo "Seed: ${SEED}"
echo "Window: ${WINDOW_HOURS}h"
echo "Batches: ${N_BATCHES}"
echo "=========================================="

python scripts/traj_scripts/eicu/eicu_circulatory_failure_bayes_trajectories.py \
    --cohort-index ${COHORT_INDEX} \
    --cohort-splits 4 \
    --window-hours ${WINDOW_HOURS} \
    --n-batches ${N_BATCHES} \
    --df-basis ${DF_BASIS} \
    --sampler ${SAMPLER}

echo "✓ Trajectory generation complete for cohort ${COHORT_INDEX} with ${SAMPLER} sampler"
