#!/usr/bin/env bash
#SBATCH --job-name=hirid_circ_fail_bayes_traj
#SBATCH --output=logs/outs/%x_%j.out
#SBATCH --error=logs/errs/%x_%j.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu-killable

# HiRiD Circulatory Failure Bayesian Trajectory Generation
# Run: sbatch scripts/slurm/hirid/run_hirid_circulatory_failure_bayes_trajectories.slurm

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

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"
mkdir -p "$LOG_ROOT/outs" "$LOG_ROOT/errs"

# Keep Python, Numba, Matplotlib, and PyTensor caches off /tmp so jobs do not
# fail when the shared temp filesystem fills up.
CACHE_ROOT="${CACHE_ROOT:-/home/gaga/tamarw1}"
JOB_CACHE_ROOT="$CACHE_ROOT/.cache/hirid_circ_fail_bayes/${SLURM_JOB_ID:-local}"
mkdir -p \
    "$JOB_CACHE_ROOT/tmp" \
    "$JOB_CACHE_ROOT/pycache" \
    "$JOB_CACHE_ROOT/numba" \
    "$JOB_CACHE_ROOT/matplotlib" \
    "$JOB_CACHE_ROOT/arviz" \
    "$JOB_CACHE_ROOT/pytensor"
export TMPDIR="$JOB_CACHE_ROOT/tmp"
export TEMP="$JOB_CACHE_ROOT/tmp"
export TMP="$JOB_CACHE_ROOT/tmp"
export PYTHONPYCACHEPREFIX="$JOB_CACHE_ROOT/pycache"
export NUMBA_CACHE_DIR="$JOB_CACHE_ROOT/numba"
export MPLCONFIGDIR="$JOB_CACHE_ROOT/matplotlib"
export ARVIZ_DATA_HOME="$JOB_CACHE_ROOT/arviz"
export PYTENSOR_FLAGS="compiledir=$JOB_CACHE_ROOT/pytensor,base_compiledir=$JOB_CACHE_ROOT/pytensor,optimizer=fast_compile,exception_verbosity=high"

cd "$REPO_ROOT" || exit 1

# Activate conda environment
source /home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh
conda activate pymc_env

echo "=========================================="
echo "HiRiD Bayesian Trajectory Generation"
echo "=========================================="
echo "Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Cohort Index: ${COHORT_INDEX}"
echo "N Patients: ${N_PATIENTS}"
echo "Seed: ${SEED}"
echo "Window: ${WINDOW_HOURS}h"
echo "Batches: ${N_BATCHES}"
echo "=========================================="

python scripts/traj_scripts/hirid/hirid_circulatory_failure_bayes_trajectories.py \
    --cohort-index ${COHORT_INDEX} \
    --cohort-splits 4 \
    --window-hours ${WINDOW_HOURS} \
    --n-batches ${N_BATCHES} \
    --df-basis ${DF_BASIS} \
    --sampler ${SAMPLER}

echo "✓ Trajectory generation complete for cohort ${COHORT_INDEX} with ${SAMPLER} sampler"
