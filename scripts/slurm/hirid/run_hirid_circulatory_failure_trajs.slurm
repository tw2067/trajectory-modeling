#!/usr/bin/env bash
#SBATCH --job-name=hirid_circulatory_trajs
#SBATCH --output=logs/outs/hirid_circulatory_%A_%a.out
#SBATCH --error=logs/errs/hirid_circulatory_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu-killable
#SBATCH --array=0-9

# HiRiD Circulatory Failure Trajectory Computation (multi-biomarker)

echo "=========================================="
echo "HiRiD Circulatory Failure Trajectory Computation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 128G"
echo "Start time: $(date)"
echo ""

# Cohort splitting
COHORT_SPLITS=10
COHORT_INDEX=${SLURM_ARRAY_TASK_ID}

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

# Run trajectory computation for all three biomarkers
python "$REPO_ROOT/scripts/trajectory/hirid/circulatory_failure_trajs.py" \
    --input-dir "$DATA_ROOT/hirid/circulatory_failure" \
    --output-dir "$DATA_ROOT/hirid/circulatory_failure" \
    --window-hours 12.0 \
    --n-batches 10 \
    --sampler nutpie \
    --cohort-splits ${COHORT_SPLITS} \
    --cohort-index ${COHORT_INDEX}

echo ""
echo "End time: $(date)"
echo "=========================================="
