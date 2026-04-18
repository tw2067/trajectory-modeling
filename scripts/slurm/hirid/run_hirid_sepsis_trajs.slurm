#!/usr/bin/env bash
#SBATCH --job-name=hirid_sepsis_trajs
#SBATCH --output=logs/outs/hirid_sepsis_%j.out
#SBATCH --error=logs/errs/hirid_sepsis_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-shamir

# HiRiD Sepsis Trajectory Computation (multi-biomarker)
# Computes trajectory probabilities for lactate, WBC, and platelets

echo "=========================================="
echo "HiRiD Septic Shock Trajectory Computation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 128G"
echo "Start time: $(date)"
echo ""

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
python "$REPO_ROOT/scripts/traj_scripts/hirid/hirid_sepsis_trajs.py" \
    --input-dir "$DATA_ROOT/hirid/sepsis" \
    --output-dir "$DATA_ROOT/hirid/sepsis" \
    --window-days 3.0 \
    --n-batches 10 \
    --sampler pymc

echo ""
echo "End time: $(date)"
echo "=========================================="
