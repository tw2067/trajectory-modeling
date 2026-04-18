#!/usr/bin/env bash
#SBATCH --job-name=hirid_liver_trajs
#SBATCH --output=logs/outs/hirid_liver_%j.out
#SBATCH --error=logs/errs/hirid_liver_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-shamir
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:1

# HiRiD Liver Failure Trajectory Computation
# Computes bilirubin trajectory probabilities for liver failure prediction

echo "=========================================="
echo "HiRiD Liver Failure Trajectory Computation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64G"
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

# Run trajectory computation
python "$REPO_ROOT/scripts/traj_scripts/hirid/hirid_liver_trajs.py" \
    --input "$DATA_ROOT/hirid/liver/bilirubin_timeseries.csv" \
    --output "$DATA_ROOT/hirid/liver/liver_trajectory_probs_bayes.csv" \
    --window-days 5.0 \
    --flat-thr 10.0 \
    --increase-thr 20.0 \
    --n-batches 10 \
    --sampler pymc

echo ""
echo "End time: $(date)"
echo "=========================================="
