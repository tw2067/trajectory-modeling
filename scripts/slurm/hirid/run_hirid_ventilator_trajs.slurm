#!/usr/bin/env bash
#SBATCH --job-name=hirid_ventilator_trajs
#SBATCH --output=logs/outs/hirid_ventilator_%j.out
#SBATCH --error=logs/errs/hirid_ventilator_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-shamir
#SBATCH --account=gpu-research

# HiRiD Ventilator Discontinuation Trajectory Computation
# Computes P/F ratio trajectory probabilities for ventilator weaning prediction

echo "=========================================="
echo "HiRiD Ventilator Trajectory Computation"
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

# Run trajectory computation
python "$REPO_ROOT/scripts/traj_scripts/hirid/hirid_ventilator_trajs.py" \
    --input "$DATA_ROOT/hirid/ventilator/pf_ratio_timeseries.csv" \
    --output "$DATA_ROOT/hirid/ventilator/ventilator_trajectory_probs_bayes.csv" \
    --window-days 3.0 \
    --flat-thr 10.0 \
    --decline-thr 30.0 \
    --n-batches 10

echo ""
echo "End time: $(date)"
echo "=========================================="
