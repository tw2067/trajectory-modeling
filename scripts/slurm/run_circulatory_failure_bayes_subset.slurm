#!/usr/bin/env bash
#SBATCH --job-name=circ_bayes_subset
#SBATCH --output=logs/outs/circ_bayes_subset_%A_%a.out
#SBATCH --error=logs/errs/circ_bayes_subset_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --partition=cpu-killable
#SBATCH --array=0-8

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

# Activate conda environment
source /home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh
conda activate pymc_env

# Configuration
N_PATIENTS=1500
SEED=920
WINDOW_HOURS=12.0
N_BATCHES=20
COHORTS_PER_DATASET=3
SUBSET_ROOT="${SUBSET_ROOT:-$DATA_ROOT/subsets/circulatory_failure_bayes}"

# Array task mapping: 9 tasks (0-8) = 3 datasets × 3 cohorts
# Tasks 0-2: mimic (cohorts 0, 1, 2)
# Tasks 3-5: eicu (cohorts 0, 1, 2)
# Tasks 6-8: hirid (cohorts 0, 1, 2)
ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
DATASETS=("mimic" "eicu" "hirid")
DATASET_IDX=$((ARRAY_TASK_ID / COHORTS_PER_DATASET))
COHORT_IDX=$((ARRAY_TASK_ID % COHORTS_PER_DATASET))
DATASET=${DATASETS[$DATASET_IDX]}

echo "=========================================="
echo "Bayesian Circulatory Failure Subset Run"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $ARRAY_TASK_ID"
echo "Dataset: $DATASET (index $DATASET_IDX)"
echo "Cohort: $COHORT_IDX / $COHORTS_PER_DATASET"
echo "Nodes: $SLURM_NNODES"
echo "Cores per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Time limit: $SLURM_TIMELIMIT"
echo "Partition: $SLURM_JOB_PARTITION"
echo ""
echo "Parameters:"
echo "  Patients per dataset: $N_PATIENTS"
echo "  Seed: $SEED"
echo "  Window hours: $WINDOW_HOURS"
echo "  N batches: $N_BATCHES"
echo "  Cohorts per dataset: $COHORTS_PER_DATASET"
echo "  Subset root: $SUBSET_ROOT"
echo ""
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Run the subset preparation and Bayesian trajectories for this dataset/cohort
python "$REPO_ROOT/scripts/traj_scripts/run_circulatory_failure_bayes_subset.py" \
    --datasets "$DATASET" \
    --n-patients "$N_PATIENTS" \
    --seed "$SEED" \
    --window-hours "$WINDOW_HOURS" \
    --n-batches "$N_BATCHES" \
    --cohort-splits "$COHORTS_PER_DATASET" \
    --cohort-index "$COHORT_IDX" \
    --subset-root "$SUBSET_ROOT"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
