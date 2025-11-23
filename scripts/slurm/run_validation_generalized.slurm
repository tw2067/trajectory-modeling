#!/bin/bash
#SBATCH --job-name=traj_validate_parkinsons
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/generalized/validate_%A_%a.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/generalized/validate_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --array=0-23

#SBATCH --gres=gpu:1
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-shamir

# Configuration arrays
BACKENDS=("deep" "bayes" "gam")
SCENARIOS=("mixed" "heterogeneous")
DISEASES=("ckd" "hiv" "alzheimers" "parkinsons")

# Calculate indices (3 backends × 2 scenarios × 4 diseases = 24 combinations)
BACKEND_IDX=$((SLURM_ARRAY_TASK_ID / 8))
SCENARIO_IDX=$(((SLURM_ARRAY_TASK_ID / 4) % 2))
DISEASE_IDX=$((SLURM_ARRAY_TASK_ID % 4))

BACKEND=${BACKENDS[$BACKEND_IDX]}
SCENARIO=${SCENARIOS[$SCENARIO_IDX]}
DISEASE=${DISEASES[$DISEASE_IDX]}

# Print configuration
echo "========================================================================"
echo "SLURM Job Configuration"
echo "========================================================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Array Task ID:   $SLURM_ARRAY_TASK_ID"
echo "Node:            $SLURM_NODELIST"
echo "CPUs:            $SLURM_CPUS_PER_TASK"
echo "Memory:          64GB"
echo "Backend:         $BACKEND"
echo "Scenario:        $SCENARIO"
echo "Disease:         $DISEASE"
echo "========================================================================"
echo ""

# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Set number of threads to match SLURM allocation
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run validation
echo "Starting validation at $(date)"
echo ""

python scripts/validate_trajectories.py \
    --backend "$BACKEND" \
    --scenario "$SCENARIO" \
    --disease "$DISEASE" \
    --n-patients 100 \
    --seed 920

EXIT_CODE=$?

echo ""
echo "Validation completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE