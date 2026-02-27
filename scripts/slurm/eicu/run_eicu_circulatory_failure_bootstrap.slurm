#!/bin/bash
#SBATCH --job-name=eicu_circ_bootstrap
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/eicu_circ_bootstrap_%A_%a.out
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

#SBATCH --partition=cpu-killable
#SBATCH --array=0-7

# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Activate conda environment
source /home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh
conda activate pymc_env

echo "========================================================================"
echo "Starting eICU Circulatory Failure BOOTSTRAP trajectory run"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Start time: $(date)"
echo "========================================================================"

# Cohort splitting
COHORT_SPLITS=8
COHORT_INDEX=${SLURM_ARRAY_TASK_ID}

action_script=/home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/eicu/circulatory_failure_trajs_bootstrap.py

echo "Running: $action_script"
echo "Cohort: ${COHORT_INDEX}/${COHORT_SPLITS}"

python $action_script \
    --data-dir /home/gaga/tamarw1/trajectory-modeling/results/eicu/circulatory_failure \
    --pred-dataset /home/gaga/tamarw1/trajectory-modeling/results/eicu/circulatory_failure/circulatory_failure_prediction_dataset.csv \
    --merged-output /home/gaga/tamarw1/trajectory-modeling/results/eicu/circulatory_failure/circulatory_failure_prediction_dataset_with_bootstrap_probs.csv \
    --window-hours 12.0 \
    --n-bootstrap 200 \
    --n-batches 8 \
    --id-col stay_id \
    --cohort-splits ${COHORT_SPLITS} \
    --cohort-index ${COHORT_INDEX}

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "eICU Circulatory Failure BOOTSTRAP run completed"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================================================"

exit $EXIT_CODE
