#!/bin/bash
#SBATCH --job-name=mimic_circ_bootstrap
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/mimic_circ_bootstrap_%A_%a.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/mimic_circ_bootstrap_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

#SBATCH --partition=cpu-killable
#SBATCH --array=0-4

# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Activate conda environment
source /home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh
conda activate pymc_env

echo "========================================================================"
echo "Starting MIMIC Circulatory Failure BOOTSTRAP trajectory run"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Start time: $(date)"
echo "========================================================================"

# Cohort splitting
COHORT_SPLITS=5
COHORT_INDEX=${SLURM_ARRAY_TASK_ID}

action_script=/home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/mimic/circulatory_failure_trajs_bootstrap.py

echo "Running: $action_script"
echo "Cohort: ${COHORT_INDEX}/${COHORT_SPLITS}"

python $action_script \
    --data-dir /home/gaga/data/physionet/mimic/circulatory_failure \
    --pred-dataset /home/gaga/data/physionet/mimic/circulatory_failure/circulatory_failure_prediction_dataset.csv \
    --merged-output /home/gaga/data/physionet/mimic/circulatory_failure/circulatory_failure_prediction_dataset_with_bootstrap_probs.csv \
    --window-hours 12.0 \
    --n-bootstrap 200 \
    --n-batches 8 \
    --cohort-splits ${COHORT_SPLITS} \
    --cohort-index ${COHORT_INDEX}

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "MIMIC Circulatory Failure BOOTSTRAP run completed"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================================================"

exit $EXIT_CODE
