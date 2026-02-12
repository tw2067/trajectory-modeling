#!/bin/bash
#SBATCH --job-name=eicu_circulatory_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/eicu_circulatory_%A_%a.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/eicu_circulatory_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

#SBATCH --partition=cpu-killable
#SBATCH --array=0-9

# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Activate conda environment
source /home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh
conda activate pymc_env

echo "Starting eICU CIRCULATORY FAILURE TRAJS run at $(date)"
echo ""

# Cohort splitting
COHORT_SPLITS=10
COHORT_INDEX=${SLURM_ARRAY_TASK_ID}

action_script=/home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/eicu/circulatory_failure_trajs.py

echo "Running: $action_script"

python $action_script \
    --data-dir /home/gaga/tamarw1/trajectory-modeling/results/eicu/circulatory_failure \
    --pred-dataset /home/gaga/tamarw1/trajectory-modeling/results/eicu/circulatory_failure/circulatory_failure_prediction_dataset.csv \
    --merged-output /home/gaga/tamarw1/trajectory-modeling/results/eicu/circulatory_failure/circulatory_failure_prediction_dataset_with_probs.csv \
    --window-hours 12.0 \
    --n-batches 8 \
    --cohort-splits ${COHORT_SPLITS} \
    --cohort-index ${COHORT_INDEX}

EXIT_CODE=$?

echo ""
echo "eICU CIRCULATORY FAILURE TRAJS run completed at $(date) with exit code $EXIT_CODE"

exit $EXIT_CODE
