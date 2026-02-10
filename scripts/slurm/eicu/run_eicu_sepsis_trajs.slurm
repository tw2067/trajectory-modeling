#!/bin/bash
#SBATCH --job-name=eicu_sepsis_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/eicu_sepsis_%A_%a.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/eicu_sepsis_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

#SBATCH --partition=cpu-killable
#SBATCH --array=0-9


# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Activate conda environment
source /home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh
conda activate pymc_env

# Cohort splitting
COHORT_SPLITS=10
COHORT_INDEX=${SLURM_ARRAY_TASK_ID}

echo "Starting eICU SEPSIS TRAJS run at $(date)"
echo ""

python /home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/eicu/sepsis_all_biomarkers.py \
    --data-dir /home/gaga/tamarw1/trajectory-modeling/results/eicu/sepsis \
    --pred-dataset /home/gaga/tamarw1/trajectory-modeling/results/eicu/sepsis/sepsis_prediction_dataset.csv \
    --output /home/gaga/tamarw1/trajectory-modeling/results/eicu/sepsis/sepsis_trajectory_probs.csv \
    --window-days 3.0 \
    --n-batches 8 \
    --cohort-splits ${COHORT_SPLITS} \
    --cohort-index ${COHORT_INDEX}

EXIT_CODE=$?

echo ""
echo "eICU SEPSIS TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE
