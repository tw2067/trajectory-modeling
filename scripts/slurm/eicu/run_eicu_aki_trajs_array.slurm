#!/bin/bash
#SBATCH --job-name=eicu_aki_array
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/eicu_aki_%A_%a.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/eicu_aki_%A_%a.err
#SBATCH --array=0-7
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G


#SBATCH --partition=cpu-killable


# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Activate conda environment
source /home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh
conda activate pymc_env

# Array job setup: 8 cohorts total, processing cohort $SLURM_ARRAY_TASK_ID
TOTAL_COHORTS=8
COHORT_INDEX=$SLURM_ARRAY_TASK_ID

# Run eICU AKI trajectory computation for this sub-cohort
echo "Starting eICU AKI TRAJsS array job at $(date)"
echo "Processing sub-cohort: $((COHORT_INDEX + 1))/$TOTAL_COHORTS"
echo ""

python /home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/eicu/aki_trajs.py \
    --input /home/gaga/tamarw1/trajectory-modeling/results/eicu/aki/creatinine_timeseries.csv \
    --pred-dataset /home/gaga/tamarw1/trajectory-modeling/results/eicu/aki/aki_prediction_dataset.csv \
    --output /home/gaga/tamarw1/trajectory-modeling/results/eicu/aki/aki_trajectory_probs.csv \
    --window-days 7.0 \
    --flat-thr 0.1 \
    --decline-thr 0.3 \
    --nonlinear-gap 0.5 \
    --n-batches 10 \
    --cohort-splits $TOTAL_COHORTS \
    --cohort-index $COHORT_INDEX

EXIT_CODE=$?

echo ""
echo "Sub-cohort $((COHORT_INDEX + 1))/$TOTAL_COHORTS completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE
