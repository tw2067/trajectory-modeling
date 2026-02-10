#!/bin/bash
#SBATCH --job-name=eicu_aki_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/eicu_aki_%A_%a.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/eicu_aki_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

#SBATCH --partition=cpu-killable
#SBATCH --array=0-19


# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Activate conda environment
source /home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh
conda activate pymc_env

# Cohort splitting
COHORT_SPLITS=20
COHORT_INDEX=${SLURM_ARRAY_TASK_ID}

# Run eICU AKI trajectory computation
echo "Starting eICU AKI TRAJS run at $(date)"
echo "Cohort: 4,062 patients, 37,766 samples"
echo ""

python /home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/eicu/aki_trajs.py \
    --input /home/gaga/tamarw1/trajectory-modeling/results/eicu/aki/creatinine_timeseries.csv \
    --pred-dataset /home/gaga/tamarw1/trajectory-modeling/results/eicu/aki/aki_prediction_dataset.csv \
    --output /home/gaga/tamarw1/trajectory-modeling/results/eicu/aki/aki_trajectory_probs.csv \
    --probs-output /home/gaga/tamarw1/trajectory-modeling/results/eicu/aki/aki_trajectory_probs_bayes.csv \
    --window-days 7.0 \
    --flat-thr 0.1 \
    --decline-thr 0.3 \
    --nonlinear-gap 0.5 \
    --n-batches 20 \
    --cohort-splits ${COHORT_SPLITS} \
    --cohort-index ${COHORT_INDEX}

EXIT_CODE=$?

echo ""
echo "eICU AKI TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE
