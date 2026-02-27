#!/bin/bash
#SBATCH --job-name=eicu_liver_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/eicu_liver_%A_%a.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/eicu_liver_%A_%a.err
#SBATCH --time=12:00:00
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

# Run eICU liver trajectory computation
echo "Starting eICU LIVER TRAJS run at $(date)"
echo "Cohort: 795 patients, 4,978 samples"
echo ""

python /home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/eicu/liver_trajs.py \
    --input /home/gaga/data/physionet/eicu/liver/bilirubin_timeseries.csv \
    --pred-dataset /home/gaga/data/physionet/eicu/liver/liver_prediction_dataset.csv \
    --output /home/gaga/data/physionet/eicu/liver/liver_trajectory_probs.csv \
    --probs-output /home/gaga/data/physionet/eicu/liver/liver_trajectory_probs_bayes.csv \
    --window-days 5.0 \
    --flat-thr 0.5 \
    --decline-thr 1.0 \
    --nonlinear-gap 3.0 \
    --n-batches 5 \
    --cohort-splits ${COHORT_SPLITS} \
    --cohort-index ${COHORT_INDEX}

EXIT_CODE=$?

echo ""
echo "eICU LIVER TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE
