#!/bin/bash
#SBATCH --job-name=eicu_vent_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/eicu_ventilator_%A_%a.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/eicu_ventilator_%A_%a.err
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

# Run eICU ventilator trajectory computation
echo "Starting eICU VENTILATOR TRAJS run at $(date)"
echo "Cohort: 2,827 patients, 11,206 samples"
echo ""

python /home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/eicu/ventilator_trajs.py \
    --input /home/gaga/data/physionet/eicu/ventilator/pf_ratio_timeseries.csv \
    --pred-dataset /home/gaga/data/physionet/eicu/ventilator/ventilator_prediction_dataset.csv \
    --output /home/gaga/data/physionet/eicu/ventilator/ventilator_trajectory_probs.csv \
    --probs-output /home/gaga/data/physionet/eicu/ventilator/ventilator_trajectory_probs_bayes.csv \
    --window-days 3.0 \
    --flat-thr 10.0 \
    --decline-thr 30.0 \
    --nonlinear-gap 20.0 \
    --n-batches 10 \
    --cohort-splits ${COHORT_SPLITS} \
    --cohort-index ${COHORT_INDEX}

EXIT_CODE=$?

echo ""
echo "eICU VENTILATOR TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE
