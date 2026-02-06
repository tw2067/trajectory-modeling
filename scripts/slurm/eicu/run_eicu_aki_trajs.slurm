#!/bin/bash
#SBATCH --job-name=eicu_aki_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/eicu_aki_%A.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/eicu_aki_%A.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

#SBATCH --gres=gpu:1
#SBATCH --account=gpu-research
#SBATCH --partition=killable


# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Activate conda environment
source /home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh
conda activate pymc_env

# Run eICU AKI trajectory computation
echo "Starting eICU AKI TRAJS run at $(date)"
echo "Cohort: 4,062 patients, 37,766 samples"
echo ""

python /home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/eicu/aki_trajs.py \
    --input /home/gaga/tamarw1/trajectory-modeling/results/eicu/aki/creatinine_timeseries.csv \
    --pred-dataset /home/gaga/tamarw1/trajectory-modeling/results/eicu/aki/aki_prediction_dataset.csv \
    --output /home/gaga/tamarw1/trajectory-modeling/results/eicu/aki/aki_trajectory_probs_klbl.csv \
    --window-days 7.0 \
    --flat-thr 0.1 \
    --decline-thr 0.3 \
    --nonlinear-gap 0.5 \
    --n-batches 20

EXIT_CODE=$?

echo ""
echo "eICU AKI TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE
