#!/bin/bash
#SBATCH --job-name=eicu_vent_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/eicu_ventilator_%A.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/eicu_ventilator_%A.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

#SBATCH --gres=gpu:1
#SBATCH --account=gpu-research
#SBATCH --partition=killable


# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Activate conda environment
source /home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh
conda activate pymc_env

# Run eICU ventilator trajectory computation
echo "Starting eICU VENTILATOR TRAJS run at $(date)"
echo "Cohort: 2,827 patients, 11,206 samples"
echo ""

python /home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/eicu/ventilator_trajs.py \
    --input /home/gaga/tamarw1/trajectory-modeling/results/eicu/ventilator/pf_ratio_timeseries.csv \
    --pred-dataset /home/gaga/tamarw1/trajectory-modeling/results/eicu/ventilator/ventilator_prediction_dataset.csv \
    --output /home/gaga/tamarw1/trajectory-modeling/results/eicu/ventilator/ventilator_trajectory_probs.csv \
    --window-days 3.0 \
    --flat-thr 10.0 \
    --decline-thr 30.0 \
    --nonlinear-gap 20.0 \
    --n-batches 10

EXIT_CODE=$?

echo ""
echo "eICU VENTILATOR TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE
