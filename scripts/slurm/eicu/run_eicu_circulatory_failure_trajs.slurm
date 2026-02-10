#!/bin/bash
#SBATCH --job-name=eicu_circulatory_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/eicu_circulatory_%j.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/eicu_circulatory_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

#SBATCH --partition=cpu-killable

# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Activate conda environment
source /home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh
conda activate pymc_env

echo "Starting eICU CIRCULATORY FAILURE TRAJS run at $(date)"
echo ""

action_script=/home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/eicu/circulatory_failure_trajs.py

echo "Running: $action_script"

python $action_script \
    --data-dir /home/gaga/tamarw1/trajectory-modeling/results/eicu/circulatory_failure \
    --pred-dataset /home/gaga/tamarw1/trajectory-modeling/results/eicu/circulatory_failure/circulatory_failure_prediction_dataset.csv \
    --merged-output /home/gaga/tamarw1/trajectory-modeling/results/eicu/circulatory_failure/circulatory_failure_prediction_dataset_with_probs.csv \
    --window-hours 12.0 \
    --n-batches 8

EXIT_CODE=$?

echo ""
echo "eICU CIRCULATORY FAILURE TRAJS run completed at $(date) with exit code $EXIT_CODE"

exit $EXIT_CODE
