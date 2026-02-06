#!/bin/bash
#SBATCH --job-name=eicu_sepsis_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/eicu_sepsis_%A.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/eicu_sepsis_%A.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

#SBATCH --gres=gpu:1
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-shamir


# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Activate conda environment
source /home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh
conda activate pymc_env

# Run eICU sepsis trajectory computation for all 3 biomarkers
echo "Starting eICU SEPSIS TRAJS run at $(date)"
echo "Cohort: 5,767 patients, 15,054 samples"
echo ""

# Biomarker-specific parameters (matching MIMIC)
# Lactate: window=3.0, flat_thr=0.1, incline_thr=0.3, nonlinear_gap=0.5
# WBC: window=3.0, flat_thr=1.0, incline_thr=3.0, nonlinear_gap=2.0
# Platelets: window=3.0, flat_thr=-20, decline_thr=-50, nonlinear_gap=30.0 (BOTH NEGATIVE for decline detection!)

BASE_DIR="/home/gaga/tamarw1/trajectory-modeling/results/eicu/sepsis"

# Lactate trajectories
echo "\n=== Computing trajectories for lactate ==="
python /home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/eicu/sepsis_trajs.py \
    --biomarker lactate \
    --input ${BASE_DIR}/lactate_timeseries.csv \
    --pred-dataset ${BASE_DIR}/sepsis_prediction_dataset.csv \
    --output ${BASE_DIR}/lactate_trajectory_probs.csv \
    --window-days 3.0 \
    --flat-thr 0.1 \
    --decline-thr 0.3 \
    --nonlinear-gap 0.5 \
    --n-batches 8

if [ $? -ne 0 ]; then
    echo "ERROR: Lactate trajectory computation failed"
    exit 1
fi

# WBC trajectories
echo "\n=== Computing trajectories for WBC ==="
python /home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/eicu/sepsis_trajs.py \
    --biomarker wbc \
    --input ${BASE_DIR}/wbc_timeseries.csv \
    --pred-dataset ${BASE_DIR}/sepsis_prediction_dataset.csv \
    --output ${BASE_DIR}/wbc_trajectory_probs.csv \
    --window-days 3.0 \
    --flat-thr 1.0 \
    --decline-thr 3.0 \
    --nonlinear-gap 2.0 \
    --n-batches 8

if [ $? -ne 0 ]; then
    echo "ERROR: WBC trajectory computation failed"
    exit 1
fi

# Platelet trajectories (NOTE: BOTH thresholds are NEGATIVE for platelets - detecting declines)
echo "\n=== Computing trajectories for platelets ==="
python /home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/eicu/sepsis_trajs.py \
    --biomarker platelets \
    --input ${BASE_DIR}/platelets_timeseries.csv \
    --pred-dataset ${BASE_DIR}/sepsis_prediction_dataset.csv \
    --output ${BASE_DIR}/platelets_trajectory_probs.csv \
    --window-days 3.0 \
    --flat-thr -20.0 \
    --decline-thr -50.0 \
    --nonlinear-gap 30.0 \
    --n-batches 8

if [ $? -ne 0 ]; then
    echo "ERROR: Platelet trajectory computation failed"
    exit 1
fi

# Merge trajectory probabilities from all biomarkers
echo "\n=== Merging trajectory probabilities ==="
python << 'MERGE_SCRIPT'
import pandas as pd
import os

BASE_DIR = "/home/gaga/tamarw1/trajectory-modeling/results/eicu/sepsis"

# Load individual biomarker trajectory probs
lactate_traj = pd.read_csv(f'{BASE_DIR}/lactate_trajectory_probs.csv')
wbc_traj = pd.read_csv(f'{BASE_DIR}/wbc_trajectory_probs.csv')
platelet_traj = pd.read_csv(f'{BASE_DIR}/platelets_trajectory_probs.csv')

print(f"Lactate: {len(lactate_traj):,} timepoints")
print(f"WBC: {len(wbc_traj):,} timepoints")
print(f"Platelets: {len(platelet_traj):,} timepoints")

# Merge on stay_id and time_day
trajectory_probs = lactate_traj[['stay_id', 'time_day', 'prob_stable', 'prob_gradual_increase', 'prob_rapid_increase']].copy()
trajectory_probs.columns = ['stay_id', 'time_day', 'lactate_stable', 'lactate_gradual', 'lactate_rapid']

wbc_cols = wbc_traj[['stay_id', 'time_day', 'prob_stable', 'prob_gradual_increase', 'prob_rapid_increase']].copy()
wbc_cols.columns = ['stay_id', 'time_day', 'wbc_stable', 'wbc_gradual', 'wbc_rapid']
trajectory_probs = trajectory_probs.merge(wbc_cols, on=['stay_id', 'time_day'], how='outer')

platelet_cols = platelet_traj[['stay_id', 'time_day', 'prob_stable', 'prob_gradual_increase', 'prob_rapid_increase']].copy()
platelet_cols.columns = ['stay_id', 'time_day', 'platelet_stable', 'platelet_gradual', 'platelet_rapid']
trajectory_probs = trajectory_probs.merge(platelet_cols, on=['stay_id', 'time_day'], how='outer')

# Save merged trajectory probs
trajectory_probs.to_csv(f'{BASE_DIR}/sepsis_trajectory_probs.csv', index=False)
print(f"\n✓ Merged trajectory probs: {len(trajectory_probs):,} timepoints")
print(f"  Columns: {len(trajectory_probs.columns)}")
MERGE_SCRIPT

EXIT_CODE=$?

echo ""
echo "eICU SEPSIS TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE
