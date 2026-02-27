#!/bin/bash
#SBATCH --job-name=mimic_aki_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/mimic_aki_%A_%a.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/mimic_aki_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --partition=cpu-killable
#SBATCH --array=0-24


# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Activate conda environment
source /home/gaga/tamarw1/anaconda3/etc/profile.d/conda.sh
conda activate pymc_env

# Cohort splitting
COHORT_SPLITS=25
COHORT_INDEX=${SLURM_ARRAY_TASK_ID}

# Run AKI creatinine trajectories
echo "Starting MIMIC AKI TRAJS run at $(date)"
echo ""

python /home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/mimic/aki_trajs.py \
	--input /home/gaga/data/physionet/mimic/aki/creatinine_timeseries.csv \
	--pred-dataset /home/gaga/data/physionet/mimic/aki/aki_prediction_dataset.csv \
	--output /home/gaga/data/physionet/mimic/aki/aki_trajectory_probs_bayes.csv \
	--merged-output /home/gaga/data/physionet/mimic/aki/aki_prediction_dataset_with_probs.csv \
	--cohort-splits ${COHORT_SPLITS} \
	--cohort-index ${COHORT_INDEX}

EXIT_CODE=$?

echo ""
echo "MIMIC AKI TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE