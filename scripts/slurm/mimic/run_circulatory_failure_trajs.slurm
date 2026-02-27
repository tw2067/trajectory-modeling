#!/bin/bash
#SBATCH --job-name=mimic_circulatory_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/mimic_circulatory_%A_%a.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/mimic_circulatory_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

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

# Run circulatory failure trajectories for all biomarkers in one pass

echo "Starting MIMIC CIRCULATORY FAILURE TRAJS run at $(date)"
echo ""

python /home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/mimic/circulatory_failure_trajs.py \
		--biomarker all \
		--data-dir /home/gaga/data/physionet/mimic/circulatory_failure \
		--pred-dataset /home/gaga/data/physionet/mimic/circulatory_failure/circulatory_failure_prediction_dataset.csv \
		--merged-output /home/gaga/data/physionet/mimic/circulatory_failure/circulatory_failure_prediction_dataset_with_probs.csv \
		--window-hours 12.0 \
		--cohort-splits ${COHORT_SPLITS} \
		--cohort-index ${COHORT_INDEX}

EXIT_CODE=$?

echo ""
echo "MIMIC CIRCULATORY FAILURE TRAJS run completed at $(date) with exit code $EXIT_CODE"

exit $EXIT_CODE
