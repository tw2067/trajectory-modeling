#!/bin/bash
#SBATCH --job-name=mimic_ventilator_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/mimic_ventilator_%A_%a.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/mimic_ventilator_%A_%a.err
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

# Run ventilator trajectories
echo "Starting MIMIC VENTILATOR TRAJS run at $(date)"
echo ""

python /home/gaga/tamarw1/trajectory-modeling/scripts/traj_scripts/mimic/ventilator_trajs.py \
	--input /home/gaga/tamarw1/trajectory-modeling/results/mimic/ventilator/pf_ratio_timeseries.csv \
	--pred-dataset /home/gaga/tamarw1/trajectory-modeling/results/mimic/ventilator/ventilator_prediction_dataset.csv \
	--output /home/gaga/tamarw1/trajectory-modeling/results/mimic/ventilator/pf_trajectory_probs_bayes.csv \
	--merged-output /home/gaga/tamarw1/trajectory-modeling/results/mimic/ventilator/ventilator_prediction_dataset_with_probs.csv \
	--cohort-splits ${COHORT_SPLITS} \
	--cohort-index ${COHORT_INDEX}

EXIT_CODE=$?

echo ""
echo "MIMIC VENTILATOR TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE
