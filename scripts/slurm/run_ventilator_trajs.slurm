#!/bin/bash
#SBATCH --job-name=mimic_ventilator_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/mimic_ventilator_%A.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/mimic_ventilator_%A.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

#SBATCH --gres=gpu:1
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-shamir


# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Run ventilator trajectories
echo "Starting VENTILATOR TRAJS run at $(date)"
echo ""

python scripts/notebooks/ventilator_trajs.py

EXIT_CODE=$?

echo ""
echo "VENTILATOR TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE
