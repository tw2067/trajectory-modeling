#!/bin/bash
#SBATCH --job-name=mimic_aki_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/mimic_aki_%A.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/mimic_aki_%A.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

#SBATCH --gres=gpu:1
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-shamir


# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Run updrs3 representation
echo "Starting AKI CREAT TRAJS run at $(date)"
echo ""

python /home/gaga/tamarw1/trajectory-modeling/scripts/aki_trajs.py

EXIT_CODE=$?

echo ""
echo "AKI CREAT TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE