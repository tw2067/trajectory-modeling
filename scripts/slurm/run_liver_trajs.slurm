#!/bin/bash
#SBATCH --job-name=mimic_liver_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/mimic_liver_%A.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/mimic_liver_%A.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

#SBATCH --gres=gpu:1
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-shamir


# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Run liver bilirubin trajectory analysis
echo "Starting LIVER BILIRUBIN TRAJS run at $(date)"
echo ""

python scripts/liver_trajs.py

EXIT_CODE=$?

echo ""
echo "LIVER BILIRUBIN TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE
