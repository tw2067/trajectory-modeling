#!/bin/bash
#SBATCH --job-name=updrs3_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/updrs3_%A.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/updrs3_%A.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G

#SBATCH --gres=gpu:1
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-shamir


# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Run updrs3 representation
echo "Starting UPDRS3 run at $(date)"
echo ""

python code/ppmi_updrs3_bayes.py 

EXIT_CODE=$?

echo ""
echo "UPDRS3 run completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE