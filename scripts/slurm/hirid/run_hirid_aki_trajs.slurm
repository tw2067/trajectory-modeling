#!/bin/bash
#SBATCH --job-name=hirid_aki_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/hirid_aki_%A.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/hirid_aki_%A.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

#SBATCH --gres=gpu:1
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-shamir


# Navigate to project directory
cd /home/gaga/tamarw1/trajectory-modeling || exit 1

# Run HiRiD AKI trajectory computation
echo "Starting HiRiD AKI TRAJS run at $(date)"
echo ""

python /home/gaga/tamarw1/trajectory-modeling/scripts/hirid_aki_trajs.py \
    --input /home/gaga/data/physionet/hirid/aki/creatinine_timeseries.csv \
    --output /home/gaga/data/physionet/hirid/aki/aki_trajectory_probs_bayes.csv \
    --window-days 7.0 \
    --flat-thr 0.1 \
    --decline-thr 0.3 \
    --n-batches 10

EXIT_CODE=$?

echo ""
echo "HiRiD AKI TRAJS run completed at $(date) with exit code $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE
