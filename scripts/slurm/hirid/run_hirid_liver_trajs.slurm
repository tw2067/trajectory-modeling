#!/bin/bash
#SBATCH --job-name=hirid_liver_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/hirid_liver_%j.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/hirid_liver_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-shamir
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:1

# HiRiD Liver Failure Trajectory Computation
# Computes bilirubin trajectory probabilities for liver failure prediction

echo "=========================================="
echo "HiRiD Liver Failure Trajectory Computation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64G"
echo "Start time: $(date)"
echo ""

# Navigate to script directory
cd /home/gaga/tamarw1/trajectory-modeling/scripts || exit 1

# Run trajectory computation
python hirid_liver_trajs.py \
    --input /home/gaga/data/physionet/hirid/liver/bilirubin_timeseries.csv \
    --output /home/gaga/data/physionet/hirid/liver/liver_trajectory_probs_bayes.csv \
    --window-days 5.0 \
    --flat-thr 10.0 \
    --increase-thr 20.0 \
    --n-batches 10 \
    --sampler pymc

echo ""
echo "End time: $(date)"
echo "=========================================="
