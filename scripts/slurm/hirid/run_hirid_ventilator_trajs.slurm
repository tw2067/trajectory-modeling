#!/bin/bash
#SBATCH --job-name=hirid_ventilator_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/hirid_ventilator_%j.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/hirid_ventilator_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-shamir
#SBATCH --account=gpu-research

# HiRiD Ventilator Discontinuation Trajectory Computation
# Computes P/F ratio trajectory probabilities for ventilator weaning prediction

echo "=========================================="
echo "HiRiD Ventilator Trajectory Computation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 128G"
echo "Start time: $(date)"
echo ""

# Navigate to script directory
cd /home/gaga/tamarw1/trajectory-modeling/scripts || exit 1

# Run trajectory computation
python hirid_ventilator_trajs.py \
    --input /home/gaga/data/physionet/hirid/ventilator/pf_ratio_timeseries.csv \
    --output /home/gaga/data/physionet/hirid/ventilator/ventilator_trajectory_probs_bayes.csv \
    --window-days 3.0 \
    --flat-thr 10.0 \
    --decline-thr 30.0 \
    --n-batches 10

echo ""
echo "End time: $(date)"
echo "=========================================="
