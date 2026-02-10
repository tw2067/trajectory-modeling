#!/bin/bash
#SBATCH --job-name=hirid_circulatory_trajs
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/hirid_circulatory_%j.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/hirid_circulatory_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-shamir

# HiRiD Circulatory Failure Trajectory Computation (multi-biomarker)

echo "=========================================="
echo "HiRiD Circulatory Failure Trajectory Computation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 128G"
echo "Start time: $(date)"
echo ""

# Navigate to script directory
cd /home/gaga/tamarw1/trajectory-modeling/scripts || exit 1

# Run trajectory computation for all three biomarkers
python traj_scripts/hirid/hirid_circulatory_failure_trajs.py \
    --input-dir ../results/hirid/circulatory_failure \
    --output-dir ../results/hirid/circulatory_failure \
    --window-hours 12.0 \
    --n-batches 10 \
    --sampler pymc

echo ""
echo "End time: $(date)"
echo "=========================================="
