#!/bin/bash
#SBATCH --job-name=smoke-test
#SBATCH --output=/home/gaga/tamarw1/trajectory-modeling/logs/outs/smoke_%j.out
#SBATCH --error=/home/gaga/tamarw1/trajectory-modeling/logs/errs/smoke_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-shamir

pytest -q /home/gaga/tamarw1/trajectory-modeling/tests/test_smoke_backends.py -s