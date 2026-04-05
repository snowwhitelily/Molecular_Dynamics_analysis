#!/bin/bash
#SBATCH --job-name=ros1_stage2_aloop
#SBATCH --output=logs/ros1_stage2_aloop_%j.out
#SBATCH --error=logs/ros1_stage2_aloop_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda activate ros1_analysis_env
cd ~/Molecular_Dynamics_analysis
python analysis/scripts/metrics/run_stage2_aloop_state.py
