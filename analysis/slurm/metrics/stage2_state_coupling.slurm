#!/bin/bash
#SBATCH --job-name=ros1_state_coupling
#SBATCH --output=logs/ros1_state_coupling_%j.out
#SBATCH --error=logs/ros1_state_coupling_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda activate ros1_analysis_env
cd ~/Molecular_Dynamics_analysis
python analysis/scripts/metrics/run_stage2_state_coupling.py
