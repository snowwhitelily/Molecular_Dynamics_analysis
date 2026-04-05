#!/bin/bash
#SBATCH --job-name=ros1_stage2_further
#SBATCH --output=logs/ros1_stage2_further_%j.out
#SBATCH --error=logs/ros1_stage2_further_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda activate ros1_analysis_env
cd ~/Molecular_Dynamics_analysis
python analysis/scripts/metrics/run_stage2_further_analysis.py
