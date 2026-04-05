#!/bin/bash
#SBATCH --job-name=ros1_lda_qfamily
#SBATCH --output=logs/ros1_lda_qfamily_%j.out
#SBATCH --error=logs/ros1_lda_qfamily_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda activate ros1_analysis_env
cd ~/Molecular_Dynamics_analysis
python analysis/scripts/metrics/run_stage2_lda_qfamily.py
