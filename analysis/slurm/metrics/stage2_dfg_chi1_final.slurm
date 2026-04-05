#!/bin/bash
#SBATCH --job-name=ros1_stage2_dfg
#SBATCH --output=logs/ros1_stage2_dfg_%j.out
#SBATCH --error=logs/ros1_stage2_dfg_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda activate ros1_analysis_env
cd ~/Molecular_Dynamics_analysis
python analysis/scripts/metrics/run_stage2_dfg_chi1_final.py
