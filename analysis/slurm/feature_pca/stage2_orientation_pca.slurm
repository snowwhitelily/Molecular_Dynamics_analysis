#!/bin/bash
#SBATCH --job-name=ros1_stage2_orient
#SBATCH --output=logs/ros1_stage2_orient_%j.out
#SBATCH --error=logs/ros1_stage2_orient_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate ros1_analysis_env
cd ~/Molecular_Dynamics_analysis
python analysis/scripts/feature_pca/run_stage2_orientation_pca.py
