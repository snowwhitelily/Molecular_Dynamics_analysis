#!/bin/bash
#SBATCH --job-name=ros1_stage2_dist
#SBATCH --output=logs/ros1_stage2_dist_%j.out
#SBATCH --error=logs/ros1_stage2_dist_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate ros1_analysis_env
cd ~/Molecular_Dynamics_analysis
python analysis/scripts/feature_pca/run_stage2_distance_pca.py
