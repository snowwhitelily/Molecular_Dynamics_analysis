#!/bin/bash
#SBATCH --job-name=ros1_stage2_coord
#SBATCH --output=logs/ros1_stage2_coord_%j.out
#SBATCH --error=logs/ros1_stage2_coord_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate ros1_analysis_env
cd ~/Molecular_Dynamics_analysis

python analysis/scripts/coordinate_pca/run_stage2_coord_pca_branch.py "$BRANCH"
