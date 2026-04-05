#!/bin/bash
#SBATCH --job-name=ros1_stage3_coord
#SBATCH --output=logs/ros1_stage3_coord_%j.out
#SBATCH --error=logs/ros1_stage3_coord_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda activate ros1_analysis_env
cd ~/Molecular_Dynamics_analysis

python analysis/scripts/coordinate_pca/run_stage3_coord_clustering_branch.py "$BRANCH"
