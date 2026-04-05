#!/bin/bash
#SBATCH --job-name=ros1_stage1_align
#SBATCH --output=logs/ros1_stage1_align_%j.out
#SBATCH --error=logs/ros1_stage1_align_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate ros1_analysis_env

cd ~/Molecular_Dynamics_analysis
python analysis/scripts/coordinate_pca/run_stage1_common_align.py
