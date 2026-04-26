#!/bin/bash
#SBATCH --job-name=ros1_step2c
#SBATCH --output=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step2c_%j.out
#SBATCH --error=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step2c_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE

set -e

unset PYTHONPATH
unset PYTHONHOME

echo "==== JOB START ===="
date
hostname
pwd

echo "==== USING PYTHON ===="
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python -V
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python -c "import sys, numpy, pandas, matplotlib; print(sys.executable); print('basic imports ok')"

cd /homes/lkgyammerah/Molecular_Dynamics_analysis/analysis

echo "==== RUNNING STEP 2C ===="
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python 02c_pca_aloop_ctlfit.py

echo "==== JOB END ===="
date
