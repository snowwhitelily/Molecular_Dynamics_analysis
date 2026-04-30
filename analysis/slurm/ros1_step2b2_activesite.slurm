#!/bin/bash
#SBATCH --job-name=ros1_step2b2_activesite
#SBATCH --output=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step2b2_activesite_%j.out
#SBATCH --error=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step2b2_activesite_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
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
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python -c "import sys; print(sys.executable)"
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python -c "import pandas; print('pandas ok')"
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python -c "import numpy; print('numpy ok')"

cd /homes/lkgyammerah/Molecular_Dynamics_analysis/analysis

echo "==== RUNNING STEP 2B2: ACTIVE SITE PCA ===="
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python 02b2_pca_activesite.py

echo "==== JOB END ===="
date
