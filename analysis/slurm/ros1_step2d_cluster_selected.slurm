#!/bin/bash
#SBATCH --job-name=ros1_step2d_cluster_selected
#SBATCH --output=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step2d_cluster_selected_%j.out
#SBATCH --error=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step2d_cluster_selected_%j.err
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
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python -c "import sklearn; print('sklearn ok')"

cd /homes/lkgyammerah/Molecular_Dynamics_analysis/analysis

echo "==== RUNNING STEP 2D: SELECTED PCA CLUSTERING ===="
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python 02d_cluster_selected_pcas.py --prefixes actout_ctlfit,aloop

echo "==== JOB END ===="
date
