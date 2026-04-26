#!/bin/bash
#SBATCH --job-name=ros1_step2f_cluster_transitions
#SBATCH --output=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step2f_cluster_transitions_%j.out
#SBATCH --error=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step2f_cluster_transitions_%j.err
#SBATCH --time=12:00:00
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
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python -c "import sys; print(sys.executable)"
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python -c "import pandas; print('pandas ok')"
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python -c "import numpy; print('numpy ok')"

cd /homes/lkgyammerah/Molecular_Dynamics_analysis/analysis

echo "==== RUNNING STEP 2F ===="
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python 02f_cluster_transitions.py --prefixes actout_ctlfit,aloop --cluster-col cluster_hdb --drop-noise

echo "==== JOB END ===="
date
