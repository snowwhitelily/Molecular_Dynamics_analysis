#!/bin/bash
#SBATCH --job-name=ros1_step2e_map_metrics_clusters
#SBATCH --output=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step2e_map_metrics_clusters_%j.out
#SBATCH --error=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step2e_map_metrics_clusters_%j.err
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

echo "==== RUNNING STEP 2E: MAP METRICS TO CLUSTERS ===="
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python 02e_map_metrics_to_clusters.py --prefixes actout_ctlfit,aloop

echo "==== JOB END ===="
date
