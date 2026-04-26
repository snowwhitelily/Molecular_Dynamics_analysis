#!/bin/bash
#SBATCH --job-name=ros1_step5a_dock_receptors
#SBATCH --output=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step5a_dock_receptors_%j.out
#SBATCH --error=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step5a_dock_receptors_%j.err
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
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python -c "import MDAnalysis; print('MDAnalysis ok')"

cd /homes/lkgyammerah/Molecular_Dynamics_analysis/analysis

echo "==== RUNNING STEP 5A ===="
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python 05a_select_extract_docking_receptors_q2022p.py --prefixes actout_ctlfit,aloop --cluster-col cluster_hdb

echo "==== JOB END ===="
date
