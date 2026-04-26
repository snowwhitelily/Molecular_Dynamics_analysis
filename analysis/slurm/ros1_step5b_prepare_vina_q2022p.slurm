#!/bin/bash
#SBATCH --job-name=ros1_step5b_vina_prep
#SBATCH --output=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step5b_vina_prep_%j.out
#SBATCH --error=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step5b_vina_prep_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
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

cd /homes/lkgyammerah/Molecular_Dynamics_analysis/analysis

echo "==== RUNNING STEP 5B ===="
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python 05b_prepare_vina_q2022p.py \
  --ligands lorlatinib,crizotinib \
  --center-x 0 --center-y 0 --center-z 0 \
  --size-x 20 --size-y 20 --size-z 20

echo "==== JOB END ===="
date
