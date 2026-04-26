#!/bin/bash
#SBATCH --job-name=ros1_step3c_fel_selected
#SBATCH --output=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step3c_fel_selected_%j.out
#SBATCH --error=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step3c_fel_selected_%j.err
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
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python -c "import matplotlib; print('matplotlib ok')"

cd /homes/lkgyammerah/Molecular_Dynamics_analysis/analysis

echo "==== RUNNING STEP 3C: FREE ENERGY LANDSCAPES ===="
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python 03c_free_energy_landscape.py --prefixes actout_ctlfit,aloop --per-mutant

echo "==== JOB END ===="
date
