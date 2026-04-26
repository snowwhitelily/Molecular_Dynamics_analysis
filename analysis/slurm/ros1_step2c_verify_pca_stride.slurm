#!/bin/bash
#SBATCH --job-name=ros1_step2c_verify
#SBATCH --output=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step2c_verify_%j.out
#SBATCH --error=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step2c_verify_%j.err
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
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python -c "import sys, numpy, pandas, matplotlib, MDAnalysis; print(sys.executable); print('imports ok')"

cd /homes/lkgyammerah/Molecular_Dynamics_analysis/analysis
echo "==== RUNNING STEP 2C VERIFY PCA STRIDE ===="
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python 02c_verify_pca_stride.py --prefixes aloop,actout_ctlfit --traj-index 0

echo "==== JOB END ===="
date
