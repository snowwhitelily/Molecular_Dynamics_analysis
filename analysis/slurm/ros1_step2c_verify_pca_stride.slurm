#!/bin/bash
#SBATCH --job-name=ros1_step2c_verify
#SBATCH --output=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step2c_verify_%j.out
#SBATCH --error=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step2c_verify_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL

set -e

unset PYTHONPATH
unset PYTHONHOME

PYTHON=/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python
BASE=/homes/lkgyammerah/Molecular_Dynamics_analysis

# =====================================================
# Flexible settings
#
# WT replicas:
#   114 = WT / 0
#   115 = WT / 1
#   116 = WT / 2
#
# Default is WT replica 0.
# Override at submission time, for example:
#
# sbatch --export=TRAJ_INDEX=115,PREFIXES=aloop,actout_ctlfit \
#   ~/Molecular_Dynamics_analysis/analysis/slurm/ros1_step2c_verify_pca_stride.slurm
# =====================================================
TRAJ_INDEX=${TRAJ_INDEX:-114}
PREFIXES=${PREFIXES:-aloop,actout_ctlfit}

echo "==== JOB START ===="
date
hostname
pwd

echo "==== USING PYTHON ===="
${PYTHON} -V
${PYTHON} -c "import sys, numpy, pandas, matplotlib, MDAnalysis; print(sys.executable); print('imports ok')"

cd ${BASE}/analysis

echo "==== RUNNING STEP 2C VERIFY PCA STRIDE ===="
echo "TRAJ_INDEX=${TRAJ_INDEX}"
echo "PREFIXES=${PREFIXES}"

${PYTHON} 02c_verify_pca_stride.py \
  --prefixes ${PREFIXES} \
  --traj-index ${TRAJ_INDEX}

echo "==== JOB END ===="
date