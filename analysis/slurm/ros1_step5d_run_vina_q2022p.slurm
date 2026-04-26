#!/bin/bash
#SBATCH --job-name=ros1_step5d_vina_run
#SBATCH --output=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step5d_vina_run_%j.out
#SBATCH --error=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step5d_vina_run_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE

set -e

unset PYTHONPATH
unset PYTHONHOME

echo "==== JOB START ===="
date
hostname
pwd

echo "==== CHECKING VINA ===="
which vina || true
vina --help >/dev/null 2>&1 || { echo "vina not found in PATH"; exit 1; }

cd /homes/lkgyammerah/Molecular_Dynamics_analysis

echo "==== RUNNING VINA JOBS ===="
bash dock/ROS1/q2022p_vina/run_vina_q2022p.sh

echo "==== JOB END ===="
date
