#!/bin/bash
#SBATCH --job-name=ros1_step4c
#SBATCH --output=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step4c_%j.out
#SBATCH --error=/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1/step4c_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE

set -e

unset PYTHONPATH
unset PYTHONHOME

REF_GROUP="WT"
TARGETS="Q2022P,Q2022P_S1986F,Q2022P_S1986Y"
SCORES_PREFIX="actout_ctlfit"

echo "==== JOB START ===="
date
hostname
pwd

echo "==== USING PYTHON ===="
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python -V
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python -c "import sys, numpy, pandas, matplotlib; print(sys.executable); print('basic imports ok')"

cd /homes/lkgyammerah/Molecular_Dynamics_analysis/analysis

echo "==== RUNNING STEP 4C ===="
/homes/lkgyammerah/.venvs/ros1_analysis_env/bin/python 04c_generalized_pairwise_lda.py \
  --ref "${REF_GROUP}" \
  --targets "${TARGETS}" \
  --scores-prefix "${SCORES_PREFIX}" \
  --k-use 6

echo "==== JOB END ===="
date
