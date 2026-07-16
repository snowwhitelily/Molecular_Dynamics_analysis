#!/usr/bin/env bash
# run_inactive_pipeline.sh
#
# Complete pipeline for inactive-form cabozantinib docking of mutant receptors.
# Run from ~/Molecular_Dynamics_analysis
#
# Steps:
#   1. Mutate WT inactive scaffold (PyMOL, system Python)
#   2. Prep PDBQTs + write Vina configs (venv, Meeko + MDAnalysis)
#   3. Run Vina docking (venv or bare shell)
#   4. Parse results (venv, pandas)
#   5. Analyze poses (venv, MDAnalysis)
#
# Usage:
#   cd ~/Molecular_Dynamics_analysis
#   bash analysis/run_inactive_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="$(dirname "$SCRIPT_DIR")"
cd "$BASE"

echo "============================================================"
echo " STEP 1: Build inactive mutant receptors (PyMOL)"
echo "============================================================"
echo ""
env -u PYTHONPATH -u VIRTUAL_ENV PATH=/usr/bin:/bin \
    /usr/bin/pymol -cq analysis/06d_mutate_inactive_receptors.py

echo ""
echo "============================================================"
echo " STEP 2: Prepare PDBQTs + Vina configs (venv)"
echo "============================================================"
echo ""
# Activate venv for Meeko + MDAnalysis
source ~/.venvs/ros1_analysis_env/bin/activate
python analysis/06e_prepare_inactive_mutants_dock.py

echo ""
echo "============================================================"
echo " STEP 3: Run Vina docking"
echo "============================================================"
echo ""
bash dock/ROS1/q2022p_vina/run_vina_cabozantinib_inactive_mutants.sh

echo ""
echo "============================================================"
echo " STEP 4: Parse all cabozantinib results"
echo "============================================================"
echo ""
python analysis/06c_rank_cabozantinib.py

echo ""
echo "============================================================"
echo " STEP 5: Analyze pose locations (type II vs ATP site)"
echo "============================================================"
echo ""
python analysis/06f_analyze_cabozantinib_poses.py

echo ""
echo "============================================================"
echo " DONE"
echo "============================================================"
echo ""
echo "Results in: results/ROS1/step6c_cabozantinib_best_by_mutant.csv"
echo ""
echo "Next: run the plotting script to generate figures."