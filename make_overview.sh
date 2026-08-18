#!/usr/bin/env bash
set -euo pipefail
DRUG="${1:?need DRUG: lorlatinib or zidesamtinib}"
SELECTION="${2:-distance}"
BASE="$HOME/Molecular_Dynamics_analysis"; ANALYSIS="$BASE/analysis/scripts"
FIGS="$BASE/figures/ROS1/ros1_prepared_final/pocket_box"
VENV="$HOME/.venvs/ros1_analysis_env/bin/activate"; cd "$BASE"
if [[ ! -f "$FIGS/prep_${SELECTION}_${DRUG}_drug.pdb" ]]; then
  echo "Run ./make_pocket_figure.sh ${SELECTION} ${DRUG} first."; exit 1; fi
echo "=== overview render ($SELECTION, $DRUG) ==="
env -u PYTHONPATH -u VIRTUAL_ENV PATH=/usr/bin:/bin /usr/bin/pymol -cq "$ANALYSIS/overview_render.py" -- "$DRUG" "$SELECTION"
echo "=== overview compose ==="
source "$VENV"; python3 "$ANALYSIS/overview_compose.py" "$DRUG" "$SELECTION"; deactivate || true
echo "Done: $FIGS/overview_${SELECTION}_${DRUG}_final.png"
