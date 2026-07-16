#!/usr/bin/env bash
# Task 10 ATP pocket figure -- prep (venv) -> render (system pymol) -> compose (venv).
#
# Usage (start with the venv ON):
#   ./make_pocket_figure.sh SELECTION DRUG [SCALE] [CUTOFF]
#     SELECTION = motif | distance
#     DRUG      = lorlatinib | zidesamtinib
#     SCALE     = eigenvector length multiplier (default 1.0)
#     CUTOFF    = distance cutoff in Angstrom (default 5.0)
set -euo pipefail

SELECTION="${1:?need SELECTION: motif or distance}"
DRUG="${2:?need DRUG: lorlatinib or zidesamtinib}"
SCALE="${3:-1.0}"
CUTOFF="${4:-5.0}"
export POCKET_SCALE="$SCALE"

BASE="$HOME/Molecular_Dynamics_analysis"
ANALYSIS="$BASE/analysis"
VENV="$HOME/.venvs/ros1_analysis_env/bin/activate"
VARIANTS=(WT S1986F Q2022P Q2022P_S1986F)
cd "$BASE"

# 1. PREP (venv ON for MDAnalysis)
echo "=== prep ($SELECTION, $DRUG, cutoff=$CUTOFF) ==="
source "$VENV"
python3 "$ANALYSIS/pocket_prep.py" "$SELECTION" "$DRUG" "$CUTOFF"
deactivate || true

# 2. RENDER each variant (venv OFF for system pymol)
for V in "${VARIANTS[@]}"; do
  echo "=== rendering $V ($SELECTION, $DRUG, scale=$SCALE) ==="
  env -u PYTHONPATH -u VIRTUAL_ENV PATH=/usr/bin:/bin /usr/bin/pymol -cq "$ANALYSIS/pocket_render.py" -- "$V" "$SELECTION" "$DRUG"
done

# 3. COMPOSE 2x2 grid (venv ON for matplotlib)
echo "=== composing 2x2 grid ==="
source "$VENV"
python3 "$ANALYSIS/pocket_compose.py" "$SELECTION" "$DRUG"
echo "Done: figures/ROS1/ros1_prepared_final/pocket_box/pocket_${SELECTION}_${DRUG}.png"
