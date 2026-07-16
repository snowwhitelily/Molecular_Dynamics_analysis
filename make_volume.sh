#!/usr/bin/env bash
set -euo pipefail
DRUG="${1:?need DRUG: lorlatinib or zidesamtinib}"
SELECTION="${2:-distance}"
BASE="$HOME/Molecular_Dynamics_analysis"
VENV="$HOME/.venvs/ros1_analysis_env/bin/activate"
cd "$BASE"
source "$VENV"
python3 "$BASE/analysis/pocket_volume.py" "$SELECTION" "$DRUG"
deactivate || true
