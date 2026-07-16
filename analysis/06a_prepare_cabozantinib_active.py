#!/usr/bin/env python3
"""
06a_prepare_cabozantinib_active.py

Dock cabozantinib into the five existing active-form receptors, reusing the
Task 5 box so the numbers are directly comparable with the other five ligands
already in the table.

Cabozantinib is a type II inhibitor and binds the DFG-out inactive
conformation, so a poor affinity here is the expected result, not a failure.
It is the control that gives the inactive-form number its meaning.

Writes configs in the same format as the existing Task 5 configs. Note that the
"log" keyword is deliberately absent: Vina 1.2.x removed it, and the run script
captures output with tee instead, which is what the existing pipeline does.

Usage (venv on):
    python 06a_prepare_cabozantinib_active.py
    bash dock/ROS1/q2022p_vina/run_vina_cabozantinib_active.sh
"""

import shutil
from pathlib import Path

BASE = Path.home() / "Molecular_Dynamics_analysis"
DOCK = BASE / "dock" / "ROS1"
VINA = DOCK / "q2022p_vina"

REC_DIR = VINA / "receptor_pdbqt"
LIG_DIR = VINA / "ligand_pdbqt"
CFG_DIR = VINA / "configs"
OUT_DIR = VINA / "vina_outputs"
LOG_DIR = VINA / "vina_logs"

LIGAND = "cabozantinib"

# Task 5 box, reused unchanged so the active-form values stay comparable
BOX = dict(center_x=68.6, center_y=64.5, center_z=23.68,
           size_x=25.0, size_y=20.0, size_z=20.0)
EXHAUSTIVENESS = 16
NUM_MODES = 20
ENERGY_RANGE = 4

for d in (CFG_DIR, OUT_DIR, LOG_DIR, LIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# the ligand PDBQTs were prepped into dock/ROS1/ligands_pdbqt by 05ab, but the
# configs point at q2022p_vina/ligand_pdbqt, so make sure it is present there
lig_pdbqt = LIG_DIR / f"{LIGAND}.pdbqt"
if not lig_pdbqt.exists():
    src = DOCK / "ligands_pdbqt" / f"{LIGAND}.pdbqt"
    if not src.exists():
        raise SystemExit(f"cannot find {LIGAND}.pdbqt in {src.parent}")
    shutil.copy2(src, lig_pdbqt)
    print(f"copied {src} -> {lig_pdbqt}")

receptors = sorted(REC_DIR.glob("*_actout_ctlfit_dominant_cluster0_rep.pdbqt"))
if not receptors:
    raise SystemExit(f"no receptor PDBQTs found in {REC_DIR}")

sh = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
for rec in receptors:
    stem = rec.stem
    cfg = CFG_DIR / f"{stem}__{LIGAND}.txt"
    out = OUT_DIR / f"{stem}__{LIGAND}.pdbqt"
    log = LOG_DIR / f"{stem}__{LIGAND}.log"

    cfg.write_text(
        f"receptor = {rec}\n"
        f"ligand = {lig_pdbqt}\n"
        f"center_x = {BOX['center_x']}\n"
        f"center_y = {BOX['center_y']}\n"
        f"center_z = {BOX['center_z']}\n"
        f"size_x = {BOX['size_x']}\n"
        f"size_y = {BOX['size_y']}\n"
        f"size_z = {BOX['size_z']}\n"
        f"exhaustiveness = {EXHAUSTIVENESS}\n"
        f"num_modes = {NUM_MODES}\n"
        f"energy_range = {ENERGY_RANGE}\n"
        f"out = {out}\n"
    )
    sh.append(f"vina --config {cfg} 2>&1 | tee {log}")
    print(f"config: {cfg.name}")

run = VINA / "run_vina_cabozantinib_active.sh"
run.write_text("\n".join(sh) + "\n")
run.chmod(0o755)
print(f"\n{len(receptors)} jobs prepared")
print(f"run script: {run}")