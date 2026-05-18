#!/usr/bin/env python3
"""
Prepare ligand PDBQT files for AutoDock Vina docking.

Converts all .sdf, .mol2, and .pdb files in the specified ligand directory
to PDBQT format using mk_prepare_ligand.py, and writes a manifest CSV
mapping each input file to its output PDBQT.

Usage:
    python3 05ab_prepare_ligands_pdbqt_q2022p.py --ligand-dir /path/to/ligands
"""

import argparse
import subprocess
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--ligand-dir", required=True, help="Directory containing ligand input files")
args = parser.parse_args()

BASE = Path.home() / "Molecular_Dynamics_analysis"
DOCK = BASE / "dock" / "ROS1"

LIG_DIR = Path(args.ligand_dir)
OUT_DIR = DOCK / "ligands_pdbqt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

# Collect all supported ligand input formats
files = []
for ext in ("*.sdf", "*.mol2", "*.pdb"):
    files.extend(sorted(LIG_DIR.glob(ext)))

for lig in files:
    out = OUT_DIR / f"{lig.stem}.pdbqt"

    cmd = [
        "mk_prepare_ligand.py",
        "-i", str(lig),
        "-o", str(out),
    ]
    subprocess.run(cmd, check=True)

    rows.append({
        "ligand_input": str(lig),
        "ligand_pdbqt": str(out),
    })

pd.DataFrame(rows).to_csv(DOCK / "ligand_pdbqt_manifest.csv", index=False)
print("Saved ligand manifest.")