#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--ligand-dir", required=True)
args = parser.parse_args()

BASE = Path.home() / "Molecular_Dynamics_analysis"
DOCK = BASE / "dock" / "ROS1"

LIG_DIR = Path(args.ligand_dir)
OUT_DIR = DOCK / "ligands_pdbqt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

files = []
for ext in ("*.sdf", "*.mol2", "*.pdb"):
    files.extend(sorted(LIG_DIR.glob(ext)))

for lig in files:
    out = OUT_DIR / f"{lig.stem}.pdbqt"

    cmd = [
        "mk_prepare_ligand.py",
        "-i", str(lig),
        "-o", str(out)
    ]

    subprocess.run(cmd, check=True)

    rows.append({
        "ligand_input": str(lig),
        "ligand_pdbqt": str(out)
    })

pd.DataFrame(rows).to_csv(DOCK / "ligand_pdbqt_manifest.csv", index=False)

print("Saved ligand manifest.")