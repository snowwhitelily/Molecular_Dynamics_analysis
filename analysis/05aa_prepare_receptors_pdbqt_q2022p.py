#!/usr/bin/env python3

import subprocess
from pathlib import Path
import pandas as pd

BASE = Path.home() / "Molecular_Dynamics_analysis"
DOCK = BASE / "dock" / "ROS1"
# Receptors written into subdirectories by 05a
REC_BASE = DOCK / "q2022p_subset_receptors"
OUT_DIR = DOCK / "receptors_pdbqt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

# Recursively find all PDB files written by 05a
pdb_files = sorted(REC_BASE.rglob("*.pdb"))

for pdb in pdb_files:
    out = OUT_DIR / f"{pdb.stem}.pdbqt"

    cmd = [
        "mk_prepare_receptor.py",
        "--read_pdb", str(pdb),   # reads PDB without ProDy
        "-o", str(out.stem),      # output basename
        "--write_pdbqt", str(out) # write PDBQT explicitly
    ]

    subprocess.run(cmd, check=True)

    rows.append({
        "receptor_pdb": str(pdb),
        "receptor_pdbqt": str(out)
    })

df = pd.DataFrame(rows)
df.to_csv(DOCK / "receptor_pdbqt_manifest.csv", index=False)

print("Saved receptor manifest.")