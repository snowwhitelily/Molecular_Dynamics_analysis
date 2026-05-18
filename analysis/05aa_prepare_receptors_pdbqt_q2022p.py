#!/usr/bin/env python3
# ROS1 Analysis Pipeline
# Script 05aa: Prepare Receptor PDBQTs — Q2022P Family
#
# Converts all receptor PDB files written by Script 05a to PDBQT format
# using mk_prepare_receptor.py from the Meeko package. Saves a manifest
# CSV mapping each PDB to its corresponding PDBQT file.

import subprocess
from pathlib import Path
import pandas as pd

BASE     = Path.home() / "Molecular_Dynamics_analysis"
DOCK     = BASE / "dock" / "ROS1"
REC_BASE = DOCK / "q2022p_subset_receptors"
OUT_DIR  = DOCK / "receptors_pdbqt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for pdb in sorted(REC_BASE.rglob("*.pdb")):
    out = OUT_DIR / f"{pdb.stem}.pdbqt"

    cmd = [
        "mk_prepare_receptor.py",
        "--read_pdb",    str(pdb),
        "-o",            str(out.stem),
        "--write_pdbqt", str(out),
    ]
    subprocess.run(cmd, check=True)

    rows.append({
        "receptor_pdb"  : str(pdb),
        "receptor_pdbqt": str(out),
    })

df = pd.DataFrame(rows)
df.to_csv(DOCK / "receptor_pdbqt_manifest.csv", index=False)
print(f"Prepared {len(rows)} receptor PDBQTs. Manifest saved.")