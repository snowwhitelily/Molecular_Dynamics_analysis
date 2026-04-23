#!/usr/bin/env python3

import subprocess
from pathlib import Path
import pandas as pd

BASE = Path.home() / "Molecular_Dynamics_analysis"
DOCK = BASE / "dock" / "ROS1"
REC_DIR = DOCK / "receptors_q2022p"
OUT_DIR = DOCK / "receptors_pdbqt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

pdb_files = sorted(REC_DIR.glob("*.pdb"))

for pdb in pdb_files:
    out = OUT_DIR / f"{pdb.stem}.pdbqt"

    cmd = [
        "mk_prepare_receptor.py",
        "-i", str(pdb),
        "-o", str(out)
    ]

    subprocess.run(cmd, check=True)

    rows.append({
        "receptor_pdb": str(pdb),
        "receptor_pdbqt": str(out)
    })

df = pd.DataFrame(rows)
df.to_csv(DOCK / "receptor_pdbqt_manifest.csv", index=False)

print("Saved receptor manifest.")