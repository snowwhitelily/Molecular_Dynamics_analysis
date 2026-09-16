#!/usr/bin/env python3
"""
06i_lorlatinib_configs.py

Re-use the 10 S1986F frame receptors already extracted by 06i and write NEW
Vina configs for LORLATINIB (the ligand that FAILED / gave N/A on the single
S1986F medoid in the original 5-ligand run). Same shared box, same locked
params + seed. Tests whether lorlatinib's failure on S1986F was a bad-frame
artifact or holds across the conformational ensemble.

Receptors are reused (no re-extraction). Only the ligand and the config/output
folders change.

Usage (venv on):
    python analysis/06i_lorlatinib_configs.py
    nohup bash dock/ROS1/q2022p_vina/s1986f_multiframe/run_vina_s1986f_lorlatinib.sh \
        > /tmp/s1986f_lorl.log 2>&1 &
"""
from pathlib import Path

import numpy as np
import MDAnalysis as mda

BASE = Path.home() / "Molecular_Dynamics_analysis"
DOCK = BASE / "dock" / "ROS1"
VINA = DOCK / "q2022p_vina"
MF = VINA / "s1986f_multiframe"
REC_PQT = MF / "receptor_pdbqt"
REC_PDB = MF / "receptor_pdb"

# lorlatinib ligand pdbqt (same location cabozantinib used)
LIG = VINA / "ligand_pdbqt" / "lorlatinib.pdbqt"
LIG_FALLBACK = DOCK / "ligands_pdbqt" / "lorlatinib.pdbqt"

SITE_REAL = [2026, 2102, 2103, 2004, 2075, 1980]
OFFSET = 1933
BOX_SIZE = (28.7, 30.3, 30.2)
SEED = 42
EXH, NMODES, ERANGE = 16, 20, 4

CFG_DIR = MF / "configs_lorlatinib"
OUT_DIR = MF / "vina_outputs_lorlatinib"
LOG_DIR = MF / "vina_logs_lorlatinib"
for d in (CFG_DIR, OUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def pocket_centre(pdb):
    u = mda.Universe(str(pdb))
    sel = "resid " + " ".join(str(r) for r in SITE_REAL) + " and not name H*"
    ag = u.select_atoms(sel)
    if ag.n_atoms == 0:
        sel = "resid " + " ".join(str(r - OFFSET) for r in SITE_REAL) + " and not name H*"
        ag = u.select_atoms(sel)
    pos = ag.positions
    return (pos.min(0) + pos.max(0)) / 2.0


def main():
    lig = LIG if LIG.exists() else LIG_FALLBACK
    if not lig.exists():
        raise SystemExit(f"lorlatinib pdbqt not found in {LIG} or {LIG_FALLBACK}")
    # ensure it is in the dir the configs reference
    if not LIG.exists():
        import shutil
        LIG.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lig, LIG)
        lig = LIG
        print(f"copied lorlatinib -> {LIG}")

    receptors = sorted(REC_PQT.glob("S1986F_frame*.pdbqt"))
    if not receptors:
        raise SystemExit(f"no frame receptors in {REC_PQT}; run 06i first")
    print(f"re-docking {len(receptors)} S1986F frames with LORLATINIB\n")

    run_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for pqt in receptors:
        tag = pqt.stem
        pdb = REC_PDB / f"{tag}.pdb"
        cx, cy, cz = pocket_centre(pdb)
        cfg = CFG_DIR / f"{tag}__lorlatinib.txt"
        out = OUT_DIR / f"{tag}__lorlatinib.pdbqt"
        log = LOG_DIR / f"{tag}__lorlatinib.log"
        cfg.write_text(
            f"receptor = {pqt}\nligand = {lig}\n"
            f"center_x = {cx:.2f}\ncenter_y = {cy:.2f}\ncenter_z = {cz:.2f}\n"
            f"size_x = {BOX_SIZE[0]}\nsize_y = {BOX_SIZE[1]}\nsize_z = {BOX_SIZE[2]}\n"
            f"out = {out}\nseed = {SEED}\nexhaustiveness = {EXH}\n"
            f"num_modes = {NMODES}\nenergy_range = {ERANGE}\n")
        run_lines.append(f"vina --config {cfg} 2>&1 | tee {log}")
        print(f"  {tag:40s} centre ({cx:.1f} {cy:.1f} {cz:.1f})")

    run_sh = MF / "run_vina_s1986f_lorlatinib.sh"
    run_sh.write_text("\n".join(run_lines) + "\n")
    print(f"\nwrote {len(receptors)} lorlatinib configs")
    print(f"run script: {run_sh}")


if __name__ == "__main__":
    main()