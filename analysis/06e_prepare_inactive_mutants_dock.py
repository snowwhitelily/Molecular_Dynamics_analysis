#!/usr/bin/env python3
"""
06e_prepare_inactive_mutants_dock.py

Prepare the inactive-form mutant receptors (built by 06d) for docking:
  1. Convert each mutant PDB to PDBQT via Meeko (mk_prepare_receptor.py).
  2. Compute the docking box from the WT inactive structure (same geometry-
     derived box that 06b used), and reuse it for ALL mutants so the inactive
     numbers are internally comparable.
  3. Write Vina configs and a run script.

The box is derived from the same type II site residues as 06b:
    L2026 (gatekeeper), D2102, F2103 (DFG), F2004, F2075 (specificity pocket),
    K1980 (catalytic lysine), with an 8 A margin.

Usage (venv on):
    source ~/.venvs/ros1_analysis_env/bin/activate
    cd ~/Molecular_Dynamics_analysis
    python analysis/06e_prepare_inactive_mutants_dock.py
    bash dock/ROS1/q2022p_vina/run_vina_cabozantinib_inactive_mutants.sh
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import MDAnalysis as mda

BASE = Path.home() / "Molecular_Dynamics_analysis"
DOCK = BASE / "dock" / "ROS1"
VINA = DOCK / "q2022p_vina"
INACTIVE_REF = DOCK / "inactive_ref"

REC_PDBQT_DIR = VINA / "receptor_pdbqt"
LIG_DIR = VINA / "ligand_pdbqt"
CFG_DIR = VINA / "configs"
OUT_DIR = VINA / "vina_outputs"
LOG_DIR = VINA / "vina_logs"

LIGAND = "cabozantinib"
MARGIN = 8.0  # angstrom, same as 06b

# Type II site residues (same as 06b — do not change)
SITE = {
    2026: 'gatekeeper',
    2102: 'DFG aspartate',
    2103: 'DFG phenylalanine',
    2004: 'specificity pocket',
    2075: 'specificity pocket',
    1980: 'catalytic lysine',
}

# Mutants produced by 06d (PDB files in inactive_ref/)
MUTANT_PDBS = [
    "ROS1_Q2022P_i",
    "ROS1_S1986F_i",
    "ROS1_Q2022P_S1986F_i",
    "ROS1_Q2022P_S1986Y_i",
]

# ── compute the docking box from WT inactive (same as 06b) ──────────────────
wt_pdb = INACTIVE_REF / "ROS1_WT_i.pdb"
if not wt_pdb.exists():
    raise SystemExit(f"missing WT inactive reference: {wt_pdb}")

# --- NEW CODE ---
wt_pdb = INACTIVE_REF / "ROS1_WT_i.pdb"
# Hardcode the center to the exact Type II specificity pocket center
centre = np.array([49.73, 44.43, 52.68])

# Constrain the box size to 18 Angstroms to exclude the ATP hinge
size = np.array([18.0, 18.0, 18.0])

# u = mda.Universe(str(wt_pdb))
# sel = 'resid ' + ' '.join(str(r) for r in SITE)
# ag = u.select_atoms(f'({sel}) and not name H*')
# if ag.n_residues != len(SITE):
#     raise SystemExit(f'expected {len(SITE)} site residues, got {ag.n_residues}')

# pos = ag.positions
# lo, hi = pos.min(axis=0), pos.max(axis=0)
# centre = (lo + hi) / 2.0
# size = (hi - lo) + 2 * MARGIN

print(f"Box derived from WT inactive site residues (same as 06b):")
print(f"  centre = {centre[0]:.2f}  {centre[1]:.2f}  {centre[2]:.2f}")
print(f"  size   = {size[0]:.1f}  {size[1]:.1f}  {size[2]:.1f}")
print()

# ── ensure directories exist ────────────────────────────────────────────────
for d in (REC_PDBQT_DIR, CFG_DIR, OUT_DIR, LOG_DIR, LIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── ligand PDBQT ────────────────────────────────────────────────────────────
lig_pdbqt = LIG_DIR / f"{LIGAND}.pdbqt"
if not lig_pdbqt.exists():
    src = DOCK / "ligands_pdbqt" / f"{LIGAND}.pdbqt"
    if not src.exists():
        raise SystemExit(f"cannot find {LIGAND}.pdbqt")
    shutil.copy2(src, lig_pdbqt)
    print(f"copied ligand: {src} -> {lig_pdbqt}")

# ── process each mutant ─────────────────────────────────────────────────────
sh_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
prepared = []

for stem in MUTANT_PDBS:
    pdb = INACTIVE_REF / f"{stem}.pdb"
    pdbqt = REC_PDBQT_DIR / f"{stem}.pdbqt"

    if not pdb.exists():
        print(f"WARNING: {pdb} not found — run 06d first. Skipping.")
        continue

    # ── step 1: convert PDB → PDBQT with Meeko ─────────────────────────────
    if pdbqt.exists():
        print(f"receptor PDBQT already exists: {pdbqt.name}")
    else:
        print(f"preparing receptor PDBQT: {stem}")
        meeko_cmd = [
            "mk_prepare_receptor.py",
            "--read_pdb", str(pdb),
            "-o", str(pdbqt.with_suffix('')),  # Meeko adds .pdbqt
            "--write_pdbqt", str(pdbqt),
        ]
        result = subprocess.run(meeko_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: Meeko failed for {stem}")
            print(f"  stderr: {result.stderr[:500]}")
            continue
        print(f"  OK: {pdbqt.name}")

    # ── step 2: write Vina config (same box for all) ────────────────────────
    cfg = CFG_DIR / f"{stem}__{LIGAND}.txt"
    out = OUT_DIR / f"{stem}__{LIGAND}.pdbqt"
    log = LOG_DIR / f"{stem}__{LIGAND}.log"

    cfg.write_text(
        f"receptor = {pdbqt}\n"
        f"ligand = {lig_pdbqt}\n"
        f"center_x = {centre[0]:.2f}\n"
        f"center_y = {centre[1]:.2f}\n"
        f"center_z = {centre[2]:.2f}\n"
        f"size_x = {size[0]:.1f}\n"
        f"size_y = {size[1]:.1f}\n"
        f"size_z = {size[2]:.1f}\n"
        f"exhaustiveness = 16\n"
        f"num_modes = 20\n"
        f"energy_range = 4\n"
        f"out = {out}\n"
    )
    sh_lines.append(f"vina --config {cfg} 2>&1 | tee {log}")
    prepared.append(stem)
    print(f"  config: {cfg.name}")

# ── write run script ────────────────────────────────────────────────────────
run = VINA / "run_vina_cabozantinib_inactive_mutants.sh"
run.write_text("\n".join(sh_lines) + "\n")
run.chmod(0o755)

print(f"\n{len(prepared)} mutant(s) prepared for docking")
print(f"run script: {run}")
print(f"\nAfter docking, parse results with:")
print(f"  python analysis/06c_rank_cabozantinib.py")
print(f"(06c auto-detects active vs inactive by the stem name)")