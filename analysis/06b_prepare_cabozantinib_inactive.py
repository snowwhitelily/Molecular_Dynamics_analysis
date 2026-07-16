#!/usr/bin/env python3
"""
06b_prepare_cabozantinib_inactive.py

Dock cabozantinib into the published inactive WT receptor (ROS1_WT_i.pdb,
Vilacha et al. 2025, Zenodo 10.5281/zenodo.15236275), verified DFG-out here by
the K1980 vertex angle (67.7 deg vs the paper's ~72 deg) and by the F2103 chi1
being trans (172.9 deg) rather than gauche.

The Task 5 box CANNOT be reused. Two independent reasons:
  1. It is expressed in the coordinate frame of our own MD receptors. The
     Zenodo structure comes from a different simulation box entirely, so those
     numbers point at empty space here.
  2. It was built around the active-form ATP site. Cabozantinib is type II: its
     fluorophenyl ring occupies the specificity pocket demarcated by F2004 and
     F2075, beyond the ATP site. An ATP-only box would truncate the site.

So the box is derived from this structure's own geometry: it is built to
enclose the ATP site (gatekeeper L2026, DFG D2102/F2103) together with the
type II crevice (F2004, F2075), plus a margin for the ligand.

Usage (venv on):
    python 06b_prepare_cabozantinib_inactive.py
"""

import shutil
from pathlib import Path

import numpy as np
import MDAnalysis as mda

BASE = Path.home() / "Molecular_Dynamics_analysis"
DOCK = BASE / "dock" / "ROS1"
VINA = DOCK / "q2022p_vina"

REC_PDB = DOCK / "inactive_ref" / "ROS1_WT_i.pdb"
REC_PDBQT = VINA / "receptor_pdbqt" / "ROS1_WT_i.pdbqt"
LIG_DIR = VINA / "ligand_pdbqt"
CFG_DIR = VINA / "configs"
OUT_DIR = VINA / "vina_outputs"
LOG_DIR = VINA / "vina_logs"

LIGAND = "cabozantinib"
MARGIN = 8.0          # angstrom of headroom around the pocket-lining residues

# residues that together define the type II site in the inactive conformation
SITE = {
    2026: 'gatekeeper, lines the ATP site',
    2102: 'DFG aspartate',
    2103: 'DFG phenylalanine, fills the ATP pocket when DFG-out',
    2004: 'demarcates the type II specificity pocket',
    2075: 'demarcates the type II specificity pocket',
    1980: 'catalytic lysine',
}

if not REC_PDB.exists():
    raise SystemExit(f'missing {REC_PDB}')

u = mda.Universe(str(REC_PDB))
sel = 'resid ' + ' '.join(str(r) for r in SITE)
ag = u.select_atoms(f'({sel}) and not name H*')
if ag.n_residues != len(SITE):
    raise SystemExit(f'expected {len(SITE)} residues, selected {ag.n_residues}')

print('site residues found:')
for res in ag.residues:
    print(f'  {res.resname}{res.resid}: {SITE[res.resid]}')

pos = ag.positions
lo, hi = pos.min(axis=0), pos.max(axis=0)
centre = (lo + hi) / 2.0
size = (hi - lo) + 2 * MARGIN

print(f'\npocket-lining atoms span {np.round(hi - lo, 1)} A')
print(f'box centre  = {centre[0]:.2f} {centre[1]:.2f} {centre[2]:.2f}')
print(f'box size    = {size[0]:.1f} {size[1]:.1f} {size[2]:.1f} '
      f'(span + 2 x {MARGIN:.0f} A margin)')

if not REC_PDBQT.exists():
    print(f'\nreceptor PDBQT not found. Prepare it first with Meeko:')
    print(f'  mk_prepare_receptor.py --read_pdb {REC_PDB} \\')
    print(f'      -o {REC_PDBQT.stem} --write_pdbqt {REC_PDBQT}')
    raise SystemExit('stopping until the receptor PDBQT exists')

lig_pdbqt = LIG_DIR / f'{LIGAND}.pdbqt'
if not lig_pdbqt.exists():
    src = DOCK / 'ligands_pdbqt' / f'{LIGAND}.pdbqt'
    if not src.exists():
        raise SystemExit(f'cannot find {LIGAND}.pdbqt')
    shutil.copy2(src, lig_pdbqt)

for d in (CFG_DIR, OUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

stem = 'ROS1_WT_i'
cfg = CFG_DIR / f'{stem}__{LIGAND}.txt'
out = OUT_DIR / f'{stem}__{LIGAND}.pdbqt'
log = LOG_DIR / f'{stem}__{LIGAND}.log'

cfg.write_text(
    f'receptor = {REC_PDBQT}\n'
    f'ligand = {lig_pdbqt}\n'
    f'center_x = {centre[0]:.2f}\n'
    f'center_y = {centre[1]:.2f}\n'
    f'center_z = {centre[2]:.2f}\n'
    f'size_x = {size[0]:.1f}\n'
    f'size_y = {size[1]:.1f}\n'
    f'size_z = {size[2]:.1f}\n'
    f'exhaustiveness = 16\n'
    f'num_modes = 20\n'
    f'energy_range = 4\n'
    f'out = {out}\n'
)

run = VINA / 'run_vina_cabozantinib_inactive.sh'
run.write_text('#!/usr/bin/env bash\nset -euo pipefail\n\n'
               f'vina --config {cfg} 2>&1 | tee {log}\n')
run.chmod(0o755)
print(f'\nconfig:     {cfg}')
print(f'run script: {run}')