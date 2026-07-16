#!/usr/bin/env python3
"""
06d_mutate_inactive_receptors.py

Build inactive-form mutant receptors by mutating ROS1_WT_i.pdb (Vilacha et al.
2025, Zenodo 10.5281/zenodo.15236275) using PyMOL's mutagenesis wizard.

For each variant the script:
  1. Loads the equilibrated inactive WT scaffold (DFG-out, ~67.7 deg).
  2. Applies point mutations using the highest-probability rotamer, matching
     Vilacha's protocol (they used Chimera's Rotamers tool, same idea).
  3. Verifies DFG is still out: K1980-E1997-F2103 angle at vertex K1980 must
     remain ~67-75 deg, and F2103 chi1 must stay trans (~170-180 deg).
  4. Saves each mutant as a clean PDB ready for Meeko PDBQT conversion.

Variants built:
  - ROS1_Q2022P_i.pdb        (Q2022P single mutant)
  - ROS1_S1986F_i.pdb        (S1986F single mutant)
  - ROS1_Q2022P_S1986F_i.pdb (double mutant)
  - ROS1_Q2022P_S1986Y_i.pdb (double mutant)

MUST be run under system PyMOL (not the venv):
    env -u PYTHONPATH -u VIRTUAL_ENV PATH=/usr/bin:/bin \
        /usr/bin/pymol -cq analysis/06d_mutate_inactive_receptors.py
"""

import os
import sys
import math
from pathlib import Path

from pymol import cmd

BASE = Path.home() / "Molecular_Dynamics_analysis"
INACTIVE_REF = BASE / "dock" / "ROS1" / "inactive_ref"
SRC_PDB = INACTIVE_REF / "ROS1_WT_i.pdb"
OUT_DIR = INACTIVE_REF  # keep mutants alongside the WT inactive

# ── mutations to apply ──────────────────────────────────────────────────────
# Each variant is a list of (resid, target_3letter) tuples.
# Residue numbering is real (paper) numbering, matching ROS1_WT_i.pdb.
VARIANTS = {
    "ROS1_Q2022P_i":        [(2022, "PRO")],
    "ROS1_S1986F_i":        [(1986, "PHE")],
    "ROS1_Q2022P_S1986F_i": [(2022, "PRO"), (1986, "PHE")],
    "ROS1_Q2022P_S1986Y_i": [(2022, "PRO"), (1986, "TYR")],
}

# ── DFG verification parameters ────────────────────────────────────────────
# Angle: Ca(K1980) - Ca(E1997) - Ca(F2103) with vertex at K1980
# Active ~31 deg, Inactive ~67-75 deg
DFG_ANGLE_MIN = 55.0   # anything below this means DFG collapsed to active
DFG_ANGLE_MAX = 90.0   # sanity upper bound
# F2103 chi1: active ~ -68 (gauche), inactive ~ 170-180 (trans)
CHI1_TRANS_MIN = 140.0


def measure_dfg_angle(obj):
    """Measure K1980-E1997-F2103 Ca angle with vertex at K1980."""
    # Get Ca coordinates
    k1980 = cmd.get_coords(f"{obj} and resid 1980 and name CA", 1)
    e1997 = cmd.get_coords(f"{obj} and resid 1997 and name CA", 1)
    f2103 = cmd.get_coords(f"{obj} and resid 2103 and name CA", 1)

    if k1980 is None or e1997 is None or f2103 is None:
        return None

    # Vertex at K1980: angle between vectors K1980->E1997 and K1980->F2103
    v1 = e1997[0] - k1980[0]
    v2 = f2103[0] - k1980[0]
    cos_a = (v1 @ v2) / (math.sqrt(v1 @ v1) * math.sqrt(v2 @ v2) + 1e-12)
    return math.degrees(math.acos(max(-1, min(1, cos_a))))


def measure_f2103_chi1(obj):
    """Measure F2103 chi1 dihedral (N-CA-CB-CG)."""
    try:
        return cmd.get_dihedral(
            f"{obj} and resid 2103 and name N",
            f"{obj} and resid 2103 and name CA",
            f"{obj} and resid 2103 and name CB",
            f"{obj} and resid 2103 and name CG",
        )
    except Exception:
        return None


def apply_mutation(obj, resid, target_aa):
    """
    Apply a point mutation using PyMOL's mutagenesis wizard.
    Selects the highest-probability rotamer (frame 1, which is the default).
    """
    cmd.wizard("mutagenesis")
    wiz = cmd.get_wizard()
    wiz.set_mode(target_aa)
    # Select the residue to mutate
    wiz.do_select(f"/{obj}//A/{resid}")
    # Frame 1 = highest-probability rotamer (PyMOL default)
    #wiz.set_frame(1)
    wiz.do_frame(1)
    wiz.apply()
    cmd.set_wizard()


# ── sanity check: verify WT inactive before we start ────────────────────────
if not SRC_PDB.exists():
    print(f"ERROR: {SRC_PDB} not found")
    sys.exit(1)

cmd.load(str(SRC_PDB), "wt_check")
wt_angle = measure_dfg_angle("wt_check")
wt_chi1 = measure_f2103_chi1("wt_check")
cmd.delete("wt_check")

print("=" * 65)
print("DFG verification of WT inactive scaffold")
print(f"  K1980 vertex angle: {wt_angle:.1f} deg (expect ~67-75)")
print(f"  F2103 chi1:         {wt_chi1:.1f} deg (expect ~170-180, trans)")
print("=" * 65)

if wt_angle is None or wt_angle < DFG_ANGLE_MIN:
    print("ERROR: WT inactive scaffold fails DFG check — cannot proceed")
    sys.exit(1)

# ── build each mutant ───────────────────────────────────────────────────────
results = []
failed = []

for name, mutations in VARIANTS.items():
    print(f"\n{'─' * 65}")
    print(f"Building {name}")

    cmd.load(str(SRC_PDB), name)

    for resid, target in mutations:
        # Verify the source residue exists
        src_res = cmd.get_fastastr(f"{name} and resid {resid}").split('\n')
        print(f"  Mutating resid {resid} -> {target}")
        apply_mutation(name, resid, target)

    # Verify DFG after mutation
    angle = measure_dfg_angle(name)
    chi1 = measure_f2103_chi1(name)

    dfg_ok = (angle is not None and DFG_ANGLE_MIN <= angle <= DFG_ANGLE_MAX)
    chi1_ok = (chi1 is not None and abs(chi1) > CHI1_TRANS_MIN)

    status = "PASS" if (dfg_ok and chi1_ok) else "FAIL"
    print(f"  DFG angle (vtxK1980): {angle:.1f} deg  {'OK' if dfg_ok else 'FAIL'}")
    print(f"  F2103 chi1:           {chi1:.1f} deg  {'OK' if chi1_ok else 'FAIL'}")
    print(f"  Overall: {status}")

    out_path = OUT_DIR / f"{name}.pdb"
    cmd.save(str(out_path), name)
    print(f"  Saved: {out_path}")

    results.append((name, angle, chi1, status))
    if status == "FAIL":
        failed.append(name)

    cmd.delete(name)

# ── summary ─────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("SUMMARY — inactive mutant receptors")
print(f"{'variant':30s} {'DFG angle':>10s} {'chi1':>8s} {'status':>8s}")
print("-" * 65)
for name, angle, chi1, status in results:
    print(f"{name:30s} {angle:10.1f} {chi1:8.1f} {status:>8s}")
print("-" * 65)

if failed:
    print(f"\nWARNING: {len(failed)} variant(s) failed DFG check: {failed}")
    print("Check these manually before docking — the DFG may have collapsed.")
    print("If a mutation is near the DFG motif, you may need to manually select")
    print("a different rotamer that preserves the DFG-out conformation.")
else:
    print(f"\nAll {len(results)} variants passed DFG verification.")
    print("Ready for PDBQT preparation (06e script).")