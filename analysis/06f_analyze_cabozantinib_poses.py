#!/usr/bin/env python3
"""
06f_analyze_cabozantinib_poses.py

Investigate where cabozantinib docking poses actually sit — in the type II
specificity pocket (expected for a type II inhibitor in the inactive form)
or wedged into the ATP site (expected in the active form, but suspicious
if it also happens in the inactive form).

This addresses the anomalous result where WT active (-10.22) scored stronger
than WT inactive (-9.76). If the inactive pose is actually sitting in the ATP
site rather than the type II pocket, that explains the anomaly.

The analysis measures:
  1. Ligand centroid distance to the ATP site centre (L2026/hinge region)
  2. Ligand centroid distance to the type II pocket centre (F2004/F2075)
  3. Minimum ligand atom distance to F2004 and F2075 (specificity pocket markers)
  4. Minimum ligand atom distance to DFG F2103
  5. Whether any ligand atoms penetrate beyond the DFG into the specificity pocket

A type II pose should be closer to F2004/F2075 than an active-form ATP pose.
An ATP-wedged pose will sit near L2026/hinge and far from F2004/F2075.

Usage (venv on):
    python analysis/06f_analyze_cabozantinib_poses.py

Reads all cabozantinib output PDBQTs and prints a comparison table.
"""

from pathlib import Path

import numpy as np
import MDAnalysis as mda

BASE = Path.home() / "Molecular_Dynamics_analysis"
DOCK = BASE / "dock" / "ROS1"
VINA = DOCK / "q2022p_vina"
OUT_DIR = VINA / "vina_outputs"
INACTIVE_REF = DOCK / "inactive_ref"

LIGAND = "cabozantinib"

# Reference residues for pocket classification
# These are in the ROS1_WT_i.pdb numbering (real numbering)
ATP_SITE_RESIDUES = [2026, 2027, 2028, 2029, 2030]   # hinge region
TYPE2_RESIDUES = [2004, 2075]                          # specificity pocket
DFG_RESIDUES = [2102, 2103]                            # DFG motif


def extract_best_pose_coords(pdbqt_path):
    """Extract coordinates of the best pose (MODEL 1) from a Vina output PDBQT."""
    coords = []
    in_model_1 = False
    for line in pdbqt_path.read_text().splitlines():
        if line.startswith("MODEL") and line.split()[1] == "1":
            in_model_1 = True
            continue
        if line.startswith("ENDMDL") and in_model_1:
            break
        if in_model_1 and (line.startswith("ATOM") or line.startswith("HETATM")):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
    return np.array(coords) if coords else None


def get_residue_ca_center(universe, resids):
    """Get the centroid of Ca atoms for the given residue IDs."""
    sel = " or ".join(f"resid {r}" for r in resids)
    ag = universe.select_atoms(f"({sel}) and name CA")
    if ag.n_atoms == 0:
        return None
    return ag.positions.mean(axis=0)


def get_residue_all_atom_positions(universe, resids):
    """Get all heavy-atom positions for the given residue IDs."""
    sel = " or ".join(f"resid {r}" for r in resids)
    ag = universe.select_atoms(f"({sel}) and not name H*")
    return ag.positions if ag.n_atoms > 0 else None


def min_distance(coords_a, coords_b):
    """Minimum distance between any atom in A and any atom in B."""
    # Use broadcasting: (N,1,3) - (1,M,3) -> (N,M,3) -> (N,M)
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2)).min()


def classify_pose(lig_centroid, atp_center, type2_center):
    """Classify as ATP-site or type-II based on centroid proximity."""
    d_atp = np.linalg.norm(lig_centroid - atp_center)
    d_type2 = np.linalg.norm(lig_centroid - type2_center)
    if d_type2 < d_atp:
        return "TYPE_II"
    else:
        return "ATP_SITE"


# ── load the inactive reference for pocket landmarks ────────────────────────
wt_inactive = INACTIVE_REF / "ROS1_WT_i.pdb"
if not wt_inactive.exists():
    raise SystemExit(f"missing {wt_inactive}")

ref = mda.Universe(str(wt_inactive))
atp_center = get_residue_ca_center(ref, ATP_SITE_RESIDUES)
type2_center = get_residue_ca_center(ref, TYPE2_RESIDUES)
dfg_positions = get_residue_all_atom_positions(ref, DFG_RESIDUES)
type2_positions = get_residue_all_atom_positions(ref, TYPE2_RESIDUES)

if atp_center is None or type2_center is None:
    raise SystemExit("could not find reference residues in WT inactive")

print(f"Reference landmarks (from {wt_inactive.name}):")
print(f"  ATP site centre (hinge Ca):   {atp_center}")
print(f"  Type II pocket centre (Ca):   {type2_center}")
print(f"  Distance between centres:     {np.linalg.norm(atp_center - type2_center):.1f} A")
print()

# ── also get landmarks from one of the active-form receptors ────────────────
# Active-form receptors use GRO numbering (real - 1933), so the residue IDs
# are different. We skip this for active poses — we just report distances
# to the WT inactive landmarks, which is sufficient for comparison.

# ── analyze all cabozantinib poses ──────────────────────────────────────────
outputs = sorted(OUT_DIR.glob(f"*__{LIGAND}.pdbqt"))
if not outputs:
    raise SystemExit(f"no {LIGAND} outputs found in {OUT_DIR}")

print(f"{'receptor':35s} {'conf':>8s} {'d_ATP':>7s} {'d_T2':>7s} "
      f"{'min_DFG':>8s} {'min_T2':>8s} {'class':>10s}")
print("-" * 95)

for path in outputs:
    stem = path.name.split(f"__{LIGAND}")[0]

    # Determine conformation
    if "_i" in stem and "actout" not in stem:
        conf = "inactive"
    else:
        conf = "active"

    lig_coords = extract_best_pose_coords(path)
    if lig_coords is None or len(lig_coords) == 0:
        print(f"{stem:35s} {conf:>8s}  -- no coordinates found --")
        continue

    lig_centroid = lig_coords.mean(axis=0)

    # Distances to reference landmarks
    d_atp = np.linalg.norm(lig_centroid - atp_center)
    d_type2 = np.linalg.norm(lig_centroid - type2_center)

    # Minimum atom-to-atom distances (only meaningful for inactive-frame poses,
    # since active poses are in a different coordinate frame)
    if conf == "inactive":
        min_dfg = min_distance(lig_coords, dfg_positions)
        min_t2 = min_distance(lig_coords, type2_positions)
        classification = classify_pose(lig_centroid, atp_center, type2_center)
    else:
        # Active-form poses are in a different coordinate frame — these
        # distances are meaningless. Flag them.
        min_dfg = float('nan')
        min_t2 = float('nan')
        classification = "diff_frame"

    print(f"{stem:35s} {conf:>8s} {d_atp:7.1f} {d_type2:7.1f} "
          f"{min_dfg:8.1f} {min_t2:8.1f} {classification:>10s}")

print("-" * 95)
print()
print("Interpretation:")
print("  d_ATP  = ligand centroid distance to hinge region (ATP site)")
print("  d_T2   = ligand centroid distance to F2004/F2075 (type II pocket)")
print("  min_DFG = closest ligand atom to DFG motif residues")
print("  min_T2  = closest ligand atom to specificity pocket residues")
print("  class  = TYPE_II if centroid closer to specificity pocket, ATP_SITE otherwise")
print("  diff_frame = active-form receptor in different coordinates (distances not comparable)")
print()
print("If WT inactive is classified as ATP_SITE, the pose is wedged into the")
print("ATP site rather than the type II pocket, which would explain why the")
print("active score (-10.22) beat the inactive score (-9.76).")