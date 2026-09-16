#!/usr/bin/env python3
"""
check_macrocycle_rigidity.py

Evidence that zidesamtinib was docked as a FROZEN RIGID ring (macrocycle never
cracked open), by measuring how much each ligand's *internal* conformation
varies across all its docked poses.

Vina outputs a rigidly-docked ligand in ONE frozen conformation -- only its
rigid-body placement and its handful of torsions change between poses -- so its
internal RMSD across poses is ~0. A correctly cracked macrocycle (repotrectinib,
used here as the positive control) samples different ring conformations, so its
internal RMSD is clearly non-zero.

Reads the best pose (MODEL 1) from every <tag>__<ligand>.pdbqt in the ensemble
docking outputs, superposes each onto the first (Kabsch, heavy atoms only),
reports mean / max internal RMSD.

Usage (venv on):
    python analysis/scripts/check_macrocycle_rigidity.py
"""
from pathlib import Path
import numpy as np

BASE   = Path.home() / "Molecular_Dynamics_analysis"
OUTDIR = BASE / "dock" / "ROS1" / "q2022p_vina" / "ensemble_all" / "vina_outputs"
LIGANDS = ["zidesamtinib", "repotrectinib"]   # suspect macrocycle + cracked control


def read_best_pose_heavy(path):
    """Heavy-atom coords of MODEL 1 (best pose) from a Vina pdbqt."""
    coords, started = [], False
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("MODEL"):
                if started:               # reached MODEL 2 -> stop
                    break
                started = True
            elif ln.startswith("ENDMDL"):
                break
            elif ln.startswith(("ATOM", "HETATM")):
                atype = ln.split()[-1]    # AutoDock atom type is the last field
                if atype in ("HD", "H"):  # skip hydrogens
                    continue
                coords.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return np.asarray(coords)


def kabsch_rmsd(P, Q):
    """RMSD of P onto Q after optimal rigid superposition."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1, 1, d]) @ Wt
    return np.sqrt(((Pc @ R - Qc) ** 2).sum() / len(P))


def main():
    if not OUTDIR.exists():
        raise SystemExit(f"pose dir not found: {OUTDIR}")
    for lig in LIGANDS:
        files = sorted(OUTDIR.glob(f"*__{lig}.pdbqt"))
        if not files:
            print(f"{lig:14s} no poses found in {OUTDIR}")
            continue
        ref = read_best_pose_heavy(files[0])
        rmsds, nbad = [], 0
        for f in files:
            P = read_best_pose_heavy(f)
            if P.shape != ref.shape:
                nbad += 1
                continue
            rmsds.append(kabsch_rmsd(P, ref))
        rmsds = np.array(rmsds)
        note = f"  ({nbad} atom-count mismatches skipped)" if nbad else ""
        print(f"{lig:14s} n_poses={len(rmsds):3d}  heavy_atoms={len(ref):3d}  "
              f"internal RMSD across poses: mean={rmsds.mean():.3f} A  "
              f"max={rmsds.max():.3f} A{note}")
    print("\n~0 A  = docked as a frozen rigid body (ring never cracked)")
    print(">0 A  = ring conformation was sampled (cracked correctly)")


if __name__ == "__main__":
    main()