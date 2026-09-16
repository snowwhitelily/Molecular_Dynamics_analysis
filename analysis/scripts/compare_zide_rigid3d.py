#!/usr/bin/env python3
"""
compare_zide_rigid3d.py

Zidesamtinib docking: ORIGINAL (flat + rigid macrocycle) vs CORRECTED
(proper 3D conformer, docked rigid). Reads best-pose Vina scores from the two
output dirs and prints per-variant medians side by side.

Run AFTER run_zide_rigid3d.sh finishes (64 files in vina_outputs_zide_rigid3d/).

Usage (venv on):
    python analysis/scripts/compare_zide_rigid3d.py
"""
from pathlib import Path
import numpy as np

BASE = Path.home() / "Molecular_Dynamics_analysis"
ENS  = BASE / "dock" / "ROS1" / "q2022p_vina" / "ensemble_all"
OLD  = ENS / "vina_outputs"                 # original flat/rigid (intact)
NEW  = ENS / "vina_outputs_zide_rigid3d"    # corrected 3D rigid
ORDER = ["WT", "S1986F", "Q2022P", "Q2022P_S1986F"]


def best_score(path):
    for ln in open(path):
        if ln.startswith("REMARK VINA RESULT"):
            return float(ln.split()[3])
    return None


def variant_of(tag):
    return tag.split("__")[0].split("_rep")[0]


def collect(d):
    out = {}
    for f in sorted(d.glob("*__zidesamtinib.pdbqt")):
        s = best_score(f)
        if s is not None:
            out.setdefault(variant_of(f.name), []).append(s)
    return out


def main():
    files = list(NEW.glob("*__zidesamtinib.pdbqt"))
    if not files:
        raise SystemExit(f"no rigid-3D poses in {NEW} — run the re-dock first")
    if len(files) < 64:
        print(f"WARNING: only {len(files)}/64 poses present — run may be unfinished\n")
    old, new = collect(OLD), collect(NEW)

    print(f"{'variant':16s} {'old med':>8s} {'new med':>8s} {'shift':>7s} "
          f"{'new min':>8s} {'new max':>8s}  {'n':>3s}")
    for v in ORDER:
        o, n = np.array(old.get(v, [])), np.array(new.get(v, []))
        if len(o) and len(n):
            print(f"{v:16s} {np.median(o):8.2f} {np.median(n):8.2f} "
                  f"{np.median(n)-np.median(o):+7.2f} {n.min():8.2f} {n.max():8.2f}  {len(n):3d}")
        else:
            print(f"{v:16s}  (old n={len(o)}, new n={len(n)} — incomplete)")

    allo = np.concatenate(list(old.values())) if old else np.array([])
    alln = np.concatenate(list(new.values())) if new else np.array([])
    if len(allo) and len(alln):
        print(f"\n{'POOLED':16s} {np.median(allo):8.2f} {np.median(alln):8.2f} "
              f"{np.median(alln)-np.median(allo):+7.2f}")
        print(f"pooled range: old {allo.min():.2f}..{allo.max():.2f}  "
              f"new {alln.min():.2f}..{alln.max():.2f}")
        print("\nExpect corrected zidesamtinib to sit in the -7 to -10 pack with the "
              "other ligands,\nnot as a -11.9 outlier. Caption it as a semi-rigid "
              "macrocycle docked from a single\n3D conformer, so the ABSOLUTE value "
              "carries more uncertainty than the flexible ligands.")


if __name__ == "__main__":
    main()