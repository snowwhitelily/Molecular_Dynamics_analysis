#!/usr/bin/env python3
"""
compare_zide_redock.py

Zidesamtinib docking: ORIGINAL (rigid, flat macrocycle) vs CORRECTED
(3D conformer + cracked ring). Reads best-pose Vina scores from the two output
dirs and prints per-variant medians side by side, so the size of the artifact
is visible at a glance.

Run AFTER run_zide_cracked.sh has finished (64 files in vina_outputs_zide_cracked/).

Usage (venv on):
    python analysis/scripts/compare_zide_redock.py
"""
from pathlib import Path
import numpy as np

BASE = Path.home() / "Molecular_Dynamics_analysis"
ENS  = BASE / "dock" / "ROS1" / "q2022p_vina" / "ensemble_all"
OLD  = ENS / "vina_outputs"                 # original rigid/flat poses (intact)
NEW  = ENS / "vina_outputs_zide_cracked"    # corrected 3D + cracked
ORDER = ["WT", "S1986F", "Q2022P", "Q2022P_S1986F"]


def best_score(path):
    """First MODEL's Vina affinity = best pose."""
    for ln in open(path):
        if ln.startswith("REMARK VINA RESULT"):
            return float(ln.split()[3])
    return None


def variant_of(tag):
    # WT_rep0_f1283_open__zidesamtinib.pdbqt -> WT
    # Q2022P_S1986F_rep0_f3339_closed__... -> Q2022P_S1986F
    return tag.split("__")[0].split("_rep")[0]


def collect(d):
    out = {}
    for f in sorted(d.glob("*__zidesamtinib.pdbqt")):
        s = best_score(f)
        if s is not None:
            out.setdefault(variant_of(f.name), []).append(s)
    return out


def main():
    if not NEW.exists() or not any(NEW.glob("*__zidesamtinib.pdbqt")):
        raise SystemExit(f"no corrected poses yet in {NEW} — run the re-dock first")
    old, new = collect(OLD), collect(NEW)

    print(f"{'variant':16s} {'old med':>8s} {'new med':>8s} {'shift':>7s} "
          f"{'new min':>8s} {'new max':>8s}")
    for v in ORDER:
        o, n = np.array(old.get(v, [])), np.array(new.get(v, []))
        if len(o) and len(n):
            print(f"{v:16s} {np.median(o):8.2f} {np.median(n):8.2f} "
                  f"{np.median(n)-np.median(o):+7.2f} {n.min():8.2f} {n.max():8.2f}")
        else:
            print(f"{v:16s}  (old n={len(o)}, new n={len(n)} — incomplete)")

    allo = np.concatenate(list(old.values())) if old else np.array([])
    alln = np.concatenate(list(new.values())) if new else np.array([])
    if len(allo) and len(alln):
        print(f"\n{'POOLED':16s} {np.median(allo):8.2f} {np.median(alln):8.2f} "
              f"{np.median(alln)-np.median(allo):+7.2f}")
        print(f"\nContext: other ligands' medians span roughly -7 to -10. "
              f"If corrected zidesamtinib\nlands in that band, the old "
              f"'strongest binder' claim was the rigid-ring artifact.")


if __name__ == "__main__":
    main()