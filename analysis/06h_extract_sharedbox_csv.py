#!/usr/bin/env python3
"""
06h_extract_sharedbox_csv.py

Read the 10 shared-box cabozantinib docks (vina_outputs_sharedbox) and write a
CSV in the SAME schema as the old step6c file, so the existing plotter can use
it unchanged:

    mutant,conformation,ligand,affinity_kcal_mol,receptor_stem

These are the CORRECTED numbers: real mutant receptors (06d2, mutate+minimize)
docked with ONE shared box (06g). They replace the old phantom-mutant +
mismatched-box values.

Usage (venv on):
    python analysis/06h_extract_sharedbox_csv.py
"""
import csv
import re
from pathlib import Path

BASE = Path.home() / "Molecular_Dynamics_analysis"
OUT_DIR = BASE / "dock" / "ROS1" / "q2022p_vina" / "vina_outputs_sharedbox"
RESULTS = BASE / "results" / "ROS1"
RESULTS.mkdir(parents=True, exist_ok=True)
CSV_OUT = RESULTS / "step6g_cabozantinib_sharedbox.csv"


def variant_and_conf(stem):
    """Map a receptor stem to (variant, conformation)."""
    if stem.endswith("_actout_ctlfit_dominant_cluster0_rep"):
        variant = stem.replace("_actout_ctlfit_dominant_cluster0_rep", "")
        return variant, "active"
    m = re.match(r"ROS1_(.+)_i$", stem)
    if m:
        return m.group(1), "inactive"
    raise ValueError(f"cannot parse receptor stem: {stem}")


def best_affinity(pdbqt):
    """First REMARK VINA RESULT = best mode; 4th token is affinity."""
    with open(pdbqt) as fh:
        for ln in fh:
            if ln.startswith("REMARK VINA RESULT"):
                return float(ln.split()[3])
    raise ValueError(f"no VINA RESULT in {pdbqt}")


def main():
    files = sorted(OUT_DIR.glob("*__cabozantinib.pdbqt"))
    if len(files) != 10:
        print(f"WARNING: expected 10 outputs, found {len(files)}")

    rows = []
    for f in files:
        stem = f.name.replace("__cabozantinib.pdbqt", "")
        variant, conf = variant_and_conf(stem)
        aff = best_affinity(f)
        rows.append(dict(mutant=variant, conformation=conf,
                         ligand="cabozantinib", affinity_kcal_mol=aff,
                         receptor_stem=stem))

    # sort for readability: variant, then active before inactive
    rows.sort(key=lambda r: (r["mutant"], r["conformation"]))

    with open(CSV_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mutant", "conformation", "ligand",
                                           "affinity_kcal_mol", "receptor_stem"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows -> {CSV_OUT}\n")
    print(f"{'variant':16s} {'active':>9s} {'inactive':>9s}  winner")
    by = {}
    for r in rows:
        by.setdefault(r["mutant"], {})[r["conformation"]] = r["affinity_kcal_mol"]
    for v in ["WT", "Q2022P", "S1986F", "Q2022P_S1986F", "Q2022P_S1986Y"]:
        if v in by:
            a = by[v].get("active"); i = by[v].get("inactive")
            if a is not None and i is not None:
                win = "active" if a < i else ("inactive" if i < a else "tie")
                print(f"{v:16s} {a:9.2f} {i:9.2f}  {win}")


if __name__ == "__main__":
    main()