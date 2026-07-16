#!/usr/bin/env python3
"""
06c_rank_cabozantinib.py

Parse the cabozantinib Vina outputs (active and inactive) into CSVs with the
same columns as the Task 5 step5c files, so the existing plotting code can read
them without modification.

Vina writes one MODEL per pose, each preceded by a line of the form
    REMARK VINA RESULT:    -7.43   0.000   0.000
and ranks them best first, so pose 1 is the best pose.

Outputs:
    results/ROS1/step6c_cabozantinib_all_poses.csv
    results/ROS1/step6c_cabozantinib_best_by_mutant.csv

Usage (venv on):
    python 06c_rank_cabozantinib.py
"""

from pathlib import Path

import pandas as pd

BASE = Path.home() / "Molecular_Dynamics_analysis"
OUT_DIR = BASE / "dock" / "ROS1" / "q2022p_vina" / "vina_outputs"
RESULTS = BASE / "results" / "ROS1"
RESULTS.mkdir(parents=True, exist_ok=True)

LIGAND = "cabozantinib"


def parse_poses(path):
    """Affinities in file order, i.e. best first."""
    vals = []
    for line in path.read_text().splitlines():
        if line.startswith("REMARK VINA RESULT:"):
            vals.append(float(line.split()[3]))
    return vals

# --- NEW CODE ---
def label(stem):
    """Map a receptor stem to (mutant, conformation)."""
    # 1. Check if it's an inactive mutant from the 06d run (ends with _i)
    if stem.endswith("_i") or "_i_" in stem:
        # Strip ROS1_ prefix and _i suffix to extract clean mutant name
        mutant_name = stem.replace("ROS1_", "").replace("_i", "")
        return mutant_name, "inactive"
    
    # 2. Otherwise, handle active trajectory cluster stems
    mutant_name = stem.split("_actout_ctlfit")[0]
    return mutant_name, "active"

# def label(stem):
#     """Map a receptor stem to (mutant, conformation)."""
#     if stem.startswith("ROS1_WT_i"):
#         return "WT", "inactive"
#     return stem.split("_actout_ctlfit")[0], "active"


rows = []
for path in sorted(OUT_DIR.glob(f"*__{LIGAND}.pdbqt")):
    stem = path.name.split(f"__{LIGAND}")[0]
    mutant, conf = label(stem)
    poses = parse_poses(path)
    if not poses:
        print(f"WARNING: no poses parsed from {path.name}")
        continue
    for rank, aff in enumerate(poses, 1):
        rows.append({
            "mutant": mutant,
            "conformation": conf,
            "ligand": LIGAND,
            "pose_rank": rank,
            "affinity_kcal_mol": aff,
            "receptor_stem": stem,
        })

if not rows:
    raise SystemExit(f"no {LIGAND} outputs found in {OUT_DIR}")

all_poses = pd.DataFrame(rows)
best = (all_poses.sort_values("affinity_kcal_mol")
        .groupby(["mutant", "conformation", "ligand"], as_index=False)
        .first()[["mutant", "conformation", "ligand",
                  "affinity_kcal_mol", "receptor_stem"]])

p1 = RESULTS / "step6c_cabozantinib_all_poses.csv"
p2 = RESULTS / "step6c_cabozantinib_best_by_mutant.csv"
all_poses.to_csv(p1, index=False)
best.to_csv(p2, index=False)

print(f"\nBest pose per system ({LIGAND})")
print("-" * 62)
print(f'{"mutant":18s} {"conformation":13s} {"best":>8s} {"poses":>6s} '
      f'{"worst":>8s}')
print("-" * 62)
for _, r in best.sort_values(["conformation", "affinity_kcal_mol"]).iterrows():
    sub = all_poses[(all_poses["mutant"] == r["mutant"]) &
                    (all_poses["conformation"] == r["conformation"])]
    print(f'{r["mutant"]:18s} {r["conformation"]:13s} '
          f'{r["affinity_kcal_mol"]:8.2f} {len(sub):6d} '
          f'{sub["affinity_kcal_mol"].max():8.2f}')
print("-" * 62)

act = best[best["conformation"] == "active"]
ina = best[best["conformation"] == "inactive"]
if not act.empty and not ina.empty:
    wt_a = act[act["mutant"] == "WT"]["affinity_kcal_mol"]
    wt_i = ina[ina["mutant"] == "WT"]["affinity_kcal_mol"]
    if not wt_a.empty and not wt_i.empty:
        d = wt_i.values[0] - wt_a.values[0]
        print(f"\nWT cabozantinib: active {wt_a.values[0]:.2f}, "
              f"inactive {wt_i.values[0]:.2f}, difference {d:+.2f} kcal/mol")
        print("Cabozantinib is type II, so a stronger (more negative) inactive")
        print("value is the expected direction.")

print(f"\nSaved: {p1}")
print(f"Saved: {p2}")