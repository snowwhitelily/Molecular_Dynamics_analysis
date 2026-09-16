#!/usr/bin/env python3
"""
06p_analyze_openness_dock.py

Join the ensemble docking scores to per-frame openness (manifest) and test:
  1. Do OPEN frames dock stronger than CLOSED frames? (openness -> binding)
  2. Do Q2022P and Q2022P_S1986F distributions overlap? (real diff or noise)
  3. Continuous: does pocket_front_dist correlate with docking score?

Reports numbers + a strip/box figure. Honest about effect sizes.

Usage (venv on):
    python analysis/06p_analyze_openness_dock.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path.home() / "Molecular_Dynamics_analysis"
OUT = BASE / "dock" / "ROS1" / "q2022p_vina" / "openness_ensemble"
VOUT = OUT / "vina_outputs"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final"
LIGANDS = ["cabozantinib", "lorlatinib"]


def best_aff(pdbqt):
    for ln in open(pdbqt):
        if ln.startswith("REMARK VINA RESULT"):
            return float(ln.split()[3])
    return np.nan


def main():
    man = pd.read_csv(OUT / "manifest.csv")
    rows = []
    for _, r in man.iterrows():
        for lig in LIGANDS:
            f = VOUT / f"{r['tag']}__{lig}.pdbqt"
            if f.exists():
                rows.append(dict(variant=r["variant"], ligand=lig,
                                 open_state=int(r["pocket_open_state"]),
                                 front=r["pocket_front_dist_nm"],
                                 score=best_aff(f)))
    df = pd.DataFrame(rows).dropna(subset=["score"])
    df["state"] = df["open_state"].map({1: "open", 0: "closed"})
    print(f"{len(df)} scored docks\n")

    # ---- Test 1: open vs closed (per ligand, pooled variants) ----
    print("=== Test 1: open vs closed frames (mean best affinity) ===")
    for lig in LIGANDS:
        d = df[df["ligand"] == lig]
        o = d[d["state"] == "open"]["score"]; c = d[d["state"] == "closed"]["score"]
        print(f"  {lig:13s} open {o.mean():6.2f}+/-{o.std():.2f} (n={len(o)}) | "
              f"closed {c.mean():6.2f}+/-{c.std():.2f} (n={len(c)}) | "
              f"diff {o.mean()-c.mean():+.2f}")

    # ---- Test 2: Q2022P vs double (per ligand) ----
    print("\n=== Test 2: Q2022P vs Q2022P_S1986F (mean best affinity) ===")
    for lig in LIGANDS:
        d = df[df["ligand"] == lig]
        q = d[d["variant"] == "Q2022P"]["score"]
        dm = d[d["variant"] == "Q2022P_S1986F"]["score"]
        print(f"  {lig:13s} Q2022P {q.mean():6.2f}+/-{q.std():.2f} | "
              f"double {dm.mean():6.2f}+/-{dm.std():.2f} | "
              f"diff {dm.mean()-q.mean():+.2f}")

    # ---- Test 3: correlation front-distance vs score ----
    print("\n=== Test 3: front-distance vs score (Pearson r) ===")
    for lig in LIGANDS:
        d = df[df["ligand"] == lig]
        if len(d) > 3:
            r = np.corrcoef(d["front"], d["score"])[0, 1]
            print(f"  {lig:13s} r = {r:+.2f}  "
                  f"(negative = more open -> stronger binding)")

    # ---- figure: score by state, split by ligand & variant ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, lig in zip(axes, LIGANDS):
        d = df[df["ligand"] == lig]
        cats = [("Q2022P", "open"), ("Q2022P", "closed"),
                ("Q2022P_S1986F", "open"), ("Q2022P_S1986F", "closed")]
        xs, data, labs = [], [], []
        for i, (v, st) in enumerate(cats):
            s = d[(d["variant"] == v) & (d["state"] == st)]["score"].values
            data.append(s); xs.append(i)
            labs.append(f"{v.replace('Q2022P_S1986F','Q+S')}\n{st}")
        bp = ax.boxplot(data, positions=xs, widths=0.6, patch_artist=True,
                        showfliers=False)
        cols = ["#1976D2", "#90CAF9", "#9C27B0", "#CE93D8"]
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_alpha(0.8)
        for i, s in enumerate(data):
            ax.scatter(np.random.normal(i, 0.05, len(s)), s, s=18,
                       color="#333", alpha=0.6, zorder=3)
        ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=8)
        ax.set_title(lig, fontsize=11)
        ax.invert_yaxis()
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Docking affinity (kcal/mol)")
    fig.suptitle("Ensemble docking by pocket openness", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "openness_ensemble_dock.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()