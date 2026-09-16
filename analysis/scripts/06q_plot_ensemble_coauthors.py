#!/usr/bin/env python3
"""
06q_plot_ensemble_coauthors.py

Presentation-ready figure of the openness-stratified ensemble docking.
Message: Q2022P and the double mutant bind comparably, and pocket openness
does not meaningfully change docking score -> the single-frame difference was
noise; Q2022P dominates.

Two panels (cabozantinib, lorlatinib). Each shows the 4 groups
(Q2022P open/closed, double open/closed) as jittered points + mean bars,
with the mean values annotated. Clean style.

Usage (venv on):
    python analysis/scripts/06q_plot_ensemble_coauthors.py
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
                                 state="open" if r["pocket_open_state"] == 1 else "closed",
                                 score=best_aff(f)))
    df = pd.DataFrame(rows).dropna(subset=["score"])

    # variant label prettifier
    vlab = {"Q2022P": "Q2022P", "Q2022P_S1986F": "Q2022P\n+ S1986F"}
    groups = [("Q2022P", "open"), ("Q2022P", "closed"),
              ("Q2022P_S1986F", "open"), ("Q2022P_S1986F", "closed")]
    colors = {"Q2022P": "#1976D2", "Q2022P_S1986F": "#9C27B0"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, lig in zip(axes, LIGANDS):
        d = df[df["ligand"] == lig]
        for i, (v, st) in enumerate(groups):
            s = d[(d["variant"] == v) & (d["state"] == st)]["score"].values
            jitter = rng.uniform(-0.12, 0.12, len(s))
            fill = colors[v] if st == "open" else "white"
            ax.scatter(i + jitter, s, s=42, facecolor=fill,
                       edgecolor=colors[v], linewidth=1.4, alpha=0.85, zorder=3)
            # mean bar
            ax.plot([i - 0.28, i + 0.28], [s.mean()] * 2, color="black",
                    lw=2.4, zorder=4)
            ax.text(i, s.max() + 0.18, f"{s.mean():.2f}", ha="center",
                    va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"{vlab[v]}\n{st}" for v, st in groups], fontsize=8.5)
        ax.set_title(lig, fontsize=12, fontweight="bold")
        ax.invert_yaxis()
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.yaxis.grid(True, color="#ececec", lw=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    axes[0].set_ylabel("Docking affinity (kcal/mol)", fontsize=11)
    # legend: filled = open, hollow = closed
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker='o', color='none', markerfacecolor='#555',
                  markeredgecolor='#555', markersize=8, label='open pocket'),
           Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
                  markeredgecolor='#555', markersize=8, label='closed pocket')]
    axes[1].legend(handles=leg, frameon=False, fontsize=9, loc="lower right")

    fig.suptitle("Ensemble docking: Q2022P vs double mutant, by pocket state",
                 fontsize=13, y=1.02)
    fig.text(0.5, -0.04,
             "Each point = one MD frame (8 open + 8 closed per variant), docked "
             "in the shared box. Bars = group means. Q2022P and the double "
             "mutant bind comparably, and\nopen vs closed frames differ by "
             "<0.3 kcal/mol (within scatter) \u2014 docking score is insensitive "
             "to pocket openness here.",
             ha="center", va="top", fontsize=7.5, color="#555")

    plt.tight_layout()
    out = FIG_DIR / "ensemble_docking_coauthors.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    # print the summary numbers too
    for lig in LIGANDS:
        d = df[df["ligand"] == lig]
        for v in ["Q2022P", "Q2022P_S1986F"]:
            s = d[d["variant"] == v]["score"]
            print(f"  {lig:13s} {v:16s} mean {s.mean():.2f} +/- {s.std():.2f}")


if __name__ == "__main__":
    main()