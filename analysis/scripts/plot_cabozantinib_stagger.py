#!/usr/bin/env python3
"""
plot_cabozantinib_stagger.py

Cabozantinib per-frame docking across the four variants, in the staggering-plot
style (Tsjerk): one jittered dot per MD frame, median bar per variant. Shows the
honest frame-to-frame spread (~2-3 kcal/mol) that a single median hides.

Open vs closed pocket frames are coloured so the reader can see the open/closed
effect is negligible (dots interleave), backing "docking is insensitive to
pocket openness here".

Reads the CORRECTED ensemble scores (step6r_ensemble_scores_all3d.csv), so the
values are the re-docked 3D-ligand numbers.

Usage (venv on):
    python analysis/scripts/plot_cabozantinib_stagger.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BASE    = Path.home() / "Molecular_Dynamics_analysis"
CSV     = BASE / "results" / "ROS1" / "step6r_ensemble_scores_all3d.csv"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final"
FIG_DIR.mkdir(parents=True, exist_ok=True)

VORD  = ["WT", "S1986F", "Q2022P", "Q2022P_S1986F"]
VLAB  = {"WT": "WT", "S1986F": "S1986F", "Q2022P": "Q2022P",
         "Q2022P_S1986F": "Q2022P\n+ S1986F"}
STATEC = {"open": "#0072B2", "closed": "#E69F00"}


def main():
    d = pd.read_csv(CSV)
    c = d[d.ligand == "cabozantinib"].copy()
    if c.empty:
        raise SystemExit("no cabozantinib rows in " + str(CSV))

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    rng = np.random.default_rng(0)
    for i, v in enumerate(VORD):
        sub = c[c.variant == v]
        for state in ("open", "closed"):
            s = sub[sub.state == state]["score"].values
            jit = rng.uniform(-0.14, 0.14, len(s))
            ax.scatter(i + jit, s, s=60, color=STATEC.get(state, "#888"),
                       alpha=0.8, edgecolor="black", linewidth=0.5, zorder=3)
        med = sub["score"].median()
        ax.plot([i - 0.28, i + 0.28], [med, med], color="black", lw=2.6, zorder=4)
        ax.text(i + 0.32, med, f"{med:.2f}", va="center", ha="left",
                fontsize=9.5, fontweight="bold")
        # medoid = frame whose score is closest to the median (red ring)
        _mv = sub["score"].values
        _medoid = _mv[np.argmin(np.abs(_mv - np.median(_mv)))]
        ax.scatter([i], [_medoid], s=190, facecolor="none",
                   edgecolor="#E63946", linewidth=2.3, zorder=6)

    ax.set_xticks(range(len(VORD)))
    ax.set_xticklabels([VLAB[v] for v in VORD], fontsize=11)
    ax.set_ylabel("Docking affinity (kcal/mol)", fontsize=11)
    ax.invert_yaxis()                       # stronger (more negative) higher
    ax.set_xlim(-0.5, len(VORD) - 0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color="#eee")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title("Cabozantinib: per-frame docking across variants",
                 fontsize=13, pad=10)

    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=STATEC["open"],
                  markeredgecolor="black", markersize=9, label="open-pocket frame"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=STATEC["closed"],
                  markeredgecolor="black", markersize=9, label="closed-pocket frame"),
           Line2D([0], [0], color="black", lw=2.6, label="median (16 frames)"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                  markeredgecolor="#E63946", markeredgewidth=2.0, markersize=11,
                  label="medoid frame")]
    ax.legend(handles=leg, frameon=False, fontsize=9.5,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.text(0.5, -0.02,
             "Each dot = one MD frame docked in the shared box (8 open + 8 closed "
             "per variant). Bar = median over 16 frames. More negative = stronger.",
             ha="center", va="top", fontsize=8, color="#666")

    plt.tight_layout()
    out = FIG_DIR / "cabozantinib_stagger.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    for v in VORD:
        s = c[c.variant == v]["score"]
        print(f"  {v:16s} median {s.median():6.2f}  spread {s.min():.2f}..{s.max():.2f}")


if __name__ == "__main__":
    main()