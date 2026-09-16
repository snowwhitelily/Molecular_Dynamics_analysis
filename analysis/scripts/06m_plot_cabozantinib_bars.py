#!/usr/bin/env python3
"""
06m_plot_cabozantinib_bars.py

Cabozantinib active-vs-inactive bar chart, clean style, publication focus set.

- Variants: WT, Q2022P, Q2022P_S1986F, S1986F  (Q2022P_S1986Y dropped).
- ACTIVE values come from the heatmap run (step6k) so the bar chart and the
  heatmap show identical active-cabozantinib numbers.
- INACTIVE values come from the cabozantinib inactive shared-box run (step6g).
- Clean style: two flat colours (active / inactive), thin bars, no top/right
  spines, light horizontal grid, small labels -- matching the requested look.

Usage (venv on):
    python analysis/scripts/06m_plot_cabozantinib_bars.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HEATMAP_CSV = RESULTS / "step6k_heatmap_sharedbox.csv"        # active source
CABO_CSV = RESULTS / "step6g_cabozantinib_sharedbox.csv"       # inactive source

VARIANTS = ["WT", "Q2022P", "Q2022P_S1986F", "S1986F"]

# two flat colours, Image-2 clean
C_ACTIVE = "#1f77b4"     # muted blue
C_INACT = "#e0a458"      # warm muted amber


def main():
    # ACTIVE cabozantinib from the heatmap run
    hm = pd.read_csv(HEATMAP_CSV)
    act = (hm[(hm["ligand"] == "cabozantinib")]
           .set_index("mutant")["affinity_kcal_mol"])

    # INACTIVE cabozantinib from the inactive shared-box run
    cabo = pd.read_csv(CABO_CSV)
    inact = (cabo[(cabo["conformation"] == "inactive")]
             .set_index("mutant")["affinity_kcal_mol"])

    act_vals = [act.get(v, np.nan) for v in VARIANTS]
    inact_vals = [inact.get(v, np.nan) for v in VARIANTS]

    x = np.arange(len(VARIANTS))
    w = 0.42                                  # fatter bars; pair nearly touches
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.bar(x - w / 2, act_vals, w, label="Active (DFG-in)",
           color=C_ACTIVE, edgecolor="none")
    ax.bar(x + w / 2, inact_vals, w, label="Inactive (DFG-out)",
           color=C_INACT, edgecolor="none")

    # value labels just below each bar tip (bars hang downward from 0)
    for xi, v in zip(x - w / 2, act_vals):
        if not np.isnan(v):
            ax.text(xi, v - 0.15, f"{v:.2f}", ha="center", va="top",
                    fontsize=8.5, color="#333")
    for xi, v in zip(x + w / 2, inact_vals):
        if not np.isnan(v):
            ax.text(xi, v - 0.15, f"{v:.2f}", ha="center", va="top",
                    fontsize=8.5, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS, fontsize=10)
    ax.set_ylabel("Binding affinity (kcal/mol)", fontsize=11)
    # 0 at the TOP, stronger (more negative) hangs DOWN -- like the reference
    ax.set_ylim(-13, 0)
    ax.set_title("Cabozantinib: active vs inactive receptor conformation",
                 fontsize=12, pad=28)   # extra pad to leave room for legend

    # clean chrome: drop top/right spines, light horizontal grid only
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    # x-axis (labels) sits at the TOP now since bars hang down; keep ticks at 0
    ax.xaxis.set_ticks_position("bottom")
    # legend ABOVE the plot, horizontal, out of the bars
    ax.legend(frameon=False, fontsize=10, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))

    fig.text(0.5, -0.03,
             "Shared docking box (ATP site + type II crevice); active and "
             "inactive directly comparable. Active values match the heatmap.",
             ha="center", va="top", fontsize=7.5, color="#666")

    plt.tight_layout()
    out = FIG_DIR / "cabozantinib_active_vs_inactive_clean.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    print(f"{'variant':16s} {'active':>8s} {'inactive':>9s}")
    for v, a, i in zip(VARIANTS, act_vals, inact_vals):
        print(f"{v:16s} {a:8.2f} {i:9.2f}")


if __name__ == "__main__":
    main()