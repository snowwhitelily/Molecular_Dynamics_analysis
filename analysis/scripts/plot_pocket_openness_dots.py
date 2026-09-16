#!/usr/bin/env python3
"""
plot_pocket_openness_dots.py

Pocket-openness figure, DOTS-ONLY version.

With only n = 3 replicas a bar + SD implies more precision than there is, and for
S1986F the SD (+/-34) was actively misleading -- it was stretched by one replica
that sat in a closed state (rep2, 26.6% open). Showing the three replica dots
makes that dissenting replica visible instead of burying it in a whisker.

  - 3 dots per variant  = the individual replicas (colour = variant)
  - short black crossbar = replicate mean (value labelled)
  - dashed line at 50%   = open / closed midpoint
  - NO bar, NO SD whisker by default (the dots carry the spread)

Numbers are recomputed from step3a_frame_metrics.csv (pocket_open_state fraction
per replica) -- never hand-typed. Reproduces WT 80.7 / S1986F 66.0 / Q2022P 65.1
/ double 59.8 (replicate means).

Usage (venv on):
    python analysis/scripts/plot_pocket_openness_dots.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BASE    = Path.home() / "Molecular_Dynamics_analysis"
CSV     = BASE / "results" / "ROS1" / "step3a_frame_metrics.csv"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final"
FIG_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["WT", "S1986F", "Q2022P", "Q2022P_S1986F"]
LABEL    = {"WT": "WT", "S1986F": "S1986F", "Q2022P": "Q2022P",
            "Q2022P_S1986F": "Q2022P + S1986F"}
COL      = {"WT": "#0072B2", "S1986F": "#009E73",
            "Q2022P": "#E69F00", "Q2022P_S1986F": "#CC79A7"}

# --- toggles (both off = the clean default shown to Tsjerk) -----------------
SHOW_SD     = False   # thin +/-1 SD whisker on the mean crossbar
SHOW_POOLED = False   # add frame-weighted pooled value as a small open marker


def main():
    d = pd.read_csv(CSV, usecols=["mutant", "replica", "pocket_open_state"])
    d = d[d.mutant.isin(VARIANTS)]

    perrep = (d.groupby(["mutant", "replica"])["pocket_open_state"]
              .mean().mul(100).rename("pct").reset_index())
    pooled = d.groupby("mutant")["pocket_open_state"].mean().mul(100)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    rng = np.random.default_rng(1)
    ax.axhline(50, ls="--", lw=1, color="#bbbbbb", zorder=1)

    for i, v in enumerate(VARIANTS):
        pts = perrep.loc[perrep.mutant == v, "pct"].values
        jit = rng.uniform(-0.07, 0.07, len(pts))
        ax.scatter(i + jit, pts, s=130, color=COL[v], edgecolor="black",
                   linewidth=1.1, zorder=3, alpha=0.9)
        m, sd = pts.mean(), pts.std(ddof=1)
        if SHOW_SD:
            ax.errorbar(i, m, yerr=sd, fmt="none", ecolor="#555",
                        elinewidth=1.2, capsize=4, capthick=1.2, zorder=2)
        ax.plot([i - 0.22, i + 0.22], [m, m], color="black", lw=2.6, zorder=4)
        ax.text(i + 0.28, m, f"{m:.1f}%", va="center", ha="left",
                fontsize=11, fontweight="bold")
        if SHOW_POOLED:
            ax.scatter([i], [pooled[v]], s=120, facecolor="none",
                       edgecolor="#333", linewidth=1.6, zorder=5)

    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels([LABEL[v] for v in VARIANTS], fontsize=11)
    ax.set_ylabel("Open-pocket frames (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, len(VARIANTS) - 0.1)
    ax.set_title("ATP-pocket openness across variants", fontsize=14, pad=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color="#eeeeee", lw=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.text(len(VARIANTS) - 0.55, 51.5, "open / closed midpoint",
            fontsize=8, color="#999999", ha="right", va="bottom")

    leg = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#888",
                  markeredgecolor="black", markersize=11,
                  label="individual replica (n = 3)"),
           Line2D([0], [0], color="black", lw=2.6, label="replicate mean")]
    if SHOW_POOLED:
        leg.append(Line2D([0], [0], marker="o", color="none",
                          markerfacecolor="none", markeredgecolor="#333",
                          markersize=11, label="pooled (frame-weighted)"))
    ax.legend(handles=leg, frameon=False, fontsize=9.5, loc="lower left")

    fig.text(0.5, -0.02,
             "Each dot = one MD replica; line = mean of the 3 replicas. "
             "Open = P-loop\u2013hinge \u2265 16.9 \u00c5 or gatekeeper\u2013DFG-Phe "
             "\u2265 16.2 \u00c5 (dataset-median thresholds).",
             ha="center", va="top", fontsize=8, color="#666")

    plt.tight_layout()
    out = FIG_DIR / "pocket_openness_dots.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    for v in VARIANTS:
        pts = perrep.loc[perrep.mutant == v, "pct"].values
        print(f"  {v:16s} reps {np.array2string(np.sort(pts)[::-1], precision=1)}"
              f"  mean {pts.mean():5.1f}  pooled {pooled[v]:5.1f}")


if __name__ == "__main__":
    main()