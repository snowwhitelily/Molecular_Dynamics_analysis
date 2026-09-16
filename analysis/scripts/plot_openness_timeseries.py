#!/usr/bin/env python3
"""
plot_openness_timeseries.py

Supplementary figure: ATP-pocket P-loop-hinge distance across each MD run, one
panel per variant, three replicas per panel. Shows whether the between-replica
openness spread reflects genuine open<->closed exchange (traces cross the
threshold and wander) or a run trapped on one side (a trace that never crosses).

Faint line = raw per-frame distance; bold line = running mean (window 150 frames)
so the trend is legible; red dashed line = the open/closed threshold (dataset
median of the distance). Legend gives each replica's open fraction.

Front distance by default; set POCKET_METRIC=dfg for the gatekeeper-DFG-Phe metric.

Usage (venv on):
    python analysis/scripts/plot_openness_timeseries.py
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE    = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METRICS = RESULTS / "step3a_frame_metrics.csv"
VARIANTS = ["WT", "S1986F", "Q2022P", "Q2022P_S1986F"]
LABELS   = {"WT": "WT", "S1986F": "S1986F", "Q2022P": "Q2022P",
            "Q2022P_S1986F": "Q2022P + S1986F"}

METRIC = os.environ.get("POCKET_METRIC", "front")
COL    = "pocket_front_dist_nm" if METRIC == "front" else "pocket_dfg_dist_nm"
NAME   = "P-loop\u2013hinge" if METRIC == "front" else "gatekeeper\u2013DFG-Phe"
WIN    = 150   # running-mean window (frames)
REP_COLORS = {0: "#0072B2", 1: "#E69F00", 2: "#009E73"}   # Okabe-Ito


def main():
    fm = pd.read_csv(METRICS)
    thr = fm[COL].median() * 10.0   # nm -> A

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharey=True)
    for ax, v in zip(axes.ravel(), VARIANTS):
        s = fm[fm["mutant"] == v]
        for rep, g in s.groupby("replica"):
            g = g.sort_values("frame_index_local")
            x = g["frame_index_local"].values
            y = g[COL].values * 10.0
            openpct = 100.0 * (y >= thr).mean()
            c = REP_COLORS.get(rep, "#555")
            ax.plot(x, y, lw=0.5, alpha=0.16, color=c, zorder=2)
            roll = pd.Series(y).rolling(WIN, center=True, min_periods=1).mean().values
            ax.plot(x, roll, lw=1.7, alpha=0.95, color=c, zorder=3,
                    label=f"replica {rep} \u2014 {openpct:.0f}% open")
        ax.axhline(thr, color="#c00", lw=1.3, ls="--", zorder=5)
        ax.set_title(LABELS[v], fontsize=12, fontweight="bold")
        ax.legend(fontsize=8.5, frameon=False, loc="upper right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#eee", lw=0.8)
        ax.set_axisbelow(True)

    for ax in axes[:, 0]:
        ax.set_ylabel(f"{NAME} distance (\u00c5)", fontsize=10)
    for ax in axes[1, :]:
        ax.set_xlabel("Simulation frame", fontsize=10)

    fig.suptitle(f"ATP-pocket {NAME} distance across the simulation, by replica",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.text(0.5, -0.01,
             "Faint line = per-frame distance; bold line = running mean "
             f"({WIN}-frame window); red dashed = open/closed threshold "
             f"({thr:.1f} \u00c5, the dataset-median distance). Traces that cross "
             "the threshold sample both open and closed states; a trace held on "
             "one side reflects incomplete open\u2013closed exchange within the run.",
             ha="center", va="top", fontsize=8.5, color="#666")

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    out = FIG_DIR / f"openness_timeseries_{METRIC}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"threshold ({NAME}) = {thr:.2f} A")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()