#!/usr/bin/env python3
"""
plot_pocket_openness.py

Pocket-openness panel. Openness = percentage of MD frames in which the ATP pocket
is open: a frame is open when the P-loop-hinge distance >= 16.9 A OR the
gatekeeper-DFG-Phe distance >= 16.2 A (each threshold is the dataset-wide median
of that distance).

Bars = mean of the 3 replicas (each run weighted equally); error bars = +/-1 SD
across replicas; dots = the individual replicas. The frame-weighted POOLED value
is also computed and printed -- it can differ from the replicate mean when
replicas differ in length (e.g. S1986F, whose short replica 2 sat in a closed
pocket: pooling down-weights it to 75%, replicate-mean weights it equally -> 66%).

HEADLINE selects the bar height:
  'replicate_mean' (default) -- mean of the 3 replica %s; matches dots + SD
  'pooled'                   -- frame-weighted over all frames (no SD shown)

Usage (venv on):
    python analysis/scripts/plot_pocket_openness.py
"""
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

METRICS  = RESULTS / "step3a_frame_metrics.csv"
HEADLINE = "replicate_mean"          # or "pooled"

VARIANTS = ["WT", "S1986F", "Q2022P", "Q2022P_S1986F"]
LABELS   = {"WT": "WT", "S1986F": "S1986F", "Q2022P": "Q2022P",
            "Q2022P_S1986F": "Q2022P + S1986F"}
PALETTE  = {"WT": "#0072B2", "S1986F": "#009E73",
            "Q2022P": "#E69F00", "Q2022P_S1986F": "#CC79A7"}


def main():
    fm = pd.read_csv(METRICS)
    rep_mean, rep_sd, pooled, reps = {}, {}, {}, {}
    for v in VARIANTS:
        s = fm[fm["mutant"] == v]
        per = 100.0 * s.groupby("replica")["pocket_open_state"].mean()
        reps[v]     = per.values
        rep_mean[v] = float(per.mean())
        rep_sd[v]   = float(per.std(ddof=1))
        pooled[v]   = 100.0 * s["pocket_open_state"].mean()
        print(f"{v:16s} replicate-mean {rep_mean[v]:5.1f} +/- {rep_sd[v]:4.1f}   "
              f"pooled {pooled[v]:5.1f}   replicas {np.round(reps[v], 1)}")

    head = rep_mean if HEADLINE == "replicate_mean" else pooled

    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    for i, v in enumerate(VARIANTS):
        ax.bar(i, head[v], width=0.62, color=PALETTE[v], edgecolor="none", zorder=2)
        if HEADLINE == "replicate_mean":
            ax.errorbar(i, rep_mean[v], yerr=rep_sd[v], fmt="none",
                        ecolor="#333", elinewidth=1.3, capsize=5, capthick=1.3, zorder=5)
        pr = reps[v]
        jit = np.linspace(-0.13, 0.13, len(pr)) if len(pr) > 1 else [0]
        ax.scatter(i + np.array(jit), pr, s=36, facecolor="white",
                   edgecolor="#333", linewidth=1.1, zorder=6)
        ytop = head[v] + (rep_sd[v] if HEADLINE == "replicate_mean" else 0)
        ax.text(i, min(max(ytop, max(pr)) + 3, 104), f"{head[v]:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                clip_on=False)

    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels([LABELS[v] for v in VARIANTS], fontsize=10)
    ax.set_ylabel("Open-pocket frames (%)", fontsize=11)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.axhline(50, color="#cccccc", lw=0.8, ls="--", zorder=1)
    ax.set_title("ATP-pocket openness across variants", fontsize=12, pad=16)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color="#eeeeee", lw=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    mode_txt = ("bars = mean of 3 replicas, error bars = \u00b11 SD"
                if HEADLINE == "replicate_mean"
                else "bars = frame-weighted pooled value")
    pooled_txt = "  |  ".join(f"{LABELS[v]} {pooled[v]:.0f}%" for v in VARIANTS)
    fig.text(0.5, -0.02,
             f"Percentage of MD frames in which the ATP pocket is open; {mode_txt}; "
             f"dots = individual replicas.\nOpen = P-loop\u2013hinge \u2265 16.9 \u00c5 "
             f"or gatekeeper\u2013DFG-Phe \u2265 16.2 \u00c5 (dataset-median thresholds). "
             f"Frame-weighted pooled: {pooled_txt}.",
             ha="center", va="top", fontsize=7.0, color="#666")

    plt.tight_layout()
    out = FIG_DIR / "pocket_openness.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()