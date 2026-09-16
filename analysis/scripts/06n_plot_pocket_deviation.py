#!/usr/bin/env python3
"""
06n_plot_pocket_deviation.py

Per-variant backbone-average deviation from WT, computed over the analyzed
body (GRO 6-277 / real 1939-2210), excluding the flexible C-terminal tail --
consistent with the PCA selection. Descriptive figure for coauthors.

Clean style to match the cabozantinib bar chart.

Usage (venv on):
    python analysis/scripts/06n_plot_pocket_deviation.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path.home() / "Molecular_Dynamics_analysis"
PB = BASE / "figures/ROS1/ros1_prepared_final/pocket_box"
FIG_DIR = BASE / "figures/ROS1/ros1_prepared_final"

LO, HI = 6, 277   # BODY: GRO 6-277 (real 1939-2210); tail excluded


def ca(pdb):
    d = {}
    for ln in open(pdb):
        if ln[:4] == "ATOM" and ln[12:16].strip() == "CA":
            d[int(ln[22:26])] = np.array([float(ln[30:38]), float(ln[38:46]),
                                          float(ln[46:54])])
    return d


def main():
    wt = ca(PB / "prep_avg_backbone_WT.pdb")
    variants = ["S1986F", "Q2022P_S1986F", "Q2022P"]     # least -> most
    labels = ["S1986F", "Q2022P\n+ S1986F", "Q2022P"]
    colors = ["#2E7D32", "#9C27B0", "#1976D2"]           # match panel palette

    means, maxes = [], []
    for v in variants:
        m = ca(PB / f"prep_avg_backbone_{v}.pdb")
        common = [r for r in range(LO, HI + 1) if r in wt and r in m]
        disp = np.array([np.linalg.norm(m[r] - wt[r]) for r in common])
        means.append(disp.mean()); maxes.append(disp.max())

    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    ax.bar(x, means, 0.62, color=colors, edgecolor="none", zorder=3)
    # max markers (thin caps) so the spread is visible without overclaiming
    for xi, mx in zip(x, maxes):
        ax.plot([xi - 0.2, xi + 0.2], [mx, mx], color="#555", lw=1.4, zorder=4)
        ax.text(xi + 0.24, mx, f"max {mx:.1f}", va="center", fontsize=8,
                color="#555")
    for xi, mn in zip(x, means):
        ax.text(xi, mn + 0.06, f"{mn:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("CA deviation from WT (\u00c5)", fontsize=11)
    ax.set_title("Backbone-average pocket deviation from WT", fontsize=12, pad=10)
    ax.set_ylim(0, max(maxes) * 1.15)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    fig.text(0.5, -0.03,
             "Mean (bars) and maximum (caps) CA deviation of backbone-average "
             "structures vs WT, over the analyzed body\n(GRO 6-277 / real "
             "1939-2210); flexible C-terminal tail excluded, consistent with "
             "the PCA. Descriptive only.",
             ha="center", va="top", fontsize=7, color="#666")

    plt.tight_layout()
    out = FIG_DIR / "pocket_deviation_from_wt.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    for v, mn, mx in zip(variants, means, maxes):
        print(f"  {v:16s} mean {mn:.2f}  max {mx:.2f}")


if __name__ == "__main__":
    main()