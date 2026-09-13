#!/usr/bin/env python3
"""
plot_mutation_motion_violin.py

REAL violins of per-frame mutation-site motion, from pocket_pc1_perframe.csv
(written by pocket_render.py using the SAME whole-pocket PCA that produces the
figures and the reported motion_index). Each frame contributes
motion = |PC1 score(frame)| x site loading, so the distribution's spread
reproduces the reported motion_index (= score_span x loading) while the median
shows the typical per-frame value.

Headline it should show: residue 1986 in the double mutant towers over every
other site (median ~4.6 vs ~0.2-0.5), matching the paper.

Usage (venv on):
    python analysis/scripts/plot_mutation_motion_violin.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

BASE    = Path.home() / "Molecular_Dynamics_analysis"
CSV     = BASE / "results" / "ROS1" / "pocket_pc1_perframe.csv"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final"
FIG_DIR.mkdir(parents=True, exist_ok=True)

VORD  = ["WT", "S1986F", "Q2022P", "Q2022P_S1986F"]
VLAB  = {"WT": "WT", "S1986F": "S1986F", "Q2022P": "Q2022P",
         "Q2022P_S1986F": "Q2022P\n+ S1986F"}
SITEC = {1986: "#0072B2", 2022: "#E69F00"}
OFF   = {1986: -0.2, 2022: 0.2}


def main():
    if not CSV.exists():
        raise SystemExit(f"{CSV} not found -- run pocket_render.py first "
                         f"(it writes the per-frame motion there).")
    df = pd.read_csv(CSV)
    df = (df.groupby(["variant", "site_real", "frame"])["motion"]
            .mean().reset_index())

    import os
    LOG = os.environ.get("LOG", "0") != "0"

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    printed = []
    for site in (1986, 2022):
        for vi, v in enumerate(VORD):
            vals = df[(df.variant == v) & (df.site_real == site)]["motion"].values
            if len(vals) == 0:
                continue
            if LOG:
                vals = vals[vals > 0]
                vals = np.clip(vals, 0.01, None)
            pos = vi + OFF[site]
            vp = ax.violinplot([vals], positions=[pos], widths=0.34,
                               showmeans=False, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(SITEC[site]); b.set_alpha(0.5)
                b.set_edgecolor(SITEC[site]); b.set_linewidth(1.0)
            med = float(np.median(vals))
            ax.plot([pos - 0.15, pos + 0.15], [med, med], color="black",
                    lw=1.8, zorder=5)
            printed.append((v, site, med, len(vals)))

    ax.set_xticks(range(len(VORD)))
    ax.set_xticklabels([VLAB[v] for v in VORD], fontsize=11)
    ax.set_ylabel("Per-frame mutation-site motion\n"
                  "(|PC1 score| \u00d7 loading, a.u.)", fontsize=11)
    ax.set_xlim(-0.5, len(VORD) - 0.5)
    if LOG:
        ax.set_yscale("log")
        ax.set_ylim(0.01, 30)
    else:
        ax.set_ylim(bottom=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color="#eee"); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.set_title("Mutation-site motion along the dominant pocket mode",
                 fontsize=13, pad=10)
    ax.legend(handles=[Patch(facecolor=SITEC[s], alpha=0.55, label=f"residue {s}")
                       for s in (1986, 2022)],
              frameon=False, fontsize=10, title="site", loc="upper left")

    top = max(printed, key=lambda r: r[2])
    if (not LOG) and top[0] == "Q2022P_S1986F" and top[1] == 1986:
        tx = VORD.index("Q2022P_S1986F") + OFF[1986]
        ax.annotate("1986 dominates\nin the double mutant",
                    xy=(tx, top[2]), xytext=(tx - 1.15, top[2] + 1.5),
                    fontsize=9.5, color="#333", ha="left",
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.3))

    fig.text(0.5, -0.02,
             "Each violin = per-frame |PC1 score| \u00d7 site loading (whole-pocket "
             "PCA, 3 replicas pooled). Black tick = median. Higher = the site moves "
             "more along the variant\u2019s dominant pocket mode.",
             ha="center", va="top", fontsize=8, color="#666")

    plt.tight_layout()
    out = FIG_DIR / ("mutation_site_motion_violin_log.png" if LOG
                     else "mutation_site_motion_violin.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}\n")
    print(f"{'variant':16s} {'site':>5s} {'median':>8s} {'n_frames':>9s}")
    for v, site, med, n in sorted(printed):
        print(f"{v:16s} {site:5d} {med:8.2f} {n:9d}")


if __name__ == "__main__":
    main()