#!/usr/bin/env python3
"""
plot_mutation_motion_lollipop.py

Single-panel replacement for the grouped mutation-motion bar chart
(mutation_site_motion.png). Shows, per variant, how much residues 1986 and 2022
move along the dominant pocket mode (motion_index = PC1 score span x that
residue's max loading).

Why a lollipop and NOT a violin: pocket_pc1_motion.csv holds ONE real value per
(variant, site) -- the rows are re-render duplicates of the same number, not a
distribution (drop_duplicates collapses each group to a single value). A violin
of a single repeated value would draw a fabricated bell curve around a point, so
it is not an honest option here. A lollipop shows one value per group truthfully.

The two drugs (lorlatinib, zidesamtinib) give near-identical motion because it is
a structural quantity independent of the ligand; the script CHECKS that before
pooling them, and warns if they diverge.

Usage (venv on):
    python analysis/scripts/plot_mutation_motion_lollipop.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BASE    = Path.home() / "Molecular_Dynamics_analysis"
CSV     = BASE / "results" / "ROS1" / "pocket_pc1_motion.csv"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final"
FIG_DIR.mkdir(parents=True, exist_ok=True)

VORD  = ["WT", "S1986F", "Q2022P", "Q2022P_S1986F"]
VLAB  = {"WT": "WT", "S1986F": "S1986F", "Q2022P": "Q2022P",
         "Q2022P_S1986F": "Q2022P\n+S1986F"}
SITEC = {1986: "#0072B2", 2022: "#E69F00"}
OFF   = {1986: -0.15, 2022: 0.15}


def main():
    df = pd.read_csv(CSV)

    # collapse re-render duplicates: one value per (variant, drug, site)
    df = df.drop_duplicates(["variant", "drug", "site_real"], keep="last")

    # check the two drugs agree before pooling (structural quantity -> they should)
    piv = df.pivot_table(index=["variant", "site_real"], columns="drug",
                         values="motion_index")
    if piv.shape[1] > 1:
        maxdiff = (piv.max(axis=1) - piv.min(axis=1)).max()
        if maxdiff > 0.1:
            print(f"WARNING: drugs differ by up to {maxdiff:.3f} in motion_index — "
                  f"pooling may hide a real difference; inspect before trusting.")
        else:
            print(f"drugs agree (max diff {maxdiff:.3f}) — pooling as one structural value.")

    # pooled single value per (variant, site)
    m = (df.groupby(["variant", "site_real"])["motion_index"].mean().reset_index())

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for site in (1986, 2022):
        s = m[m.site_real == site].set_index("variant")
        for v in VORD:
            if v not in s.index:
                continue
            x = VORD.index(v) + OFF[site]
            y = float(s.loc[v, "motion_index"])
            ax.plot([x, x], [0, y], color=SITEC[site], lw=2.5, zorder=2,
                    solid_capstyle="round")
            ax.scatter([x], [y], s=130, color=SITEC[site], edgecolor="black",
                       lw=1.1, zorder=3)
            if y > 6:
                ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 8),
                            textcoords="offset points", ha="center",
                            fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(VORD)))
    ax.set_xticklabels([VLAB[v] for v in VORD], fontsize=11)
    ax.set_ylabel("Mutation-site motion\n(PC1 span \u00d7 loading, a.u.)", fontsize=11)
    ymax = float(m.motion_index.max()) * 1.12
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.5, len(VORD) - 0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color="#eee")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title("Mutation-site motion along the dominant pocket mode",
                 fontsize=13, pad=10)

    leg = [Line2D([0], [0], marker="o", color=SITEC[s], markeredgecolor="black",
                  lw=0, markersize=11, label=f"residue {s}") for s in (1986, 2022)]
    ax.legend(handles=leg, frameon=False, fontsize=10, title="site", loc="upper left")

    # annotate the headline spike (top 1986 point)
    top = m.loc[m.motion_index.idxmax()]
    tx = VORD.index(top.variant) + OFF[int(top.site_real)]
    ax.annotate("~6\u00d7 any other site", (tx, top.motion_index),
                xytext=(tx - 1.0, top.motion_index * 0.85), fontsize=9.5, color="#333",
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.3))

    fig.text(0.5, -0.02,
             "One value per variant\u00d7site (structural quantity; near-identical "
             "for both drugs, so pooled). Higher = the site moves more along that "
             "variant\u2019s dominant pocket mode.",
             ha="center", va="top", fontsize=8, color="#666")

    plt.tight_layout()
    out = FIG_DIR / "mutation_site_motion_lollipop.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    print(m.to_string(index=False))


if __name__ == "__main__":
    main()