#!/usr/bin/env python3
"""
plot_mutation_motion.py

Answers "which mutation site moves most / least, across all variants and both
drugs" from the records pocket_render.py logs to pocket_pc1_motion.csv.

motion_index = PC1 score span x that residue's max loading = the amplitude of the
residue's motion along its variant's dominant pocket mode (scale-free; comparable
as "motion along each variant's own PC1"). Keeps the last row per
(variant, drug, site) in case panels were re-rendered.

Prints a ranked table and writes a grouped bar plot (1986 vs 2022 per variant,
one panel per drug).

Usage (venv on):
    python analysis/scripts/plot_mutation_motion.py
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
CSV     = RESULTS / "pocket_pc1_motion.csv"

VORDER = ["WT", "S1986F", "Q2022P", "Q2022P_S1986F"]
VLAB   = {"WT": "WT", "S1986F": "S1986F", "Q2022P": "Q2022P",
          "Q2022P_S1986F": "Q2022P\n+ S1986F"}
SITE_C = {1986: "#0072B2", 2022: "#E69F00"}


def main():
    if not CSV.exists():
        raise SystemExit(f"{CSV} not found - render the pocket panels first "
                         f"(the render logs motion there).")
    df = pd.read_csv(CSV)
    # keep the most recent record per variant/drug/site
    df = df.drop_duplicates(subset=["variant", "drug", "site_real"], keep="last")

    print("\n=== mutation-site motion (motion_index = span x loading) ===")
    print("higher = moves more along that variant's dominant pocket mode\n")
    for drug in sorted(df["drug"].unique()):
        d = df[df["drug"] == drug].sort_values("motion_index", ascending=False)
        print(f"[{drug}]")
        for _, r in d.iterrows():
            print(f"   {r['variant']:16s} site {int(r['site_real'])}  "
                  f"loading {r['max_loading']:.3f}  motion {r['motion_index']:6.2f}")
        top = d.iloc[0]; bot = d.iloc[-1]
        print(f"   -> MOST:  {top['variant']} / {int(top['site_real'])} "
              f"({top['motion_index']:.2f})")
        print(f"   -> LEAST: {bot['variant']} / {int(bot['site_real'])} "
              f"({bot['motion_index']:.2f})\n")

    drugs = sorted(df["drug"].unique())
    fig, axes = plt.subplots(1, len(drugs), figsize=(5.2 * len(drugs), 4.4),
                             sharey=True, squeeze=False)
    for ax, drug in zip(axes[0], drugs):
        d = df[df["drug"] == drug]
        x = np.arange(len(VORDER)); w = 0.38
        for k, site in enumerate((1986, 2022)):
            vals = [float(d[(d.variant == v) & (d.site_real == site)]["motion_index"].iloc[0])
                    if len(d[(d.variant == v) & (d.site_real == site)]) else 0.0
                    for v in VORDER]
            ax.bar(x + (k - 0.5) * w, vals, w, color=SITE_C[site], label=f"res {site}")
        ax.set_xticks(x); ax.set_xticklabels([VLAB[v] for v in VORDER], fontsize=9)
        ax.set_title(drug, fontsize=12, fontweight="bold")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.yaxis.grid(True, color="#eee"); ax.set_axisbelow(True); ax.tick_params(length=0)
    axes[0][0].set_ylabel("Mutation-site motion\n(PC1 span \u00d7 loading, a.u.)", fontsize=10)
    axes[0][-1].legend(frameon=False, fontsize=9, title="site")
    fig.suptitle("Mutation-site motion along the dominant pocket mode",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02,
             "Amplitude of each mutation residue's motion along its variant's own "
             "PC1 (scale-free). Higher = moves more; compared within each variant's "
             "dominant mode, not a common axis.",
             ha="center", va="top", fontsize=8, color="#666")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = FIG_DIR / "mutation_site_motion.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()