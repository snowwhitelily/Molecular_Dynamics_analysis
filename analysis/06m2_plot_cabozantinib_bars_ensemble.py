#!/usr/bin/env python3
"""
06m2_plot_cabozantinib_bars_ensemble.py

Cabozantinib active-vs-inactive bar chart, ENSEMBLE version.

The honest asymmetry, made visible:
  - ACTIVE bar  = ensemble MEDIAN over 16 active MD frames
                  (from step6r_ensemble_summary.csv), with a whisker spanning
                  the frame-to-frame min..max so the sampling is visible.
  - INACTIVE bar = single minimised DFG-out structure (from step6g). There is no
                  inactive-state MD trajectory, so it cannot be ensembled — the
                  flat bar with no whisker shows that plainly.

Colours: Okabe-Ito blue / orange (colour-blind safe; blue matches WT in the PCA
figure so the paper stays consistent).

Usage (venv on):
    python analysis/06m2_plot_cabozantinib_bars_ensemble.py
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

SUMMARY  = RESULTS / "step6r_ensemble_summary.csv"            # active ensemble
CABO_CSV = RESULTS / "step6g_cabozantinib_sharedbox.csv"      # inactive single

VARIANTS = ["WT", "Q2022P", "Q2022P_S1986F", "S1986F"]
C_ACTIVE = "#0072B2"   # Okabe-Ito blue
C_INACT  = "#E69F00"   # Okabe-Ito orange


def main():
    summ = pd.read_csv(SUMMARY)
    cabo = summ[summ["ligand"] == "cabozantinib"].set_index("variant")

    med = [cabo.loc[v, "median"] if v in cabo.index else np.nan for v in VARIANTS]
    lo  = [cabo.loc[v, "min"]    if v in cabo.index else np.nan for v in VARIANTS]
    hi  = [cabo.loc[v, "max"]    if v in cabo.index else np.nan for v in VARIANTS]

    # inactive: single DFG-out structure (step6g)
    ina_df = pd.read_csv(CABO_CSV)
    inact = (ina_df[ina_df["conformation"] == "inactive"]
             .set_index("mutant")["affinity_kcal_mol"])
    inact_vals = [inact.get(v, np.nan) for v in VARIANTS]

    # asymmetric whisker on the active bar: median down to min, up to max
    yerr_lower = [m - l for m, l in zip(med, lo)]   # toward more negative
    yerr_upper = [h - m for m, h in zip(med, hi)]   # toward less negative

    x = np.arange(len(VARIANTS)); w = 0.42
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    ax.bar(x - w/2, med, w, label="Active (DFG-in)",
           color=C_ACTIVE, edgecolor="none")
    ax.errorbar(x - w/2, med, yerr=[yerr_lower, yerr_upper], fmt="none",
                ecolor="#333", elinewidth=1.3, capsize=4, capthick=1.3, zorder=5)
    ax.bar(x + w/2, inact_vals, w, label="Inactive (DFG-out)",
           color=C_INACT, edgecolor="none")

    for xi, v in zip(x - w/2, med):
        if not np.isnan(v):
            ax.text(xi, v - 0.12, f"{v:.2f}", ha="center", va="top",
                    fontsize=8.5, color="#333")
    for xi, v in zip(x + w/2, inact_vals):
        if not np.isnan(v):
            ax.text(xi, v - 0.12, f"{v:.2f}", ha="center", va="top",
                    fontsize=8.5, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS, fontsize=10)
    ax.set_ylabel("Binding affinity (kcal/mol)", fontsize=11)
    ax.set_ylim(-13, 0)
    ax.set_title("Cabozantinib: active vs inactive receptor conformation",
                 fontsize=12, pad=28)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=10, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))

    fig.text(0.5, -0.03,
             "Active = median over 16 MD frames; whisker = frame min..max. "
             "Inactive = single minimised DFG-out structure (no inactive MD "
             "ensemble). Shared box, identical params.",
             ha="center", va="top", fontsize=7.5, color="#666")

    plt.tight_layout()
    out = FIG_DIR / "cabozantinib_active_vs_inactive_ensemble.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    print(f"{'variant':16s} {'act.median':>10s} {'act.min':>8s} {'act.max':>8s} {'inactive':>9s}")
    for v, m, l, h, i in zip(VARIANTS, med, lo, hi, inact_vals):
        print(f"{v:16s} {m:10.2f} {l:8.2f} {h:8.2f} {i:9.2f}")


if __name__ == "__main__":
    main()