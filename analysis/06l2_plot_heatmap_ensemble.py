#!/usr/bin/env python3
"""
06l2_plot_heatmap_ensemble.py

Active-form docking heatmap, ENSEMBLE version.

Differences from 06l:
  - cells are the ensemble MEDIAN over 16 frames (from step6r_ensemble_summary.csv),
    not a single medoid.
  - colourmap is viridis (colour-blind safe, perceptually uniform) instead of
    RdYlGn_r.
  - the dashed highlight box around the cabozantinib column is REMOVED.
  - label colour is chosen from cell luminance so every number stays readable.

Usage (venv on):
    python analysis/06l2_plot_heatmap_ensemble.py
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

SUMMARY = RESULTS / "step6r_ensemble_summary.csv"

VARIANTS = ["WT", "Q2022P", "Q2022P_S1986F", "S1986F"]
LIGANDS  = ["ceritinib", "crizotinib", "lorlatinib",
            "repotrectinib", "zidesamtinib", "cabozantinib"]   # cabo last
CMAP = "viridis"


def main():
    summ = pd.read_csv(SUMMARY)
    pivot = (summ.pivot(index="variant", columns="ligand", values="median")
             .loc[VARIANTS, LIGANDS])

    fig, ax = plt.subplots(figsize=(max(8, len(LIGANDS) * 1.3), 4.2))
    vmin, vmax = np.nanmin(pivot.values), np.nanmax(pivot.values)
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    norm = plt.Normalize(vmin, vmax)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if np.isnan(val):
                ax.text(j, i, "-", ha="center", va="center", color="grey")
                continue
            r, g, b, _ = cmap(norm(val))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if lum < 0.5 else "#1a1a1a")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    # NOTE: no dashed cabozantinib box — deliberately removed.

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Binding affinity (kcal/mol)", fontsize=10)
    ax.set_title("Docking affinities \u2014 active-form receptors "
                 "(ensemble median, shared box)", fontsize=12, pad=12)
    fig.text(0.5, -0.04,
             "Each cell is the median best-pose affinity over 16 MD frames "
             "(8 open + 8 closed), one shared box, identical params and seed.",
             ha="center", va="top", fontsize=7.5, color="#555")

    plt.tight_layout()
    out = FIG_DIR / "docking_heatmap_ensemble.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    print("\nmedian matrix:")
    print(pivot.round(2).to_string())


if __name__ == "__main__":
    main()