#!/usr/bin/env python3
"""
06l_plot_heatmap_sharedbox.py

Build the active-form docking heatmap from the shared-box results ONLY
(06k outputs). One consistent method for every cell: same box, same params,
same seed. No old per-ligand box, no N/A rescue.

Focus variants: WT, Q2022P, Q2022P_S1986F, S1986F.
Ligands: cabozantinib, ceritinib, crizotinib, lorlatinib, repotrectinib,
         zidesamtinib.

Reads the pdbqt outputs directly; writes a tidy CSV + the heatmap PNG.

Usage (venv on):
    python analysis/06l_plot_heatmap_sharedbox.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path.home() / "Molecular_Dynamics_analysis"
OUT_DIR = BASE / "dock" / "ROS1" / "q2022p_vina" / "vina_outputs_heatmap_sharedbox"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final"
RESULTS.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["WT", "Q2022P", "Q2022P_S1986F", "S1986F"]
LIGANDS = ["ceritinib", "crizotinib", "lorlatinib", "repotrectinib",
           "zidesamtinib", "cabozantinib"]   # cabo last (matches old layout)


def best_affinity(pdbqt):
    with open(pdbqt) as fh:
        for ln in fh:
            if ln.startswith("REMARK VINA RESULT"):
                return float(ln.split()[3])
    return np.nan


def main():
    rows = []
    for v in VARIANTS:
        stem = f"{v}_actout_ctlfit_dominant_cluster0_rep"
        for lig in LIGANDS:
            f = OUT_DIR / f"{stem}__{lig}.pdbqt"
            aff = best_affinity(f) if f.exists() else np.nan
            rows.append(dict(mutant=v, ligand=lig, affinity_kcal_mol=aff))
    df = pd.DataFrame(rows)
    csv = RESULTS / "step6k_heatmap_sharedbox.csv"
    df.to_csv(csv, index=False)
    print(f"wrote {csv}")

    pivot = df.pivot(index="mutant", columns="ligand",
                     values="affinity_kcal_mol")
    pivot = pivot.loc[VARIANTS, LIGANDS]

    fig, ax = plt.subplots(figsize=(max(8, len(LIGANDS) * 1.3), 4.2))
    failed = pivot > 0
    valid = pivot.where(~failed)
    vmin, vmax = valid.min().min(), valid.max().max()
    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(valid.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if failed.iloc[i, j]:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=10,
                        fontweight="bold", color="#555")
            elif np.isnan(val):
                ax.text(j, i, "-", ha="center", va="center", color="grey")
            else:
                nv = (val - vmin) / (vmax - vmin + 1e-9)
                tc = "white" if nv < 0.4 or nv > 0.8 else "#1a1a1a"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=tc)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    if "cabozantinib" in pivot.columns:
        ci = list(pivot.columns).index("cabozantinib")
        ax.add_patch(plt.Rectangle((ci - 0.5, -0.5), 1, len(pivot.index),
                     lw=2.5, edgecolor="#333", facecolor="none", ls="--"))

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Binding affinity (kcal/mol)", fontsize=10)
    ax.set_title("Docking affinities \u2014 active-form receptors "
                 "(shared box, single method)", fontsize=12, pad=12)
    fig.text(0.5, -0.04,
             "All cells docked in one shared box (ATP site + type II crevice), "
             "identical params and seed.",
             ha="center", va="top", fontsize=7.5, color="#555")

    plt.tight_layout()
    out = FIG_DIR / "docking_heatmap_sharedbox.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    print("\nmatrix:")
    print(pivot.round(2).to_string())


if __name__ == "__main__":
    main()