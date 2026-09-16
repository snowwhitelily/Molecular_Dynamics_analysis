#!/usr/bin/env python3
"""
plot_cabozantinib_results_sharedbox.py

CORRECTED version of plot_cabozantinib_results.py. Two changes only:

  1. Reads the shared-box CSV (step6g_cabozantinib_sharedbox.csv) produced by
     06h -- real mutant receptors (06d2) docked with ONE shared box (06g).

  2. Figure B no longer carries the "active and inactive use DIFFERENT boxes,
     compare within conformation only" caveat. That caveat existed because the
     old active/inactive boxes were mismatched. They are now the SAME box, so
     the across-conformation comparison is legitimate; the note states that.

Everything else (palette, order, heatmap N/A handling) is unchanged.

Usage (venv on):
    python analysis/scripts/plot_cabozantinib_results_sharedbox.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CABO_CSV = RESULTS / "step6g_cabozantinib_sharedbox.csv"   # NEW shared-box CSV

PALETTE = {
    "WT":            "#1565C0",
    "Q2022P":        "#E63946",
    "Q2022P_S1986F": "#F4511E",
    "Q2022P_S1986Y": "#F9A825",
    "S1986F":        "#2E7D32",
}
VARIANT_ORDER = ["WT", "Q2022P", "Q2022P_S1986F", "Q2022P_S1986Y", "S1986F"]


def make_heatmap():
    existing_csv = RESULTS / "step5c_q2022p_vina_best_by_mutant_ligand.csv"
    if not existing_csv.exists():
        print(f"WARNING: {existing_csv} not found - skipping heatmap.")
        return None
    if not CABO_CSV.exists():
        print(f"WARNING: {CABO_CSV} not found - run 06h first.")
        return None

    existing = pd.read_csv(existing_csv)
    cabo = pd.read_csv(CABO_CSV)
    cabo_active = cabo[cabo["conformation"] == "active"].copy()

    for df in [existing, cabo_active]:
        if "variant" in df.columns and "mutant" not in df.columns:
            df.rename(columns={"variant": "mutant"}, inplace=True)
        if "best_affinity" in df.columns and "affinity_kcal_mol" not in df.columns:
            df.rename(columns={"best_affinity": "affinity_kcal_mol"}, inplace=True)

    cols_keep = ["mutant", "ligand", "affinity_kcal_mol"]
    combined = pd.concat([
        existing[cols_keep] if all(c in existing.columns for c in cols_keep)
        else existing.iloc[:, :3].set_axis(cols_keep, axis=1),
        cabo_active[cols_keep],
    ], ignore_index=True)

    pivot = combined.pivot_table(index="mutant", columns="ligand",
                                 values="affinity_kcal_mol", aggfunc="min")
    row_order = [v for v in VARIANT_ORDER if v in pivot.index]
    pivot = pivot.loc[row_order]
    cols = [c for c in pivot.columns if c != "cabozantinib"]
    if "cabozantinib" in pivot.columns:
        cols.append("cabozantinib")
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.4), 5))
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
                        fontweight="bold", color="#555555")
            elif np.isnan(val):
                ax.text(j, i, "-", ha="center", va="center", fontsize=10,
                        color="grey")
            else:
                norm_val = (val - vmin) / (vmax - vmin + 1e-9)
                text_col = "white" if norm_val < 0.4 or norm_val > 0.8 else "#1a1a1a"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_col)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    if "cabozantinib" in pivot.columns:
        cabo_idx = list(pivot.columns).index("cabozantinib")
        ax.add_patch(plt.Rectangle((cabo_idx - 0.5, -0.5), 1, len(pivot.index),
                     linewidth=2.5, edgecolor="#333", facecolor="none",
                     linestyle="--"))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Binding affinity (kcal/mol)", fontsize=10)
    ax.set_title("Docking affinities - active-form receptors\n", fontsize=12, pad=12)
    plt.tight_layout()
    out = FIG_DIR / "docking_heatmap_with_cabozantinib_sharedbox.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return out


def make_active_vs_inactive():
    if not CABO_CSV.exists():
        print(f"WARNING: {CABO_CSV} not found - run 06h first.")
        return None

    cabo = pd.read_csv(CABO_CSV)
    active = cabo[cabo["conformation"] == "active"].set_index("mutant")
    inactive = cabo[cabo["conformation"] == "inactive"].set_index("mutant")
    all_variants = [v for v in VARIANT_ORDER
                    if v in active.index or v in inactive.index]
    if not all_variants:
        print("WARNING: no cabozantinib data.")
        return None

    fig, ax = plt.subplots(figsize=(max(7, len(all_variants) * 1.5), 5))
    x = np.arange(len(all_variants)); width = 0.35
    act_vals = [active.loc[v, "affinity_kcal_mol"] if v in active.index else np.nan
                for v in all_variants]
    inact_vals = [inactive.loc[v, "affinity_kcal_mol"] if v in inactive.index else np.nan
                  for v in all_variants]

    bars_a = ax.bar(x - width / 2, act_vals, width, label="Active (DFG-in)",
                    color=[PALETTE.get(v, "#888") for v in all_variants],
                    edgecolor="black", linewidth=0.8)
    bars_i = ax.bar(x + width / 2, inact_vals, width, label="Inactive (DFG-out)",
                    color=[PALETTE.get(v, "#888") for v in all_variants],
                    edgecolor="black", linewidth=0.8, alpha=0.5, hatch="///")

    for bars in [bars_a, bars_i]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.35,
                        f"{h:.2f}", ha="center", va="center", fontsize=8.5,
                        fontweight="bold", color="#1a1a1a")

    ax.set_xticks(x)
    ax.set_xticklabels(all_variants, fontsize=10)
    ax.set_ylabel("Binding affinity (kcal/mol)", fontsize=11)
    ax.set_ylim(-13.0, 0)
    ax.set_title("Cabozantinib: active vs inactive receptor conformation",
                 fontsize=12, pad=10)
    ax.legend(fontsize=10, loc="upper right")
    ax.invert_yaxis()

    # Method note as a small caption UNDER the plot (not a box stamped on the
    # bars). Cleaner for a publication figure; keep the full sentence for the
    # figure legend in the manuscript.
    fig.text(0.5, -0.02,
             "Shared docking box (ATP site + type II crevice), identical size "
             "for all receptors, centred per receptor on the same pocket "
             "residues; active and inactive are directly comparable.",
             ha="center", va="top", fontsize=7, color="#555555")

    plt.tight_layout()
    out = FIG_DIR / "cabozantinib_active_vs_inactive_sharedbox.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    print("Cabozantinib docking figures (shared box)\n")
    print("Figure A: extended heatmap"); make_heatmap()
    print("\nFigure B: active vs inactive"); make_active_vs_inactive()