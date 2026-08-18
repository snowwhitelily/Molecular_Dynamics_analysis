#!/usr/bin/env python3
"""
plot_cabozantinib_results.py

Generate two figures from the cabozantinib docking results:

  Figure A: Active-form cabozantinib as a sixth column alongside the existing
            five-ligand heatmap. Same box, same method, directly comparable.

  Figure B: Active-vs-inactive comparison for cabozantinib across all variants.
            Different boxes (active box vs type II box), so this is a separate
            figure with a clear methods note.

Reads:
    results/ROS1/step5c_q2022p_vina_best_by_mutant_ligand.csv   (UPDATED: Correct baseline file)
    results/ROS1/step6c_cabozantinib_best_by_mutant.csv

Outputs:
    figures/ROS1/ros1_prepared_final/docking_heatmap_with_cabozantinib.png
    figures/ROS1/ros1_prepared_final/cabozantinib_active_vs_inactive.png

Usage (venv on):
    python analysis/plot_cabozantinib_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Palette from the handover
PALETTE = {
    "WT":            "#1565C0",
    "Q2022P":        "#E63946",
    "Q2022P_S1986F": "#F4511E",
    "Q2022P_S1986Y": "#F9A825",
    "S1986F":        "#2E7D32",
}

VARIANT_ORDER = ["WT", "Q2022P", "Q2022P_S1986F", "Q2022P_S1986Y", "S1986F"]


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE A: Extended heatmap (6 ligands x 5 variants, all active-form)
# ═══════════════════════════════════════════════════════════════════════════

def make_heatmap():
    """Add cabozantinib as a sixth column to the existing docking heatmap."""

    # Load the existing five-ligand docking results
    existing_csv = RESULTS / "step5c_q2022p_vina_best_by_mutant_ligand.csv"
    if not existing_csv.exists():
        print(f"WARNING: {existing_csv} not found — cannot build extended heatmap.")
        print("  The existing 5-ligand CSV is needed.")
        return None

    existing = pd.read_csv(existing_csv)

    # Load cabozantinib active results
    cabo_csv = RESULTS / "step6c_cabozantinib_best_by_mutant.csv"
    if not cabo_csv.exists():
        print(f"WARNING: {cabo_csv} not found — run 06c first.")
        return None

    cabo = pd.read_csv(cabo_csv)
    cabo_active = cabo[cabo["conformation"] == "active"].copy()

    # Normalize columns across both files
    for df in [existing, cabo_active]:
        if "variant" in df.columns and "mutant" not in df.columns:
            df.rename(columns={"variant": "mutant"}, inplace=True)
        if "best_affinity" in df.columns and "affinity_kcal_mol" not in df.columns:
            df.rename(columns={"best_affinity": "affinity_kcal_mol"}, inplace=True)

    # Merge
    cols_keep = ["mutant", "ligand", "affinity_kcal_mol"]
    combined = pd.concat([
        existing[cols_keep] if all(c in existing.columns for c in cols_keep)
        else existing.iloc[:, :3].set_axis(cols_keep, axis=1),
        cabo_active[cols_keep],
    ], ignore_index=True)

    # Pivot to matrix
    pivot = combined.pivot_table(
        index="mutant", columns="ligand", values="affinity_kcal_mol",
        aggfunc="min"  # best (most negative) affinity
    )

    # Reorder rows
    row_order = [v for v in VARIANT_ORDER if v in pivot.index]
    pivot = pivot.loc[row_order]

    # Ensure cabozantinib is the last column
    cols = [c for c in pivot.columns if c != "cabozantinib"]
    if "cabozantinib" in pivot.columns:
        cols.append("cabozantinib")
    pivot = pivot[cols]

    # Plot
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.4), 5))

    # A positive affinity means the docking failed to find a valid pose, not a
    # real (weak) binding energy. Treat those as N/A: keep them out of the
    # colour scale (so one +18 doesn't blow out the whole range) and label N/A.
    failed = pivot > 0
    valid = pivot.where(~failed)              # positives -> NaN

    vmin = valid.min().min()
    vmax = valid.max().max()
    cmap = plt.cm.RdYlGn_r.copy()             # red = strong binding, green = weak
    cmap.set_bad("#d9d9d9")                    # grey for N/A cells

    im = ax.imshow(valid.values, cmap=cmap, aspect="auto",
                   vmin=vmin, vmax=vmax)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if failed.iloc[i, j]:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=10,
                        fontweight="bold", color="#555555")
            elif np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10,
                        color="grey")
            else:
                # Choose text colour for contrast
                norm_val = (val - vmin) / (vmax - vmin + 1e-9)
                text_col = "white" if norm_val < 0.4 or norm_val > 0.8 else "#1a1a1a"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_col)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    # Highlight the cabozantinib column with a border
    if "cabozantinib" in pivot.columns:
        cabo_idx = list(pivot.columns).index("cabozantinib")
        rect = plt.Rectangle(
            (cabo_idx - 0.5, -0.5), 1, len(pivot.index),
            linewidth=2.5, edgecolor="#333", facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Binding affinity (kcal/mol)", fontsize=10)

    ax.set_title("Docking affinities — active-form receptors\n",
                 fontsize=12, pad=12)

    plt.tight_layout()
    out = FIG_DIR / "docking_heatmap_with_cabozantinib.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE B: Active vs inactive comparison (cabozantinib only)
# ═══════════════════════════════════════════════════════════════════════════

def make_active_vs_inactive():
    """Grouped bar chart comparing active and inactive cabozantinib affinities."""

    cabo_csv = RESULTS / "step6c_cabozantinib_best_by_mutant.csv"
    if not cabo_csv.exists():
        print(f"WARNING: {cabo_csv} not found — run 06c first.")
        return None

    cabo = pd.read_csv(cabo_csv)
    active = cabo[cabo["conformation"] == "active"].set_index("mutant")
    inactive = cabo[cabo["conformation"] == "inactive"].set_index("mutant")

    # Only include variants that have at least one value
    all_variants = [v for v in VARIANT_ORDER
                    if v in active.index or v in inactive.index]

    if not all_variants:
        print("WARNING: no cabozantinib data found.")
        return None

    fig, ax = plt.subplots(figsize=(max(7, len(all_variants) * 1.5), 5))

    x = np.arange(len(all_variants))
    width = 0.35

    act_vals = [active.loc[v, "affinity_kcal_mol"]
                if v in active.index else np.nan for v in all_variants]
    inact_vals = [inactive.loc[v, "affinity_kcal_mol"]
                  if v in inactive.index else np.nan for v in all_variants]

    bars_a = ax.bar(x - width / 2, act_vals, width, label="Active (DFG-in)",
                    color=[PALETTE.get(v, "#888") for v in all_variants],
                    edgecolor="black", linewidth=0.8)
    bars_i = ax.bar(x + width / 2, inact_vals, width, label="Inactive (DFG-out)",
                    color=[PALETTE.get(v, "#888") for v in all_variants],
                    edgecolor="black", linewidth=0.8, alpha=0.5,
                    hatch="///")

    # Value labels on bars (FIXED: Dark text + adjusted positioning inside inverted axis)
    for bars in [bars_a, bars_i]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                # Using h + 0.35 pushes the label down, keeping it cleanly inside the bar
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.35,
                        f"{h:.2f}", ha="center", va="center", fontsize=8.5,
                        fontweight="bold", color="#1a1a1a")

    ax.set_xticks(x)
    ax.set_xticklabels(all_variants, fontsize=10)
    ax.set_ylabel("Binding affinity (kcal/mol)", fontsize=11)

    # Expand vertical limit slightly so labels at -10 kcal/mol don't hit the border
    ax.set_ylim(-13.0, 0)

    ax.set_title("Cabozantinib: active vs inactive receptor conformation",
                 fontsize=12, pad=10)
    ax.legend(fontsize=10, loc="upper right")

    # Invert y-axis so more negative (stronger) is taller
    ax.invert_yaxis()

    # Move the note box up top out of the way of the bars
    ax.annotate(
        "Note: active and inactive use different docking boxes.\n"
        "Active = ATP-site box. Inactive = type II pocket box.\n"
        "Compare within conformation, not across.",
        xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=7, color="#333333", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4",
                  edgecolor="grey", alpha=0.9),
    )

    plt.tight_layout()
    out = FIG_DIR / "cabozantinib_active_vs_inactive.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Cabozantinib docking figures")
    print("=" * 60)
    print()

    print("Figure A: Extended heatmap")
    print("-" * 40)
    make_heatmap()

    print()
    print("Figure B: Active vs inactive comparison")
    print("-" * 40)
    make_active_vs_inactive()