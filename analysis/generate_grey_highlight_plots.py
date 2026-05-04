#!/usr/bin/env python3
"""
generate_grey_highlight_plots.py
=================================
Generates 39 grey-background highlight plots (1 per system) from the
actout_ctlfit PCA scores. Each plot shows ALL systems in light grey and
the focal system in colour, making its conformational footprint immediately
visible against the full panel.

Run on the cluster (outside the venv is fine — only needs numpy + matplotlib):
    python generate_grey_highlight_plots.py

Expected input files (adjust BASE_DIR if needed):
    actout_ctlfit_scores.npy   — shape (N_total_frames, n_pcs)
    x_owner.npy                — shape (N_total_frames,)  integer 0..38

Output:
    figures/grey_highlight/<name>_highlight.png  (39 files, 150 dpi)
    figures/grey_highlight/panel_all_highlight.png  (6x7 summary panel)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

# ── 1. PATHS ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.expanduser(
    "~/Molecular_Dynamics_analysis/results/ROS1/ros1_prepared_final"
)
SCORES_FILE = os.path.join(BASE_DIR, "actout_ctlfit_scores.npy")
OWNER_FILE  = os.path.join(BASE_DIR, "x_owner.npy")
OUT_DIR     = os.path.join(BASE_DIR, "figures", "grey_highlight")

# ── 2. ORDERED LABEL LIST (index 0..38 matching x_owner integer values) ──────
# Order must match the order used when scores were concatenated in the pipeline.
# Edit this list only if your pipeline used a different ordering.
LABELS = [
    "WT",
    "C2060G",
    "D1988N",
    "D2033N",
    "D2113G",
    "D2113N",
    "E1935G",
    "E1990G",
    "E2020K",
    "E2131Q",
    "F1994L",
    "F2004C",
    "F2004L",
    "F2004V",
    "F2075V",
    "G1957A",
    "G1971E",
    "G2032K",
    "G2032R",
    "G2048A_NC",
    "G2101A",
    "H1999Q_NC",
    "L1947R",
    "L1951R",
    "L1982F",
    "L1982V",
    "L2010M",
    "L2026M",
    "L2053V_NC",
    "L2086F",
    "L2155S",
    "Q2022P",
    "Q2022P_S1986F",
    "Q2022P_S1986Y",
    "R2078K",
    "S1986F",
    "S1986Y",
    "V2089M",
    "V2098I",
]

# ── 3. COLOUR MAP — interesting mutants get distinctive colours ───────────────
# Everyone not listed here gets the default red (#E63946).
COLOUR_MAP = {
    "WT":              "#2196F3",   # blue — reference
    "Q2022P":          "#E63946",   # red
    "Q2022P_S1986F":   "#FF6B35",   # orange
    "Q2022P_S1986Y":   "#FFBE0B",   # amber
    "S1986F":          "#4CAF50",   # green
    "S1986Y":          "#8BC34A",   # light green
    "G2032K":          "#9C27B0",   # purple
    "G2032R":          "#673AB7",   # deep purple
    "L1982F":          "#F06292",   # pink
    "L1982V":          "#EC407A",   # dark pink
    "F2004C":          "#00BCD4",   # cyan
    "F2004L":          "#0097A7",   # dark cyan
    "F2004V":          "#26C6DA",   # light cyan
    "D2113G":          "#FF5722",   # deep orange
    "D2113N":          "#FF7043",   # orange red
    "G2048A_NC":       "#607D8B",   # blue grey (NC)
    "H1999Q_NC":       "#90A4AE",   # light blue grey (NC)
    "L2053V_NC":       "#B0BEC5",   # pale blue grey (NC)
}
DEFAULT_COLOUR = "#E63946"

# ── 4. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading scores and owner arrays...")
if not os.path.exists(SCORES_FILE):
    sys.exit(f"ERROR: Cannot find scores file:\n  {SCORES_FILE}\n"
             "Check BASE_DIR at the top of this script.")
if not os.path.exists(OWNER_FILE):
    sys.exit(f"ERROR: Cannot find owner file:\n  {OWNER_FILE}")

scores = np.load(SCORES_FILE)   # (N, n_pcs)
owner  = np.load(OWNER_FILE)    # (N,)

print(f"  scores shape : {scores.shape}")
print(f"  owner shape  : {owner.shape}")
print(f"  unique owners: {np.unique(owner)}")
print(f"  n labels     : {len(LABELS)}")

pc1 = scores[:, 0]
pc2 = scores[:, 1]

# Axis limits — fixed across all plots so they're directly comparable
x_lo, x_hi = np.percentile(pc1, 0.5), np.percentile(pc1, 99.5)
y_lo, y_hi = np.percentile(pc2, 0.5), np.percentile(pc2, 99.5)
pad = 0.05
x_range = x_hi - x_lo
y_range = y_hi - y_lo
xlim = (x_lo - pad * x_range, x_hi + pad * x_range)
ylim = (y_lo - pad * y_range, y_hi + pad * y_range)

# ── 5. INDIVIDUAL PLOTS ───────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

for i, name in enumerate(LABELS):
    mask_grey = owner != i
    mask_mut  = owner == i

    if mask_mut.sum() == 0:
        print(f"  WARNING: no frames found for index {i} ({name}), skipping.")
        continue

    colour = COLOUR_MAP.get(name, DEFAULT_COLOUR)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    fig.patch.set_facecolor("white")

    # Grey background — all other systems
    ax.scatter(pc1[mask_grey], pc2[mask_grey],
               c="#CCCCCC", s=0.8, alpha=0.25, rasterized=True,
               linewidths=0, label="Other variants")

    # Focal mutant in colour
    ax.scatter(pc1[mask_mut], pc2[mask_mut],
               c=colour, s=1.5, alpha=0.75, rasterized=True,
               linewidths=0, label=name)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("PC1 (actout_ctlfit)", fontsize=9)
    ax.set_ylabel("PC2 (actout_ctlfit)", fontsize=9)
    ax.set_title(f"{name}  (n={mask_mut.sum():,} frames)", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

    # Minimal legend
    handles = [
        plt.scatter([], [], c="#CCCCCC", s=10, alpha=0.5, label="All other variants"),
        plt.scatter([], [], c=colour,    s=10, alpha=0.9, label=name),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right",
              framealpha=0.7, markerscale=2)

    out_path = os.path.join(OUT_DIR, f"{name}_highlight.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [{i+1:02d}/{len(LABELS)}] Saved: {name}_highlight.png")

# ── 6. SUMMARY PANEL (6 columns × 7 rows = 42 slots, 39 used) ─────────────────
print("\nGenerating summary panel...")
n_cols = 6
n_rows = 7  # ceil(39/6) = 7
fig_panel = plt.figure(figsize=(n_cols * 3.5, n_rows * 3.5), dpi=120)
fig_panel.patch.set_facecolor("white")
gs = gridspec.GridSpec(n_rows, n_cols, figure=fig_panel, hspace=0.4, wspace=0.3)

for i, name in enumerate(LABELS):
    row = i // n_cols
    col = i  % n_cols
    ax  = fig_panel.add_subplot(gs[row, col])

    mask_grey = owner != i
    mask_mut  = owner == i
    colour    = COLOUR_MAP.get(name, DEFAULT_COLOUR)

    ax.scatter(pc1[mask_grey], pc2[mask_grey],
               c="#DDDDDD", s=0.3, alpha=0.2, rasterized=True, linewidths=0)
    if mask_mut.sum() > 0:
        ax.scatter(pc1[mask_mut], pc2[mask_mut],
                   c=colour, s=0.8, alpha=0.8, rasterized=True, linewidths=0)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(name, fontsize=7, fontweight="bold", pad=2)
    ax.set_xticks([])
    ax.set_yticks([])

# Turn off empty subplots (slots 39..41)
for j in range(len(LABELS), n_rows * n_cols):
    row = j // n_cols
    col = j  % n_cols
    fig_panel.add_subplot(gs[row, col]).axis("off")

fig_panel.suptitle(
    "actout_ctlfit PCA: Per-Mutant Highlight Plots (all 39 systems)",
    fontsize=13, fontweight="bold", y=1.01
)

panel_path = os.path.join(OUT_DIR, "panel_all_highlight.png")
fig_panel.savefig(panel_path, dpi=120, bbox_inches="tight", facecolor="white")
plt.close(fig_panel)
print(f"  Summary panel saved: panel_all_highlight.png")
print(f"\nAll done. Output directory: {OUT_DIR}")
print(f"Total files: {len(LABELS)} individual + 1 panel = {len(LABELS)+1}")
