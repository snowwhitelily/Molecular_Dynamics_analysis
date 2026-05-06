#!/usr/bin/env python3
"""
generate_grey_highlight_plots.py
=================================
Uses actout_ctlfit_fel_frame_points.csv — confirmed to include all 38 mutants + WT.
F1994L is correctly absent (excluded from pipeline at analysis stage).

Run:
  python3 generate_grey_highlight_plots.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

BASE_DIR = '/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1'
FP_FILE  = os.path.join(BASE_DIR, 'actout_ctlfit_fel_frame_points.csv')
OUT_DIR  = '/homes/lkgyammerah/Molecular_Dynamics_analysis/figures/ROS1/ros1_prepared_final/grey_highlight'

# 38 mutants + WT (F1994L excluded — not in frame_points)
MUTANTS = [
    'C2060G','D1988N','D2033N','D2113G','D2113N',
    'E1935G','E1990G','E2020K','E2131Q',
    'F2004C','F2004L','F2004V','F2075V',
    'G1957A','G1971E','G2032K','G2032R','G2048A_NC','G2101A',
    'H1999Q_NC','L1947R','L1951R','L1982F','L1982V',
    'L2010M','L2026M','L2053V_NC','L2086F','L2155S',
    'Q2022P','Q2022P_S1986F','Q2022P_S1986Y',
    'R2078K','S1986F','S1986Y','V2089M','V2098I','WT',
]

NC_VARIANTS = {'G2048A_NC','H1999Q_NC','L2053V_NC'}

# All colours chosen to be clearly visible against light grey background
COLOUR_MAP = {
    # Reference
    'WT':               '#1565C0',   # strong blue
    # Q2022P family
    'Q2022P':           '#E63946',   # vivid red
    'Q2022P_S1986F':    '#F4511E',   # deep orange-red
    'Q2022P_S1986Y':    '#F9A825',   # strong amber
    # S1986 series
    'S1986F':           '#2E7D32',   # dark green
    'S1986Y':           '#66BB6A',   # medium green
    # L1982 series
    'L1982F':           '#E91E63',   # vivid pink
    'L1982V':           '#AD1457',   # deep magenta-pink
    # G2032 series
    'G2032K':           '#7B1FA2',   # strong purple
    'G2032R':           '#CE93D8',   # light purple
    # F2004 series
    'F2004C':           '#00838F',   # dark cyan
    'F2004L':           '#00BCD4',   # bright cyan
    'F2004V':           '#80DEEA',   # light cyan
    # D2113 series
    'D2113G':           '#E65100',   # deep orange
    'D2113N':           '#FF8F00',   # amber-orange
    # P-loop arginine pair
    'L1947R':           '#F50057',   # vivid hot pink
    'L1951R':           '#FF4081',   # bright pink
    # Other resistance mutants — all clearly distinct, no grey tones
    'C2060G':           '#6A1B9A',   # deep violet
    'D1988N':           '#00695C',   # dark teal
    'D2033N':           '#43A047',   # medium green
    'E1935G':           '#FB8C00',   # bright orange
    'E1990G':           '#039BE5',   # sky blue
    'E2020K':           '#0277BD',   # dark sky blue
    'E2131Q':           '#283593',   # dark indigo
    'F2075V':           '#FF6F00',   # deep amber
    'G1957A':           '#00897B',   # teal
    'G1971E':           '#00ACC1',   # cyan-teal
    'G2101A':           '#D81B60',   # deep rose
    'L2010M':           '#558B2F',   # olive green
    'L2026M':           '#EF6C00',   # dark orange (gatekeeper)
    'L2086F':           '#4527A0',   # deep indigo
    'L2155S':           '#37474F',   # dark blue-grey (dark enough to see)
    'R2078K':           '#006064',   # dark cyan-teal
    'V2089M':           '#C62828',   # dark red
    'V2098I':           '#1A237E',   # deep navy blue
    # Negative controls — visible but clearly grouped
    'G2048A_NC':        '#5C6BC0',   # medium indigo
    'H1999Q_NC':        '#7E57C2',   # medium purple
    'L2053V_NC':        '#AB47BC',   # medium violet
}

# ── LOAD ───────────────────────────────────────────────────────────────────────
print('Loading frame_points CSV...')
if not os.path.exists(FP_FILE):
    sys.exit(f'ERROR: File not found:\n  {FP_FILE}')

fp = pd.read_csv(FP_FILE)
pc1_all    = fp['PC1'].values
pc2_all    = fp['PC2'].values
mutant_col = fp['mutant'].values
print(f'  {len(fp):,} frames, {fp.mutant.nunique()} systems')

# Fixed axis limits (from all frames including WT)
pad = 0.05
x0,x1 = np.percentile(pc1_all,0.5), np.percentile(pc1_all,99.5)
y0,y1 = np.percentile(pc2_all,0.5), np.percentile(pc2_all,99.5)
xlim = (x0-pad*(x1-x0), x1+pad*(x1-x0))
ylim = (y0-pad*(y1-y0), y1+pad*(y1-y0))
print(f'  PC1: {xlim[0]:.4f} → {xlim[1]:.4f}')
print(f'  PC2: {ylim[0]:.4f} → {ylim[1]:.4f}')

os.makedirs(OUT_DIR, exist_ok=True)

# ── INDIVIDUAL PLOTS ───────────────────────────────────────────────────────────
print(f'\nGenerating {len(MUTANTS)} plots...')
for i, name in enumerate(MUTANTS):
    mask_mut  = mutant_col == name
    mask_grey = ~mask_mut
    colour    = COLOUR_MAP.get(name, '#E63946')
    n_frames  = mask_mut.sum()

    fig, ax = plt.subplots(figsize=(5,5), dpi=150)
    fig.patch.set_facecolor('white')

    # Grey background — very faint
    ax.scatter(pc1_all[mask_grey], pc2_all[mask_grey],
               c='#CCCCCC', s=0.5, alpha=0.15, rasterized=True, linewidths=0)

    # Focal system — bold and clear
    ax.scatter(pc1_all[mask_mut], pc2_all[mask_mut],
               c=colour, s=3, alpha=0.95, rasterized=True, linewidths=0)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel('PC1 (actout_ctlfit)', fontsize=9)
    ax.set_ylabel('PC2 (actout_ctlfit)', fontsize=9)
    ax.tick_params(labelsize=8)
    suffix = ' [NC]' if name in NC_VARIANTS else ''
    ax.set_title(f'{name}{suffix}  (n={n_frames:,})', fontsize=9, fontweight='bold')

    handles = [
        plt.scatter([],[],c='#CCCCCC',s=12,alpha=0.4,label='All other variants'),
        plt.scatter([],[],c=colour,   s=12,alpha=0.95,label=f'{name}{suffix}'),
    ]
    ax.legend(handles=handles, fontsize=7, loc='upper right',
              framealpha=0.7, markerscale=1.5)

    fig.savefig(os.path.join(OUT_DIR, f'{name}_highlight.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [{i+1:02d}/{len(MUTANTS)}] {name:25s} {n_frames:,} frames  ✓')

# ── SUMMARY PANEL 6×7 ─────────────────────────────────────────────────────────
print('\nGenerating summary panel...')
n_cols, n_rows = 6, 7
fig_p = plt.figure(figsize=(n_cols*3.2, n_rows*3.2), dpi=110)
fig_p.patch.set_facecolor('white')
gs = gridspec.GridSpec(n_rows, n_cols, figure=fig_p, hspace=0.45, wspace=0.3)

for i, name in enumerate(MUTANTS):
    ax = fig_p.add_subplot(gs[i//n_cols, i%n_cols])
    mask_mut  = mutant_col == name
    mask_grey = ~mask_mut
    colour    = COLOUR_MAP.get(name, '#E63946')

    ax.scatter(pc1_all[mask_grey], pc2_all[mask_grey],
               c='#DDDDDD', s=0.15, alpha=0.12, rasterized=True, linewidths=0)
    ax.scatter(pc1_all[mask_mut], pc2_all[mask_mut],
               c=colour, s=1.5, alpha=0.9, rasterized=True, linewidths=0)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xticks([]); ax.set_yticks([])
    suffix = ' NC' if name in NC_VARIANTS else ''
    ax.set_title(f'{name}{suffix}', fontsize=6.5, fontweight='bold', pad=2)

for j in range(len(MUTANTS), n_rows*n_cols):
    fig_p.add_subplot(gs[j//n_cols, j%n_cols]).axis('off')

fig_p.suptitle(
    'actout_ctlfit PCA — Per-Mutant Highlight Plots (38 mutants + WT)\n'
    'NC = negative control   F1994L excluded from comparative analysis',
    fontsize=11, fontweight='bold', y=1.01)

fig_p.savefig(os.path.join(OUT_DIR, 'panel_all_highlight.png'),
              dpi=110, bbox_inches='tight', facecolor='white')
plt.close(fig_p)

print(f'\nDone. Output: {OUT_DIR}')
print(f'Files: {len(MUTANTS)} individual PNGs + panel_all_highlight.png')
