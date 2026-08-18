"""
Plot AutoDock Vina docking results for the paper.

Generates two figures:
    1. All 20 poses per system per ligand — jittered dot plot with best pose bar
       One panel per ligand (5 panels), all variants on x-axis.
    2. Best pose summary heatmap — variants x ligands.
       Failed docking runs (positive affinity) shown as grey with asterisk.

Output:
    figures/ROS1/ros1_prepared_final/docking_all_poses_paper.png
    figures/ROS1/ros1_prepared_final/docking_heatmap_paper.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = '/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1'
FIGS_DIR = '/homes/lkgyammerah/Molecular_Dynamics_analysis/figures/ROS1/ros1_prepared_final'
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Load results ───────────────────────────────────────────────────────────────
all_poses = pd.read_csv(os.path.join(BASE, 'step5c_q2022p_vina_all_poses.csv'))
best      = pd.read_csv(os.path.join(BASE, 'step5c_q2022p_vina_best_by_mutant_ligand.csv'))

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = {
    'WT':            '#1565C0',
    'Q2022P':        '#E63946',
    'Q2022P_S1986F': '#F4511E',
    'Q2022P_S1986Y': '#F9A825',
    'S1986F':        '#2E7D32',
}

VARIANTS = ['WT', 'Q2022P', 'Q2022P_S1986F', 'Q2022P_S1986Y', 'S1986F']
LIGANDS  = ['crizotinib', 'ceritinib', 'lorlatinib', 'repotrectinib', 'zidesamtinib']

LABELS = {
    'WT':            'WT',
    'Q2022P':        'Q2022P',
    'Q2022P_S1986F': 'Q2022P\nS1986F',
    'Q2022P_S1986Y': 'Q2022P\nS1986Y',
    'S1986F':        'S1986F',
}

# ── Figure 1 — All poses jittered dot plot ─────────────────────────────────────
np.random.seed(42)

fig, axes = plt.subplots(1, 5, figsize=(22, 7), dpi=130)
fig.patch.set_facecolor('white')

for ax, lig in zip(axes, LIGANDS):
    sub_all  = all_poses[all_poses['ligand'] == lig]
    sub_best = best[best['ligand'] == lig]

    for i, mutant in enumerate(VARIANTS):
        poses  = sub_all[sub_all['mutant'] == mutant]['affinity_kcal_mol'].values
        colour = PALETTE.get(mutant, '#888888')

        if len(poses) == 0:
            continue

        poses_clipped = np.clip(poses, -15, 5)
        jitter = np.random.uniform(-0.18, 0.18, len(poses))

        ax.scatter(i + jitter, poses_clipped, color=colour,
                   alpha=0.6, s=22, zorder=2, linewidths=0)

        best_val = max(poses.min(), -15)
        ax.plot([i - 0.32, i + 0.32],
                [best_val, best_val],
                color=colour, lw=2.5, zorder=3, solid_capstyle='round')

    wt_row = sub_best[sub_best['mutant'] == 'WT']
    if not wt_row.empty:
        wt_best = wt_row['affinity_kcal_mol'].values[0]
        ax.axhline(wt_best, color='#1565C0', lw=1.2, ls='--',
                   alpha=0.5, label=f'WT best ({wt_best:.2f})')
        ax.legend(fontsize=7, frameon=False)

    ax.set_ylim(5, -15)
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels([LABELS[v] for v in VARIANTS],
                       rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('Predicted binding affinity (kcal/mol)', fontsize=9)
    ax.set_title(lig.capitalize(), fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(
    'AutoDock Vina — All 20 Docking Poses per System\n'
    'Horizontal bar = best pose  |  Dots = all poses (jittered)  |  '
    'Y-axis inverted: upward = stronger predicted binding',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
out1 = os.path.join(FIGS_DIR, 'docking_all_poses_paper.png')
plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out1}')

# ── Figure 2 — Best pose heatmap ───────────────────────────────────────────────
heatmap_data = best.pivot(index='mutant', columns='ligand', values='affinity_kcal_mol')
heatmap_data = heatmap_data.reindex(index=VARIANTS, columns=LIGANDS)

# Mask failed docking runs (positive affinity)
failed_mask = heatmap_data.values > 0
plot_data   = heatmap_data.values.copy()
plot_data[failed_mask] = np.nan

fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
fig.patch.set_facecolor('white')

# Plot valid values
im = ax.imshow(plot_data, cmap='RdYlBu', aspect='auto', vmin=-14, vmax=0)

# Grey out failed cells
failed_overlay = np.zeros((*plot_data.shape, 4))
failed_overlay[failed_mask] = [0.75, 0.75, 0.75, 1.0]
ax.imshow(failed_overlay, aspect='auto')

# Annotate cells
for i in range(len(VARIANTS)):
    for j in range(len(LIGANDS)):
        val = heatmap_data.values[i, j]
        if failed_mask[i, j]:
            # Failed docking — show asterisk and note
            ax.text(j, i, 'N/A*', ha='center', va='center',
                    fontsize=9, fontweight='bold', color='#555555')
        elif not np.isnan(val):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if val < -9 else 'black')

ax.set_xticks(range(len(LIGANDS)))
ax.set_xticklabels([l.capitalize() for l in LIGANDS], fontsize=10)
ax.set_yticks(range(len(VARIANTS)))
ax.set_yticklabels(VARIANTS, fontsize=10)

for i, v in enumerate(VARIANTS):
    ax.get_yticklabels()[i].set_color(PALETTE.get(v, '#333333'))
    ax.get_yticklabels()[i].set_fontweight('bold')

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Best predicted binding affinity (kcal/mol)', fontsize=9)

ax.set_title(
    'Best-pose predicted binding affinities — AutoDock Vina v1.2.5\n'
    'More negative = stronger predicted binding  |  N/A* = docking failed (no valid pose found)',
    fontsize=11, fontweight='bold'
)
plt.tight_layout()
out2 = os.path.join(FIGS_DIR, 'docking_heatmap_paper.png')
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out2}')

print('\nDone. Output files:')
print(f'  {out1}')
print(f'  {out2}')