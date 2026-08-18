#!/usr/bin/env python3
"""
plot_cabozantinib_bars_split.py

Split the combined active-vs-inactive cabozantinib bar chart
into two SEPARATE charts -- one for the active (DFG-in) receptors and one for
the inactive (DFG-out) receptors -- each showing all five variants.

Why splitting is actually the cleaner thing to do: the active arm used the
original ATP-site box; the inactive arm used a larger box that also covers the
type II specificity pocket. In the combined chart those two boxes sat side by
side in every group. Splitting them puts each chart on one consistent box.
The two share a y-axis so they still line up if placed next to each other.

Reads:  results/ROS1/step6c_cabozantinib_best_by_mutant.csv
Writes: figures/ROS1/ros1_prepared_final/cabozantinib_active_only.png
        figures/ROS1/ros1_prepared_final/cabozantinib_inactive_only.png

Usage (venv on):
    python plot_cabozantinib_bars_split.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.expanduser('~/Molecular_Dynamics_analysis')
CSV  = os.environ.get('CABO_CSV',
        os.path.join(BASE, 'results/ROS1/step6c_cabozantinib_best_by_mutant.csv'))
FIGS = os.environ.get('CABO_FIGS',
        os.path.join(BASE, 'figures/ROS1/ros1_prepared_final'))
os.makedirs(FIGS, exist_ok=True)

# same order and palette as the docking figures
VARIANTS = ['WT', 'Q2022P', 'Q2022P_S1986F', 'Q2022P_S1986Y', 'S1986F']
PALETTE = {
    'WT':            '#1565C0',
    'Q2022P':        '#E63946',
    'Q2022P_S1986F': '#F4511E',
    'Q2022P_S1986Y': '#F9A825',
    'S1986F':        '#2E7D32',
}

df = pd.read_csv(CSV)
df = df[df['ligand'] == 'cabozantinib']

# shared y-limit so the two charts are directly comparable side by side.
# axis is inverted (0 at the bottom, stronger/more-negative binding upward),
# matching the original combined chart.
strongest = df['affinity_kcal_mol'].min()
YTOP = min(strongest, -10.0) - 1.2


def draw(conf, hatch, fname, label):
    sub = df[df['conformation'] == conf].set_index('mutant')
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor('white')

    for i, v in enumerate(VARIANTS):
        if v not in sub.index:
            continue
        aff = float(sub.loc[v, 'affinity_kcal_mol'])
        ax.bar(i, aff, width=0.62, color=PALETTE[v], edgecolor='black',
               linewidth=1.1, hatch=hatch, zorder=3)
        # label just inside the tip of the bar, on a small white patch so it
        # stays legible over the diagonal hatching on the inactive chart
        ax.text(i, aff + 0.30, f'{aff:.2f}', ha='center', va='top',
                fontsize=11, fontweight='bold', color='black', zorder=5,
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                          edgecolor='none', alpha=0.85))

    ax.set_ylim(0, YTOP)                      # inverted: 0 bottom, negative up
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels(VARIANTS, fontsize=10)
    ax.set_ylabel('Binding affinity (kcal/mol)', fontsize=11)
    ax.set_title(f'Cabozantinib \u2014 {label} receptor conformation',
                 fontsize=13, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIGS, fname)
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out}')


# active = solid (as before), inactive = hatched (keeps the DFG-out visual cue)
draw('active',   '',    'cabozantinib_active_only.png',   'active (DFG-in)')
draw('inactive', '///', 'cabozantinib_inactive_only.png', 'inactive (DFG-out)')