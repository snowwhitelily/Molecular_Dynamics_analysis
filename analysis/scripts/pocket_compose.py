"""pocket_compose.py -- 2x2 grid with titles, legend, dynamic title (venv/matplotlib).
Usage: python pocket_compose.py SELECTION DRUG"""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D

SELECTION = sys.argv[1] if len(sys.argv) > 1 else 'distance'
DRUG      = sys.argv[2] if len(sys.argv) > 2 else 'lorlatinib'
drugcap   = DRUG.capitalize()

BASE      = os.path.join(os.environ['HOME'], 'Molecular_Dynamics_analysis')
PANEL_DIR = os.path.join(BASE, 'figures/ROS1/ros1_prepared_final/pocket_box')
OUT_PNG   = os.path.join(PANEL_DIR, f'pocket_{SELECTION}_{DRUG}.png')

PALETTE = {'WT': '#555555', 'S1986F': '#00A651',
           'Q2022P': '#1976D2', 'Q2022P_S1986F': '#9C27B0'}
LABELS  = {'WT': 'non-mutated (WT)', 'S1986F': 'S1986F', 'Q2022P': 'Q2022P',
           'Q2022P_S1986F': 'Q2022P + S1986F'}
ORDER   = ['WT', 'S1986F', 'Q2022P', 'Q2022P_S1986F']

def panel(v):
    return os.path.join(PANEL_DIR, f'panel_{SELECTION}_{DRUG}_{v}.png')

missing = [v for v in ORDER if not os.path.exists(panel(v))]
if missing:
    raise SystemExit(f'Missing panel(s): {missing}. Render them first.')

fig, axes = plt.subplots(2, 2, figsize=(11.5, 12.3), dpi=150)
fig.patch.set_facecolor('white')
PANEL_LETTER = ['A', 'B', 'C', 'D']
for ax, v, letter in zip(axes.ravel(), ORDER, PANEL_LETTER):
    ax.imshow(mpimg.imread(panel(v)))
    ax.set_title(LABELS[v], fontsize=15, fontweight='bold', color=PALETTE[v], pad=8)
    ax.text(0.01, 0.99, letter, transform=ax.transAxes, fontsize=20, fontweight='bold',
            va='top', ha='left', color='black')
    # mutation-site text labels (fixed positions: camera is shared across panels,
    # so both clusters sit at the same screen spot in every panel and both drugs)
    import matplotlib.patheffects as _pe
    # label (text) and the cluster it points to (arrow tip), axes fractions.
    # 2022 = upper: label high, arrow points DOWN to cluster.
    # 1986 = lower: label low, arrow points UP to cluster.
    _lab = {'2022': dict(txt=(0.82, 0.72), tip=(0.74, 0.60)),
            '1986': dict(txt=(0.86, 0.26), tip=(0.80, 0.40))}
    for _name, _d in _lab.items():
        ax.annotate(_name, xy=_d['tip'], xytext=_d['txt'],
                    xycoords='axes fraction', textcoords='axes fraction',
                    fontsize=13, fontweight='bold', color='black',
                    va='center', ha='left',
                    path_effects=[_pe.withStroke(linewidth=3, foreground='white')],
                    arrowprops=dict(arrowstyle='-|>', color='black', lw=2.0,
                                    mutation_scale=16, shrinkA=4, shrinkB=4))
    ax.axis('off')

legend = [
    Line2D([0], [0], color='#E6C200', lw=6, label=f'{drugcap} (docked pose)'),
    Line2D([0], [0], color='#A0A0A0', lw=4, label='WT reference pocket'),
    Line2D([0], [0], color='#888888', lw=5, label='Mutant pocket (per variant)'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor='#888888',
           markersize=12, label='Mutation sites (1986 / 2022)'),
    Line2D([0], [0], color='#888888', lw=6, label='PC1 mobility (boxplot)'),
]
fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, 0.005))

fig.suptitle(
    f'ATP-binding pocket comparison across WT and mutants \u2014 {drugcap}\n'
    f'WT reference (grey) overlaid with each mutant; PC1 motility as boxplots',
    fontsize=14, fontweight='bold', y=0.995)

plt.subplots_adjust(wspace=0.03, hspace=0.07)
plt.tight_layout(rect=[0, 0.04, 1, 0.955])
plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT_PNG}')