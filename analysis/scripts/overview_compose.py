"""overview_compose.py -- title + colour legend for the overview (venv/matplotlib).
Tightened layout so the enlarged protein fills the page.
Usage: python overview_compose.py DRUG [SELECTION]"""
import os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

DRUG = sys.argv[1] if len(sys.argv) > 1 else 'lorlatinib'
SELECTION = sys.argv[2] if len(sys.argv) > 2 else 'distance'
drugcap = DRUG.capitalize()

BASE = os.path.join(os.environ['HOME'], 'Molecular_Dynamics_analysis')
PANEL_DIR = os.path.join(BASE, 'figures/ROS1/ros1_prepared_final/pocket_box')
IN_PNG = os.path.join(PANEL_DIR, f'overview_{SELECTION}_{DRUG}.png')
OUT_PNG = os.path.join(PANEL_DIR, f'overview_{SELECTION}_{DRUG}_final.png')
if not os.path.exists(IN_PNG):
    raise SystemExit(f'Missing {IN_PNG}; run the overview render first.')

LEGEND = [('P-loop (1951-1959)', '#228B22'), ('aC-helix (1986-2003, E1997)', '#FF8000'),
          ('hinge (2026-2033, gatekeeper L2026)', '#8080FF'),
          ('catalytic loop (2077-2084)', '#0080FF'), ('DFG (2102-2104)', '#BF00BF')]

fig, ax = plt.subplots(figsize=(11, 10), dpi=150); fig.patch.set_facecolor('white')
ax.imshow(mpimg.imread(IN_PNG)); ax.axis('off')
ax.margins(0)

fig.suptitle(f'Structural overview of the ROS1 kinase domain with {drugcap} bound',
             fontsize=16, fontweight='bold', y=0.99)

handles = [Patch(facecolor=c, edgecolor='none', label=n) for n, c in LEGEND]
handles += [Line2D([0], [0], marker='o', color='none', markerfacecolor='hotpink',
                   markersize=12, label='mutation sites (S1986F, Q2022P)'),
            Line2D([0], [0], color='#E6C200', lw=6, label=f'{drugcap} (docked pose)')]
fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=10.5,
           frameon=False, bbox_to_anchor=(0.5, 0.005))

plt.tight_layout(rect=[0, 0.10, 1, 0.965])
plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
print(f'Saved: {OUT_PNG}')