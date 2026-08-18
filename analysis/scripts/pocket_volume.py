"""
pocket_volume.py -- put a NUMBER on the pocket difference between variants, so
S1986F / Q2022P / the double mutant can be told apart even when the eye can't.

For each variant it averages the pocket Ca positions (each frame aligned to the
WT N-lobe, same frame as the figures), then reports:
  * the convex-hull volume of those Ca  -> "how wide is the pocket"
  * the mean Ca shift from WT           -> "how much did the pocket rearrange"
and draws a two-panel bar chart. Runs in the venv (MDAnalysis + scipy).

Usage (venv on):  python pocket_volume.py SELECTION DRUG
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.environ['HOME'], 'Molecular_Dynamics_analysis', 'analysis'))
import pocket_lib as PL
import MDAnalysis as mda
from MDAnalysis.analysis import align as _align
from scipy.spatial import ConvexHull
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SELECTION = sys.argv[1] if len(sys.argv) > 1 else 'distance'
DRUG      = sys.argv[2] if len(sys.argv) > 2 else 'lorlatinib'

BASE = os.path.join(os.environ['HOME'], 'Molecular_Dynamics_analysis')
TRAJ = os.path.join(BASE, 'trajectories/ros1_prepared_final')
FIGS = os.path.join(BASE, 'figures/ROS1/ros1_prepared_final/pocket_box')
VARIANTS = ['WT', 'S1986F', 'Q2022P', 'Q2022P_S1986F']
LABELS = {'WT': 'WT', 'S1986F': 'S1986F', 'Q2022P': 'Q2022P', 'Q2022P_S1986F': 'Q2022P+S1986F'}
COLOR  = {'WT': '#A0A0A0', 'S1986F': '#00A651', 'Q2022P': '#1976D2', 'Q2022P_S1986F': '#9C27B0'}

resids = [r for r in open(os.path.join(
    FIGS, f'prep_{SELECTION}_{DRUG}_resids.txt')).read().strip().split(',') if r.strip()]
sel = '(resid ' + ' '.join(resids) + ') and name CA'

wt = PL.ntl_reference(TRAJ, 'WT')

def mean_ca(variant):
    csum, n, wu = None, 0, None
    for top, trj in PL.replica_paths(TRAJ, variant):
        u = mda.Universe(top, trj); wu = u; ag = u.select_atoms(sel)
        for _ in u.trajectory:
            _align.alignto(u, wt, select=PL.NTL_SEL, weights='mass')
            p = ag.positions
            csum = p.copy() if csum is None else csum + p; n += 1
    wu.trajectory[0]; ag = wu.select_atoms(sel)
    return ag.resids.copy(), csum / n

data = {v: mean_ca(v) for v in VARIANTS}
wt_rids, wt_mean = data['WT']
wt_map = {int(r): c for r, c in zip(wt_rids, wt_mean)}

vol, disp = {}, {}
for v in VARIANTS:
    rids, mean = data[v]
    vol[v] = ConvexHull(mean).volume
    d = [np.linalg.norm(c - wt_map[int(r)]) for r, c in zip(rids, mean) if int(r) in wt_map]
    disp[v] = float(np.mean(d))

wtv = vol['WT']
print(f'\nATP pocket size ({SELECTION}, {DRUG}) -- convex hull of {len(resids)} pocket Ca')
print(f'{"variant":16s} {"hull vol (A^3)":16s} {"vs WT":8s} {"mean Ca shift vs WT (A)"}')
for v in VARIANTS:
    print(f'{LABELS[v]:16s} {vol[v]:10.1f}       {100*(vol[v]-wtv)/wtv:+6.1f}%   {disp[v]:.3f}')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150); fig.patch.set_facecolor('white')
xs = [LABELS[v] for v in VARIANTS]; cols = [COLOR[v] for v in VARIANTS]
axes[0].bar(xs, [vol[v] for v in VARIANTS], color=cols, edgecolor='black', linewidth=0.6)
axes[0].axhline(wtv, color='grey', ls='--', lw=1)
axes[0].set_ylabel('pocket hull volume (A$^3$)'); axes[0].set_title('ATP-pocket size per variant')
axes[1].bar(xs, [disp[v] for v in VARIANTS], color=cols, edgecolor='black', linewidth=0.6)
axes[1].set_ylabel('mean C$\\alpha$ shift from WT (A)'); axes[1].set_title('Pocket rearrangement vs WT')
for ax in axes:
    ax.tick_params(axis='x', rotation=20)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.suptitle(f'ATP-binding pocket, {DRUG.capitalize()} \u2014 size and shift per variant',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(FIGS, f'pocket_volume_{SELECTION}_{DRUG}.png')
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
print(f'\nSaved: {out}')

csv_out = os.path.join(FIGS, f'pocket_volume_{SELECTION}_{DRUG}.csv')
with open(csv_out, 'w') as fh:
    fh.write('variant,hull_volume_A3,vs_WT_percent,mean_CA_shift_from_WT_A\n')
    for v in VARIANTS:
        fh.write(f'{LABELS[v]},{vol[v]:.1f},{100*(vol[v]-wtv)/wtv:+.1f},{disp[v]:.3f}\n')
print(f'Saved: {csv_out}')