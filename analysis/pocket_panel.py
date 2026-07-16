"""
pocket_panel.py -- render ONE variant's ATP pocket panel for Task 10.

Uses Tsjerk's pca.py (princomp / drawmean / drawcomp) to compute the pocket PCA
fresh from the trajectories and draw the PC1 eigenvector as a boxplot, matching
Christa's Figure 2. The drug pose is frame corrected from the ctlfit docking
frame into the NTL analysis frame by pocket_lib and shown as a fixed reference.

Run one variant at a time, headless:
    pymol -cq analysis/pocket_panel.py -- VARIANT SELECTION DRUG
      VARIANT   = WT | S1986F | Q2022P | Q2022P_S1986F
      SELECTION = motif | distance
      DRUG      = lorlatinib | zidesamtinib
"""

import os
import sys
import numpy as np

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
if len(argv) < 3:
    raise SystemExit('usage: pocket_panel.py -- VARIANT SELECTION DRUG')
VARIANT, SELECTION, DRUG = argv[0], argv[1], argv[2]

SCALE = float(os.environ.get('POCKET_SCALE', '1.0'))

BASE     = '/homes/lkgyammerah/Molecular_Dynamics_analysis'
TRAJ_DIR = os.path.join(BASE, 'trajectories/ros1_prepared_final')
DOCK_OUT = os.path.join(BASE, 'dock/ROS1/q2022p_vina/vina_outputs')
RECEPTOR = os.path.join(BASE, 'dock/ROS1/q2022p_subset_receptors/actout_ctlfit/WT/'
                              'WT_actout_ctlfit_dominant_cluster0_rep.pdb')
FIGS_DIR = os.path.join(BASE, 'figures/ROS1/ros1_prepared_final/pocket_box')
os.makedirs(FIGS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE, 'analysis', 'scripts'))
sys.path.insert(0, os.path.join(BASE, 'analysis'))
import pocket_lib as PL

from pymol import cmd

def hex_rgb(h):
    h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

PALETTE = {
    'WT':            hex_rgb('#1565C0'),
    'S1986F':        hex_rgb('#2E7D32'),
    'Q2022P':        hex_rgb('#E63946'),
    'Q2022P_S1986F': hex_rgb('#F4511E'),
}
COLOUR = PALETTE[VARIANT]

wt_ref  = PL.ntl_reference(TRAJ_DIR, 'WT')
pose    = PL.read_pdbqt_pose1(os.path.join(
    DOCK_OUT, f'WT_actout_ctlfit_dominant_cluster0_rep__{DRUG}.pdbqt'))
drug_ntl, rmsd = PL.correct_drug_to_ntl(pose, RECEPTOR, wt_ref)
print(f'drug {DRUG}: frame-correction receptor RMSD {rmsd:.3f} A, '
      f'{len(drug_ntl)} atoms placed')

if SELECTION == 'motif':
    resids = PL.pocket_resids('motif')
elif SELECTION == 'distance':
    env = os.environ.get('POCKET_RESIDS')
    if env:
        resids = [int(x) for x in env.split(',')]
    else:
        resids = PL.pocket_resids('distance', wt_ref, drug_ntl, cutoff=5.0)
    print(f'distance pocket: {len(resids)} residues (GRO) {resids}')
else:
    raise SystemExit("SELECTION must be motif or distance")

resid_sel = 'resid ' + '+'.join(str(r) for r in resids) + ' and name CA'

cmd.reinitialize()
cmd.bg_color('white')
cmd.set('ray_opaque_background', 1)
cmd.set('orthoscopic', 1)
cmd.set('ray_shadows', 0)

rep_objs = []
for i, (top, trj) in enumerate(PL.replica_paths(TRAJ_DIR, VARIANT)):
    name = f'{VARIANT}_r{i}'
    cmd.load(top, name)
    cmd.load_traj(trj, name)
    cmd.intra_fit(f'{name} and ({PL.NTL_SEL})')
    rep_objs.append(name)

pca_sel = '(' + ' or '.join(rep_objs) + f') and ({resid_sel})'

import pca
import time as _time, ast as _ast
pca.time = _time
pca.ast = _ast
pca.cmd = cmd
name = f'pc_{VARIANT}'
pca.princomp(pca_sel, name=name, maxvec=2, states=(0,))
pca.drawcomp(name, comp=0, draw='boxplot', radius=0.25, scale=SCALE, color=COLOUR)
pca.drawmean(name)

mean_obj = f'{name}_mean'
cmd.hide('everything', mean_obj)
cmd.show('spheres', mean_obj)
cmd.set('sphere_scale', 0.35, mean_obj)
cmd.set_color(f'c_{VARIANT}', list(COLOUR))
cmd.color(f'c_{VARIANT}', mean_obj)

drug_pdb = os.path.join(FIGS_DIR, f'_drug_{DRUG}.pdb')
with open(drug_pdb, 'w') as fh:
    for i, (x, y, z) in enumerate(drug_ntl, 1):
        fh.write(f'HETATM{i:>5}  C   LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}'
                 f'  1.00  0.00           C\n')
    fh.write('END\n')
cmd.load(drug_pdb, 'drug')
cmd.hide('everything', 'drug')
cmd.show('sticks', 'drug')
cmd.color('yellow', 'drug')
cmd.set('stick_radius', 0.18, 'drug')

for r in rep_objs:
    cmd.disable(r)
cmd.orient(f'{mean_obj} or drug')
cmd.zoom(f'{mean_obj} or drug', buffer=4.0)
out = os.path.join(FIGS_DIR, f'panel_{SELECTION}_{DRUG}_{VARIANT}.png')
cmd.png(out, width=1200, height=1200, dpi=300, ray=1)
print(f'saved {out}')
