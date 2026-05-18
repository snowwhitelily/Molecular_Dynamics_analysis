"""
Script B: PC1 loading violin on WT structure using Tsjerk's princomp.py
Loads WT trajectory frames, aligns on CTL, runs Princomp on Ca atoms,
draws density-modulated cylinders coloured blue-white-red by PC1 loading.

Usage (from cluster):
    cd /homes/lkgyammerah/Molecular_Dynamics_analysis
    pymol -c analysis/pymol_B_pc1_violin.py

Output:
    figures/ROS1/ros1_prepared_final/pc1_violin_wt.png
    figures/ROS1/ros1_prepared_final/pc1_violin_wt_rotated.png
"""

import sys
import os
import numpy as np
from pymol import cmd, stored

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = '/homes/lkgyammerah/Molecular_Dynamics_analysis'
SCRIPTS = f'{BASE}/analysis/scripts'
TRAJ    = f'{BASE}/trajectories/ros1_prepared_final'
OUTDIR  = f'{BASE}/figures/ROS1/ros1_prepared_final'
DEBUG   = f'{BASE}/analysis/violin_debug.txt'
os.makedirs(OUTDIR, exist_ok=True)

sys.path.insert(0, SCRIPTS)
from princomp import Princomp
from colorinator import BWR

# ── File-based debug logger ───────────────────────────────────────────────────
open(DEBUG, 'w').close()
def dlog(msg):
    with open(DEBUG, 'a') as f:
        f.write(msg + '\n')

dlog("=== Script B debug log ===")

# ── Load WT trajectory ────────────────────────────────────────────────────────
for rep in ['0', '1', '2']:
    xtc = f'{TRAJ}/WT/{rep}/WT-MD-prot.xtc'
    pdb = f'{TRAJ}/WT/{rep}/WT-MD-prot.pdb'
    if os.path.exists(xtc) and os.path.exists(pdb):
        dlog(f"Loading WT rep{rep}...")
        cmd.load(pdb, 'WT_traj')
        cmd.load_traj(xtc, 'WT_traj', interval=20)
        dlog(f"  rep{rep}: {cmd.count_states('WT_traj')} frames total so far")
    else:
        dlog(f"  rep{rep} not found: {xtc}")

n_frames = cmd.count_states('WT_traj')
dlog(f"Total frames loaded: {n_frames}")

if n_frames < 2:
    dlog("ERROR: Not enough frames loaded.")
    cmd.quit()

# ── Align all frames on CTL Ca (resi 100-292, no chain filter) ───────────────
dlog("Aligning all frames on CTL Ca (resi 100-292)...")
n = cmd.count_states('WT_traj')
for state in range(2, n + 1):
    cmd.fit('WT_traj and name CA and resi 100-292',
            'WT_traj and name CA and resi 100-292',
            mobile_state=state,
            target_state=1)
dlog("Alignment done.")

# ── Run Princomp ──────────────────────────────────────────────────────────────
dlog("Running PCA on Ca atoms...")
P = Princomp('WT_traj and name CA', ncomponents=5)

total_var = P.variances.sum()
dlog(f"PC1 variance explained: {P.variances[0]/total_var*100:.1f}%")
dlog(f"PC2 variance explained: {P.variances[1]/total_var*100:.1f}%")
dlog(f"PC1+PC2 combined:       {(P.variances[0]+P.variances[1])/total_var*100:.1f}%")

# ── Draw PC1 violin ───────────────────────────────────────────────────────────
pc1 = P[1]
dlog(f"PC1 score range (raw): {pc1.scores.min():.4f} to {pc1.scores.max():.4f}")

# Scale scores to ±40 A
scores_norm = pc1.scores / np.abs(pc1.scores).max() * 5.0
pc1_scores_orig = pc1.scores
pc1.scores = scores_norm

# Draw mean structure as grey tube
P.drawmean('pc_mean')
cmd.show('cartoon', 'pc_mean')
cmd.cartoon('tube', 'pc_mean')
cmd.set('cartoon_tube_radius', 0.3, 'pc_mean')
cmd.set('cartoon_color', 'grey70', 'pc_mean')
cmd.set('cartoon_transparency', 0.4, 'pc_mean')
dlog("Mean structure drawn.")

# Draw violin
violin = pc1.violin(radius=0.2, bw=0.8)
violin.recolor(BWR.kde, scores_norm, scores_norm)
violin.draw('pc1_violin')
dlog("Violin drawn.")

# Restore scores
pc1.scores = pc1_scores_orig

# ── Hide trajectory — only mean + violin visible ──────────────────────────────
cmd.hide('everything', 'WT_traj')
dlog("WT_traj hidden.")

# ── Rendering ─────────────────────────────────────────────────────────────────
cmd.bg_color('white')
cmd.set('ray_shadows', 0)
cmd.set('ray_opaque_background', 1)
cmd.set('antialias', 2)
cmd.orient('pc_mean')
cmd.zoom('pc_mean', buffer=5)

cmd.ray(1600, 1200)
cmd.png(f'{OUTDIR}/pc1_violin_wt.png', dpi=150)
dlog(f"Saved: {OUTDIR}/pc1_violin_wt.png")

cmd.rotate('y', 90)
cmd.ray(1600, 1200)
cmd.png(f'{OUTDIR}/pc1_violin_wt_rotated.png', dpi=150)
dlog(f"Saved: {OUTDIR}/pc1_violin_wt_rotated.png")

cmd.save(f'{OUTDIR}/pc1_violin_wt.pse')
dlog(f"Saved session: {OUTDIR}/pc1_violin_wt.pse")
dlog("=== DONE ===")
cmd.quit()