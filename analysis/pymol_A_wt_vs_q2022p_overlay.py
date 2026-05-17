"""
Script A: WT vs Q2022P structural overlay — FIXED for GRO numbering
Chain X, residues 1-292 (GRO offset 1933 from ROS1 numbering)

Usage:
    cd /homes/lkgyammerah/Molecular_Dynamics_analysis/analysis
    pymol -c pymol_A_wt_vs_q2022p_overlay.py
"""

from pymol import cmd
import os

BASE   = '/homes/lkgyammerah/Molecular_Dynamics_analysis'
RECEPT = f'{BASE}/dock/ROS1/q2022p_subset_receptors/actout_ctlfit'
OUTDIR = f'{BASE}/figures/ROS1/ros1_prepared_final'
os.makedirs(OUTDIR, exist_ok=True)

WT_PDB  = f'{RECEPT}/WT/WT_actout_ctlfit_dominant_cluster0_rep.pdb'
Q22_PDB = f'{RECEPT}/Q2022P/Q2022P_actout_ctlfit_dominant_cluster0_rep.pdb'

# ── Load ──────────────────────────────────────────────────────────────────────
cmd.load(WT_PDB,  'WT')
cmd.load(Q22_PDB, 'Q2022P')
print(f"Loaded WT and Q2022P")
print(f"WT states: {cmd.count_states('WT')}, chains: {cmd.get_chains('WT')}")

# ── Align on CTL backbone (GRO resi 100-292, chain X) ─────────────────────────
# CTL = ROS1 2033-2225 = GRO 100-292
rms = cmd.align('Q2022P and chain X and backbone and resi 100-292',
                'WT and chain X and backbone and resi 100-292')
print(f"CTL alignment RMSD: {rms[0]:.3f} A over {rms[1]} atoms")

# ── Representation ────────────────────────────────────────────────────────────
cmd.hide('everything')
cmd.show('cartoon')

# Thesis colours: WT=blue, Q2022P=red
cmd.color('0x4A90C8', 'WT')
cmd.color('0xD85050', 'Q2022P')
cmd.set('cartoon_transparency', 0.2, 'WT')
cmd.set('cartoon_transparency', 0.2, 'Q2022P')

# αC-helix: GRO 42-67 — teal
cmd.color('0x2B7A6E', 'WT and resi 42-67')
cmd.color('0x2B7A6E', 'Q2022P and resi 42-67')

# Hinge: GRO 85-95 — amber
cmd.color('0xC47A00', 'WT and resi 85-95')
cmd.color('0xC47A00', 'Q2022P and resi 85-95')

# P-loop: GRO 2-24 — purple
cmd.color('0x534AB7', 'WT and resi 2-24')
cmd.color('0x534AB7', 'Q2022P and resi 2-24')

# Solvent front: GRO 99-103 — red
cmd.color('0xC0392B', 'WT and resi 99-103')
cmd.color('0xC0392B', 'Q2022P and resi 99-103')

# DFG: GRO 169-171 — red
cmd.color('0xC0392B', 'WT and resi 169-171')
cmd.color('0xC0392B', 'Q2022P and resi 169-171')

# Q2022P mutation site sphere: GRO 89
cmd.show('spheres', 'Q2022P and resi 89 and name CA')
cmd.color('0xB82020', 'Q2022P and resi 89 and name CA')
cmd.set('sphere_scale', 0.7, 'Q2022P and resi 89')

# Also show WT equivalent for comparison
cmd.show('spheres', 'WT and resi 89 and name CA')
cmd.color('0x4A90C8', 'WT and resi 89 and name CA')
cmd.set('sphere_scale', 0.5, 'WT and resi 89')

# ── Rendering ─────────────────────────────────────────────────────────────────
cmd.bg_color('white')
cmd.set('ray_shadows', 0)
cmd.set('ray_opaque_background', 1)
cmd.set('antialias', 2)
cmd.set('cartoon_fancy_helices', 1)
cmd.set('cartoon_smooth_loops', 1)

cmd.orient()
cmd.zoom('all', buffer=8)
cmd.ray(1600, 1200)
cmd.png(f'{OUTDIR}/wt_vs_q2022p_overlay.png', dpi=150)
print(f"Saved: {OUTDIR}/wt_vs_q2022p_overlay.png")

cmd.rotate('y', 90)
cmd.ray(1600, 1200)
cmd.png(f'{OUTDIR}/wt_vs_q2022p_overlay_rotated.png', dpi=150)
print(f"Saved: {OUTDIR}/wt_vs_q2022p_overlay_rotated.png")

cmd.save(f'{OUTDIR}/wt_vs_q2022p_overlay.pse')
print(f"Saved session: {OUTDIR}/wt_vs_q2022p_overlay.pse")
cmd.quit()
