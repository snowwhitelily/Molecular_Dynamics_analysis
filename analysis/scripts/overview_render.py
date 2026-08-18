"""overview_render.py -- whole ROS1 kinase domain with the drug in the pocket,
face-on into the cleft. Enlarged/zoomed to fill the frame; thicker motif cartoon;
slightly larger ligand and mutation spheres. Naming is done in overview_compose.py."""
import os, sys
import numpy as np
from pymol import cmd, util

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
DRUG = argv[0]; SELECTION = argv[1] if len(argv) > 1 else 'distance'

BASE = '/homes/lkgyammerah/Molecular_Dynamics_analysis'
FIGS = os.path.join(BASE, 'figures/ROS1/ros1_prepared_final/pocket_box')
domain_pdb = os.path.join(FIGS, 'prep_avg_backbone_WT.pdb')
drug_pdb   = os.path.join(FIGS, f'prep_{SELECTION}_{DRUG}_drug.pdb')
out_png    = os.path.join(FIGS, f'overview_{SELECTION}_{DRUG}.png')

MOTIFS = [(range(18, 27), 'forest'), (range(53, 71), 'orange'), (range(93, 101), 'slate'),
          (range(144, 152), 'marine'), (range(169, 172), 'purple')]
MUT = '53+89'

cmd.reinitialize(); cmd.bg_color('white')
cmd.set('ray_opaque_background', 1); cmd.set('orthoscopic', 1); cmd.set('ray_shadows', 0)
# thicker cartoon so the coloured motifs stand out
cmd.set('cartoon_rect_width', 1.6); cmd.set('cartoon_oval_width', 0.55)
cmd.set('cartoon_loop_radius', 0.35); cmd.set('cartoon_tube_radius', 0.7)

cmd.load(domain_pdb, 'domain'); cmd.hide('everything', 'domain')
cmd.dss('domain'); cmd.show('cartoon', 'domain'); cmd.color('grey80', 'domain')
for rng, col in MOTIFS:
    cmd.color(col, 'domain and resid ' + '+'.join(str(r) for r in rng))

cmd.show('spheres', f'domain and name CA and resid {MUT}')
cmd.set('sphere_scale', 1.25, f'domain and name CA and resid {MUT}')
cmd.color('hotpink', f'domain and name CA and resid {MUT}')

cmd.load(drug_pdb, 'drug'); cmd.hide('everything', 'drug')
cmd.show('sticks', 'drug'); cmd.set('stick_radius', 0.33, 'drug')
cmd.show('spheres', 'drug'); cmd.set('sphere_scale', 0.35, 'drug'); util.cbay('drug')

# face the drug so the ligand is always visible, then pull back to show the domain
# --- aim the camera down the pocket mouth so the drug cannot be occluded ----
# Fitting on the CTL or NTL cannot solve this: a fit moves the molecule into a
# shared frame, it does not move the camera, and cmd.orient re-derives the view
# from scratch afterwards regardless. What does solve it is deriving the view
# from the geometry. The drug is enclosed by protein on every side except the
# one facing solvent, so summing the vectors from the surrounding protein atoms
# to the drug centre gives the direction in which the protein mass is thinnest,
# which is the pocket mouth. Point the camera down that vector and the drug sits
# nearest the viewer with the protein behind it, by construction.
prot = np.array(cmd.get_model('domain').get_coord_list())
drug = np.array(cmd.get_model('drug').get_coord_list())
dcom = drug.mean(axis=0)

near = prot[np.linalg.norm(prot - dcom, axis=1) < 15.0]
if len(near) < 10:
    near = prot
out = (dcom - near).sum(axis=0)
out = out / np.linalg.norm(out)
if os.environ.get('OVERVIEW_FLIP'):
    out = -out

# camera basis with +z along the outward vector. PyMOL's view matrix maps model
# space to camera space and camera z increases towards the viewer, so making the
# outward vector the third row puts the drug at the largest z.
tmp = np.array([0.0, 0.0, 1.0]) if abs(out[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
cx = np.cross(tmp, out); cx = cx / np.linalg.norm(cx)
cy = np.cross(out, cx)
R = np.array([cx, cy, out])

cmd.orient('domain')                      # sets a sane origin, rotation replaced below
view = list(cmd.get_view())
view[0:9] = [float(v) for v in R.flatten()]
cmd.set_view(view)
cmd.turn('y', float(os.environ.get('OVERVIEW_TURN', 0)))   # optional nudge
cmd.zoom('domain', buffer=1.0, complete=1)
cmd.png(out_png, width=1700, height=1500, dpi=300, ray=1)
print(f'saved {out_png}')