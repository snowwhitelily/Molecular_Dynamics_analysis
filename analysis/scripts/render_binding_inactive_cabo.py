"""
render_binding_inactive_cabo.py -- cabozantinib in the INACTIVE (DFG-out) pocket.

Cabozantinib is a type II inhibitor: it engages the inactive DFG-out ROS1
conformation. This renders the corrected 3D cabozantinib pose in the full-atom
inactive receptor (both in the same frame -> NO frame-correction needed).

CARTOON-based, not surface: a surface occludes the drug and the DFG-Phe, which is
exactly what this figure needs to show. Cartoon + sticks shows both cleanly and
matches render_binding_annotated.py (the keeper style). Functional regions coloured;
DFG-Phe (F2103) as thick sticks so the DFG-OUT state is visible. Palette + legend
match render_smoke.py so an active/inactive pair is visually consistent.

IMPORTANT: the inactive receptor uses REAL residue numbering (no GRO offset).
Verified on disk: F2103 is a full-side-chain PHE; pose z-range ~25 A (genuinely 3D).

Prints DFG-Phe CZ -> drug distance: LARGE (~>10 A) confirms the Phe is flipped OUT.

    PYTHONPATH=$VENV_SP pymol -cq analysis/scripts/render_binding_inactive_cabo.py -- [TX TY TZ]
      TX TY TZ = optional extra rotation (deg) about x,y,z to tune the angle,
                 e.g. `-- 0 90 0` if the drug faces away.
"""
import os, sys

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
# tuned keeper camera (deg about x,y,z); override by passing TX TY TZ on the CLI
TX = float(argv[0]) if len(argv) > 0 else 20.0
TY = float(argv[1]) if len(argv) > 1 else 200.0
TZ = float(argv[2]) if len(argv) > 2 else 20.0
overridden = len(argv) > 0

BASE = os.path.join(os.environ['HOME'], 'Molecular_Dynamics_analysis')
REC  = os.path.join(BASE, 'dock/ROS1/inactive_ref/ROS1_WT_i.pdb')
POSE = os.path.join(BASE, 'dock/ROS1/q2022p_vina/vina_outputs_sharedbox_3d/'
                          'ROS1_WT_i__cabozantinib.pdbqt')
FIGS = os.path.join(BASE, 'figures/ROS1/ros1_prepared_final/pocket_box')

# region colours (RGB) + legend swatch hex -- avoids drug element palette
# (yellow C / red O / blue N / green F). matches render_smoke.py.
COL = {
    'orange': ((1.00, 0.50, 0.00), '#F0921E'),
    'purple': ((0.42, 0.12, 0.78), '#7333BF'),
    'rose':   ((0.92, 0.15, 0.42), '#E64073'),
    'char':   ((0.10, 0.10, 0.10), '#1F1F1F'),
}
# REAL numbering (inactive file has no GRO offset)
REGIONS = {
    'P-loop': (range(1948, 1956), 'orange', 'P-loop (glycine-rich)'),
    'hinge':  (range(2027, 2031), 'purple', 'hinge (H-bond region)'),
    'DFG':    (range(2102, 2105), 'rose',   'DFG motif (D-F-G)'),
}
DFG_PHE = 2103

from pymol import cmd, util

cmd.reinitialize()
cmd.bg_color('white')
cmd.set('ray_opaque_background', 1)
cmd.set('orthoscopic', 1)
cmd.set('ray_shadows', 0)
cmd.set('cartoon_transparency', 0.15)   # faint see-through cartoon, drug stays clear
for name, (rgb, _hex) in COL.items():
    cmd.set_color('reg_%s' % name, list(rgb))

cmd.load(REC, 'prot')
cmd.load(POSE, 'drug')

def sel(rr):
    return 'prot and resid ' + '+'.join(str(r) for r in rr)   # REAL numbering

# --- DFG-out gate: DFG-Phe CZ -> drug centroid (LARGE = Phe flipped out) ---
try:
    phe = cmd.get_model('prot and resid %d and name CZ' % DFG_PHE)
    lig = cmd.get_model('drug')
    if phe.atom and lig.atom:
        pz = phe.atom[0].coord
        n = len(lig.atom)
        lc = [sum(a.coord[i] for a in lig.atom) / n for i in range(3)]
        dist = sum((pz[i] - lc[i]) ** 2 for i in range(3)) ** 0.5
        state = 'FLIPPED OUT (DFG-out) - good' if dist > 10 else \
                'in-pocket-ish (check!)' if dist < 8 else 'intermediate'
        print('DFG-Phe %d CZ -> drug centroid: %.1f A  -> %s' % (DFG_PHE, dist, state))
    else:
        print('(DFG-Phe gate: no CZ or no ligand atoms found)')
except Exception as e:
    print('(DFG-Phe gate skipped: %s)' % e)

# whole domain: faint grey cartoon for context (reads as a kinase)
cmd.hide('everything', 'prot')
cmd.dss('prot')
cmd.show('cartoon', 'prot')
cmd.color('grey80', 'prot')

# region patches: colour cartoon + show side chains as sticks
for name, (rr, ckey, label) in REGIONS.items():
    s = sel(rr)
    cmd.color('reg_%s' % ckey, s)
    cmd.show('sticks', '%s and not name C+N+O' % s)
    cmd.set('stick_radius', 0.15, s)

# DFG-Phe: thick charcoal sticks -- the flip residue, must read clearly
phe_sel = 'prot and resid %d' % DFG_PHE
cmd.show('sticks', phe_sel)
cmd.set('stick_radius', 0.40, phe_sel)
cmd.color('reg_char', phe_sel)

# drug: element colours kept, drawn on top
cmd.hide('everything', 'drug')
cmd.show('sticks', 'drug')
cmd.set('stick_radius', 0.28, 'drug')
cmd.color('yellow', 'drug')
util.cnc('drug')

# view: frame the pocket AND the drug AND the Phe -- the three things the figure
# is about -- then apply optional extra rotation to look INTO the pocket.
frame = ('drug or ' + sel(range(1948, 1956)) + ' or ' + sel(range(2027, 2031))
         + ' or ' + sel(range(2102, 2105)))
cmd.orient(frame)
cmd.zoom(frame, buffer=2.0)
if TX: cmd.turn('x', TX)
if TY: cmd.turn('y', TY)
if TZ: cmd.turn('z', TZ)

tag = ('_x%dy%dz%d' % (int(TX), int(TY), int(TZ))) if overridden else ''
raw = os.path.join(FIGS, 'binding_inactive_cabozantinib%s_raw.png' % tag)
cmd.png(raw, width=1600, height=1300, dpi=300, ray=1)
print('saved %s   (rotation x=%s y=%s z=%s)' % (raw, TX, TY, TZ))

# ---- composite legend (matches render_smoke.py) ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D

img = mpimg.imread(raw)
h, w = img.shape[:2]

# add white headroom at the top so the (smaller) legend sits ABOVE the structure
pad = int(h * 0.17)
if img.ndim == 3:
    band = np.ones((pad, w, img.shape[2]), dtype=img.dtype)
    if img.shape[2] == 4:
        band[..., 3] = 1.0
else:
    band = np.ones((pad, w), dtype=img.dtype)
img2 = np.vstack([band, img])
H = h + pad

fig = plt.figure(figsize=(w / 300.0, H / 300.0), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img2); ax.axis('off')

handles = [mpatches.Patch(facecolor=COL[ckey][1], edgecolor='black', linewidth=0.5,
                          label=label)
           for (_rr, ckey, label) in REGIONS.values()]
handles += [
    Line2D([0], [0], color=COL['char'][1], lw=5, label='DFG-Phe (F2103, flipped OUT)'),
    Line2D([0], [0], color='#E8C000', lw=5, label='cabozantinib: C yellow, O red, N blue'),
]
leg = ax.legend(handles=handles, loc='upper left', frameon=True, fontsize=8,
                borderpad=0.5, labelspacing=0.35, handlelength=1.1, handleheight=0.9,
                title='ROS1 inactive (DFG-out) pocket',
                bbox_to_anchor=(0.01, 0.995))
leg.get_title().set_fontsize(8.5)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_alpha(0.9)
leg.get_frame().set_edgecolor('#cccccc')
# provenance caption, matching the active panels
ax.text(0.01, 0.01,
        'docked pose · homology-model inactive receptor · DFG angle ~70° (DFG-out)',
        transform=ax.transAxes, fontsize=6.5, color='#666', va='bottom', ha='left')

out = os.path.join(FIGS, 'binding_inactive_cabozantinib%s.png' % tag)
fig.savefig(out, dpi=300)
plt.close()
print('saved %s' % out)