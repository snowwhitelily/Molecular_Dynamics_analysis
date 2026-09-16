"""
render_binding_active.py -- active-state binding-site cartoon for one drug.

Shows HOW the drug sits in the active-state pocket. NOTE ON DFG STATE: the active
receptors are AlphaFold activation-loop-OUT models; the DFG-Phe was measured NOT
canonically DFG-in (Phe-CZ -> hinge-CA ~13-16 A across all WT frames, vs 19.6 A for
the true DFG-out inactive ref and ~5-7 A expected for canonical DFG-in). So this is
labelled "active-state (activation-loop-out)", NOT "DFG-in". The DFG-Phe is still
drawn, but is not asserted to be in. The gate below prints the number as a diagnostic.

Renders the CORRECTED 3D pose (ensemble_all/vina_outputs_all3d) in its OWN
full-atom active-frame receptor (ensemble_all/receptor_pdb) -- receptor and pose
share the frame tag, so they are the SAME frame (no cross-frame mixing).

Representative frame per drug = the WT frame nearest that drug's median score in
step6r_ensemble_scores_all3d.csv (both are 'open' frames, for a comparable pair):
    zidesamtinib -> WT_rep2_f8932_open   (score -8.43, at median)
    cabozantinib -> WT_rep1_f11211_open  (score -9.02, median-adjacent)
This is a SINGLE representative pose, NOT the ensemble -- caption it as such.

Numbering: these ensemble_all receptors are a 292-residue slice renumbered from 1.
DFG-Phe (real 2103) sits at local 170 -> offset = 1933 (verified from the sequence).

CARTOON style + legend match render_binding_inactive_cabo.py so the active/inactive
panels form a set. Prints DFG-Phe CZ -> drug distance; for an active DFG-IN frame the
Phe should be IN-pocket (SMALL distance). We report whatever it actually is.

    PYTHONPATH=$VENV_SP pymol -cq analysis/scripts/render_binding_active.py -- DRUG [TX TY TZ]
      DRUG     = zidesamtinib | cabozantinib
      TX TY TZ = optional extra rotation (deg) about x,y,z to tune the angle
"""
import os, sys

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
DRUG = argv[0] if argv else 'zidesamtinib'
# per-drug default camera rotation (deg about x,y,z) — the tuned keeper angles.
# override by passing TX TY TZ on the command line.
DEFAULT_ROT = {
    'zidesamtinib': (20.0, 300.0, 20.0),
    'cabozantinib': (20.0, 180.0, 20.0),
}
_dx, _dy, _dz = DEFAULT_ROT.get(DRUG, (0.0, 0.0, 0.0))
TX = float(argv[1]) if len(argv) > 1 else _dx
TY = float(argv[2]) if len(argv) > 2 else _dy
TZ = float(argv[3]) if len(argv) > 3 else _dz
OFFSET = 1933   # real = local + 1933 (DFG-Phe real 2103 -> local 170)

# representative frame per drug (nearest WT median, both open)
FRAME = {
    'zidesamtinib': 'WT_rep2_f8932_open',
    'cabozantinib': 'WT_rep1_f11211_open',
}
if DRUG not in FRAME:
    sys.exit('DRUG must be one of: %s' % ', '.join(FRAME))
tagf = FRAME[DRUG]

BASE = os.path.join(os.environ['HOME'], 'Molecular_Dynamics_analysis')
ENS  = os.path.join(BASE, 'dock/ROS1/q2022p_vina/ensemble_all')
REC  = os.path.join(ENS, 'receptor_pdb', '%s.pdb' % tagf)
POSE = os.path.join(ENS, 'vina_outputs_all3d', '%s__%s.pdbqt' % (tagf, DRUG))
FIGS = os.path.join(BASE, 'figures/ROS1/ros1_prepared_final/pocket_box')

# guard: refuse to render if inputs are missing (no silent empty figure)
for p in (REC, POSE):
    if not os.path.exists(p):
        sys.exit('MISSING INPUT: %s' % p)

# region colours (RGB) + legend swatch hex -- match the inactive script
COL = {
    'orange': ((1.00, 0.50, 0.00), '#F0921E'),
    'purple': ((0.42, 0.12, 0.78), '#7333BF'),
    'rose':   ((0.92, 0.15, 0.42), '#E64073'),
    'char':   ((0.10, 0.10, 0.10), '#1F1F1F'),
}
# region ranges in REAL numbering -> converted to local by (r - OFFSET)
REGIONS = {
    'P-loop': (range(1948, 1956), 'orange', 'P-loop (glycine-rich)'),
    'hinge':  (range(2027, 2031), 'purple', 'hinge (H-bond region)'),
    'DFG':    (range(2102, 2105), 'rose',   'DFG motif (D-F-G)'),
}
DFG_PHE_LOCAL = 2103 - OFFSET   # = 170

from pymol import cmd, util

cmd.reinitialize()
cmd.bg_color('white')
cmd.set('ray_opaque_background', 1)
cmd.set('orthoscopic', 1)
cmd.set('ray_shadows', 0)
cmd.set('cartoon_transparency', 0.15)
for name, (rgb, _hex) in COL.items():
    cmd.set_color('reg_%s' % name, list(rgb))

cmd.load(REC, 'prot')
cmd.load(POSE, 'drug')

def sel(rr):   # real range -> local resid selection
    return 'prot and resid ' + '+'.join(str(r - OFFSET) for r in rr)

# --- DFG-in gate: DFG-Phe CZ -> drug centroid (SMALL = Phe in-pocket = DFG-in) ---
try:
    phe = cmd.get_model('prot and resid %d and name CZ' % DFG_PHE_LOCAL)
    lig = cmd.get_model('drug')
    if phe.atom and lig.atom:
        pz = phe.atom[0].coord
        n = len(lig.atom)
        lc = [sum(a.coord[i] for a in lig.atom) / n for i in range(3)]
        dist = sum((pz[i] - lc[i]) ** 2 for i in range(3)) ** 0.5
        state = ('IN-pocket (DFG-in) - good' if dist < 8 else
                 'flipped out (check! unexpected for active)' if dist > 10 else
                 'intermediate')
        print('DFG-Phe 2103 (local %d) CZ -> drug centroid: %.1f A  -> %s'
              % (DFG_PHE_LOCAL, dist, state))
    else:
        print('(DFG-Phe gate: no CZ or no ligand atoms found)')
except Exception as e:
    print('(DFG-Phe gate skipped: %s)' % e)

# whole domain: faint grey cartoon
cmd.hide('everything', 'prot')
cmd.dss('prot')
cmd.show('cartoon', 'prot')
cmd.color('grey80', 'prot')

# region patches: colour + side-chain sticks
for name, (rr, ckey, label) in REGIONS.items():
    s = sel(rr)
    cmd.color('reg_%s' % ckey, s)
    cmd.show('sticks', '%s and not name C+N+O' % s)
    cmd.set('stick_radius', 0.15, s)

# DFG-Phe: thick charcoal sticks (the flip residue)
phe_sel = 'prot and resid %d' % DFG_PHE_LOCAL
cmd.show('sticks', phe_sel)
cmd.set('stick_radius', 0.40, phe_sel)
cmd.color('reg_char', phe_sel)

# drug: element colours kept
cmd.hide('everything', 'drug')
cmd.show('sticks', 'drug')
cmd.set('stick_radius', 0.28, 'drug')
cmd.color('yellow', 'drug')
util.cnc('drug')

# view: orient on the drug + immediate pocket so the ligand sits in-frame and
# reasonably large (like the inactive render), then optional extra rotation.
# tighter buffer => bigger drug. bump BUF up if the pocket walls get clipped.
BUF = 2.0
frame = ('drug or ' + sel(range(1948, 1956)) + ' or ' + sel(range(2027, 2031))
         + ' or ' + sel(range(2102, 2105)))
cmd.orient(frame)
cmd.zoom(frame, buffer=BUF)
if TX: cmd.turn('x', TX)
if TY: cmd.turn('y', TY)
if TZ: cmd.turn('z', TZ)

# filename: the tuned per-drug default writes the PLAIN name (binding_active_DRUG.png);
# an explicit CLI override (args beyond DRUG) gets the angle suffix so tuning runs
# don't clobber the keeper.
overridden = len(argv) > 1
rtag = ('_x%dy%dz%d' % (int(TX), int(TY), int(TZ))) if overridden else ''
raw = os.path.join(FIGS, 'binding_active_%s%s_raw.png' % (DRUG, rtag))
cmd.png(raw, width=1600, height=1300, dpi=300, ray=1)
print('saved %s   (frame %s, rotation x=%s y=%s z=%s)' % (raw, tagf, TX, TY, TZ))

# ---- composite legend (matches inactive script: headroom band, small font) ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D

img = mpimg.imread(raw)
h, w = img.shape[:2]
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
    Line2D([0], [0], color=COL['char'][1], lw=5, label='DFG-Phe (F2103)'),
    Line2D([0], [0], color='#E8C000', lw=5, label='%s: C yellow, O red, N blue' % DRUG),
]
leg = ax.legend(handles=handles, loc='upper left', frameon=True, fontsize=8,
                borderpad=0.5, labelspacing=0.35, handlelength=1.1, handleheight=0.9,
                title='ROS1 active-state pocket — %s' % DRUG,
                bbox_to_anchor=(0.01, 0.995))
leg.get_title().set_fontsize(8.5)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_alpha(0.9)
leg.get_frame().set_edgecolor('#cccccc')
# small provenance caption (bottom-left), keeps the exact frame + DFG-state honest
ax.text(0.01, 0.01,
        'representative docked pose · frame %s · DFG angle ~94° (activation-loop-out)' % tagf,
        transform=ax.transAxes, fontsize=6.5, color='#666', va='bottom', ha='left')

out = os.path.join(FIGS, 'binding_active_%s%s.png' % (DRUG, rtag))
fig.savefig(out, dpi=300)
plt.close()
print('saved %s' % out)