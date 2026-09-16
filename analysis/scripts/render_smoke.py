"""
render_smoke.py -- headless surface binding figure (single-surface version).

Back to ONE moderately-transparent surface (like the early working renders) instead
of the bulk/patch split that occluded the drug. Region patches are given boosted
base colours so they still read saturated through the transparency. Drug keeps
element colours; region palette (orange/purple/rose/charcoal) avoids drug atoms.
Prints DFG-Phe -> drug distance so we can confirm DFG-in before calling it that.

    # NOTE: run with the venv ACTIVE is not required; just need venv packages on
    # PYTHONPATH for the matplotlib legend step. If 'deactivate' errors with
    # 'command not found' that just means no venv was active -- harmless.
    PYTHONPATH=$VENV_SP pymol -cq analysis/scripts/render_smoke.py -- DRUG [TX TY TZ]
      DRUG     = zidesamtinib | cabozantinib
      TX TY TZ = optional extra rotation (deg) about x,y,z
"""
import os, sys

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
DRUG = argv[0] if argv else 'cabozantinib'
TX = float(argv[1]) if len(argv) > 1 else 0.0
TY = float(argv[2]) if len(argv) > 2 else 0.0
TZ = float(argv[3]) if len(argv) > 3 else 0.0
OFFSET = 1933

BASE = os.path.join(os.environ['HOME'], 'Molecular_Dynamics_analysis')
FIGS = os.path.join(BASE, 'figures/ROS1/ros1_prepared_final/pocket_box')
prot_pdb = os.path.join(FIGS, 'prep_avg_backbone_WT.pdb')
drug_pdb = os.path.join(FIGS, 'prep_distance_%s_drug.pdb' % DRUG)

# region colours (RGB) + matching legend swatch hex. avoids drug element palette
# (yellow C / red O / blue N / green F). base colours pushed vivid to survive the
# transparency wash of the single surface.
COL = {
    'orange': ((1.00, 0.50, 0.00), '#F0921E'),
    'purple': ((0.42, 0.12, 0.78), '#7333BF'),
    'rose':   ((0.92, 0.15, 0.42), '#E64073'),
    'char':   ((0.10, 0.10, 0.10), '#1F1F1F'),
}
REGIONS = {
    'P-loop': (range(1948, 1956), 'orange', 'P-loop (glycine-rich)'),
    'hinge':  (range(2027, 2031), 'purple', 'hinge (H-bond region)'),
    'DFG':    (range(2102, 2105), 'rose',   'DFG motif (D-F-G)'),
}
DFG_PHE_GRO = 2103 - OFFSET

from pymol import cmd, util

cmd.reinitialize()
cmd.bg_color('white')
cmd.set('ray_opaque_background', 1)
cmd.set('orthoscopic', 1)
for name, (rgb, _hex) in COL.items():
    cmd.set_color('reg_%s' % name, list(rgb))

# lighting: soft, no black interior
cmd.set('surface_quality', 2)
cmd.set('two_sided_lighting', 1)
cmd.set('ray_interior_color', 'grey80')
cmd.set('ambient', 0.42)
cmd.set('direct', 0.58)
cmd.set('specular', 0.15)
cmd.set('ray_shadows', 0)

cmd.load(prot_pdb, 'prot')
cmd.load(drug_pdb, 'drug')

def gro(rr):
    return 'prot and resid ' + '+'.join(str(r - OFFSET) for r in rr)

# --- DFG-in sanity: DFG-Phe CZ -> drug centroid (small = Phe in-pocket = DFG-in) ---
try:
    phe = cmd.get_model('prot and resid %d and name CZ' % DFG_PHE_GRO)
    lig = cmd.get_model('drug')
    if phe.atom and lig.atom:
        pz = phe.atom[0].coord
        n = len(lig.atom)
        lc = [sum(a.coord[i] for a in lig.atom) / n for i in range(3)]
        dist = sum((pz[i] - lc[i]) ** 2 for i in range(3)) ** 0.5
        print('DFG-Phe %d CZ -> drug centroid: %.1f A  '
              '(small ~<8 A = Phe in-pocket = DFG-in; large ~>10 A = flipped out)'
              % (DFG_PHE_GRO + OFFSET, dist))
    else:
        print('(DFG-Phe check: no CZ atom or no ligand atoms found)')
except Exception as e:
    print('(DFG-Phe check skipped: %s)' % e)

# ---- ONE surface, moderate transparency (the version that showed the drug) ----
cmd.hide('everything', 'prot')
cmd.show('surface', 'prot')
cmd.set('transparency', 0.45, 'prot')     # a touch less see-through than 0.55
cmd.color('grey80', 'prot')

# colour the region patches on that same surface
for name, (rr, ckey, label) in REGIONS.items():
    cmd.color('reg_%s' % ckey, gro(rr))

# DFG-Phe: thick charcoal sticks (the flip residue)
phe_sel = 'prot and resid %d' % DFG_PHE_GRO
cmd.show('sticks', phe_sel)
cmd.set('stick_radius', 0.40, phe_sel)
cmd.color('reg_char', phe_sel)

# drug: element colours kept, drawn AFTER the surface so it isn't occluded
cmd.hide('everything', 'drug')
cmd.show('sticks', 'drug')
cmd.set('stick_radius', 0.28, 'drug')
cmd.color('yellow', 'drug')
util.cnc('drug')

cmd.orient('drug')
cmd.zoom('drug', 8)
if TX: cmd.turn('x', TX)
if TY: cmd.turn('y', TY)
if TZ: cmd.turn('z', TZ)

tag = DRUG + (('_x%dy%dz%d' % (int(TX), int(TY), int(TZ))) if (TX or TY or TZ) else '')
raw = os.path.join(FIGS, 'smoke_%s_raw.png' % tag)
cmd.png(raw, width=1600, height=1300, dpi=300, ray=1)
print('saved %s   (rotation x=%s y=%s z=%s)' % (raw, TX, TY, TZ))

# ---- composite legend (guaranteed placement) ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

img = mpimg.imread(raw)
h, w = img.shape[:2]
fig = plt.figure(figsize=(w / 300.0, h / 300.0), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis('off')

handles = [mpatches.Patch(facecolor=COL[ckey][1], edgecolor='black', linewidth=0.6,
                          label=label)
           for (_rr, ckey, label) in REGIONS.values()]
handles += [
    Line2D([0], [0], color=COL['char'][1], lw=6, label='DFG-Phe (F2103, flips in/out)'),
    Line2D([0], [0], color='#E8C000', lw=6, label='%s: C yellow, O red, N blue' % DRUG),
]
leg = ax.legend(handles=handles, loc='upper left', frameon=True, fontsize=11,
                borderpad=0.8, labelspacing=0.6, handlelength=1.3)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_alpha(0.85)
leg.get_frame().set_edgecolor('#cccccc')

out = os.path.join(FIGS, 'smoke_%s.png' % tag)
fig.savefig(out, dpi=300)
plt.close()
print('saved %s' % out)