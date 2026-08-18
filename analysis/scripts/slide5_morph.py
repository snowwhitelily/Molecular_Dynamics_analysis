"""
slide5_morph.py -- Slide 5 movie: the kinase changing shape, open <-> closed
(system PyMOL).

Interpolates between two real WT reference structures -- the active (DFG-in,
"open") and inactive (DFG-out, "closed") kinase -- and renders the tween forward
then backward, so the clip loops open -> closed -> open. This is an ILLUSTRATIVE
morph: the two ends are real structures, the in-between is a straight-line
interpolation, not a simulated pathway. Caption it as such.

Frames land in an output folder; stitch to video with ffmpeg afterwards
(command printed at the end). Headless PyMOL needs ray=1, so a full run takes a
few minutes -- do the small test first.

Env knobs (all optional):
  MORPH_A        active/open pdb    (default dock/ROS1/inactive_ref/ROS1_wt.pdb)
  MORPH_B        inactive/closed    (default dock/ROS1/inactive_ref/ROS1_WT_i.pdb)
  MORPH_STEPS    tween steps        (default 40)
  MORPH_HOLD     frames held at each end (default 8)
  MORPH_DEG      total rotation deg (default 0 = static camera)
  MORPH_W MORPH_H  pixel size       (default 1280 x 720)
  MORPH_RAY      1 = ray trace      (default 1; required headless)
  MORPH_BG       background         (default white)
  MORPH_COLOR    cartoon hex        (default #3B7EA1)
  MORPH_OUT      output folder      (default figures/.../slide5_morph_frames)

Usage (system pymol, venv OFF):
  env -u PYTHONPATH -u VIRTUAL_ENV PATH=/usr/bin:/bin /usr/bin/pymol -cq \
      analysis/slide5_morph.py
"""

import os
import sys

import numpy as np
from pymol import cmd

BASE = os.path.join(os.environ.get('HOME', ''), 'Molecular_Dynamics_analysis')
REF = os.path.join(BASE, 'dock', 'ROS1', 'inactive_ref')

A       = os.environ.get('MORPH_A', os.path.join(REF, 'ROS1_wt.pdb'))
B       = os.environ.get('MORPH_B', os.path.join(REF, 'ROS1_WT_i.pdb'))
STEPS   = int(os.environ.get('MORPH_STEPS', 40))
HOLD    = int(os.environ.get('MORPH_HOLD', 8))
DEG     = float(os.environ.get('MORPH_DEG', 0))
TURN_X  = float(os.environ.get('MORPH_TURN_X', 0))   # tilt the camera to face
TURN_Y  = float(os.environ.get('MORPH_TURN_Y', 0))   # the groove toward the
TURN_Z  = float(os.environ.get('MORPH_TURN_Z', 0))   # screen
PREVIEW = int(os.environ.get('MORPH_PREVIEW', 0))    # 1 = two stills, then stop
HL_CUT   = float(os.environ.get('MORPH_HL_CUT', 6.0))     # A: residues moving >= this get highlighted
HL_COLOR = os.environ.get('MORPH_HL_COLOR', '#E63946')    # highlight colour (default red)
HL_RESIDUES = os.environ.get('MORPH_HL_RESIDUES', '').strip()  # e.g. '2102-2125'; overrides motion cutoff
W       = int(os.environ.get('MORPH_W', 1280))
H       = int(os.environ.get('MORPH_H', 720))
RAY     = int(os.environ.get('MORPH_RAY', 1))
BG      = os.environ.get('MORPH_BG', 'white')
COLOR   = os.environ.get('MORPH_COLOR', '#3B7EA1')
HILITE   = int(os.environ.get('MORPH_HIGHLIGHT', 1))          # 1 = colour landmarks
LOOP_SEL = os.environ.get('MORPH_LOOP', 'resid 2102-2125')   # activation loop
HELIX_SEL = os.environ.get('MORPH_HELIX', 'resid 1986-2003')  # aC-helix
LOOP_COL = os.environ.get('MORPH_LOOP_COLOR', '#E8792B')      # orange
HELIX_COL = os.environ.get('MORPH_HELIX_COLOR', '#E8C020')    # gold
OUTDIR  = os.environ.get(
    'MORPH_OUT',
    os.path.join(BASE, 'figures', 'ROS1', 'ros1_prepared_final', 'slide5_morph_frames'))

os.makedirs(OUTDIR, exist_ok=True)
for p in (A, B):
    if not os.path.exists(p):
        sys.exit(f'missing {p}')

def hexrgb(h):
    h = h.lstrip('#')
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]

cmd.reinitialize()
cmd.bg_color(BG)
cmd.set('ray_opaque_background', 1)
cmd.set('orthoscopic', 1)
cmd.set('ray_shadows', 0)
cmd.set('antialias', 2)
cmd.set('cartoon_fancy_helices', 1)

cmd.load(A, 'act')
cmd.load(B, 'ina')
na, nb = cmd.count_atoms('act'), cmd.count_atoms('ina')
print(f'active {na} atoms   inactive {nb} atoms')
if na != nb:
    sys.exit(f'atom counts differ ({na} vs {nb}); this manual morph needs the '
             f'two structures to share identical topology')

# --- manual linear morph (open-source PyMOL has no "morph" command) ---
# read both coordinate sets in the object's atom order
xa = np.asarray(cmd.get_coords('act', 1), dtype=float)
xb = np.asarray(cmd.get_coords('ina', 1), dtype=float)
if xa is None or xb is None or xa.shape != xb.shape:
    sys.exit('could not read matching coordinate arrays')

# CA mask, in the same order as the coordinate arrays
ca = []
cmd.iterate('act', 'ca.append(name == "CA")', space={'ca': ca})
ca = np.asarray(ca, dtype=bool)


def kabsch(P, Q):
    """rigid R, t least-squares fitting P onto Q."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, Q.mean(0) - R @ P.mean(0)


# superimpose inactive onto active on CA, so we interpolate the internal shape
# change rather than a rigid-body slide across space
R, t = kabsch(xb[ca], xa[ca])
xb = (R @ xb.T).T + t
rmsd = float(np.sqrt(((xb[ca] - xa[ca]) ** 2).sum(1).mean()))
print(f'superimposed inactive onto active: CA RMSD {rmsd:.2f} A '
      f'over {int(ca.sum())} residues  (a sane few-A value means the atoms '
      f'line up; a huge value means they do not)')

# build the morph object: STEPS+1 states, linear blend active -> inactive
cmd.create('mo', 'act', 1, 1)
for s in range(2, STEPS + 2):
    cmd.create('mo', 'act', 1, s)
for k in range(STEPS + 1):
    frac = k / STEPS
    cmd.load_coords((1.0 - frac) * xa + frac * xb, 'mo', state=k + 1)
nstates = cmd.count_states('mo')
print(f'morph "mo": {nstates} states, {cmd.count_atoms("mo")} atoms')
if nstates < 2:
    sys.exit('morph build failed')

cmd.disable('act'); cmd.disable('ina')
cmd.hide('everything')
cmd.set_color('c_morph', hexrgb(COLOR))
cmd.dss('mo')
cmd.show('cartoon', 'mo')
cmd.color('c_morph', 'mo')

# measure per-residue displacement (kept for the diagnostic printout)
disp = np.sqrt(((xb - xa) ** 2).sum(1))
resa = []
cmd.iterate('act', 'resa.append(resi)', space={'resa': resa})
res_disp = {}
for r, d in zip(resa, disp):
    if d > res_disp.get(r, -1.0):
        res_disp[r] = float(d)
top = sorted(res_disp.items(), key=lambda kv: -kv[1])[:8]
print('top movers (resid: A moved) -> '
      + ', '.join(f'{r}:{d:.1f}' for r, d in top))

cmd.set_color('c_hot', hexrgb(HL_COLOR))
if HL_RESIDUES:
    # explicit residue range (numbering confirmed) -- clean, named highlight
    cmd.color('c_hot', f'mo and resid {HL_RESIDUES}')
    print(f'highlighted named residues: {HL_RESIDUES}')
else:
    hot = [r for r, d in res_disp.items() if d >= HL_CUT]
    if hot:
        cmd.color('c_hot', 'mo and resid ' + '+'.join(hot))
        print(f'highlighted {len(hot)} residues moving >= {HL_CUT} A')
    else:
        print(f'nothing moves >= {HL_CUT} A -- lower MORPH_HL_CUT to highlight')

# frame on the whole motion so nothing clips as it opens/closes
cmd.set('all_states', 0)
cmd.frame(1)
cmd.orient('mo')
cmd.zoom('mo', buffer=8)

# override the automatic camera so the meaningful face (pocket + activation
# loop) can be turned to face the screen
cmd.turn('x', TURN_X)
cmd.turn('y', TURN_Y)
cmd.turn('z', TURN_Z)

# fast orientation hunt: render just the two end shapes and stop
if PREVIEW:
    for st, tag in ((1, 'open'), (nstates, 'closed')):
        cmd.frame(st)
        cmd.png(os.path.join(OUTDIR, f'preview_{tag}.png'),
                width=W, height=H, dpi=150, ray=RAY)
    print(f'\npreview stills in {OUTDIR}: preview_open.png, preview_closed.png')
    print(f'current turn  x={TURN_X}  y={TURN_Y}  z={TURN_Z}')
    print('adjust MORPH_TURN_X/Y/Z and re-preview until the groove faces you')
    os._exit(0)

# ping-pong: hold open, tween to closed, hold closed, tween back
order = ([1] * HOLD + list(range(2, nstates + 1))
         + [nstates] * HOLD + list(range(nstates - 1, 0, -1)))
deg_per = DEG / len(order)

for i, st in enumerate(order):
    cmd.frame(st)
    if DEG:
        cmd.turn('y', deg_per)
    cmd.png(os.path.join(OUTDIR, f'frame_{i:04d}.png'),
            width=W, height=H, dpi=150, ray=RAY)
    if i % 10 == 0:
        print(f'  frame {i + 1}/{len(order)}  (state {st})')

print(f'\ndone. {len(order)} frames in {OUTDIR}')
print('stitch to mp4 (needs ffmpeg):')
print(f'  ffmpeg -y -framerate 30 -i {OUTDIR}/frame_%04d.png '
      f'-c:v libx264 -pix_fmt yuv420p -crf 18 {OUTDIR}/../slide5_morph.mp4')