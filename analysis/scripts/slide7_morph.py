"""
slide7_morph.py -- Slide 7 movie: the ATP pocket clamping shut, WT (open) ->
Q2022P (compact) (system PyMOL).

Interpolates between the two backbone-average structures from the pocket
pipeline (prep_avg_backbone_WT.pdb and _Q2022P.pdb). They share identical
backbone topology, so we blend coordinates directly -- and because we
superimpose on the C-LOBE, the N-lobe visibly clamps inward for Q2022P, which
is the 84% -> 49% pocket-closing story made visual. The Q2022P mutation site
(GRO resid 89 = real 2022) is marked with a labelled sphere.

Same illustrative-morph caveat as Slide 5: the ends are real average structures,
the in-between is a straight-line interpolation, not a pathway.

Env knobs (all optional):
  M7_A / M7_B    open / compact pdb (default WT / Q2022P prep_avg_backbone)
  M7_CLOBE       C-lobe align range, GRO (default 120-283)
  M7_MUT         mutation resid, GRO (default 89 -> labelled 2022)
  M7_MUT_COLOR   mutation sphere colour (default #E63946)
  M7_STEPS       tween steps        (default 40)
  M7_HOLD        frames held at each end (default 8)
  M7_TURN_X/Y/Z  camera turns       (default 0)
  M7_W / M7_H    pixel size         (default 1280 x 720)
  M7_RAY         1 = ray trace      (default 1; required headless)
  M7_BG          background         (default white)
  M7_COLOR       cartoon hex        (default #3B7EA1)
  M7_HL_CUT      A; motion highlight cutoff (default 999 = off)
  M7_HL_COLOR    motion highlight colour    (default #F9A825)
  M7_PREVIEW     1 = two stills, then stop

Usage (system pymol, venv OFF):
  env -u PYTHONPATH -u VIRTUAL_ENV PATH=/usr/bin:/bin /usr/bin/pymol -cq \
      analysis/slide7_morph.py
"""

import os
import sys

import numpy as np
from pymol import cmd

BASE = os.path.join(os.environ.get('HOME', ''), 'Molecular_Dynamics_analysis')
PB = os.path.join(BASE, 'figures', 'ROS1', 'ros1_prepared_final', 'pocket_box')

A       = os.environ.get('M7_A', os.path.join(PB, 'prep_avg_backbone_WT.pdb'))
B       = os.environ.get('M7_B', os.path.join(PB, 'prep_avg_backbone_Q2022P.pdb'))
CLOBE   = os.environ.get('M7_CLOBE', '120-283')
MUT     = os.environ.get('M7_MUT', '89')
MUT_COL = os.environ.get('M7_MUT_COLOR', '#E63946')
STEPS   = int(os.environ.get('M7_STEPS', 40))
HOLD    = int(os.environ.get('M7_HOLD', 8))
TURN_X  = float(os.environ.get('M7_TURN_X', 0))
TURN_Y  = float(os.environ.get('M7_TURN_Y', 0))
TURN_Z  = float(os.environ.get('M7_TURN_Z', 0))
W       = int(os.environ.get('M7_W', 1280))
H       = int(os.environ.get('M7_H', 720))
RAY     = int(os.environ.get('M7_RAY', 1))
BG      = os.environ.get('M7_BG', 'white')
COLOR   = os.environ.get('M7_COLOR', '#3B7EA1')
HL_CUT  = float(os.environ.get('M7_HL_CUT', 999))
HL_COL  = os.environ.get('M7_HL_COLOR', '#F9A825')
PREVIEW = int(os.environ.get('M7_PREVIEW', 0))
SHOW    = os.environ.get('M7_SHOW', '6-283')   # cartoon only this range (hide floppy ends)
REGIONS = int(os.environ.get('M7_REGIONS', 0)) # 1 = colour by thesis regions
# thesis region colours, GRO numbering (real - 1933); base lobes first, then features
REGION_ORDER = [
    ('nlobe', '12-85',   '#87CEEB'),
    ('clobe', '95-287',  '#C7CCD1'),
    ('aC',    '55-70',   '#EE1111'),
    ('ploop', '18-26',   '#FFD400'),
    ('hinge', '86-94',   '#FF8C1A'),
    ('dfg',   '169-171', '#FF12FF'),
]
OFFSET  = 1933
OUTDIR  = os.environ.get(
    'M7_OUT', os.path.join(BASE, 'figures', 'ROS1', 'ros1_prepared_final',
                           'slide7_morph_frames'))

os.makedirs(OUTDIR, exist_ok=True)
for p in (A, B):
    if not os.path.exists(p):
        sys.exit(f'missing {p}')


def hexrgb(h):
    h = h.lstrip('#')
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, Q.mean(0) - R @ P.mean(0)


cmd.reinitialize()
cmd.bg_color(BG)
cmd.set('ray_opaque_background', 1)
cmd.set('orthoscopic', 1)
cmd.set('ray_shadows', 0)
cmd.set('antialias', 2)
cmd.set('cartoon_fancy_helices', 1)

cmd.load(A, 'opn')
cmd.load(B, 'cmp')
na, nb = cmd.count_atoms('opn'), cmd.count_atoms('cmp')
print(f'open(WT) {na} atoms   compact(Q2022P) {nb} atoms')
if na != nb:
    sys.exit(f'atom counts differ ({na} vs {nb}); need identical backbone topology')

xa = np.asarray(cmd.get_coords('opn', 1), dtype=float)
xb = np.asarray(cmd.get_coords('cmp', 1), dtype=float)

# per-atom (resid, atom name), in coordinate-array order
info = []
cmd.iterate('opn', 'info.append((int(resi), name))', space={'info': info})
resnum = np.array([r for r, _ in info])
atname = np.array([n for _, n in info])

lo, hi = (int(v) for v in CLOBE.split('-'))
clobe_ca = (atname == 'CA') & (resnum >= lo) & (resnum <= hi)
if clobe_ca.sum() < 10:
    sys.exit(f'C-lobe mask hit only {int(clobe_ca.sum())} CA atoms in {CLOBE}; '
             f'check the M7_CLOBE range against this file numbering')

# superimpose the compact form onto the open form ON THE C-LOBE, so the N-lobe
# shows its full clamp instead of a rigid-body slide
R, t = kabsch(xb[clobe_ca], xa[clobe_ca])
xb = (R @ xb.T).T + t
crmsd = float(np.sqrt(((xb[clobe_ca] - xa[clobe_ca]) ** 2).sum(1).mean()))
print(f'C-lobe aligned on GRO {CLOBE}: RMSD {crmsd:.2f} A over '
      f'{int(clobe_ca.sum())} CA (small = good lock on the C-lobe)')

# build morph object: STEPS+1 states, linear blend open -> compact
cmd.create('mo', 'opn', 1, 1)
for s in range(2, STEPS + 2):
    cmd.create('mo', 'opn', 1, s)
for k in range(STEPS + 1):
    frac = k / STEPS
    cmd.load_coords((1.0 - frac) * xa + frac * xb, 'mo', state=k + 1)
nstates = cmd.count_states('mo')
print(f'morph "mo": {nstates} states, {cmd.count_atoms("mo")} atoms')

cmd.disable('opn'); cmd.disable('cmp')
cmd.hide('everything')
cmd.set_color('c_morph', hexrgb(COLOR))
cmd.dss('mo')
cmd.show('cartoon', f'mo and resid {SHOW}')
if REGIONS:
    for nm, rng, hx in REGION_ORDER:
        cmd.set_color(f'c_{nm}', hexrgb(hx))
        cmd.color(f'c_{nm}', f'mo and resid {rng}')
    print('coloured by thesis regions (N-lobe/C-lobe/aC/P-loop/hinge/DFG)')
else:
    cmd.color('c_morph', 'mo')

# how much does each residue move (after C-lobe lock)? the N-lobe clamp.
disp = np.sqrt(((xb - xa) ** 2).sum(1))
res_disp = {}
for r, d in zip(resnum, disp):
    if d > res_disp.get(r, -1.0):
        res_disp[r] = float(d)
top = sorted(res_disp.items(), key=lambda kv: -kv[1])[:8]
print('top movers after C-lobe lock (resid: A) -> '
      + ', '.join(f'{r}:{d:.1f}' for r, d in top))

# optional motion highlight of the clamping N-lobe (off unless M7_HL_CUT set low)
hot = [str(r) for r, d in res_disp.items() if d >= HL_CUT]
if hot and not REGIONS:
    cmd.set_color('c_hot', hexrgb(HL_COL))
    cmd.color('c_hot', 'mo and resid ' + '+'.join(hot))
    print(f'highlighted {len(hot)} residues moving >= {HL_CUT} A')

# mark the mutation site
real = int(MUT) + OFFSET
cmd.set_color('c_mut', hexrgb(MUT_COL))
msel = f'mo and resid {MUT} and name CA'
cmd.show('spheres', msel)
cmd.set('sphere_scale', 0.9, msel)
cmd.color('c_mut', msel)
cmd.set('label_size', 18)
cmd.set('label_color', 'black')
cmd.set('label_outline_color', 'white')
cmd.set('label_position', [0, 2.5, 0])
cmd.label(msel, repr(str(real)))
print(f'marked mutation site GRO {MUT} = real {real}')

# camera
cmd.set('all_states', 0)
cmd.frame(1)
cmd.orient('mo')
cmd.zoom('mo', buffer=6)
cmd.turn('x', TURN_X)
cmd.turn('y', TURN_Y)
cmd.turn('z', TURN_Z)

if PREVIEW:
    for st, tag in ((1, 'open'), (nstates, 'compact')):
        cmd.frame(st)
        cmd.png(os.path.join(OUTDIR, f'preview_{tag}.png'),
                width=W, height=H, dpi=150, ray=RAY)
    print(f'\npreview stills in {OUTDIR}: preview_open.png, preview_compact.png')
    print(f'current turn  x={TURN_X}  y={TURN_Y}  z={TURN_Z}')
    print('adjust M7_TURN_X/Y/Z and re-preview until the pocket faces you')
    os._exit(0)

# ping-pong: hold open, clamp to compact, hold compact, open back up
order = ([1] * HOLD + list(range(2, nstates + 1))
         + [nstates] * HOLD + list(range(nstates - 1, 0, -1)))
for i, st in enumerate(order):
    cmd.frame(st)
    cmd.png(os.path.join(OUTDIR, f'frame_{i:04d}.png'),
            width=W, height=H, dpi=150, ray=RAY)
    if i % 10 == 0:
        print(f'  frame {i + 1}/{len(order)}  (state {st})')

print(f'\ndone. {len(order)} frames in {OUTDIR}')
print('stitch to mp4 (needs ffmpeg):')
print(f'  ffmpeg -y -framerate 30 -i {OUTDIR}/frame_%04d.png '
      f'-c:v libx264 -pix_fmt yuv420p -crf 18 {OUTDIR}/../slide7_morph.mp4')