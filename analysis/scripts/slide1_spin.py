"""
slide1_spin.py -- Slide 1 title movie: ROS1 backbone in motion (system PyMOL).

Renders a sequence of PNG frames of the WT kinase backbone. Two looks, chosen
with POCKET... no -- with the SPIN_ENSEMBLE knob:

  SPIN_ENSEMBLE=0 (default): the trajectory PLAYS -- the backbone visibly
      flexes -- while the camera slowly rotates. "The protein, in motion."
  SPIN_ENSEMBLE=1: every frame overlaid at once as a translucent cloud
      (the classic MD ensemble), spinning. No playback, just the spread.

Frames land in an output folder; stitch them to a video afterwards with ffmpeg
(command is printed at the end). Headless PyMOL has no OpenGL, so ray=1 is
required -- that's why a full run takes a few minutes. Do a fast test first.

Env knobs (all optional):
  SPIN_VARIANT   variant name         (default WT)
  SPIN_REP       replica 0/1/2        (default 0)
  SPIN_FRAMES    number of frames     (default 120)
  SPIN_DEG       total rotation deg   (default 360)
  SPIN_W SPIN_H  pixel size           (default 1280 x 720)
  SPIN_RAY       1 = ray trace        (default 1; required when headless)
  SPIN_STRIDE    load every Nth frame (default 10)
  SPIN_ENSEMBLE  0 play, 1 cloud      (default 0)
  SPIN_BG        background colour    (default white)
  SPIN_COLOR     cartoon hex colour   (default #3B7EA1)
  SPIN_OUT       output folder        (default figures/.../slide1_spin_frames)

Usage (system pymol, venv OFF):
  env -u PYTHONPATH -u VIRTUAL_ENV PATH=/usr/bin:/bin /usr/bin/pymol -cq \
      analysis/slide1_spin.py
"""

import os
import sys

from pymol import cmd

BASE = os.path.join(os.environ.get('HOME', ''), 'Molecular_Dynamics_analysis')
TRAJ = os.path.join(BASE, 'trajectories', 'ros1_prepared_final')

VARIANT  = os.environ.get('SPIN_VARIANT', 'WT')
REP      = os.environ.get('SPIN_REP', '0')
NFRAMES  = int(os.environ.get('SPIN_FRAMES', 120))
DEG      = float(os.environ.get('SPIN_DEG', 360))
W        = int(os.environ.get('SPIN_W', 1280))
H        = int(os.environ.get('SPIN_H', 720))
RAY      = int(os.environ.get('SPIN_RAY', 1))
STRIDE   = int(os.environ.get('SPIN_STRIDE', 10))
FIT_SEL  = os.environ.get('SPIN_FIT_SEL', 'name CA')
ENSEMBLE = int(os.environ.get('SPIN_ENSEMBLE', 0))
BG       = os.environ.get('SPIN_BG', 'white')
COLOR    = os.environ.get('SPIN_COLOR', '#3B7EA1')
REGIONS  = int(os.environ.get('SPIN_REGIONS', 0))   # 1 = colour by thesis regions
# thesis region colours, GRO numbering (real - 1933); base lobes first, then features
REGION_ORDER = [
    ('nlobe', '12-85',   '#87CEEB'),
    ('clobe', '95-287',  '#C7CCD1'),
    ('aC',    '55-70',   '#EE1111'),
    ('ploop', '18-26',   '#FFD400'),
    ('hinge', '86-94',   '#FF8C1A'),
    ('dfg',   '169-171', '#FF12FF'),
]
OUTDIR   = os.environ.get(
    'SPIN_OUT',
    os.path.join(BASE, 'figures', 'ROS1', 'ros1_prepared_final', 'slide1_spin_frames'))

os.makedirs(OUTDIR, exist_ok=True)

top = os.path.join(TRAJ, VARIANT, REP, f'{VARIANT}-MD-prot.pdb')
trj = os.path.join(TRAJ, VARIANT, REP, f'{VARIANT}-MD-prot.xtc')
for p in (top, trj):
    if not os.path.exists(p):
        sys.exit(f'missing {p}')

cmd.reinitialize()
cmd.bg_color(BG)
cmd.set('ray_opaque_background', 1)
cmd.set('orthoscopic', 1)
cmd.set('ray_shadows', 0)
cmd.set('antialias', 2)
cmd.set('cartoon_fancy_helices', 1)

cmd.load(top, 'mol')
cmd.load_traj(trj, 'mol', state=1, interval=STRIDE)
nstates = cmd.count_states('mol')
print(f'loaded {VARIANT} rep {REP}: {nstates} frames (stride {STRIDE})')

# superimpose every frame on the first, on the backbone, so the molecule spins
# cleanly instead of tumbling -- only the internal loop flexing is left over
if nstates > 1:
    rms = cmd.intra_fit(f'mol and ({FIT_SEL})', 1)
    good = [r for r in rms if isinstance(r, (int, float)) and r >= 0]
    if good:
        print(f'aligned frames on "{FIT_SEL}": mean backbone RMSD '
              f'{sum(good) / len(good):.2f} A over {len(good)} frames')

def hexrgb(h):
    h = h.lstrip('#')
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]

cmd.set_color('c_spin', hexrgb(COLOR))
cmd.hide('everything', 'mol')

if ENSEMBLE:
    # all frames at once -> a translucent conformational cloud, then just spin
    cmd.set('all_states', 1)
    cmd.show('ribbon', 'mol')
    cmd.set('ribbon_width', 1.0)
    cmd.set('ribbon_transparency', 0.55)
    cmd.color('c_spin', 'mol')
else:
    # single state shown; we step through the trajectory as we rotate
    cmd.set('all_states', 0)
    cmd.dss('mol')
    cmd.show('cartoon', 'mol')
    cmd.color('c_spin', 'mol')

# frame the molecule once, then rotate away from that fixed framing
if REGIONS:
    for nm, rng, hx in REGION_ORDER:
        cmd.set_color(f'c_{nm}', hexrgb(hx))
        cmd.color(f'c_{nm}', f'mol and resid {rng}')
    print('coloured by thesis regions (N-lobe/C-lobe/aC/P-loop/hinge/DFG)')

cmd.frame(1)
cmd.orient('mol')
cmd.zoom('mol', buffer=5)

deg_per = DEG / NFRAMES
for i in range(NFRAMES):
    if not ENSEMBLE and nstates > 1:
        state = 1 + round(i * (nstates - 1) / max(1, NFRAMES - 1))
        cmd.frame(state)
    cmd.turn('y', deg_per)
    out = os.path.join(OUTDIR, f'frame_{i:04d}.png')
    cmd.png(out, width=W, height=H, dpi=150, ray=RAY)
    if i % 10 == 0:
        print(f'  frame {i + 1}/{NFRAMES}')

print(f'\ndone. {NFRAMES} frames in {OUTDIR}')
print('stitch to mp4 (needs ffmpeg):')
print(f'  ffmpeg -y -framerate 30 -i {OUTDIR}/frame_%04d.png '
      f'-c:v libx264 -pix_fmt yuv420p -crf 18 {OUTDIR}/../slide1_spin.mp4')