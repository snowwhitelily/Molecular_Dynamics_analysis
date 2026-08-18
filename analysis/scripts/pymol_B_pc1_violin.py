"""
pymol_B_pc1_violin.py -- thesis Figure 4.1B: PC1 loading violin on the WT mean
structure, in the actout_ctlfit inter-lobe subspace.

Written fresh against Tsjerk's princomp.py (which is where violin() actually
lives; pca.py has no violin). princomp.py needs only numpy, PyMOL and
colorinator, so this runs directly in system PyMOL. No MDAnalysis, and no venv
split of the kind the pocket figure needed.

Spec (thesis section 4.1, Figure 4.1 caption, and Appendix Table C1):
  space     actout_ctlfit (global inter-lobe geometry)
  fit       CTL backbone Ca, internal 120-283 (real 2053-2216)
  measured  whole body backbone Ca, internal 6-277 (real 1940-2210)
  stride    10 (every 500 ps at a 50 ps save interval)
  variant   WT only
  drawing   density modulated cylinders, radius from the KDE of frame sampling
            along PC1, coloured blue-white-red by PC1 score.

Parameters from the original handover: scale 10, radius 0.3, bw 1.0.

Usage (venv OFF, system PyMOL):
  env -u PYTHONPATH -u VIRTUAL_ENV PATH=/usr/bin:/bin /usr/bin/pymol -cq \
      analysis/pymol_B_pc1_violin.py

Notes on the library, verified by reading it:
  * violin() ignores its own colors= argument (it hardcodes the BOX palette),
    so the blue-white-red must be applied afterwards. We map BWR along the
    violin points, which is the direct analogue of what violin() does with BOX.
  * AtomicCGO.recolor(BWR, ...) raises NotImplementedError, because BWR is a
    Colorinator and that branch is unfinished. Its bw argument is also never
    passed through. We therefore colour with BWR.map() and the %= operator,
    which is the supported path.
  * CoordinateArray builds state indices with np.arange(count_states), which is
    0 based, while PyMOL states are 1 based. frames=slice(1, None) keeps every
    index a genuine state and avoids state 0 meaning "current state".
"""

import os
import sys

import numpy as np
from pymol import cmd

BASE = '/homes/lkgyammerah/Molecular_Dynamics_analysis'
TRAJ = os.path.join(BASE, 'trajectories/ros1_prepared_final')
FIGS = os.path.join(BASE, 'figures/ROS1/ros1_prepared_final')
sys.path.insert(0, os.path.join(BASE, 'analysis', 'scripts'))

from princomp import Princomp
from colorinator import BWR

VARIANT = 'WT'
STRIDE = int(os.environ.get('VIOLIN_STRIDE', 10))
# Length of the longest violin rod, in angstrom. Scores alone are the wrong
# handle here: the eigenvector is unit normalised across all measured atoms, so
# a raw "scale 10" gives only about 10/sqrt(natoms) = 0.6 A of motion per atom,
# which renders as a speck. We therefore divide by the largest per atom loading,
# so this number is the real displacement of the most mobile atom.
TARGET_DISP = float(os.environ.get('VIOLIN_DISP', 8.0))
RADIUS = float(os.environ.get('VIOLIN_RADIUS', 0.3))
BW = float(os.environ.get('VIOLIN_BW', 1.0))

FIT_SEL = 'name CA and resid 120-283'   # CTL, the fixed reference lobe
MEAS_SEL = 'name CA and resid 6-277'    # whole body, measured in the PCA
OUT_PNG = os.path.join(FIGS, 'pc1_violin_wt.png')

os.makedirs(FIGS, exist_ok=True)

cmd.reinitialize()
cmd.bg_color('white')
cmd.set('ray_opaque_background', 1)
cmd.set('orthoscopic', 1)
cmd.set('ray_shadows', 0)

# --- load the WT replicas and fit them on the CTL -------------------------
# Fitting on the CTL is what makes this the actout_ctlfit space: the C-terminal
# lobe is held fixed, so NTL motion appears explicitly in the PC scores.
objects = []
first = None
for rep in ('0', '1', '2'):
    stem = os.path.join(TRAJ, VARIANT, rep, f'{VARIANT}-MD-prot')
    if not (os.path.exists(stem + '.pdb') and os.path.exists(stem + '.xtc')):
        print(f'replica {rep}: not found, skipping')
        continue
    raw = f'raw_{rep}'
    cmd.load(stem + '.pdb', raw)
    cmd.load_traj(stem + '.xtc', raw, interval=STRIDE)   # stride at load time
    cmd.intra_fit(f'{raw} and ({FIT_SEL})')              # CTL fit within replica
    if first is None:
        first = raw
    else:
        cmd.align(f'{raw} and ({FIT_SEL})',              # CTL fit across replicas
                  f'{first} and ({FIT_SEL})', cycles=0)
    name = f'{VARIANT}_r{rep}'
    cmd.create(name, raw)          # bake the alignment into the coordinates
    if raw != first:
        cmd.delete(raw)
    objects.append(name)

if first is not None:
    cmd.delete(first)
if not objects:
    raise SystemExit(f'No {VARIANT} replicas found under {TRAJ}')

nstates = sum(cmd.count_states(o) for o in objects)
natoms = cmd.count_atoms(f'{objects[0]} and ({MEAS_SEL})')
print(f'{VARIANT}: {len(objects)} replicas, {nstates} states loaded '
      f'(stride {STRIDE}), {natoms} measured Ca atoms')

# --- PCA over all replicas ------------------------------------------------
# frames=slice(1, None) skips index 0, so every state index passed to PyMOL is
# a real 1 based state rather than 0, which PyMOL reads as "current state".
P = Princomp(MEAS_SEL, frames=slice(1, None))
frac = P.variances[0] / P.variances.sum()
print(f'PC1 explains {100 * frac:.1f}% of variance '
      f'({len(P.data.states)} frames, {P.mean.shape[0]} atoms)')

pc1 = P[1]

# --- scale so the longest rod is TARGET_DISP angstrom ---------------------
# displacement of an atom = score * that atom's loading vector, so the visible
# length is set by (max score) * (max per atom loading).
peak = np.abs(pc1.scores).max()
atomload = np.linalg.norm(pc1.loadings.reshape(-1, 3), axis=1).max()
if peak == 0 or atomload == 0:
    raise SystemExit('PC1 scores or loadings are all zero, nothing to draw.')

# report the true physical amplitude before any rescaling, so the display scale
# can be compared against what the simulation actually sampled.
raw_lo, raw_hi = pc1.scores.min(), pc1.scores.max()
raw_span = (raw_hi - raw_lo) * atomload
print(f'raw PC1 score range {raw_lo:.2f} to {raw_hi:.2f}; '
      f'max per atom loading {atomload:.4f}')
print(f'TRUE physical amplitude of the most mobile atom: {raw_span:.2f} A '
      f'(tip to tip)')

if os.environ.get('VIOLIN_RAW'):
    print('VIOLIN_RAW set: keeping the true amplitude, no display rescaling')
else:
    pc1.scores = pc1.scores / (peak * atomload) * TARGET_DISP
    span = (pc1.scores.max() - pc1.scores.min()) * atomload
    print(f'display rescaled: {TARGET_DISP:.1f} A from centre, '
          f'{span:.1f} A tip to tip')

# --- build the violin and colour it blue-white-red by PC1 score ------------
violin = pc1.violin(radius=RADIUS, bw=BW)
violin %= BWR.map(violin.points)     # blue at one extreme, white mid, red at the other
violin.draw('pc1_violin')
print(f'violin drawn: {len(violin.points)} points, '
      f'score range {pc1.scores.min():.2f} to {pc1.scores.max():.2f} A')

# --- WT mean structure underneath -----------------------------------------
P.drawmean('wt_mean')
cmd.hide('everything', 'wt_mean')
cmd.set('cartoon_trace_atoms', 1, 'wt_mean')   # Ca only model, so trace it
cmd.cartoon('tube', 'wt_mean')
cmd.show('cartoon', 'wt_mean')
cmd.set('cartoon_tube_radius', 0.15, 'wt_mean')   # thin, so the violins lead
cmd.color('grey80', 'wt_mean')

for o in objects:
    cmd.delete(o)

# zoom on the true scene extent. A CGO name is not an atom selection, so it
# cannot be zoomed on directly. Pinning an axis aligned bounding box does not
# work either: once the camera rotates, its corners project wider than the rods
# actually reach, which pads the frame. So pin the real rod tips. Every atom has
# a rod, and CTL rods are near zero length, so these tips also cover the
# structure itself. The view survives deleting them.
tips = np.concatenate([violin.cylinders[:, 0, 1:4],
                       violin.cylinders[:, -1, 4:7]])
for x, y, z in tips:
    cmd.pseudoatom('extent_pts', pos=[float(x), float(y), float(z)])

cmd.orient('wt_mean')
cmd.zoom('extent_pts', buffer=1.0, complete=1)
cmd.delete('extent_pts')
cmd.png(OUT_PNG, width=1600, height=1400, dpi=300, ray=1)
print(f'saved {OUT_PNG}')