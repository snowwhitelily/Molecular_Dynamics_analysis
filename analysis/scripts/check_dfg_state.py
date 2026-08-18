"""
check_dfg_state.py -- decide DFG-in vs DFG-out using three independent measures,
so the answer does not rest on one ambiguous definition.

Usage (venv on):
    python check_dfg_state.py FILE [FILE ...]

Run it on structures of KNOWN state first (your own active receptors) to
calibrate, then on the unknowns. A metric that cannot separate your known
active receptors from the published inactive one is not measuring DFG state.

Measures reported per structure:
  1. CA angle K1980 - E1997 - F2103, vertex at E1997. This is the Vilacha et al.
     "DFG rotation angle" as far as it can be reconstructed from the text, but
     the vertex atom is not stated explicitly, so it is calibrated, not trusted.
     Reference: active 31 +/- 6 deg, inactive 72 +/- 8 deg.
  2. Distance from the F2103 ring (CZ) to the gatekeeper L2026 CA. In DFG-out
     the phenylalanine swings into the ATP pocket, which the gatekeeper lines,
     so this distance is SHORT when out and LONG when in.
  3. Distance from the F2103 ring (CZ) to the E1997 CA on the aC-helix. In
     DFG-in the phenylalanine sits in the back pocket under the aC-helix, so
     this is SHORT when in and LONG when out.

Measures 2 and 3 move in opposite directions, so their difference is a robust
discriminator that does not depend on any angle convention.
"""

import sys

import numpy as np
import MDAnalysis as mda

TARGETS = {'K1980': ('LYS', 1980), 'E1997': ('GLU', 1997),
           'L2026': ('LEU', 2026), 'F2103': ('PHE', 2103)}


def pick(u, offset):
    got = {}
    for label, (want, real) in TARGETS.items():
        ag = u.select_atoms(f'resid {real + offset}')
        if len(ag) == 0:
            return None
        got[label] = ag
    return got


def angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def ca(ag):
    sel = ag.select_atoms('name CA')
    return sel.positions[0]


def ring(ag):
    """F2103 ring centre: prefer CZ, else the mean of the aromatic carbons."""
    cz = ag.select_atoms('name CZ')
    if len(cz):
        return cz.positions[0]
    ar = ag.select_atoms('name CG CD1 CD2 CE1 CE2 CZ')
    if len(ar) == 0:
        return None
    return ar.positions.mean(axis=0)


print(f'{"structure":42s} {"angle":>7s} {"F2103-L2026":>12s} '
      f'{"F2103-E1997":>12s} {"call":>10s}')
print('-' * 90)

for path in sys.argv[1:]:
    try:
        u = mda.Universe(path)
    except Exception as exc:
        print(f'{path[-42:]:42s}  could not read: {exc}')
        continue

    got = None
    for offset in (0, -1933):
        got = pick(u, offset)
        if got is not None:
            break
    if got is None:
        print(f'{path.split("/")[-1][:42]:42s}  residues not found')
        continue

    names = {k: v.residues[0].resname for k, v in got.items()}
    bad = [k for k, (want, _) in TARGETS.items() if names[k] != want]

    ang = angle(ca(got['K1980']), ca(got['E1997']), ca(got['F2103']))
    r = ring(got['F2103'])
    d_gate = np.linalg.norm(r - ca(got['L2026'])) if r is not None else float('nan')
    d_ac = np.linalg.norm(r - ca(got['E1997'])) if r is not None else float('nan')

    # DFG-out puts the ring in the ATP pocket (near the gatekeeper) and away
    # from the aC-helix, so d_gate < d_ac. DFG-in is the reverse.
    call = 'OUT?' if d_gate < d_ac else 'IN?'

    flag = '  <- resname mismatch: ' + ','.join(bad) if bad else ''
    print(f'{path.split("/")[-1][:42]:42s} {ang:7.1f} {d_gate:12.1f} '
          f'{d_ac:12.1f} {call:>10s}{flag}')

print('-' * 90)
print('Calibrate on structures of known state before trusting any column.')
print('If the known-active receptors do not separate cleanly from the')
print('published inactive structure, the metric is not measuring DFG state.')