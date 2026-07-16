"""
check_dfg_calibrate.py -- work out which angle convention Vilacha et al. used,
by testing all three against structures of known state, then apply it.

The paper (ACS Omega 2025, 10, 22837-22846) defines DFG rotation as "an angle
based on the alpha carbon of residues K1980-E1997-F2103", and reports:
    active   ~31 deg
    inactive ~72 deg, WT inactive peak at 75 deg (DFG Phe fills the ATP pocket)
It does not state which atom is the vertex. A triangle has three, so all three
are computed here. The correct convention is the one that reads ~31 deg on
structures we already know are DFG-in.

The DFG motif is located by SEQUENCE (Asp-Phe-Gly), not by residue number, so a
numbering offset in any file cannot silently point us at the wrong Phe.

Usage (venv on):
    python check_dfg_calibrate.py FILE [FILE ...]

Put the known active receptors first. Whatever convention makes them ~31 deg is
the paper's convention.
"""

import sys

import numpy as np
import MDAnalysis as mda


def angle_at(vertex, a, b):
    v1, v2 = a - vertex, b - vertex
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def atom(sel, name):
    a = sel.select_atoms(f'name {name}')
    return a.positions[0] if len(a) else None


hdr = (f'{"structure":34s} {"DFG resid":>9s} {"vtxE1997":>9s} {"vtxK1980":>9s} '
       f'{"vtxF2103":>9s} {"Phe chi1":>9s} {"F2103-L2026":>11s}')
print(hdr)
print('-' * len(hdr))

for path in sys.argv[1:]:
    name = path.split('/')[-1][:34]
    try:
        u = mda.Universe(path)
    except Exception as exc:
        print(f'{name:34s}  unreadable: {exc}')
        continue

    prot = u.select_atoms('protein')
    seq = [r.resname for r in prot.residues]

    # locate DFG by sequence; keep the one nearest resid 2102 if several
    hits = [i for i in range(len(seq) - 2)
            if (seq[i], seq[i + 1], seq[i + 2]) == ('ASP', 'PHE', 'GLY')]
    if not hits:
        print(f'{name:34s}  no Asp-Phe-Gly motif found')
        continue
    i = min(hits, key=lambda j: abs(int(prot.residues[j].resid) - 2102))
    dres, fres = prot.residues[i], prot.residues[i + 1]

    k = prot.select_atoms('resid 1980')
    e = prot.select_atoms('resid 1997')
    g = prot.select_atoms('resid 2026')
    if not (k.n_residues and e.n_residues and g.n_residues):
        print(f'{name:34s}  K1980/E1997/L2026 not found at those numbers')
        continue

    ca_k, ca_e = atom(k, 'CA'), atom(e, 'CA')
    ca_f = atom(fres.atoms, 'CA')
    ca_g = atom(g, 'CA')

    a_e = angle_at(ca_e, ca_k, ca_f)      # vertex E1997 (what we assumed)
    a_k = angle_at(ca_k, ca_e, ca_f)      # vertex K1980
    a_f = angle_at(ca_f, ca_k, ca_e)      # vertex F2103

    n, ca, cb, cg = (atom(fres.atoms, x) for x in ('N', 'CA', 'CB', 'CG'))
    chi = dihedral(n, ca, cb, cg) if all(x is not None for x in (n, ca, cb, cg)) \
        else float('nan')

    cz = atom(fres.atoms, 'CZ')
    dg = np.linalg.norm(cz - ca_g) if cz is not None else float('nan')

    print(f'{name:34s} {int(dres.resid):9d} {a_e:9.1f} {a_k:9.1f} '
          f'{a_f:9.1f} {chi:9.1f} {dg:11.1f}')

print('-' * len(hdr))
print('Paper: active ~31 deg, inactive ~72 deg (WT inactive peak 75 deg).')
print('Pick the vertex column that reads ~31 on the known ACTIVE receptors;')
print('that column is the paper\'s convention. DFG-out should also put F2103')
print('IN the ATP pocket, i.e. a SHORTER F2103-L2026 distance than active.')