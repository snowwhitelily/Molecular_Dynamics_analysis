"""DFG rotation angle: Ca angle K1980 - E1997 - F2103, vertex at E1997.
Active ~31 +/- 6 deg, inactive ~72 +/- 8 deg (Vilacha et al. 2025)."""
import sys
import numpy as np
import MDAnalysis as mda

u = mda.Universe(sys.argv[1])
for off, label in ((0, 'real numbering'), (-1933, 'GRO numbering')):
    k = u.select_atoms(f'name CA and resid {1980 + off}')
    e = u.select_atoms(f'name CA and resid {1997 + off}')
    f = u.select_atoms(f'name CA and resid {2103 + off}')
    if len(k) == len(e) == len(f) == 1:
        print(f'using {label}')
        break
else:
    raise SystemExit('could not locate K1980 / E1997 / F2103 CA atoms')

print(f'identities: {k.resnames[0]} (want LYS), {e.resnames[0]} (want GLU), '
      f'{f.resnames[0]} (want PHE)')
v1 = k.positions[0] - e.positions[0]
v2 = f.positions[0] - e.positions[0]
ang = np.degrees(np.arccos(np.clip(
    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
print(f'DFG rotation angle = {ang:.1f} deg')
print('-> ~31 deg means ACTIVE (DFG-in); ~72 deg means INACTIVE (DFG-out)')
