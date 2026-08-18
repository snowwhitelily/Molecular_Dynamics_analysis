"""
pocket_lib.py -- shared helpers for the ATP pocket figure.

Prepares coordinates, selections and the frame-corrected drug pose. The active
site PCA and averaged pocket live in the NTL fitted frame (backbone, GRO 5-97,
name CA). The docking receptor and drug poses live in the ctlfit frame. The drug
is moved into the analysis frame by superposing the ctlfit receptor onto the
NTL-fitted WT structure using the NTL region only (the stable N-lobe carrying the
pocket), then applying that transform to the drug.

Motif set (real numbering; GRO = real - 1933): G-loop 1951-1959, beta3 K1980,
aC-helix 1988-2003, hinge 2026-2033, catalytic 2077-2084, DFG 2102-2104.
"""

import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.align import rotation_matrix

OFFSET = 1933
NTL_SEL    = 'backbone and resid 5:97 and name CA'
NTL_BB_SEL = 'backbone and resid 5:97'
BB_SEL     = 'protein and backbone'

MOTIF_REAL = [
    (1951, 1959), (1980, 1980), (1988, 2003),
    (2026, 2033), (2077, 2084), (2102, 2104),
]
MOTIF_GRO = [r - OFFSET for lo, hi in MOTIF_REAL for r in range(lo, hi + 1)]


def replica_paths(traj_dir, variant, replicas=('0', '1', '2')):
    out = []
    for rep in replicas:
        stem = os.path.join(traj_dir, variant, rep, f'{variant}-MD-prot')
        if os.path.exists(stem + '.pdb') and os.path.exists(stem + '.xtc'):
            out.append((stem + '.pdb', stem + '.xtc'))
    if not out:
        raise SystemExit(f'No trajectories found for {variant} in {traj_dir}')
    return out


def ntl_reference(traj_dir, variant='WT'):
    top, trj = replica_paths(traj_dir, variant)[0]
    u = mda.Universe(top, trj)
    u.trajectory[0]
    ntl = u.select_atoms(NTL_SEL)
    expect = 97 - 5 + 1
    if ntl.n_atoms != expect:
        raise SystemExit(f'NTL selection got {ntl.n_atoms} atoms, expected {expect}.')
    return u


def read_pdbqt_pose1(path):
    xyz, recording = [], False
    for line in open(path):
        s = line.rstrip('\n')
        if s.startswith('MODEL'):
            recording = s.split()[-1] == '1'; continue
        if s.startswith('ENDMDL'):
            if recording: break
            continue
        if recording and s.startswith(('ATOM', 'HETATM')):
            xyz.append([float(s[30:38]), float(s[38:46]), float(s[46:54])])
    if not xyz:
        raise SystemExit(f'No pose 1 atoms parsed from {path}')
    return np.asarray(xyz)


def correct_drug_to_ntl(drug_xyz, ctlfit_receptor_pdb, ntl_ref_universe):
    recep = mda.Universe(ctlfit_receptor_pdb)
    mob = recep.select_atoms(NTL_BB_SEL).positions.copy()
    ref = ntl_ref_universe.select_atoms(NTL_BB_SEL).positions.copy()
    if mob.shape != ref.shape:
        raise SystemExit(
            f'NTL backbone mismatch: receptor {mob.shape} vs reference {ref.shape}.')
    mc, rc = mob.mean(0), ref.mean(0)
    R, rmsd = rotation_matrix(mob - mc, ref - rc)
    drug_ntl = (drug_xyz - mc) @ R.T + rc
    return drug_ntl, rmsd


def averaged_backbone(traj_dir, variant, ntl_ref_universe):
    bb_sum = None; n = 0; writer_u = None
    for top, trj in replica_paths(traj_dir, variant):
        u = mda.Universe(top, trj)
        writer_u = u
        for _ in u.trajectory:
            align.alignto(u, ntl_ref_universe, select=NTL_SEL, weights='mass')
            p = u.select_atoms(BB_SEL).positions
            bb_sum = p.copy() if bb_sum is None else bb_sum + p
            n += 1
    bb_mean = bb_sum / n
    writer_u.trajectory[0]
    bb = writer_u.select_atoms(BB_SEL)
    bb.positions = bb_mean
    return bb, n


def pocket_resids(selection, ntl_ref_universe=None, drug_ntl=None, cutoff=5.0):
    if selection == 'motif':
        return sorted(MOTIF_GRO)
    if selection == 'distance':
        if ntl_ref_universe is None or drug_ntl is None:
            raise SystemExit('distance selection needs the WT universe and drug.')
        prot = ntl_ref_universe.select_atoms('protein')
        pos = prot.positions
        mind = np.full(len(pos), np.inf)
        for d in drug_ntl:
            mind = np.minimum(mind, np.sqrt(((pos - d) ** 2).sum(axis=1)))
        return sorted(set(int(r) for r in prot[mind <= cutoff].resids))
    raise SystemExit(f"selection must be 'motif' or 'distance', got {selection}")


def gro_to_real(gro):
    return gro + OFFSET
