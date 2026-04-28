import numpy as np
import MDAnalysis as mda


def ensure_meta(meta=None, top=None, selection=None, natoms=None):
    if meta is None:
        if top is None:
            raise ValueError("Need `top` when meta is None.")
        if selection is None:
            raise ValueError("Need `selection` when meta is None.")
        u = mda.Universe(top)
        meta = u.select_atoms(selection)

    if natoms is not None and len(meta) != natoms:
        raise ValueError(f"meta atoms ({len(meta)}) != expected natoms ({natoms})")

    return meta


def ensure_exists_all(Multiverse, exists_all=None):
    natoms = Multiverse.shape[2]
    if exists_all is not None and len(exists_all) == natoms:
        return exists_all

    eps = 1e-6
    valid_sys_atom = np.any(np.linalg.norm(Multiverse, axis=3) > eps, axis=1)
    exists_all = np.all(valid_sys_atom, axis=0)
    return exists_all


def build_standard_masks(meta, exists_all=None):
    natoms = len(meta)
    if exists_all is None:
        exists_all = np.ones(natoms, dtype=bool)

    CA = (meta.names == "CA")
    BB = np.isin(meta.names, ["N", "CA", "C", "O"])

    # BODY: core kinase domain backbone
    # Internal residues 6-277 = real residues 1940-2210
    # Already correctly trimmed — no changes needed here.
    # Used by: ACT-IN, ACT-OUT, ACT-OUT CTL-fit PCAs (all clean)
    BODY = np.isin(meta.resids, np.arange(6, 278))

    resids_atom = meta.resids.astype(int) + 1933

    # ACT (activation loop): real residues 2045-2070
    # Internal residues 112-137
    # No changes needed here.
    ACT = (meta.resids >= (2045 - 1933)) & (meta.resids <= (2070 - 1933))

    # NTL (N-terminal lobe): real residues 1934-2030
    # ORIGINAL: started at internal residue 1 (real 1934 = ILE)
    # PROBLEM:  internal residues 1-4 (ILE, GLU, ASN, LEU) are floppy
    #           N-terminal ends with no upstream structure to anchor them.
    #           They dominated NTL PC1 with a loading magnitude of 1.4,
    #           which is an artifact not a biological signal.
    # FIX:      start at internal residue 5 (real residue 1938 = PRO)
    #           This removes the 4 floppy terminal residues.
    # Confirmed from PDB: resid 1=ILE, 2=GLU, 3=ASN, 4=LEU, 5=PRO
    NTL = (meta.resids >= 5) & (meta.resids <= (2030 - 1933))

    # CTL (C-terminal lobe): real residues 2031-2225
    # ORIGINAL: ended at internal residue 292 (real 2225 = SER)
    # PROBLEM:  internal residues 291-292 (ASN, SER) are floppy
    #           C-terminal ends with no downstream structure to anchor them.
    #           They dominated CTL PC2 with a loading magnitude of 1.0,
    #           which is an artifact not a biological signal.
    # FIX:      end at internal residue 290 (real residue 2223 = LEU)
    #           This removes the 2 floppy terminal residues.
    # Confirmed from PDB: resid 290=LEU, 291=ASN, 292=SER
    CTL = (meta.resids >= (2031 - 1933)) & (meta.resids <= 290)

    return {
        "CA":          CA & exists_all,
        "BB":          BB & exists_all,
        "BODY":        BODY & exists_all,
        "ACT":         ACT & exists_all,
        "NTL":         NTL & exists_all,
        "CTL":         CTL & exists_all,
        "resids_atom": resids_atom,
        "exists_all":  exists_all,
    }


def combine_masks(*masks):
    out = masks[0].copy()
    for m in masks[1:]:
        out &= m
    return out


def traj_frames_atoms(xyz, mask):
    return xyz[:, mask].reshape(xyz.shape[0], -1)


def principal_axis(coords):
    _, _, Vt = np.linalg.svd(coords, full_matrices=False)
    v = Vt[0]
    return v / (np.linalg.norm(v) + 1e-12)


def angle(u, v):
    c = np.clip(np.dot(u, v), -1.0, 1.0)
    return np.arccos(c)