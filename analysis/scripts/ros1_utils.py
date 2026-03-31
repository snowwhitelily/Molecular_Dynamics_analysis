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
    BODY = np.isin(meta.resids, np.arange(6, 278))

    resids_atom = meta.resids.astype(int) + 1933

    ACT = (meta.resids >= (2045 - 1933)) & (meta.resids <= (2070 - 1933))
    NTL = (meta.resids >= (1934 - 1933)) & (meta.resids <= (2030 - 1933))
    CTL = (meta.resids >= (2031 - 1933)) & (meta.resids <= (2225 - 1933))

    return {
        "CA": CA & exists_all,
        "BB": BB & exists_all,
        "BODY": BODY & exists_all,
        "ACT": ACT & exists_all,
        "NTL": NTL & exists_all,
        "CTL": CTL & exists_all,
        "resids_atom": resids_atom,
        "exists_all": exists_all,
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
