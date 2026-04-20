import numpy as np
import MDAnalysis as mda
import molly
from pathlib import Path


def _prot_files(d):
    """
    Resolve the mutant-specific prepared protein files from a run directory.

    Expected layout
    ---------------
    d = .../<MUT>/<REP>/

    Files inside
    ------------
    <MUT>-MD-prot.pdb
    <MUT>-MD-prot.xtc
    """
    d = Path(d)
    mut = d.parent.name
    pdb = d / f"{mut}-MD-prot.pdb"
    xtc = d / f"{mut}-MD-prot.xtc"
    return pdb, xtc


def subsampled_len(d, stride=1):
    """
    Number of frames after subsampling by stride.
    """
    pdb, xtc = _prot_files(d)
    u = mda.Universe(str(pdb), str(xtc))
    return len(np.arange(len(u.trajectory))[::stride])


def get_frame_idx(d, stride=1, nsub=None):
    """
    Return subsampled frame indices for one trajectory.
    If nsub is provided, truncate to that length.
    """
    pdb, xtc = _prot_files(d)
    u = mda.Universe(str(pdb), str(xtc))
    idx = np.arange(len(u.trajectory))[::stride]
    if nsub is not None:
        idx = idx[:nsub]
    return idx


def xtc2array_varlen_by_indices(d, atom_indices, frame_selection=None):
    """
    Reads ONLY atom_indices (0-based indices into Universe atoms)

    Returns
    -------
    xyz : np.ndarray
        Shape (T, nsel, 3)
    pbc : np.ndarray
        Shape (T, 3, 3)
    """
    _, xtc = _prot_files(d)
    m = molly.XTCReader(str(xtc))
    nframes = len(m.read_frames(atom_selection=[], frame_selection=frame_selection))
    m.home()

    out = np.zeros((nframes, len(atom_indices), 3), dtype=np.float32)
    pbc = np.empty((nframes, 3, 3), dtype=np.float32)

    m.read_into_array(
        out,
        pbc,
        atom_selection=[int(i) for i in atom_indices],
        frame_selection=frame_selection
    )

    pbc = np.array(pbc).transpose((0, 2, 1))  # (T,3,3)
    return out, pbc


def selection_keys_and_indices(d, selection):
    """
    Returns:
      keys: set of (resid, name) for atoms in selection
      key_to_index: dict mapping key -> atom_index (0-based)
    """
    pdb, _ = _prot_files(d)
    u = mda.Universe(str(pdb))
    ag = u.select_atoms(selection)

    keys = []
    key_to_index = {}
    for a in ag:
        k = (int(a.resid), str(a.name))
        keys.append(k)
        key_to_index[k] = int(a.ix)

    return set(keys), key_to_index


def align_traj_to_ref(xyz, fit_mask, ref_fit_centered):
    """
    Align trajectory on COMMON atoms.

    Parameters
    ----------
    xyz : np.ndarray
        Shape (T, natoms, 3)
    fit_mask : np.ndarray
        Boolean mask of length natoms
    ref_fit_centered : np.ndarray
        Shape (nfit, 3), already centered

    Returns
    -------
    np.ndarray
        Aligned xyz, shape (T, natoms, 3), dtype float32
    """
    xyz = xyz.copy().astype(np.float64)

    xyz -= xyz[:, fit_mask].mean(axis=1, keepdims=True)
    xfit = xyz[:, fit_mask]

    m = np.einsum(
        "ij,tjk->tik",
        (ref_fit_centered.T / (3 * ref_fit_centered.shape[0])),
        xfit
    )

    u, s, vt = np.linalg.svd(m)
    r = u @ vt

    xyz = xyz @ np.transpose(r, (0, 2, 1))
    return xyz.astype(np.float32)