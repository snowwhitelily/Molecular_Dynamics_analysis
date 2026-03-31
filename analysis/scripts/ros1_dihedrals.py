import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_dihedrals


def get_atom_indices_for_resid(u, resid, names=("N", "CA", "CB", "CG")):
    """
    Return atom indices for the requested residue/atom names.

    Parameters
    ----------
    u : MDAnalysis.Universe
    resid : int
    names : tuple[str, ...]

    Returns
    -------
    list[int]
        Atom indices in the same order as names
    """
    idxs = []
    for nm in names:
        ag = u.select_atoms(f"protein and resid {resid} and name {nm}")
        if len(ag) != 1:
            raise ValueError(
                f"[DFG χ1] Could not uniquely select resid {resid} atom {nm}. "
                f"Found {len(ag)} atoms. Check numbering/topology."
            )
        idxs.append(int(ag.indices[0]))
    return idxs


def compute_chi1_deg_for_traj(pdb, xtc, resid):
    """
    Compute chi1 dihedral in degrees for all frames of one trajectory.

    Parameters
    ----------
    pdb : str
    xtc : str
    resid : int

    Returns
    -------
    np.ndarray
        Chi1 values in degrees, shape (nframes,)
    """
    u = mda.Universe(pdb, xtc)
    iN, iCA, iCB, iCG = get_atom_indices_for_resid(u, resid)

    n_local = u.trajectory.n_frames
    coords_N = np.empty((n_local, 3), dtype=np.float32)
    coords_CA = np.empty((n_local, 3), dtype=np.float32)
    coords_CB = np.empty((n_local, 3), dtype=np.float32)
    coords_CG = np.empty((n_local, 3), dtype=np.float32)

    for fi, ts in enumerate(u.trajectory):
        coords_N[fi] = u.atoms[iN].position
        coords_CA[fi] = u.atoms[iCA].position
        coords_CB[fi] = u.atoms[iCB].position
        coords_CG[fi] = u.atoms[iCG].position

    chi1 = calc_dihedrals(coords_N, coords_CA, coords_CB, coords_CG)
    return np.degrees(chi1).astype(np.float32)
