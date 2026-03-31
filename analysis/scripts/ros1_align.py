import numpy as np


def align_traj_to_ref_by_fit(xyz, fit_mask, ref_fit_centered):
    """
    Align one trajectory to a reference using a fit mask.

    Parameters
    ----------
    xyz : np.ndarray
        Shape (T, natoms, 3)
    fit_mask : np.ndarray
        Boolean mask of length natoms
    ref_fit_centered : np.ndarray
        Reference coordinates on fit atoms, already centered.
        Shape (nfit, 3)

    Returns
    -------
    np.ndarray
        Aligned trajectory, shape (T, natoms, 3), dtype float32
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


def kabsch_R(p, q):
    """
    Rotation R such that (p @ R) ~= q

    Parameters
    ----------
    p, q : np.ndarray
        Shape (n, 3), already centered

    Returns
    -------
    np.ndarray
        Rotation matrix shape (3, 3)
    """
    c = p.T @ q
    u, s, vt = np.linalg.svd(c)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r
