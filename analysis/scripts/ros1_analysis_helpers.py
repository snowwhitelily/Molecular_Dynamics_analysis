import numpy as np
import matplotlib.pyplot as plt


def switch_count(x01):
    x01 = np.asarray(x01)
    return int(np.sum(x01[1:] != x01[:-1]))


def mean_dwell_frames(x01):
    x = np.asarray(x01).astype(int)
    cuts = np.where(x[1:] != x[:-1])[0] + 1
    run_lengths = np.diff(np.r_[0, cuts, len(x)])
    return float(run_lengths.mean())


def cohens_d(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    sa = a.std(ddof=1) + 1e-12
    sb = b.std(ddof=1) + 1e-12
    sp = np.sqrt(((len(a) - 1) * sa * sa + (len(b) - 1) * sb * sb) / (len(a) + len(b) - 2) + 1e-12)
    return (a.mean() - b.mean()) / sp


def summarize_pair(A, B, ld1_frames, names_pair):
    a = np.concatenate([ld1_frames[i] for i in range(len(names_pair)) if names_pair[i] == A])
    b = np.concatenate([ld1_frames[i] for i in range(len(names_pair)) if names_pair[i] == B])

    d = cohens_d(a, b)

    stdA = np.mean([ld1_frames[i].std() for i in range(len(names_pair)) if names_pair[i] == A])
    stdB = np.mean([ld1_frames[i].std() for i in range(len(names_pair)) if names_pair[i] == B])

    sep = abs(a.mean() - b.mean())
    within = 0.5 * (stdA + stdB) + 1e-12
    ratio = sep / within

    return {
        "cohens_d": float(d),
        "sep_over_within": float(ratio),
        "meanA": float(a.mean()),
        "meanB": float(b.mean()),
        "stdA_within_traj": float(stdA),
        "stdB_within_traj": float(stdB),
        "nframesA": int(a.size),
        "nframesB": int(b.size),
    }


def lda_multiclass(X, y, eps=1e-8):
    classes = np.unique(y)
    k = X.shape[1]
    means = {c: X[y == c].mean(axis=0, keepdims=True) for c in classes}
    overall_mean = X.mean(axis=0, keepdims=True)

    sw = np.zeros((k, k), dtype=np.float64)
    for c in classes:
        xc = X[y == c] - means[c]
        sw += xc.T @ xc

    sb = np.zeros((k, k), dtype=np.float64)
    for c in classes:
        n_c = (y == c).sum()
        diff = (means[c] - overall_mean)
        sb += n_c * (diff.T @ diff)

    sw_reg = sw + eps * np.eye(k)
    m = np.linalg.solve(sw_reg, sb)
    eigvals, eigvecs = np.linalg.eig(m)
    order = np.argsort(eigvals.real)[::-1]
    eigvals = eigvals.real[order]
    w = eigvecs.real[:, order]
    return w, eigvals


def lda_2class_direction(X, y01, eps=1e-10):
    x0 = X[y01 == 0]
    x1 = X[y01 == 1]
    m0 = x0.mean(axis=0)
    m1 = x1.mean(axis=0)
    sw = (x0 - m0).T @ (x0 - m0) + (x1 - m1).T @ (x1 - m1)
    sw = sw + eps * np.eye(X.shape[1])
    w = np.linalg.solve(sw, (m1 - m0))
    w = w / (np.linalg.norm(w) + 1e-12)
    return w


def lda_2class_w(X, y01, eps=1e-10):
    return lda_2class_direction(X, y01, eps=eps)


def _as_mean_xyz(mean_obj, n_atoms):
    m = np.asarray(mean_obj)
    if m.ndim == 2 and m.shape == (n_atoms, 3):
        return m.astype(float)
    if m.ndim == 1 and m.size == 3 * n_atoms:
        return m.reshape(n_atoms, 3).astype(float)
    raise ValueError(f"Mean has shape {m.shape}, not compatible with n_atoms={n_atoms}")


def _infer_n_atoms_from_loadings(L):
    l = np.asarray(L)
    if l.ndim != 2 or (l.shape[0] % 3 != 0):
        raise ValueError(f"Loadings shape {l.shape} not xyz-style")
    return l.shape[0] // 3


def _mean_atom_count(obj):
    obj = np.asarray(obj)
    if obj.ndim == 2 and obj.shape[1] == 3:
        return obj.shape[0]
    if obj.ndim == 1 and obj.size % 3 == 0:
        return obj.size // 3
    return None


def _get_sel_idx_from_mask(mask):
    mask = np.asarray(mask).astype(bool)
    return np.where(mask)[0].astype(int)


def _resolve_sel_idx(globals_dict, n_atoms, sel_idx_name=None, sele_name=None):
    if sel_idx_name is not None and sel_idx_name in globals_dict:
        idx = np.asarray(globals_dict[sel_idx_name]).astype(int)
        if idx.size == n_atoms:
            return idx

    if sele_name is not None and sele_name in globals_dict:
        idx = _get_sel_idx_from_mask(globals_dict[sele_name])
        if idx.size == n_atoms:
            return idx

    for nm in ["sel_idx_loop", "sel_idx_actout", "sel_idx_act", "sel_idx_global", "sel_idx"]:
        if nm in globals_dict:
            idx = np.asarray(globals_dict[nm]).astype(int)
            if idx.size == n_atoms:
                return idx

    for nm in ["sele_actout", "sele_act", "sele", "sele_loop"]:
        if nm in globals_dict:
            idx = _get_sel_idx_from_mask(globals_dict[nm])
            if idx.size == n_atoms:
                return idx

    return None


def write_mode_movie_pdb(template_pdb, sel_idx, mean_xyz, mode_xyz, out_pdb, alpha=2.0, nsteps=21):
    import MDAnalysis as mda

    mean_xyz = np.asarray(mean_xyz, float).reshape(-1, 3)
    mode_xyz = np.asarray(mode_xyz, float).reshape(-1, 3)

    u = mda.Universe(template_pdb)
    ag = u.atoms[sel_idx]

    mode_xyz = mode_xyz / (np.linalg.norm(mode_xyz) + 1e-12)

    tvals = np.linspace(-1, 1, nsteps)
    with mda.Writer(out_pdb, ag.n_atoms, multiframe=True) as W:
        for t in tvals:
            ag.positions = mean_xyz + (t * alpha) * mode_xyz
            W.write(ag)

    print(f"[WROTE] {out_pdb}")


def pairwise_ld1_scatter(ld1_traj, y01, A, B, title=None):
    plt.figure(figsize=(6, 2.8))
    plt.scatter(ld1_traj[y01 == 0], np.zeros_like(ld1_traj[y01 == 0]), s=80, alpha=0.85, label=A)
    plt.scatter(ld1_traj[y01 == 1], np.ones_like(ld1_traj[y01 == 1]), s=80, alpha=0.85, label=B)
    plt.xlabel("LD1")
    plt.yticks([0, 1], [A, B])
    plt.title(title or f"Pairwise LDA (trajectory means): {A} vs {B}")
    plt.legend()
    plt.show()
