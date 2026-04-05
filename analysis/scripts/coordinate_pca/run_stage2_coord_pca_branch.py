import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.scripts.timer import Timer
from analysis.scripts.ros1_utils import ensure_meta, build_standard_masks, combine_masks
from analysis.scripts.ros1_align import align_traj_to_ref_by_fit
from analysis.scripts.ros1_plots import plot_scree

tim = Timer()

STAGE1 = Path("analysis/outputs/stage1_common_align")


def load_list_npz(path):
    z = np.load(path, allow_pickle=True)
    keys = sorted(z.files, key=lambda x: int(x.split("_")[1]))
    return [z[k] for k in keys]


def save_list_npz(path, arr_list):
    np.savez_compressed(path, **{f"traj_{i}": np.asarray(arr) for i, arr in enumerate(arr_list)})


def branch_config(branch, meta, masks):
    BB = masks["BB"]
    BODY = masks["BODY"]
    ACT = masks["ACT"]
    NTL = masks["NTL"]
    CTL = masks["CTL"]

    if branch == "actin":
        sele = combine_masks(BB, BODY)
        fit = None
        split_mode = "frac80"
        prefix = "actin"
        title = "ACT-IN"
        ncomponents = 10

    elif branch == "actout_plain":
        sele = combine_masks(BB, BODY, ~ACT)
        fit = sele.copy()
        split_mode = "frac80"
        prefix = "actout"
        title = "ACT-OUT"
        ncomponents = 10

    elif branch == "actout_ctlfit":
        sele = combine_masks(BB, BODY, ~ACT)
        fit = combine_masks(BB, CTL, ~ACT)
        split_mode = "frac80"
        prefix = "actout_ctlfit"
        title = "ACT-OUT CTL-fit"
        ncomponents = 10

    elif branch == "ntl":
        sele = combine_masks(BB, NTL)
        fit = sele.copy()
        split_mode = "frac80"
        prefix = "ntl"
        title = "NTL"
        ncomponents = 10

    elif branch == "ctl":
        sele = combine_masks(BB, CTL, ~ACT)
        fit = sele.copy()
        split_mode = "frac80"
        prefix = "ctl"
        title = "CTL"
        ncomponents = 10

    elif branch == "aloop":
        sele = combine_masks(ACT, BB)
        fit = sele.copy()
        split_mode = "first1"
        prefix = "aloop"
        title = "A-loop"
        ncomponents = 10

    elif branch == "tyrloop":
        loop_res = np.arange(170, 191)
        loop_atoms = np.isin(meta.resids, loop_res) & masks["exists_all"]
        sele = combine_masks(loop_atoms, BB)
        fit = combine_masks(BB, BODY)
        split_mode = "first1"
        prefix = "tyrloop"
        title = "Tyrosine loop"
        ncomponents = 10

    else:
        raise ValueError(f"Unknown branch: {branch}")

    return {
        "sele": sele,
        "fit": fit,
        "split_mode": split_mode,
        "prefix": prefix,
        "title": title,
        "ncomponents": ncomponents,
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_stage2_coord_pca_branch.py <branch>")

    branch = sys.argv[1]
    cfg_out = None

    F = np.load(STAGE1 / "F.npy", allow_pickle=True).tolist()
    per_sim_indices = np.load(STAGE1 / "per_sim_indices.npy", allow_pickle=True)
    ref_centered = np.load(STAGE1 / "ref_centered.npy")
    traj_aligned = load_list_npz(STAGE1 / "traj_aligned_list.npz")

    natoms = traj_aligned[0].shape[1]
    u0 = mda.Universe(f"{F[0]}/md-prot.pdb")
    meta = u0.atoms[np.asarray(per_sim_indices[0], dtype=int)]
    meta = ensure_meta(meta=meta, natoms=natoms)

    masks = build_standard_masks(meta)
    cfg = branch_config(branch, meta, masks)
    cfg_out = cfg

    OUTDIR = Path(f"analysis/outputs/stage2_{cfg['prefix']}_pca")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    sele = cfg["sele"]
    fit = cfg["fit"]
    split_mode = cfg["split_mode"]
    ncomponents = cfg["ncomponents"]

    print("=" * 80)
    print(f"STAGE 2: {cfg['title']} PCA")
    print("=" * 80)
    print("Loaded trajectories:", len(F))
    print("Example shape:", traj_aligned[0].shape)
    print(f"{cfg['title']} sele atoms:", int(sele.sum()), "fit atoms:", int(fit.sum()) if fit is not None else int(sele.sum()))

    if fit is None:
        traj_use = traj_aligned
    else:
        ref_fit = ref_centered[fit]
        ref_fit = ref_fit - ref_fit.mean(axis=0, keepdims=True)
        with tim(f"{cfg['title']}: aligning all trajectories on fit"):
            traj_use = [align_traj_to_ref_by_fit(xyz, fit, ref_fit) for xyz in traj_aligned]

    with tim(f"{cfg['title']}: vectorizing deviations + building P/X"):
        all_frames = []
        for xyz in traj_use:
            all_frames.append(xyz[:, sele, :].reshape(xyz.shape[0], -1))
        all_frames = np.vstack(all_frames)

        mean_vec = all_frames.mean(axis=0)
        d = mean_vec.size

        P_list, X_list = [], []
        P_owner, X_owner = [], []

        for ti, xyz in enumerate(traj_use):
            dev = xyz[:, sele, :].reshape(xyz.shape[0], -1) - mean_vec

            if split_mode == "frac80":
                split_idx = int(0.8 * dev.shape[0])
            elif split_mode == "first1":
                split_idx = min(1, dev.shape[0])
            else:
                raise ValueError(split_mode)

            P_i = dev[:split_idx]
            X_i = dev[split_idx:]

            if P_i.shape[0]:
                P_list.append(P_i.astype(np.float32))
                P_owner.append(np.full(P_i.shape[0], ti, dtype=int))
            if X_i.shape[0]:
                X_list.append(X_i.astype(np.float32))
                X_owner.append(np.full(X_i.shape[0], ti, dtype=int))

        P = np.vstack(P_list) if P_list else np.zeros((0, d), dtype=np.float32)
        X = np.vstack(X_list) if X_list else np.zeros((0, d), dtype=np.float32)
        P_owner = np.concatenate(P_owner) if P_owner else np.zeros((0,), dtype=int)
        X_owner = np.concatenate(X_owner) if X_owner else np.zeros((0,), dtype=int)

    print(f"{cfg['title']} P shape:", P.shape)
    print(f"{cfg['title']} X shape:", X.shape)

    with tim(f"{cfg['title']}: PCA diagonalization"):
        C = X.T @ (X / len(X))
        evals, loadings = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        loadings = loadings[:, order]

    plot_scree(evals, n=25, title=f"Scree ({cfg['title']} PCA)")
    print(f"{cfg['title']} cumulative variance (first 25):")
    print(np.round(np.cumsum(evals[:25]) / evals.sum(), 2))

    scale = np.sqrt(mean_vec.size)
    with tim(f"{cfg['title']}: projections on first {ncomponents} PCs"):
        pscores_flat = (P @ loadings[:, :ncomponents]) / scale
        scores_flat = (X @ loadings[:, :ncomponents]) / scale

    # quick scatter
    if scores_flat.shape[0] > 0:
        plt.figure(figsize=(7, 7))
        plt.scatter(scores_flat[:, 0], scores_flat[:, 1], s=2, alpha=0.25, linewidths=0)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"{cfg['title']} PCA")
        plt.tight_layout()
        plt.savefig(f"figures/{cfg['prefix']}_pca_scatter.png", dpi=300, bbox_inches="tight")
        plt.close()

    # save generic
    np.save(OUTDIR / "mean_vec.npy", mean_vec.astype(np.float32))
    np.save(OUTDIR / "evals.npy", evals.astype(np.float32))
    np.save(OUTDIR / "loadings.npy", loadings.astype(np.float32))
    np.save(OUTDIR / "pscores_flat.npy", pscores_flat.astype(np.float32))
    np.save(OUTDIR / "scores_flat.npy", scores_flat.astype(np.float32))
    np.save(OUTDIR / "P_owner.npy", P_owner.astype(int))
    np.save(OUTDIR / "X_owner.npy", X_owner.astype(int))
    np.save(OUTDIR / "sele_mask.npy", sele.astype(bool))
    np.save(OUTDIR / "fit_mask.npy", (fit if fit is not None else sele).astype(bool))
    save_list_npz(OUTDIR / "scores_list.npz", [scores_flat[X_owner == ti].copy() for ti in range(len(F))])

    # save compatibility names for special branches
    if branch == "actout_ctlfit":
        np.save(OUTDIR / "scores_flat_actout_ctlfit.npy", scores_flat.astype(np.float32))
        np.save(OUTDIR / "pscores_flat_actout_ctlfit.npy", pscores_flat.astype(np.float32))
        np.save(OUTDIR / "X_owner_actout_ctlfit.npy", X_owner.astype(int))
        np.save(OUTDIR / "P_owner_actout_ctlfit.npy", P_owner.astype(int))
        save_list_npz(OUTDIR / "scores_actout_ctlfit.npz", [scores_flat[X_owner == ti].copy() for ti in range(len(F))])

    with open(OUTDIR / "summary.txt", "w") as fh:
        fh.write(f"branch={branch}\n")
        fh.write(f"ntraj={len(F)}\n")
        fh.write(f"sele_atoms={int(sele.sum())}\n")
        fh.write(f"fit_atoms={int((fit if fit is not None else sele).sum())}\n")
        fh.write(f"P_shape={P.shape}\n")
        fh.write(f"X_shape={X.shape}\n")

    print("\nSaved outputs to:", OUTDIR)
    print("DONE.")


if __name__ == "__main__":
    main()
