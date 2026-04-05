import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.scripts.timer import Timer
from analysis.scripts.ros1_utils import ensure_meta, build_standard_masks, combine_masks, principal_axis, angle

tim = Timer()
STAGE1 = Path("analysis/outputs/stage1_common_align")
OUTDIR = Path("analysis/outputs/stage2_orientation_pca")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_list_npz(path):
    z = np.load(path, allow_pickle=True)
    keys = sorted(z.files, key=lambda x: int(x.split("_")[1]))
    return [z[k] for k in keys]


def plot_scree(evals, title):
    var_ratio = evals / evals.sum()
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(evals) + 1), var_ratio, marker="o")
    plt.xlabel("PC")
    plt.ylabel("Explained variance ratio")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("figures/orientation_scree.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 80)
    print("STAGE 2: Orientation PCA")
    print("=" * 80)

    F = np.load(STAGE1 / "F.npy", allow_pickle=True).tolist()
    per_sim_indices = np.load(STAGE1 / "per_sim_indices.npy", allow_pickle=True)
    traj_aligned = load_list_npz(STAGE1 / "traj_aligned_list.npz")

    natoms = traj_aligned[0].shape[1]
    u0 = mda.Universe(f"{F[0]}/md-prot.pdb")
    meta = u0.atoms[np.asarray(per_sim_indices[0], dtype=int)]
    meta = ensure_meta(meta=meta, natoms=natoms)

    masks = build_standard_masks(meta)
    CA = masks["CA"]
    BODY = masks["BODY"]
    ACT = masks["ACT"]
    NTL = masks["NTL"]
    CTL = masks["CTL"]

    mask_ctl = combine_masks(CTL, BODY, CA, ~ACT)
    mask_ntl = combine_masks(NTL, BODY, CA)

    ctl_idx = np.where(mask_ctl)[0]
    ntl_idx = np.where(mask_ntl)[0]

    print("CTL CA atoms:", ctl_idx.size, "| NTL CA atoms:", ntl_idx.size)

    if ctl_idx.size < 5 or ntl_idx.size < 5:
        raise ValueError("CTL/NTL CA selections are too small.")

    with tim("FAST orientation features (NTL vs CTL)"):
        feats_list = []
        owner_list = []
        d_list = []

        for ti, xyz in enumerate(traj_aligned):
            ctl = xyz[:, ctl_idx, :]
            ntl = xyz[:, ntl_idx, :]

            ctl_com = ctl.mean(axis=1)
            ntl_com = ntl.mean(axis=1)

            v = ntl_com - ctl_com
            vnorm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
            vhat = v / vnorm

            nf_i = xyz.shape[0]
            feats_i = np.zeros((nf_i, 12), dtype=np.float32)

            for fj in range(nf_i):
                ctl_c = ctl[fj] - ctl_com[fj]
                ntl_c = ntl[fj] - ntl_com[fj]

                a_ctl = principal_axis(ctl_c)
                a_ntl = principal_axis(ntl_c)

                feats_i[fj, 0:3] = vhat[fj]
                feats_i[fj, 3:6] = a_ctl
                feats_i[fj, 6:9] = a_ntl
                feats_i[fj, 9] = angle(vhat[fj], a_ctl)
                feats_i[fj, 10] = angle(vhat[fj], a_ntl)
                feats_i[fj, 11] = angle(a_ctl, a_ntl)

            feats_list.append(feats_i)
            owner_list.append(np.full(nf_i, ti, dtype=int))

            dist = np.linalg.norm(ntl_com - ctl_com, axis=1) / 10.0
            d_i = np.exp(-5 * dist)
            d_list.append(d_i.astype(np.float32))

    feats = np.vstack(feats_list)
    X_owner_orient = np.concatenate(owner_list)
    d = np.concatenate(d_list)

    d -= d.min() - 1e-6
    d /= d.max()

    m = feats.mean(axis=0, keepdims=True)
    Dc = feats - m

    with tim("PCA on 12D features"):
        C = (Dc.T @ Dc) / len(Dc)
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

    yscores = Dc @ evecs[:, :3]
    scores_orient = [yscores[X_owner_orient == ti].copy() for ti in range(len(F))]

    var_ratio = evals / evals.sum()
    print("Orientation PCA variance ratios (top 6):", np.round(var_ratio[:6], 4))
    print("Orientation PCA cumulative variance (top 6):", np.round(np.cumsum(var_ratio[:6]), 4))

    plot_scree(evals, "Orientation-PCA scree (12 features)")

    plt.figure(figsize=(7, 7))
    plt.scatter(yscores[:, 0], yscores[:, 1], s=15 + 80 * d, alpha=0.5, linewidths=0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Orientation-PCA (NTL vs CTL)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("figures/orientation_pca_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    np.save(OUTDIR / "scores_orient_flat.npy", yscores.astype(np.float32))
    np.save(OUTDIR / "X_owner_orient.npy", X_owner_orient.astype(int))
    np.save(OUTDIR / "evals.npy", evals.astype(np.float32))
    np.save(OUTDIR / "evecs.npy", evecs.astype(np.float32))
    np.save(OUTDIR / "ctl_idx.npy", ctl_idx.astype(int))
    np.save(OUTDIR / "ntl_idx.npy", ntl_idx.astype(int))
    np.savez_compressed(
        OUTDIR / "scores_orient_list.npz",
        **{f"traj_{i}": arr.astype(np.float32) for i, arr in enumerate(scores_orient)}
    )

    with open(OUTDIR / "summary.txt", "w") as fh:
        fh.write(f"ntraj={len(F)}\n")
        fh.write(f"ctl_atoms={ctl_idx.size}\n")
        fh.write(f"ntl_atoms={ntl_idx.size}\n")
        fh.write(f"var_ratio_top6={np.round(var_ratio[:6], 4).tolist()}\n")

    print("\nSaved outputs to:", OUTDIR)
    print("DONE.")


if __name__ == "__main__":
    main()
