import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.scripts.ros1_utils import ensure_meta, build_standard_masks, combine_masks
from analysis.scripts.ros1_plots import plot_scree

STAGE1 = Path("analysis/outputs/stage1_common_align")
OUTDIR = Path("analysis/outputs/stage2_distance_pca")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_list_npz(path):
    z = np.load(path, allow_pickle=True)
    keys = sorted(z.files, key=lambda x: int(x.split("_")[1]))
    return [z[k] for k in keys]


def main():
    print("=" * 80)
    print("STAGE 2: Distance PCA")
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

    print("CTL CA:", len(ctl_idx), "NTL CA:", len(ntl_idx))

    D_list = []
    traj_id_list = []
    xmask_list = []

    for ti, xyz in enumerate(traj_aligned):
        MCTL = xyz[:, ctl_idx, :]
        MNTL = xyz[:, ntl_idx, :]

        Dij = np.sqrt(((MCTL[:, :, None, :] - MNTL[:, None, :, :]) ** 2).sum(axis=-1))
        D_i = Dij.reshape(xyz.shape[0], -1).astype(np.float32)
        D_list.append(D_i)

        traj_id_list.append(np.full(xyz.shape[0], ti, dtype=int))

        split_idx = min(1, xyz.shape[0])
        xmask_i = np.zeros(xyz.shape[0], dtype=bool)
        xmask_i[split_idx:] = True
        xmask_list.append(xmask_i)

    D = np.vstack(D_list)
    traj_id_dist = np.concatenate(traj_id_list)
    xmask_dist = np.concatenate(xmask_list)

    print("D:", D.shape)

    m = D.mean(axis=0, keepdims=True)
    Dc = D - m
    C = (Dc.T @ Dc) / len(Dc)

    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    plot_scree(evals, n=25, title="Scree (Distance PCA)")

    yscores_dist = Dc @ evecs[:, :3]
    print("yscores_dist:", yscores_dist.shape)

    yscores_dist_x = yscores_dist[xmask_dist]
    traj_id_dist_x = traj_id_dist[xmask_dist]

    d_dist = np.exp(-5 * D).sum(axis=1)
    d_dist -= d_dist.min() - 1e-6
    d_dist /= d_dist.max()

    plt.figure(figsize=(7, 7))
    plt.scatter(
        yscores_dist[:, 0],
        yscores_dist[:, 1],
        s=15 + 80 * d_dist,
        alpha=0.5,
        linewidths=0
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Distance-PCA")
    plt.tight_layout()
    plt.savefig("figures/distance_pca_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    np.save(OUTDIR / "yscores_dist.npy", yscores_dist.astype(np.float32))
    np.save(OUTDIR / "yscores_dist_x.npy", yscores_dist_x.astype(np.float32))
    np.save(OUTDIR / "traj_id_dist.npy", traj_id_dist.astype(int))
    np.save(OUTDIR / "traj_id_dist_x.npy", traj_id_dist_x.astype(int))
    np.save(OUTDIR / "xmask_dist.npy", xmask_dist.astype(bool))
    np.save(OUTDIR / "evals.npy", evals.astype(np.float32))
    np.save(OUTDIR / "evecs.npy", evecs.astype(np.float32))

    print("\nSaved outputs to:", OUTDIR)
    print("DONE.")


if __name__ == "__main__":
    main()
