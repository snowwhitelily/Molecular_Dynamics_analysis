import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.scripts.ros1_clustering import lda_multiclass, lda_2class_direction, summarize_pair

STAGE1 = Path("analysis/outputs/stage1_common_align")
PCA_DIR = Path("analysis/outputs/stage2_actout_ctlfit_pca")
OUTDIR = Path("analysis/outputs/stage2_lda_qfamily")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_list_npz(path):
    z = np.load(path, allow_pickle=True)
    keys = sorted(z.files, key=lambda x: int(x.split("_")[1]))
    return [np.asarray(z[k]) for k in keys]


def save_list_npz(path, arr_list):
    np.savez_compressed(path, **{f"traj_{i}": np.asarray(arr) for i, arr in enumerate(arr_list)})


def try_load_score_list():
    candidates = [
        PCA_DIR / "scores_actout_ctlfit.npz",
        PCA_DIR / "scores_list.npz",
        PCA_DIR / "scores_traj_list.npz",
    ]
    for p in candidates:
        if p.exists():
            return load_list_npz(p)

    flat_candidates = [
        (PCA_DIR / "scores_flat_actout_ctlfit.npy", PCA_DIR / "X_owner_actout_ctlfit.npy"),
        (PCA_DIR / "scores_flat.npy", PCA_DIR / "X_owner.npy"),
    ]
    for sfile, ofile in flat_candidates:
        if sfile.exists() and ofile.exists():
            scores_flat = np.load(sfile)
            x_owner = np.load(ofile).astype(int)
            ntraj = int(x_owner.max()) + 1
            return [scores_flat[x_owner == ti].copy() for ti in range(ntraj)]

    raise FileNotFoundError("Could not find saved PCA score lists in stage2_actout_ctlfit_pca.")


def main():
    print("=" * 80)
    print("STAGE 2: Q2022P-family LDA")
    print("=" * 80)

    F = np.load(STAGE1 / "F.npy", allow_pickle=True).tolist()
    traj_mut = np.array([p.split("/")[0] for p in F], dtype=object)

    PS_list = try_load_score_list()
    print("Loaded PCA score source with", len(PS_list), "trajectories")

    nPC = min(z.shape[1] for z in PS_list if z.shape[1] >= 1)
    k_use = min(6, nPC)

    X_traj = np.vstack([z[:, :k_use].mean(axis=0, keepdims=True) for z in PS_list])

    q_family = ["Q2022P", "Q2022P_S1986F", "Q2022P_S1986Y"]
    mask_q = np.isin(traj_mut, q_family)

    Xq = X_traj[mask_q]
    namesq = traj_mut[mask_q]
    uniq_sys = np.unique(namesq)

    print("Q2022P-family systems:", uniq_sys.tolist())
    print("Counts:", {u: int((namesq == u).sum()) for u in uniq_sys})

    label_map = {m: i for i, m in enumerate(uniq_sys)}
    yq = np.array([label_map[m] for m in namesq], dtype=int)

    W, evals = lda_multiclass(Xq, yq, eps=1e-8)
    ndim = min(len(uniq_sys) - 1, k_use)
    Z = Xq @ W[:, :ndim]

    print("Top eigenvalues:", np.round(evals[:min(5, len(evals))], 6))

    np.save(OUTDIR / "X_traj_qfamily.npy", Xq.astype(np.float32))
    np.save(OUTDIR / "yq.npy", yq.astype(int))
    np.save(OUTDIR / "W_multiclass.npy", W.astype(np.float32))
    np.save(OUTDIR / "evals_multiclass.npy", evals.astype(np.float32))
    np.save(OUTDIR / "Z_multiclass.npy", Z.astype(np.float32))
    np.save(OUTDIR / "names_qfamily.npy", namesq.astype(object), allow_pickle=True)
    np.save(OUTDIR / "uniq_sys.npy", uniq_sys.astype(object), allow_pickle=True)

    plt.figure(figsize=(7, 4))
    for m in uniq_sys:
        c = label_map[m]
        zz = Z[yq == c, 0]
        plt.scatter(zz, np.zeros_like(zz), s=80, alpha=0.85, label=f"{m} (n={zz.size})")
    plt.xlabel("LD1")
    plt.yticks([])
    plt.title("Multi-class LDA on Q2022P family (trajectory means)")
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(OUTDIR / "multiclass_ld1_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()
