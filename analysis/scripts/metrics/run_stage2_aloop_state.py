# 4) analysis/scripts/metrics/run_stage2_aloop_state.py
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.scripts.ros1_utils import build_standard_masks, combine_masks
from analysis.scripts.ros1_align import align_traj_to_ref_by_fit

STAGE1 = Path("analysis/outputs/stage1_common_align")
OUTDIR = Path("analysis/outputs/stage2_aloop_state_labels")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_list_npz(path):
    z = np.load(path, allow_pickle=True)
    keys = sorted(z.files, key=lambda x: int(x.split("_")[1]))
    return [z[k] for k in keys]


def plot_scree(evals, n=25, title="Scree"):
    n = min(n, len(evals))
    x = np.arange(1, n + 1)
    y = evals[:n] / evals.sum()
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel("PC")
    plt.ylabel("Explained variance ratio")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("figures/aloop_state_scree.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_list_npz(path, arr_list):
    np.savez_compressed(path, **{f"traj_{i}": np.asarray(arr) for i, arr in enumerate(arr_list)})


def main():
    print("=" * 80)
    print("STAGE 2: A-loop active/inactive labels")
    print("=" * 80)

    F = np.load(STAGE1 / "F.npy", allow_pickle=True).tolist()
    per_sim_indices = np.load(STAGE1 / "per_sim_indices.npy", allow_pickle=True)
    ref_centered = np.load(STAGE1 / "ref_centered.npy")
    traj_aligned = load_list_npz(STAGE1 / "traj_aligned_list.npz")

    u0 = mda.Universe(f"{F[0]}/md-prot.pdb")
    meta = u0.atoms[np.asarray(per_sim_indices[0], dtype=int)]
    masks = build_standard_masks(meta)

    ACT = masks["ACT"]
    BB = masks["BB"]

    sele_act = combine_masks(ACT, BB)
    fit_act = sele_act
    ACT_idx = np.where(sele_act)[0].astype(int)

    print("ACT atoms:", int(ACT.sum()))
    print("A-loop BB atoms:", len(ACT_idx))

    ref_fit_act = ref_centered[fit_act]
    ref_fit_act = ref_fit_act - ref_fit_act.mean(axis=0, keepdims=True)

    print("A-loop: aligning all trajectories on ACT fit")
    traj_act = [align_traj_to_ref_by_fit(xyz, fit_act, ref_fit_act) for xyz in traj_aligned]

    print("A-loop: vectorizing deviations + building P/X")
    all_frames = np.vstack([xyz[:, sele_act, :].reshape(xyz.shape[0], -1) for xyz in traj_act])
    mean_act = all_frames.mean(axis=0)
    d = mean_act.size

    P_list, X_list = [], []
    P_owner, X_owner = [], []
    P_state, X_state = [], []
    aloop_rmsd_list = []
    ACT_state = []

    pooled_rmsd = []
    for xyz in traj_act:
        act_xyz = xyz[:, ACT_idx, :]
        ref0 = act_xyz[0]
        diff = act_xyz - ref0[None, :, :]
        rmsd_nm = np.sqrt((diff ** 2).sum(axis=(1, 2)) / ACT_idx.size) / 10.0
        pooled_rmsd.append(rmsd_nm)

    thr = float(np.median(np.concatenate(pooled_rmsd)))

    for ti, xyz in enumerate(traj_act):
        act_xyz = xyz[:, ACT_idx, :]
        ref0 = act_xyz[0]
        diff = act_xyz - ref0[None, :, :]
        rmsd_nm = np.sqrt((diff ** 2).sum(axis=(1, 2)) / ACT_idx.size) / 10.0
        aloop_rmsd_list.append(rmsd_nm)

        state = (rmsd_nm > thr).astype(np.int8)
        ACT_state.append(state)

        dev = xyz[:, sele_act, :].reshape(xyz.shape[0], -1) - mean_act

        split_idx = int(0.8 * dev.shape[0])
        P_i, X_i = dev[:split_idx], dev[split_idx:]
        sP, sX = state[:split_idx], state[split_idx:]

        if P_i.shape[0]:
            P_list.append(P_i)
            P_owner.append(np.full(P_i.shape[0], ti, dtype=int))
            P_state.append(sP)
        if X_i.shape[0]:
            X_list.append(X_i)
            X_owner.append(np.full(X_i.shape[0], ti, dtype=int))
            X_state.append(sX)

    P = np.vstack(P_list) if len(P_list) else np.zeros((0, d))
    X = np.vstack(X_list) if len(X_list) else np.zeros((0, d))
    P_owner = np.concatenate(P_owner) if len(P_owner) else np.zeros((0,), dtype=int)
    X_owner = np.concatenate(X_owner) if len(X_owner) else np.zeros((0,), dtype=int)
    P_state = np.concatenate(P_state) if len(P_state) else np.zeros((0,), dtype=int)
    X_state = np.concatenate(X_state) if len(X_state) else np.zeros((0,), dtype=int)

    print("A-loop X shape:", X.shape)

    plt.figure(figsize=(6, 4))
    plt.hist(np.concatenate(aloop_rmsd_list), bins=100)
    plt.xlabel("A-loop RMSD (nm)")
    plt.ylabel("Counts")
    plt.title("Activation-loop RMSD distribution")
    plt.tight_layout()
    plt.savefig("figures/aloop_rmsd_hist.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Threshold (nm):", thr)
    print("Fraction inactive-like (overall):", float(np.mean(np.concatenate(ACT_state))))

    print("A-loop: diagonalization")
    C = X.T @ (X / len(X))
    evals, loadings = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    loadings = loadings[:, order]

    plot_scree(evals, n=25, title="Scree (A-loop PCA, state-labeled)")

    ncomponents = 10
    scale = np.sqrt(mean_act.size)
    pscores_flat = (P @ loadings[:, :ncomponents]) / scale
    scores_flat = (X @ loadings[:, :ncomponents]) / scale

    np.save(OUTDIR / "threshold_nm.npy", np.array([thr], dtype=np.float32))
    np.save(OUTDIR / "scores_flat.npy", scores_flat.astype(np.float32))
    np.save(OUTDIR / "pscores_flat.npy", pscores_flat.astype(np.float32))
    np.save(OUTDIR / "X_owner.npy", X_owner)
    np.save(OUTDIR / "P_owner.npy", P_owner)
    np.save(OUTDIR / "X_state.npy", X_state.astype(np.int8))
    np.save(OUTDIR / "P_state.npy", P_state.astype(np.int8))
    save_list_npz(OUTDIR / "ACT_state_list.npz", ACT_state)
    save_list_npz(OUTDIR / "aloop_rmsd_list.npz", [arr.astype(np.float32) for arr in aloop_rmsd_list])

    with open(OUTDIR / "summary.txt", "w") as fh:
        fh.write(f"threshold_nm={thr}\n")
        fh.write(f"ntraj={len(F)}\n")
        fh.write(f"aloop_atoms={len(ACT_idx)}\n")
        fh.write(f"overall_fraction_inactive_like={float(np.mean(np.concatenate(ACT_state)))}\n")
        fh.write(f"X_shape={X.shape}\n")

    print("\nSaved outputs to:", OUTDIR)
    print("DONE.")


if __name__ == "__main__":
    main()
