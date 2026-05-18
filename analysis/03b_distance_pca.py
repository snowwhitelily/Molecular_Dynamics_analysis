# ROS1 Analysis Pipeline
# Script 03b: NTL-CTL Distance PCA
#
# Computes PCA on the matrix of pairwise Calpha-Calpha distances between
# all NTL and CTL residues. This is superposition-independent and provides
# orthogonal validation of the actout_ctlfit inter-lobe geometry results.
# Trajectories are subsampled at stride 10 to reduce memory usage (~10x).

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from scripts.ros1_utils import combine_masks
from scripts.ros1_plots import plot_scree
from scripts.timer import Timer

tim = Timer()

BASE    = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "step3_distance_pca"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_PCS       = 5
DIST_STRIDE = 10

print("Running Script 3B: Distance PCA")
print(f"DIST_STRIDE = {DIST_STRIDE}")
print("Figure output folder:", FIG_DIR)


def save_current_figure(filename):
    out = FIG_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure:", out)


def traj_mutant(p):
    return Path(p).parent.name


def traj_replica(p):
    return Path(p).name


# ============================================================
# Load preprocessed data from Script 01
# ============================================================
F_all          = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
traj_aligned   = np.load(RESULTS / "traj_aligned.npy", allow_pickle=True).tolist()
frame_idx_list = np.load(RESULTS / "frame_idx_list.npy", allow_pickle=True).tolist()
masks_npz      = np.load(RESULTS / "masks.npz", allow_pickle=True)
masks          = {k: masks_npz[k] for k in masks_npz.files}

meta = SimpleNamespace(
    resids   = np.load(RESULTS / "meta_resids.npy", allow_pickle=True),
    names    = np.load(RESULTS / "meta_names.npy", allow_pickle=True),
    resnames = np.load(RESULTS / "meta_resnames.npy", allow_pickle=True),
)

# ============================================================
# Exclude F1994L: AlphaFold2 predicted this variant in a distinct
# inactive conformation not comparable to the rest of the panel.
# ============================================================
EXCLUDE_MUTANTS = {"F1994L"}
keep           = [i for i, f in enumerate(F_all) if Path(f).parent.name not in EXCLUDE_MUTANTS]
F              = [F_all[i] for i in keep]
traj_aligned   = [traj_aligned[i] for i in keep]
frame_idx_list = [frame_idx_list[i] for i in keep]
print(f"Trajectories after exclusion: {len(F)}")

mut_names = [traj_mutant(p) for p in F]
rep_names = [traj_replica(p) for p in F]

# ============================================================
# NTL and CTL Calpha atom index sets
# ============================================================
CA   = masks["CA"]
BODY = masks["BODY"]
ACT  = masks["ACT"]
NTL  = masks["NTL"]
CTL  = masks["CTL"]

mask_ctl = combine_masks(CTL, BODY, CA, ~ACT)
mask_ntl = combine_masks(NTL, BODY, CA)
ctl_idx  = np.where(mask_ctl)[0]
ntl_idx  = np.where(mask_ntl)[0]

print(f"CTL CA atoms: {len(ctl_idx)}")
print(f"NTL CA atoms: {len(ntl_idx)}")
print(f"Distance vector length per frame: {len(ctl_idx) * len(ntl_idx)}")

# ============================================================
# Build pairwise NTL-CTL distance matrix
# Subsampled at DIST_STRIDE to reduce memory from ~15 GB to ~1.5 GB.
# ============================================================
D_list        = []
traj_id_list  = []
frame_orig_list = []
mutant_list   = []
replica_list  = []

print(f"\nBuilding distance matrix (stride={DIST_STRIDE})...")
for ti, xyz in enumerate(traj_aligned):
    xyz_sub       = xyz[::DIST_STRIDE]
    frame_idx_sub = np.asarray(frame_idx_list[ti])[::DIST_STRIDE].astype(int)
    n_sub         = xyz_sub.shape[0]

    MCTL = xyz_sub[:, ctl_idx, :]
    MNTL = xyz_sub[:, ntl_idx, :]
    Dij  = np.sqrt(((MCTL[:, :, None, :] - MNTL[:, None, :, :]) ** 2).sum(axis=-1))
    D_i  = Dij.reshape(n_sub, -1).astype(np.float32)

    D_list.append(D_i)
    traj_id_list.append(np.full(n_sub, ti, dtype=int))
    frame_orig_list.append(frame_idx_sub)
    mutant_list.append(np.array([mut_names[ti]] * n_sub))
    replica_list.append(np.array([rep_names[ti]] * n_sub))

    if ti % 10 == 0:
        print(f"  {ti+1}/{len(traj_aligned)} ({mut_names[ti]}) — {n_sub} frames")

D           = np.vstack(D_list)
traj_id     = np.concatenate(traj_id_list)
frame_orig  = np.concatenate(frame_orig_list)
mutant_arr  = np.concatenate(mutant_list)
replica_arr = np.concatenate(replica_list)

print(f"\nDistance matrix shape: {D.shape}")
print(f"Memory: {D.nbytes / 1e9:.2f} GB")

# ============================================================
# PCA on the distance matrix
# ============================================================
print("\nRunning PCA...")
m            = D.mean(axis=0, keepdims=True)
Dc           = D - m
C            = (Dc.T @ Dc) / len(Dc)
evals, evecs = np.linalg.eigh(C)
order        = np.argsort(evals)[::-1]
evals        = evals[order]
evecs        = evecs[:, order]

print(f"Top 5 eigenvalues: {evals[:5]}")
print(f"Variance PC1: {100*evals[0]/evals.sum():.1f}%")
print(f"Variance PC1+PC2: {100*evals[:2].sum()/evals.sum():.1f}%")

plot_scree(evals, n=25, title="Scree: Distance PCA (NTL-CTL CA-CA distances)")
save_current_figure("step3b_distance_pca_scree.png")

scores = Dc @ evecs[:, :N_PCS]
print(f"Scores shape: {scores.shape}")

Z       = Dc + m
d_scale = np.exp(-5 * Z).sum(axis=1)
d_scale -= d_scale.min() - 1e-6
d_scale /= d_scale.max()

# ============================================================
# Save outputs
# ============================================================
np.save(RESULTS / "dist_scores.npy",        scores)
np.save(RESULTS / "dist_x_owner.npy",       traj_id)
np.save(RESULTS / "dist_density_scale.npy", d_scale)
np.save(RESULTS / "dist_evecs.npy",         evecs[:, :N_PCS])
np.save(RESULTS / "dist_evals.npy",         evals[:N_PCS])
np.save(RESULTS / "dist_mean.npy",          m)

pc_cols   = {f"PC{i+1}": scores[:, i] for i in range(N_PCS)}
scores_df = pd.DataFrame({
    "traj_index"          : traj_id,
    "mutant"              : mutant_arr,
    "replica"             : replica_arr,
    "frame_index_original": frame_orig,
    **pc_cols,
    "size_scale"          : d_scale,
})
scores_df.to_csv(RESULTS / "step3b_distance_pca_scores.csv", index=False)
print("Saved: step3b_distance_pca_scores.csv")

# ============================================================
# Plots
# ============================================================
plt.figure(figsize=(7, 7))
plt.scatter(scores[:, 0], scores[:, 1], c=traj_id,
            s=100 * d_scale, alpha=0.5, linewidths=0, cmap="tab20")
plt.colorbar(label="Trajectory index")
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Distance PCA: NTL-CTL CA-CA distances — all systems")
save_current_figure("step3b_distance_pca_pc1_pc2_all.png")

unique_muts = sorted(set(mutant_arr))
cmap_mut    = get_cmap("tab20", len(unique_muts))

plt.figure(figsize=(10, 8))
for i, mut in enumerate(unique_muts):
    mask_m = mutant_arr == mut
    pc1_m  = scores[mask_m, 0]
    pc2_m  = scores[mask_m, 1]
    plt.scatter(pc1_m, pc2_m, s=10, alpha=0.15, color=cmap_mut(i), linewidths=0)
    plt.scatter(pc1_m.mean(), pc2_m.mean(), s=120,
                color=cmap_mut(i), edgecolors="black",
                linewidths=0.8, zorder=5, label=mut)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Distance PCA: per-mutant positions")
plt.legend(fontsize=6, ncol=4, loc="upper right", framealpha=0.8)
save_current_figure("step3b_distance_pca_per_mutant.png")

plt.figure(figsize=(7, 7))
plt.scatter(scores[:, 0], scores[:, 2], c=traj_id,
            s=100 * d_scale, alpha=0.5, linewidths=0, cmap="tab20")
plt.colorbar(label="Trajectory index")
plt.xlabel("PC1")
plt.ylabel("PC3")
plt.title("Distance PCA: PC1 vs PC3")
save_current_figure("step3b_distance_pca_pc1_pc3.png")

# Top distance pairs contributing to PC1
loadings_pc1 = evecs[:, 0]
top_idx      = np.argsort(np.abs(loadings_pc1))[::-1][:20]
n_ctl        = len(ctl_idx)
n_ntl        = len(ntl_idx)
top_ctl      = top_idx // n_ntl
top_ntl      = top_idx % n_ntl
resids_abs   = np.asarray(meta.resids).astype(int) + 1933

print("\nTop 20 distance pairs contributing to PC1:")
print(f"{'Rank':<5} {'CTL_resid':<12} {'NTL_resid':<12} {'Loading':>10}")
for rank, (ci, ni, idx) in enumerate(zip(top_ctl, top_ntl, top_idx), start=1):
    print(f"{rank:<5} {int(resids_abs[ctl_idx[ci]]):<12} "
          f"{int(resids_abs[ntl_idx[ni]]):<12} "
          f"{float(loadings_pc1[idx]):>10.4f}")

print("\nScript 3B completed.")