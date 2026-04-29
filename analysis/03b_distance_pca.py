# %%
# ROS1 Analysis Pipeline
# Script 3B: Distance PCA (NTL–CTL CA–CA pairwise distances)
#
# PURPOSE:
#   Computes PCA on the matrix of pairwise CA–CA distances between
#   NTL residues and CTL residues. This captures the relative
#   arrangement of the two lobes (NTL-wrt-CTL) in a way that is
#   independent of global superposition/alignment.
#   This is directly relevant to the supervisor's request to focus
#   on NTL-wrt-CTL inter-lobe geometry.
#
# CORRECTIONS FROM ORIGINAL:
#   1. xmask dead code cleaned up — xmask was always all-True,
#      making all _x variables identical to unmasked versions.
#      Removed redundant _x variables. Saves are now non-redundant.
#   2. Mutant label added to scores CSV for downstream colouring.
#   3. n_pcs argument added — saves top N PCs (default 5, not 3).
#   4. density scaling documented with comment.
#   5. Per-mutant scatter plot added so mutant positions are visible.
#   6. Comments added throughout for clarity.
#
# WHAT IS DISTANCE PCA?
#   For each frame, compute the distance from every CTL CA atom to
#   every NTL CA atom. This gives a long vector of distances per frame.
#   PCA on this matrix finds the main axes of variation in inter-lobe
#   geometry. PC1 = direction of largest variation in NTL-CTL distances.

import os
from pathlib import Path
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from scripts.ros1_utils import ensure_meta, combine_masks
from scripts.ros1_plots import plot_scree
from scripts.timer import Timer

tim = Timer()

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "step3_distance_pca"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Running Script 3B: Distance PCA")
print("Figure output folder:", FIG_DIR)

N_PCS = 5  # number of principal components to save (was 3 in original)


def save_current_figure(filename: str):
    out = FIG_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure:", out)


def traj_mutant(p):
    return Path(p).parent.name


def traj_replica(p):
    return Path(p).name


# ============================================================
# Load preprocessed data
# ============================================================
F = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
traj_aligned = np.load(RESULTS / "traj_aligned.npy", allow_pickle=True).tolist()
frame_idx_list = np.load(RESULTS / "frame_idx_list.npy", allow_pickle=True).tolist()
masks_npz = np.load(RESULTS / "masks.npz", allow_pickle=True)
masks = {k: masks_npz[k] for k in masks_npz.files}
selection = (RESULTS / "selection.txt").read_text().strip()

meta = SimpleNamespace(
    resids=np.load(RESULTS / "meta_resids.npy", allow_pickle=True),
    names=np.load(RESULTS / "meta_names.npy", allow_pickle=True),
    resnames=np.load(RESULTS / "meta_resnames.npy", allow_pickle=True),
)

mut_names = [traj_mutant(p) for p in F]
rep_names = [traj_replica(p) for p in F]

# ============================================================
# Define NTL and CTL CA atom index sets
# ============================================================
meta_obj = ensure_meta(
    meta=meta,
    top=f"{F[0]}/{Path(F[0]).parent.name}-MD-prot.pdb",
    selection=selection,
    natoms=traj_aligned[0].shape[1]
)

CA   = masks["CA"]
BODY = masks["BODY"]
ACT  = masks["ACT"]
NTL  = masks["NTL"]
CTL  = masks["CTL"]

# CTL CA atoms, excluding activation loop residues
# (ACT excluded so A-loop does not dominate inter-lobe distances)
mask_ctl = combine_masks(CTL, BODY, CA, ~ACT)

# NTL CA atoms
mask_ntl = combine_masks(NTL, BODY, CA)

ctl_idx = np.where(mask_ctl)[0]
ntl_idx = np.where(mask_ntl)[0]

print(f"CTL CA atoms: {len(ctl_idx)}")
print(f"NTL CA atoms: {len(ntl_idx)}")
print(f"Distance vector length per frame: {len(ctl_idx) * len(ntl_idx)}")

if len(ctl_idx) == 0 or len(ntl_idx) == 0:
    raise ValueError(
        "CTL or NTL CA mask is empty. "
        "Check masks.npz — CTL and NTL masks must be non-empty."
    )

# ============================================================
# Build distance matrix D: shape (total_frames, n_ctl * n_ntl)
# ============================================================
# For each frame, compute all CTL_i -- NTL_j CA distances.
# Dij[frame, i, j] = distance between CTL atom i and NTL atom j.
# Flattened to D[frame, i*n_ntl + j].

D_list = []
traj_id_list = []
frame_orig_list = []
mutant_list = []
replica_list = []

print("\nBuilding distance matrix...")
for ti, xyz in enumerate(traj_aligned):
    MCTL = xyz[:, ctl_idx, :]   # shape: (nframes, n_ctl, 3)
    MNTL = xyz[:, ntl_idx, :]   # shape: (nframes, n_ntl, 3)

    # Pairwise distances: (nframes, n_ctl, n_ntl)
    Dij = np.sqrt(
        ((MCTL[:, :, None, :] - MNTL[:, None, :, :]) ** 2).sum(axis=-1)
    )

    # Flatten to (nframes, n_ctl * n_ntl)
    D_i = Dij.reshape(xyz.shape[0], -1).astype(np.float32)
    D_list.append(D_i)

    traj_id_list.append(np.full(xyz.shape[0], ti, dtype=int))
    frame_orig_list.append(np.asarray(frame_idx_list[ti]).astype(int))
    mutant_list.append(np.array([mut_names[ti]] * xyz.shape[0]))
    replica_list.append(np.array([rep_names[ti]] * xyz.shape[0]))

    if ti % 10 == 0:
        print(f"  Processed trajectory {ti+1}/{len(traj_aligned)} ({mut_names[ti]})")

D          = np.vstack(D_list)
traj_id    = np.concatenate(traj_id_list)
frame_orig = np.concatenate(frame_orig_list)
mutant_arr = np.concatenate(mutant_list)
replica_arr= np.concatenate(replica_list)

print(f"\nFull distance matrix shape: {D.shape}")
print(f"Total frames: {D.shape[0]}")

# ============================================================
# PCA on distance matrix
# ============================================================
print("\nRunning PCA...")
m  = D.mean(axis=0, keepdims=True)   # mean distance vector
Dc = D - m                            # mean-centred

# Covariance matrix and eigen-decomposition
C = (Dc.T @ Dc) / len(Dc)
evals, evecs = np.linalg.eigh(C)

# Sort descending
order = np.argsort(evals)[::-1]
evals = evals[order]
evecs = evecs[:, order]

print(f"Top 5 eigenvalues: {evals[:5]}")
print(f"Variance explained by PC1: {100*evals[0]/evals.sum():.1f}%")
print(f"Variance explained by PC1+PC2: {100*evals[:2].sum()/evals.sum():.1f}%")

# Scree plot
plot_scree(evals, n=25, title="Scree: Distance PCA (NTL-CTL CA-CA distances)")
save_current_figure("step3b_distance_pca_scree.png")

# Project all frames onto top N_PCS components
scores = Dc @ evecs[:, :N_PCS]   # shape: (total_frames, N_PCS)
print(f"Scores shape: {scores.shape}")

# ============================================================
# Density scaling for scatter plot point sizes
# Taken directly from supervisor's notebook (Cell 56).
#
# IMPORTANT: must add the mean back (Z = Dc + m = original distances)
# before applying exp(-5*Z). Dc is mean-centred so contains negative
# values — exp(-5 * negative) = exp(positive) = huge number, which
# would completely distort the scaling. Original distances are all
# positive, so exp(-5 * Z) decays correctly from 1 toward 0 as
# distances grow.
#
# The -5 factor is the supervisor's choice. At -5, exp(-5*Z) decays
# to ~0.007 at Z=1 Angstrom and ~0 at Z=2 Angstrom, giving
# larger points to frames with shorter (more compact) NTL-CTL distances.
# Cosmetic only — does not affect PCA results.
# ============================================================
Z = Dc + m          # add mean back: Z = original non-centred distances (all positive)
d_scale = np.exp(-5 * Z).sum(axis=1)
d_scale -= d_scale.min() - 1e-6
d_scale /= d_scale.max()

# ============================================================
# Save outputs
# ============================================================
np.save(RESULTS / "dist_scores.npy",       scores)
np.save(RESULTS / "dist_x_owner.npy",      traj_id)
np.save(RESULTS / "dist_density_scale.npy", d_scale)
np.save(RESULTS / "dist_evecs.npy",        evecs[:, :N_PCS])
np.save(RESULTS / "dist_evals.npy",        evals[:N_PCS])
np.save(RESULTS / "dist_mean.npy",         m)

print("\nSaved: dist_scores.npy, dist_x_owner.npy, dist_density_scale.npy")
print("Saved: dist_evecs.npy, dist_evals.npy, dist_mean.npy")

# Scores CSV with mutant labels (needed for downstream colouring/FEL)
pc_cols = {f"PC{i+1}": scores[:, i] for i in range(N_PCS)}
scores_df = pd.DataFrame({
    "traj_index":           traj_id,
    "mutant":               mutant_arr,     # ADDED — was missing in original
    "replica":              replica_arr,    # ADDED — was missing in original
    "frame_index_original": frame_orig,
    **pc_cols,
    "size_scale":           d_scale,
})
scores_df.to_csv(RESULTS / "step3b_distance_pca_scores.csv", index=False)
print("Saved: step3b_distance_pca_scores.csv")

# ============================================================
# Plots
# ============================================================

# --- Plot 1: All systems, coloured by trajectory index ---
plt.figure(figsize=(7, 7))
plt.scatter(
    scores[:, 0],
    scores[:, 1],
    c=traj_id,
    s=100 * d_scale,   # supervisor's notebook uses s=100*d (Cell 58)
    alpha=0.5,
    linewidths=0,
    cmap="tab20",
)
plt.colorbar(label="Trajectory index")
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Distance PCA: NTL-CTL CA-CA distances\nAll systems coloured by trajectory")
save_current_figure("step3b_distance_pca_pc1_pc2_all.png")

# --- Plot 2: Per-mutant mean positions ---
# Show where each mutant's centre of mass sits in distance-PCA space
unique_muts = sorted(set(mutant_arr))
cmap = get_cmap("tab20", len(unique_muts))

plt.figure(figsize=(10, 8))
for i, mut in enumerate(unique_muts):
    mask_m = mutant_arr == mut
    pc1_m = scores[mask_m, 0]
    pc2_m = scores[mask_m, 1]
    plt.scatter(
        pc1_m, pc2_m,
        s=10,
        alpha=0.15,
        color=cmap(i),
        linewidths=0,
    )
    plt.scatter(
        pc1_m.mean(), pc2_m.mean(),
        s=120,
        color=cmap(i),
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
        label=mut,
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Distance PCA: per-mutant positions\n"
          "(dots = all frames, filled circle = mean position)")
plt.legend(
    fontsize=6,
    ncol=4,
    loc="upper right",
    markerscale=1.2,
    framealpha=0.8,
)
save_current_figure("step3b_distance_pca_per_mutant.png")

# --- Plot 3: PC1 vs PC3 ---
plt.figure(figsize=(7, 7))
plt.scatter(
    scores[:, 0],
    scores[:, 2],
    c=traj_id,
    s=100 * d_scale,   # supervisor's notebook uses s=100*d
    alpha=0.5,
    linewidths=0,
    cmap="tab20",
)
plt.colorbar(label="Trajectory index")
plt.xlabel("PC1")
plt.ylabel("PC3")
plt.title("Distance PCA: PC1 vs PC3")
save_current_figure("step3b_distance_pca_pc1_pc3.png")

# --- Plot 4: Loadings for PC1 (top contributing distance pairs) ---
# Each loading value corresponds to one CTL_i -- NTL_j distance pair.
# High absolute loading = that pair of residues changes most along PC1.
loadings_pc1 = evecs[:, 0]
top_n = 20
top_idx = np.argsort(np.abs(loadings_pc1))[::-1][:top_n]

n_ctl = len(ctl_idx)
n_ntl = len(ntl_idx)
top_ctl = top_idx // n_ntl   # which CTL atom
top_ntl = top_idx % n_ntl    # which NTL atom

resids_abs = np.asarray(meta.resids).astype(int) + 1933
names_arr  = np.asarray(meta.names)

print("\nTop 20 distance pairs contributing to PC1 (NTL-CTL distance PCA):")
print(f"{'Rank':<5} {'CTL_resid':<12} {'NTL_resid':<12} {'Loading':>10}")
for rank, (ci, ni, idx) in enumerate(
        zip(top_ctl, top_ntl, top_idx), start=1):
    ctl_res = int(resids_abs[ctl_idx[ci]])
    ntl_res = int(resids_abs[ntl_idx[ni]])
    load    = float(loadings_pc1[idx])
    print(f"{rank:<5} {ctl_res:<12} {ntl_res:<12} {load:>10.4f}")

print("\nScript 3B completed.")
print("\nOutputs saved to:", RESULTS)
print("Figures saved to:", FIG_DIR)
print("\nNote for thesis:")
print(
    "Distance PCA was computed on the matrix of pairwise CA-CA distances "
    "between NTL (real 1938-2030) and CTL (real 2053-2216, excluding "
    "activation loop) residues. This captures inter-lobe geometry in a "
    "superposition-independent manner, directly reflecting NTL-wrt-CTL "
    "conformational changes as requested by supervisor."
)
