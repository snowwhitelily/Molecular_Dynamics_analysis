# %%
# ROS1 Analysis Pipeline
# Script 2A: PCA and Clustering (ACT-IN + ACT-OUT)

# %%
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from types import SimpleNamespace

from scripts.clustering import run_hdbscan, run_dbscan
from scripts.plotting import (
    plot_occupancy_heatmap,
    plot_cluster_population,
    plot_pca_clusters,
)

from scripts.ros1_utils import (
    combine_masks,
    traj_frames_atoms,
)

from scripts.ros1_plots import (
    plot_scree,
)

from scripts.ros1_align import (
    align_traj_to_ref_by_fit,
)

from scripts.ros1_clustering import (
    run_cluster_panel_varlen,
)

from scripts.timer import Timer
tim = Timer()

eps = 1e-6

# %%
# ============================================
# SETTINGS
# ============================================
PCA_STRIDE = 10
NCOMPONENTS = 10
PLOT_PAIRS = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "step2_fast" / "02a_actin_actout"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Running exploratory PCA mode")
print("PCA_STRIDE =", PCA_STRIDE)
print("Figure output folder:", FIG_DIR)

# %%
# ============================================
# Load outputs from Step 1
# ============================================
F = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
traj_aligned = np.load(RESULTS / "traj_aligned.npy", allow_pickle=True).tolist()

masks_npz = np.load(RESULTS / "masks.npz", allow_pickle=True)
masks = {k: masks_npz[k] for k in masks_npz.files}

CA = masks["CA"]
BB = masks["BB"]
BODY = masks["BODY"]
ACT = masks["ACT"]

sele = np.load(RESULTS / "sele.npy", allow_pickle=True)
fit = np.load(RESULTS / "fit.npy", allow_pickle=True)
ref_centered = np.load(RESULTS / "ref_centered.npy", allow_pickle=True)

meta = SimpleNamespace(
    resids=np.load(RESULTS / "meta_resids.npy", allow_pickle=True),
    names=np.load(RESULTS / "meta_names.npy", allow_pickle=True),
    resnames=np.load(RESULTS / "meta_resnames.npy", allow_pickle=True),
)

with open(RESULTS / "selection.txt") as f:
    selection = f.read().strip()

with open(RESULTS / "colors.txt") as f:
    colors = tuple(line.strip() for line in f if line.strip())

with open(RESULTS / "nrep.txt") as f:
    nrep = int(f.read().strip())

print("Step 2A setup ok.")
print("n trajectories:", len(traj_aligned))
print("example F:", F[0])
print("sele atoms:", int(sele.sum()))

# %%
# ============================================
# helpers
# ============================================

def save_current_figure(filename: str):
    out = FIG_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure:", out)

def make_traj_colors(F, colors, nrep):
    traj_colors = np.repeat(np.array(colors), nrep, axis=0)
    traj_colors = np.tile(traj_colors, int(np.ceil(len(F) / len(traj_colors))))[:len(F)]
    return traj_colors

def first_last_10_centroids(scores_by_traj):
    first_pts = []
    last_pts = []
    for Z in scores_by_traj:
        if Z.shape[0] == 0:
            first_pts.append(np.array([np.nan, np.nan]))
            last_pts.append(np.array([np.nan, np.nan]))
            continue
        k = max(1, int(np.ceil(0.10 * Z.shape[0])))
        first_pts.append(Z[:k, :2].mean(axis=0))
        last_pts.append(Z[-k:, :2].mean(axis=0))
    return np.array(first_pts), np.array(last_pts)

def plot_first_last_10_overlay(scores_by_traj, traj_colors, title, prefix):
    first_pts, last_pts = first_last_10_centroids(scores_by_traj)

    fig, ax = plt.subplots(figsize=(9, 9))

    for ti, Z in enumerate(scores_by_traj):
        if Z.shape[0] == 0:
            continue

        ax.scatter(
            Z[:, 0], Z[:, 1],
            s=2, c=[traj_colors[ti]], alpha=0.12, linewidths=0
        )

        ax.scatter(
            first_pts[ti, 0], first_pts[ti, 1],
            s=80, c=[traj_colors[ti]], marker="o",
            edgecolors="black", linewidths=0.5, zorder=5
        )

        ax.scatter(
            last_pts[ti, 0], last_pts[ti, 1],
            s=90, c=[traj_colors[ti]], marker="^",
            edgecolors="black", linewidths=0.5, zorder=6
        )

        ax.plot(
            [first_pts[ti, 0], last_pts[ti, 0]],
            [first_pts[ti, 1], last_pts[ti, 1]],
            c=traj_colors[ti], alpha=0.55, linewidth=1.2, zorder=4
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title + "\n(circle = first 10%, triangle = last 10%)")
    save_current_figure(f"{prefix}_first_last10.png")

def plot_loading_by_residue(loadings, sel_mask, meta, pc, title_prefix, prefix):
    sel_idx = np.where(sel_mask)[0].astype(int)
    L = loadings[:, pc - 1].reshape((len(sel_idx), 3))
    atom_contrib = np.sqrt((L ** 2).sum(axis=1))

    sel_resids = meta.resids[sel_idx]
    uniq = np.unique(sel_resids)
    res_contrib = np.array([atom_contrib[sel_resids == r].sum() for r in uniq])

    plt.figure(figsize=(11, 3))
    plt.plot(uniq, res_contrib)
    plt.xlabel("Residue id")
    plt.ylabel(f"PC{pc} loading magnitude")
    plt.title(f"{title_prefix}: residue contributions for PC{pc}")
    save_current_figure(f"{prefix}_pc{pc}_loadings.png")

def downsample_traj_list(traj_list, stride):
    return [xyz[::stride].copy() for xyz in traj_list]

def save_scores_table(scores_by_traj, F, prefix):
    rows = []
    for ti, Z in enumerate(scores_by_traj):
        mutant = Path(F[ti]).parent.name
        replica = Path(F[ti]).name
        for fi, row in enumerate(Z):
            rec = {
                "traj_index": ti,
                "mutant": mutant,
                "replica": replica,
                "frame_index_downsampled": fi,
            }
            for j, val in enumerate(row, start=1):
                rec[f"PC{j}"] = float(val)
            rows.append(rec)

    df = pd.DataFrame(rows)
    out = RESULTS / f"{prefix}_scores.csv"
    df.to_csv(out, index=False)
    print("Saved scores table:", out)

def save_cluster_summary(cluster_result, prefix):
    out = RESULTS / f"{prefix}_cluster_summary.txt"
    with open(out, "w") as f:
        f.write(str(cluster_result))
    print("Saved cluster summary:", out)

def run_pca_block(traj_list, sel_mask, title_prefix, prefix, traj_colors):
    with tim(f"{title_prefix}: vectorizing downsampled conformations"):
        traj_small = downsample_traj_list(traj_list, PCA_STRIDE)

        all_frames = []
        for xyz in traj_small:
            all_frames.append(traj_frames_atoms(xyz, sel_mask))
        all_frames = np.vstack(all_frames)

        mean = all_frames.mean(axis=0)

        X_list = []
        X_owner = []

        for ti, xyz in enumerate(traj_small):
            dev = traj_frames_atoms(xyz, sel_mask) - mean
            X_list.append(dev)
            X_owner.append(np.full(dev.shape[0], ti, dtype=int))

        X = np.vstack(X_list)
        X_owner = np.concatenate(X_owner)

    print(f"{title_prefix} X shape:", X.shape)

    with tim(f"{title_prefix}: diagonalization (PCA)"):
        C = X.T @ (X / len(X))
        evals, loadings = np.linalg.eigh(C)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        loadings = loadings[:, idx]

    plot_scree(evals, n=25, title=f"{title_prefix} scree plot")
    save_current_figure(f"{prefix}_scree.png")

    scale = np.sqrt(mean.size)

    with tim(f"{title_prefix}: projections on first {NCOMPONENTS} components"):
        scores_flat = (X @ loadings[:, :NCOMPONENTS]) / scale

    scores_by_traj = [scores_flat[X_owner == ti].copy() for ti in range(len(F))]
    print(f"{title_prefix} scores prepared.")

    save_scores_table(scores_by_traj, F, prefix)

    for a, b in PLOT_PAIRS:
        fig, ax = plt.subplots(figsize=(9, 9))
        for ti in range(len(F)):
            Zi = scores_by_traj[ti]
            ax.scatter(
                Zi[:, a], Zi[:, b],
                s=2, c=[traj_colors[ti]], alpha=0.15, linewidths=0
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"PC{a+1}")
        ax.set_ylabel(f"PC{b+1}")
        ax.set_title(f"{title_prefix} scores plot (PC{a+1} vs PC{b+1})")
        save_current_figure(f"{prefix}_pc{a+1}_pc{b+1}.png")

    plot_first_last_10_overlay(
        scores_by_traj=scores_by_traj,
        traj_colors=traj_colors,
        title=f"{title_prefix} PCA",
        prefix=prefix,
    )

    cluster_result = run_cluster_panel_varlen(
        scores_flat=scores_flat,
        x_owner=X_owner,
        F=F,
        prefix=prefix,
        title_prefix=title_prefix,
        n_pcs=5,
        min_cluster_size=200,
        eps=0.9,
        min_samples=60,
        run_hdbscan=run_hdbscan,
        run_dbscan=run_dbscan,
        plot_occupancy_heatmap=plot_occupancy_heatmap,
        plot_cluster_population=plot_cluster_population,
        plot_pca_clusters=plot_pca_clusters,
        display_fn=print,
    )

    save_cluster_summary(cluster_result, prefix)

    plot_loading_by_residue(loadings, sel_mask, meta, pc=1, title_prefix=f"{title_prefix} PCA", prefix=prefix)
    plot_loading_by_residue(loadings, sel_mask, meta, pc=2, title_prefix=f"{title_prefix} PCA", prefix=prefix)

    np.save(RESULTS / f"{prefix}_loadings.npy", loadings)
    np.save(RESULTS / f"{prefix}_mean.npy", mean)
    np.save(RESULTS / f"{prefix}_scores.npy", scores_flat)
    np.save(RESULTS / f"{prefix}_x_owner.npy", X_owner)
    print("Saved arrays:", prefix)

    return {
        "mean": mean.copy(),
        "loadings": loadings.copy(),
        "scores": scores_flat.copy(),
        "X_owner": X_owner.copy(),
        "scores_by_traj": [z.copy() for z in scores_by_traj],
        "F": np.array(F, dtype=object),
        "sele": sel_mask.copy(),
        "cluster_results": cluster_result,
        "pca_stride": PCA_STRIDE,
    }

# %%
traj_colors = make_traj_colors(F, colors, nrep)

# %%
# ============================================
# ACT-IN PCA
# ============================================
PCA_ACT_IN = run_pca_block(
    traj_list=traj_aligned,
    sel_mask=sele,
    title_prefix="ACT-IN",
    prefix="actin",
    traj_colors=traj_colors,
)

# %%
# ============================================
# ACT-OUT setup + alignment
# ============================================
sele_actout = combine_masks(BB, BODY, ~ACT)
fit_actout = sele_actout

print("ACT-OUT sele atoms:", int(sele_actout.sum()))
print("ACT-OUT fit atoms:", int(fit_actout.sum()))

ref_actout_centered = ref_centered.copy()
ref_fit_actout = ref_actout_centered[fit_actout]
ref_fit_actout = ref_fit_actout - ref_fit_actout.mean(axis=0, keepdims=True)

with tim("ACT-OUT: aligning all trajectories on fit"):
    traj_actout = [align_traj_to_ref_by_fit(xyz, fit_actout, ref_fit_actout) for xyz in traj_aligned]

plt.figure(figsize=(7, 6))
A = []
for xyz in traj_actout:
    a = xyz[:, sele_actout, 1:].reshape((-1, 2))
    mask_nonzero = ~((np.abs(a[:, 0]) < 1e-6) & (np.abs(a[:, 1]) < 1e-6))
    A.append(a[mask_nonzero])
A = np.vstack(A)

plt.hist2d(A[:, 0], A[:, 1], bins=256)
plt.gca().set_aspect("equal")
plt.title("ACT-OUT aligned coordinate density (YZ)")
plt.xlabel("Y")
plt.ylabel("Z")
save_current_figure("actout_density_yz.png")

# %%
# ============================================
# ACT-OUT PCA
# ============================================
PCA_ACT_OUT = run_pca_block(
    traj_list=traj_actout,
    sel_mask=sele_actout,
    title_prefix="ACT-OUT",
    prefix="actout",
    traj_colors=traj_colors,
)

# %%
print("Script 2A completed:")
print("- ACT-IN")
print("- ACT-OUT")
print("Flow used: Scree -> Scores -> Clusters -> Loadings -> first 10% / last 10%")
print("Mode used: exploratory downsampled PCA, stride =", PCA_STRIDE)
print("Saved figures to:", FIG_DIR)
print("Saved analysis outputs to:", RESULTS)
