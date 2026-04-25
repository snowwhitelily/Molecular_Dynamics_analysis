import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.ros1_utils import traj_frames_atoms
from scripts.ros1_plots import plot_scree


def save_current_figure(fig_dir, filename: str):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / filename
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


def plot_first_last_10_overlay(scores_by_traj, traj_colors, title, prefix, fig_dir):
    first_pts, last_pts = first_last_10_centroids(scores_by_traj)
    fig, ax = plt.subplots(figsize=(9, 9))

    for ti, Z in enumerate(scores_by_traj):
        if Z.shape[0] == 0:
            continue
        ax.scatter(Z[:, 0], Z[:, 1], s=2, c=[traj_colors[ti]], alpha=0.12, linewidths=0)
        ax.scatter(first_pts[ti, 0], first_pts[ti, 1], s=80, c=[traj_colors[ti]], marker="o", edgecolors="black", linewidths=0.5, zorder=5)
        ax.scatter(last_pts[ti, 0], last_pts[ti, 1], s=90, c=[traj_colors[ti]], marker="^", edgecolors="black", linewidths=0.5, zorder=6)
        ax.plot([first_pts[ti, 0], last_pts[ti, 0]], [first_pts[ti, 1], last_pts[ti, 1]], c=traj_colors[ti], alpha=0.55, linewidth=1.2, zorder=4)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title + "\n(circle = first 10%, triangle = last 10%)")
    save_current_figure(fig_dir, f"{prefix}_first_last10.png")


def plot_loading_by_residue(loadings, sel_mask, meta, pc, title_prefix, prefix, fig_dir):
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
    save_current_figure(fig_dir, f"{prefix}_pc{pc}_loadings.png")


def downsample_traj_list(traj_list, stride):
    return [xyz[::stride].copy() for xyz in traj_list]


def frame_indices_for_pca(frame_idx_list, traj_small, pca_stride):
    frame_idx_small_by_traj = []
    for ti, xyz_small in enumerate(traj_small):
        idx = np.asarray(frame_idx_list[ti]).astype(int)[::pca_stride]
        if len(idx) != xyz_small.shape[0]:
            raise ValueError(
                f"PCA frame mapping mismatch for traj {ti}: "
                f"{len(idx)} frame indices vs {xyz_small.shape[0]} downsampled frames"
            )
        frame_idx_small_by_traj.append(idx.copy())
    return frame_idx_small_by_traj


def save_scores_table(scores_by_traj, F, prefix, frame_idx_small_by_traj, results_dir):
    rows = []
    for ti, Z in enumerate(scores_by_traj):
        p = Path(F[ti])
        mutant = p.parent.name
        replica = p.name
        frame_idx_orig = np.asarray(frame_idx_small_by_traj[ti]).astype(int)
        if len(frame_idx_orig) != Z.shape[0]:
            raise ValueError(f"Frame-index mapping length mismatch for traj {ti}: {len(frame_idx_orig)} vs {Z.shape[0]}")
        for fi, row in enumerate(Z):
            rec = {
                "traj_index": ti,
                "mutant": mutant,
                "replica": replica,
                "frame_index_downsampled": fi,
                "frame_index_original": int(frame_idx_orig[fi]),
            }
            for j, val in enumerate(row, start=1):
                rec[f"PC{j}"] = float(val)
            rows.append(rec)

    df = pd.DataFrame(rows)
    out = Path(results_dir) / f"{prefix}_scores.csv"
    df.to_csv(out, index=False)
    print("Saved scores table:", out)


def save_pca_metadata(prefix, title_prefix, pca_stride, ncomponents, results_dir, selection_atom_count=None):
    out = Path(results_dir) / f"{prefix}_pca_metadata.json"
    payload = {
        "prefix": prefix,
        "title_prefix": title_prefix,
        "pca_stride": int(pca_stride),
        "ncomponents": int(ncomponents),
        "selection_atom_count": None if selection_atom_count is None else int(selection_atom_count),
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print("Saved PCA metadata:", out)


def run_pca_block(
    traj_list,
    sel_mask,
    title_prefix,
    prefix,
    traj_colors,
    F,
    frame_idx_list,
    meta,
    results_dir,
    fig_dir,
    pca_stride=10,
    ncomponents=10,
    plot_pairs=((0, 1), (0, 2), (1, 2), (0, 3), (1, 3)),
    timer=None,
):
    """Run coordinate PCA for one trajectory representation and save standard ROS1 outputs."""
    ctx = timer(f"{title_prefix}: vectorizing downsampled conformations") if timer else nullcontext()
    with ctx:
        traj_small = downsample_traj_list(traj_list, pca_stride)
        frame_idx_small_by_traj = frame_indices_for_pca(frame_idx_list, traj_small, pca_stride)

        all_frames = [traj_frames_atoms(xyz, sel_mask) for xyz in traj_small]
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

    ctx = timer(f"{title_prefix}: diagonalization (PCA)") if timer else nullcontext()
    with ctx:
        C = X.T @ (X / len(X))
        evals, loadings = np.linalg.eigh(C)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        loadings = loadings[:, idx]

    plot_scree(evals, n=25, title=f"{title_prefix} scree plot")
    save_current_figure(fig_dir, f"{prefix}_scree.png")

    scale = np.sqrt(mean.size)
    ctx = timer(f"{title_prefix}: projections on first {ncomponents} components") if timer else nullcontext()
    with ctx:
        scores_flat = (X @ loadings[:, :ncomponents]) / scale

    scores_by_traj = [scores_flat[X_owner == ti].copy() for ti in range(len(F))]
    print(f"{title_prefix} scores prepared.")

    save_scores_table(scores_by_traj, F, prefix, frame_idx_small_by_traj, results_dir)

    for a, b in plot_pairs:
        fig, ax = plt.subplots(figsize=(9, 9))
        for ti in range(len(F)):
            Zi = scores_by_traj[ti]
            ax.scatter(Zi[:, a], Zi[:, b], s=2, c=[traj_colors[ti]], alpha=0.15, linewidths=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"PC{a+1}")
        ax.set_ylabel(f"PC{b+1}")
        ax.set_title(f"{title_prefix} scores plot (PC{a+1} vs PC{b+1})")
        save_current_figure(fig_dir, f"{prefix}_pc{a+1}_pc{b+1}.png")

    plot_first_last_10_overlay(scores_by_traj, traj_colors, f"{title_prefix} PCA", prefix, fig_dir)
    plot_loading_by_residue(loadings, sel_mask, meta, 1, f"{title_prefix} PCA", prefix, fig_dir)
    plot_loading_by_residue(loadings, sel_mask, meta, 2, f"{title_prefix} PCA", prefix, fig_dir)

    results_dir = Path(results_dir)
    np.save(results_dir / f"{prefix}_loadings.npy", loadings)
    np.save(results_dir / f"{prefix}_mean.npy", mean)
    np.save(results_dir / f"{prefix}_scores.npy", scores_flat)
    np.save(results_dir / f"{prefix}_x_owner.npy", X_owner)
    save_pca_metadata(prefix, title_prefix, pca_stride, ncomponents, results_dir, selection_atom_count=int(np.sum(sel_mask)))
    print("Saved arrays:", prefix)

    return {
        "mean": mean.copy(),
        "loadings": loadings.copy(),
        "scores": scores_flat.copy(),
        "X_owner": X_owner.copy(),
        "scores_by_traj": [z.copy() for z in scores_by_traj],
        "F": np.array(F, dtype=object),
        "sele": sel_mask.copy(),
        "pca_stride": pca_stride,
    }


class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False
