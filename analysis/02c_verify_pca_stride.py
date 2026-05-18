# ROS1 Analysis Pipeline
# Script 02c_verify: PCA Stride Stability Verification
#
# Verifies that PCA_STRIDE=10 does not meaningfully change the dominant
# PC directions compared to finer strides (1 and 5). For each PCA space,
# PCA is computed at strides 1, 5, and 10 on a single selected trajectory
# and the cosine similarity between PC loading vectors is reported.
#
# Run once after 02c and before 02d to confirm stride 10 is valid.
#
# Usage:
#   python 02c_verify_pca_stride.py
#   python 02c_verify_pca_stride.py --prefixes aloop --traj-index 0
#   python 02c_verify_pca_stride.py --prefixes aloop,actout_ctlfit --traj-index 3

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from types import SimpleNamespace

try:
    from scripts.ros1_utils import combine_masks, traj_frames_atoms
    from scripts.ros1_align import align_traj_to_ref_by_fit
    _HELPERS = True
except ImportError:
    _HELPERS = False
    print("[WARN] Pipeline helpers not importable. Using inline fallbacks.")

# ============================================================
# Settings
# ============================================================
STRIDES_TO_TEST = [1, 5, 10]
NCOMPONENTS     = 10
STRIDE_COLORS   = {1: "#2166ac", 5: "#f4a582", 10: "#d6604d"}

BASE    = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "02c_verify_stride"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Inline fallbacks (used only when scripts/ is not importable)
# ============================================================

def _traj_frames_atoms(xyz, mask):
    """Extract selected atoms and flatten to (n_frames, n_atoms*3)."""
    return xyz[:, mask, :].reshape(xyz.shape[0], -1).astype(np.float64)


def _align_traj_to_ref(xyz, fit_mask, ref_fit):
    """Kabsch alignment of each frame onto ref_fit."""
    out = xyz.copy()
    for fi in range(xyz.shape[0]):
        mob   = xyz[fi, fit_mask, :]
        mob_c = mob - mob.mean(axis=0)
        H     = mob_c.T @ ref_fit
        U, _, Vt = np.linalg.svd(H)
        d  = np.linalg.det(Vt.T @ U.T)
        D  = np.diag([1.0, 1.0, d])
        R  = Vt.T @ D @ U.T
        t  = -mob.mean(axis=0) @ R.T
        out[fi] = xyz[fi] @ R.T + t
    return out


if _HELPERS:
    _traj_frames_atoms = traj_frames_atoms
    _align_fn          = align_traj_to_ref_by_fit
else:
    _align_fn = _align_traj_to_ref


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Verify PCA stride stability.")
    p.add_argument("--prefixes", default="aloop,actout_ctlfit",
                   help="Comma-separated PCA prefixes to verify.")
    p.add_argument("--traj-index", type=int, default=0,
                   help="Trajectory index to use for comparison (default: 0).")
    p.add_argument("--n-frames-max", type=int, default=0,
                   help="Cap frames used at stride 1 (0 = no cap).")
    return p.parse_args()


# ============================================================
# PCA at a given stride
# ============================================================

def run_pca_at_stride(traj_xyz, sel_mask, stride, n_components=NCOMPONENTS):
    """
    Run PCA on a single trajectory subsampled at the given stride.

    Parameters
    ----------
    traj_xyz     : ndarray (n_frames, n_atoms, 3)
    sel_mask     : boolean ndarray (n_atoms,)
    stride       : int
    n_components : int

    Returns
    -------
    dict with keys: loadings, evals, scores, explained_variance_ratio,
                    n_frames_used, stride
    """
    xyz_s = traj_xyz[::stride]
    X     = _traj_frames_atoms(xyz_s, sel_mask)
    mean  = X.mean(axis=0)
    Xc    = X - mean

    C              = Xc.T @ (Xc / len(Xc))
    evals, evecs   = np.linalg.eigh(C)
    order          = np.argsort(evals)[::-1]
    evals, evecs   = evals[order], evecs[:, order]

    total_var = evals.sum()
    exp_var   = evals[:n_components] / (total_var + 1e-12)
    scale     = np.sqrt(mean.size)
    scores    = (Xc @ evecs[:, :n_components]) / scale

    return {
        "loadings"                : evecs[:, :n_components],
        "evals"                   : evals,
        "scores"                  : scores,
        "explained_variance_ratio": exp_var,
        "n_frames_used"           : len(xyz_s),
        "stride"                  : stride,
    }


# ============================================================
# Cosine similarity metric
# ============================================================

def cosine_similarity(v1, v2):
    """
    Sign-invariant cosine similarity between two vectors.
    Returns |cos θ| in [0, 1]; 1.0 = identical direction.
    """
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return np.nan
    return float(np.abs(v1 @ v2) / (n1 * n2))


def compare_strides(results_dict, reference_stride=10):
    """
    Compare PC loading directions at each stride against the reference stride.

    Returns a DataFrame with columns: PC, stride, cosine_sim_vs_s{ref},
    explained_variance_ratio, n_frames_used.
    """
    ref = results_dict[reference_stride]
    rows = []
    for stride, res in results_dict.items():
        for pc in range(NCOMPONENTS):
            cos = cosine_similarity(res["loadings"][:, pc], ref["loadings"][:, pc])
            rows.append({
                "PC"                        : pc + 1,
                "stride"                    : stride,
                f"cosine_sim_vs_s{reference_stride}": cos,
                "explained_variance_ratio"  : float(res["explained_variance_ratio"][pc]),
                "n_frames_used"             : res["n_frames_used"],
            })
    return pd.DataFrame(rows)


# ============================================================
# Plots
# ============================================================

def plot_pc1_pc2_overlay(results_dict, prefix, traj_index, fig_dir):
    """Overlay PC1 vs PC2 score clouds at each stride."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for stride, res in results_dict.items():
        sc = res["scores"]
        ax.scatter(sc[:, 0], sc[:, 1], s=1, alpha=0.3,
                   color=STRIDE_COLORS[stride], label=f"stride {stride}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{prefix} — PC1 vs PC2 by stride (traj {traj_index})")
    ax.legend(markerscale=5)
    out = fig_dir / f"{prefix}_traj{traj_index}_stride_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


def plot_cosine_heatmap(comparison_df, prefix, traj_index, fig_dir):
    """Heatmap of cosine similarities for PCs 1-5 at each stride."""
    sub    = comparison_df[comparison_df["PC"] <= 5].copy()
    col    = [c for c in sub.columns if c.startswith("cosine_sim")][0]
    pivot  = sub.pivot(index="stride", columns="PC", values=col)
    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(pivot.values, vmin=0.9, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"PC{c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"stride {s}" for s in pivot.index])
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.4f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"{prefix} — cosine similarity vs stride 10 (traj {traj_index})")
    out = fig_dir / f"{prefix}_traj{traj_index}_cosine_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


def plot_explained_variance(results_dict, prefix, traj_index, fig_dir):
    """Scree plots at each stride overlaid."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for stride, res in results_dict.items():
        evr = res["explained_variance_ratio"]
        ax.plot(range(1, len(evr) + 1), evr * 100,
                marker="o", markersize=3, label=f"stride {stride}",
                color=STRIDE_COLORS[stride])
    ax.set_xlabel("PC")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title(f"{prefix} — explained variance by stride (traj {traj_index})")
    ax.legend()
    out = fig_dir / f"{prefix}_traj{traj_index}_explained_variance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# ============================================================
# Load pipeline outputs
# ============================================================

def load_pipeline_outputs():
    """Load all shared outputs from Script 01 and 02c."""
    EXCLUDE_MUTANTS = {"F1994L"}

    F              = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
    traj_aligned   = np.load(RESULTS / "traj_aligned.npy", allow_pickle=True).tolist()
    frame_idx_list = np.load(RESULTS / "frame_idx_list.npy", allow_pickle=True).tolist()

    keep           = [i for i, f in enumerate(F) if Path(f).parent.name not in EXCLUDE_MUTANTS]
    F              = [F[i] for i in keep]
    traj_aligned   = [traj_aligned[i] for i in keep]
    frame_idx_list = [frame_idx_list[i] for i in keep]

    masks_npz  = np.load(RESULTS / "masks.npz", allow_pickle=True)
    masks      = {k: masks_npz[k] for k in masks_npz.files}
    ref_centered = np.load(RESULTS / "ref_centered.npy", allow_pickle=True)

    meta = SimpleNamespace(
        resids   = np.load(RESULTS / "meta_resids.npy", allow_pickle=True),
        names    = np.load(RESULTS / "meta_names.npy", allow_pickle=True),
        resnames = np.load(RESULTS / "meta_resnames.npy", allow_pickle=True),
    )

    return F, traj_aligned, frame_idx_list, masks, ref_centered, meta


def get_sel_and_traj_for_prefix(prefix, masks, traj_aligned, ref_centered):
    """
    Return the atom selection mask and aligned trajectory list
    appropriate for the given PCA prefix.
    """
    BB   = masks["BB"]
    BODY = masks["BODY"]
    CTL  = masks["CTL"]
    ACT  = masks["ACT"]
    NTL  = masks["NTL"]

    if prefix == "aloop":
        sel_mask  = combine_masks(BB, ACT) if _HELPERS else (BB & ACT)
        traj_list = traj_aligned

    elif prefix == "actout_ctlfit":
        sel_mask   = combine_masks(BB, BODY) if _HELPERS else (BB & BODY)
        fit_mask   = combine_masks(BB, CTL)  if _HELPERS else (BB & CTL)
        ref_fit    = ref_centered[fit_mask] - ref_centered[fit_mask].mean(axis=0, keepdims=True)
        traj_list  = [_align_fn(xyz, fit_mask, ref_fit) for xyz in traj_aligned]

    elif prefix == "activesite":
        resids_abs = np.asarray(masks.get("resids_atom", np.zeros(BB.shape[0]))).astype(int) + 1933
        REGIONS    = [(1957, 1962), (1983, 1993), (2019, 2025), (2031, 2038), (2042, 2044)]
        as_mask    = np.zeros(len(resids_abs), dtype=bool)
        for start, end in REGIONS:
            as_mask |= (resids_abs >= start) & (resids_abs <= end)
        sel_mask   = as_mask & BB
        fit_mask   = combine_masks(BB, NTL) if _HELPERS else (BB & NTL)
        ref_fit    = ref_centered[fit_mask] - ref_centered[fit_mask].mean(axis=0, keepdims=True)
        traj_list  = [_align_fn(xyz, fit_mask, ref_fit) for xyz in traj_aligned]

    else:
        raise ValueError(f"Unknown prefix '{prefix}'. "
                         "Supported: aloop, actout_ctlfit, activesite.")

    return sel_mask, traj_list


# ============================================================
# Interpretation summary
# ============================================================

def interpret_results(comparison_df, prefix):
    """Print a plain-English summary of the cosine similarity results."""
    print(f"\n{'='*60}")
    print(f"STRIDE STABILITY SUMMARY — {prefix}")
    print(f"{'='*60}")

    for pc in [1, 2]:
        thresh = 0.99 if pc == 1 else 0.95
        for stride in [1, 5]:
            row = comparison_df[(comparison_df["PC"] == pc) & (comparison_df["stride"] == stride)]
            if row.empty:
                continue
            cos    = float(row["cosine_sim_vs_s10"].iloc[0])
            status = "STABLE" if cos >= thresh else "UNSTABLE — investigate"
            print(f"  PC{pc} | stride {stride:2d} vs stride 10 | cos={cos:.4f} | {status}")

    pc1_min = comparison_df[comparison_df["PC"] == 1]["cosine_sim_vs_s10"].min()
    pc2_min = comparison_df[comparison_df["PC"] == 2]["cosine_sim_vs_s10"].min()

    print()
    if pc1_min >= 0.99 and pc2_min >= 0.95:
        print(f"  Stride 10 is stable for {prefix}. PC1 min cos={pc1_min:.4f}, PC2 min cos={pc2_min:.4f}.")
    elif pc1_min >= 0.95:
        print(f"  PC1 is stable but PC2 drifts (min cos={pc2_min:.4f}). Consider stride 5.")
    else:
        print(f"  PC directions are NOT stable at stride 10 (PC1 min cos={pc1_min:.4f}). Use stride 1 or 5.")
    print(f"{'='*60}\n")


# ============================================================
# Main
# ============================================================

def main():
    args       = parse_args()
    prefixes   = [x.strip() for x in args.prefixes.split(",") if x.strip()]
    traj_index = args.traj_index

    print(f"Prefixes: {prefixes} | Strides: {STRIDES_TO_TEST} | Output: {FIG_DIR}\n")

    F, traj_aligned, frame_idx_list, masks, ref_centered, meta = load_pipeline_outputs()

    # Resolve traj_index — fall back to WT if out of range after exclusions
    if traj_index >= len(F):
        print(f"traj_index={traj_index} out of range after exclusions (n={len(F)}). Searching for WT...")
        wt_index = next(
            (i for i, f in enumerate(F) if Path(f).parent.name == "WT"),
            None
        )
        if wt_index is None:
            raise ValueError("Could not find WT trajectory in F_paths.csv.")
        traj_index = wt_index
        print(f"Using WT at traj_index={traj_index}: {F[traj_index]}")

    print(f"Trajectory: {Path(F[traj_index]).parent.name} / {Path(F[traj_index]).name}")
    print(f"Frames: {traj_aligned[traj_index].shape[0]}\n")

    all_summary_rows = []

    for prefix in prefixes:
        print(f"{'─'*50}\nPREFIX: {prefix}\n{'─'*50}")

        sel_mask, traj_list = get_sel_and_traj_for_prefix(prefix, masks, traj_aligned, ref_centered)
        print(f"Atom selection: {int(sel_mask.sum())} atoms | Feature vector: {int(sel_mask.sum()) * 3} dims")

        xyz_test = traj_list[traj_index]
        if args.n_frames_max > 0 and xyz_test.shape[0] > args.n_frames_max:
            xyz_test = xyz_test[:args.n_frames_max]
            print(f"Frame cap: {args.n_frames_max} frames")

        results_dict = {}
        for stride in STRIDES_TO_TEST:
            n_s = len(range(0, xyz_test.shape[0], stride))
            print(f"  stride {stride:2d} ({n_s} frames) ...", end=" ")
            results_dict[stride] = run_pca_at_stride(xyz_test, sel_mask, stride, NCOMPONENTS)
            ev12 = results_dict[stride]["explained_variance_ratio"][:2].sum()
            print(f"done  [PC1+PC2: {ev12*100:.1f}%]")

        comparison_df = compare_strides(results_dict, reference_stride=10)
        print("\nCosine similarity (|cos θ| vs stride-10):")
        print(comparison_df[comparison_df["PC"] <= 5].to_string(index=False))

        interpret_results(comparison_df, prefix)

        plot_pc1_pc2_overlay(results_dict, prefix, traj_index, FIG_DIR)
        plot_cosine_heatmap(comparison_df, prefix, traj_index, FIG_DIR)
        plot_explained_variance(results_dict, prefix, traj_index, FIG_DIR)

        out_csv = RESULTS / f"stride_verify_{prefix}_traj{traj_index}.csv"
        comparison_df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

        comparison_df.insert(0, "prefix", prefix)
        comparison_df.insert(1, "traj_index", traj_index)
        comparison_df.insert(2, "mutant", Path(F[traj_index]).parent.name)
        all_summary_rows.append(comparison_df)

    if all_summary_rows:
        summary     = pd.concat(all_summary_rows, ignore_index=True)
        combined_out = RESULTS / f"stride_verify_summary_traj{traj_index}.csv"
        summary.to_csv(combined_out, index=False)
        print(f"\nCombined summary saved: {combined_out}")

        print(f"\n{'='*60}\nFINAL PASS/FAIL\n{'='*60}")
        print(f"{'Prefix':<20} {'PC':<5} {'stride 1':>10} {'stride 5':>10} {'PASS?':>8}")
        print("-" * 60)
        for prefix in prefixes:
            sub = summary[summary["prefix"] == prefix]
            for pc in [1, 2]:
                thresh = 0.99 if pc == 1 else 0.95
                c1 = sub[(sub["PC"] == pc) & (sub["stride"] == 1)]["cosine_sim_vs_s10"]
                c5 = sub[(sub["PC"] == pc) & (sub["stride"] == 5)]["cosine_sim_vs_s10"]
                c1 = float(c1.iloc[0]) if not c1.empty else np.nan
                c5 = float(c5.iloc[0]) if not c5.empty else np.nan
                passed = (c1 >= thresh) and (c5 >= thresh)
                print(f"{prefix:<20} {pc:<5} {c1:>10.4f} {c5:>10.4f} {'PASS' if passed else 'FAIL':>8}")
        print("=" * 60)

    print(f"\nScript 02c_verify completed. Figures: {FIG_DIR} | Tables: {RESULTS}")


if __name__ == "__main__":
    main()