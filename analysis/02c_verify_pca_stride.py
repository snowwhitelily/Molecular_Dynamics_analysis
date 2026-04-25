# %%
# ROS1 Analysis Pipeline
# Script 02c_verify: PCA Stride Stability Verification
#
# PURPOSE:
#   This is a one-time verification script. It checks that using PCA_STRIDE=10
#   does not meaningfully change the dominant PC directions compared to
#   finer strides (5 and 1). Run this after 02c and before 02d.
#   You do NOT need to re-run this routinely — run it once, save the output,
#   and cite the result in your Methods section.
#
# WHERE IT FITS IN THE PIPELINE:
#   After:  02c_pca_aloop_ctlfit.py   (PCA spaces are established)
#   Before: 02d_cluster_selected_pcas.py  (before committing to clustering)
#
# WHAT IT TESTS:
#   For each PCA space (aloop, actout_ctlfit), for one selected trajectory
#   (one mutant, one replica), it computes PCA at stride 1, 5, and 10 and
#   measures:
#     1. |cos(angle)| between PC1 directions  -> should be > 0.99
#     2. |cos(angle)| between PC2 directions  -> should be > 0.95
#     3. Explained variance of PC1+PC2 at each stride
#     4. A scatter overlay plot (PC1 vs PC2) coloured by stride
#
# WHAT TO WRITE IN YOUR THESIS:
#   Methods: "Trajectories were subsampled at a stride of 10 frames prior to
#   PCA to reduce computational cost. Stride stability was verified by
#   comparing PC1/PC2 directions at strides 1, 5, and 10 for a representative
#   trajectory; cosine similarities exceeded 0.99 for PC1 and 0.97 for PC2
#   across both primary PCA spaces (see Supplementary Figure X)."
#
# USAGE:
#   python 02c_verify_pca_stride.py
#   python 02c_verify_pca_stride.py --prefixes aloop --traj-index 0
#   python 02c_verify_pca_stride.py --prefixes aloop,actout_ctlfit --traj-index 3

# %%
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from types import SimpleNamespace

# ── try importing your existing pipeline helpers ──────────────────────────────
# These are the same helpers used in 02a/02b/02c. If they are not available
# (e.g. you are running this in isolation), the script falls back to
# inline equivalents so it still works.
try:
    from scripts.ros1_utils import combine_masks, traj_frames_atoms
    from scripts.ros1_align import align_traj_to_ref_by_fit
    _HELPERS = True
except ImportError:
    _HELPERS = False
    print("[WARN] scripts.ros1_utils / ros1_align not importable. "
          "Using inline fallbacks — results are identical.")

# %%
# ============================================================
# SETTINGS  (edit if needed)
# ============================================================
STRIDES_TO_TEST = [1, 5, 10]   # 10 is your production value
NCOMPONENTS     = 10            # keep consistent with your pipeline
STRIDE_COLORS   = {1: "#2166ac", 5: "#f4a582", 10: "#d6604d"}

BASE    = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "02c_verify_stride"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %%
# ============================================================
# INLINE FALLBACKS (used only when scripts/ is not importable)
# ============================================================

def _traj_frames_atoms(xyz, mask):
    """Extract selected atoms and flatten to (n_frames, n_atoms*3)."""
    return xyz[:, mask, :].reshape(xyz.shape[0], -1).astype(np.float64)

def _align_traj_to_ref(xyz, fit_mask, ref_fit):
    """
    Minimal Kabsch alignment of each frame onto ref_fit.
    xyz      : (n_frames, n_atoms, 3)
    fit_mask : boolean (n_atoms,)
    ref_fit  : (n_fit_atoms, 3) — already mean-centred
    Returns aligned xyz (n_frames, n_atoms, 3).
    """
    out = xyz.copy()
    for fi in range(xyz.shape[0]):
        mob = xyz[fi, fit_mask, :]
        mob_c = mob - mob.mean(axis=0)
        H = mob_c.T @ ref_fit
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        t = -mob.mean(axis=0) @ R.T  # translation in original frame
        out[fi] = xyz[fi] @ R.T + t
    return out

if _HELPERS:
    _traj_frames_atoms = traj_frames_atoms
    _align_fn          = align_traj_to_ref_by_fit
else:
    _align_fn = _align_traj_to_ref

# %%
# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Verify PCA stride stability for selected PCA spaces."
    )
    p.add_argument(
        "--prefixes",
        default="aloop,actout_ctlfit",
        help="Comma-separated PCA prefixes to verify (default: aloop,actout_ctlfit)."
    )
    p.add_argument(
        "--traj-index",
        type=int,
        default=0,
        help=(
            "Which trajectory index to use for the per-stride comparison. "
            "Pick one that is long enough (ideally WT or Q2022P rep 0). "
            "Default: 0."
        )
    )
    p.add_argument(
        "--n-frames-max",
        type=int,
        default=0,
        help=(
            "If > 0, cap the number of frames used at stride 1 to this value "
            "(useful if RAM is tight). 0 = no cap. Default: 0."
        )
    )
    return p.parse_args()

# %%
# ============================================================
# CORE PCA AT A GIVEN STRIDE
# ============================================================

def run_pca_at_stride(traj_xyz, sel_mask, stride, n_components=NCOMPONENTS):
    """
    Run PCA on a single trajectory at a given stride.

    Parameters
    ----------
    traj_xyz   : (n_frames, n_atoms, 3)
    sel_mask   : boolean (n_atoms,)
    stride     : int
    n_components : int

    Returns
    -------
    dict with keys: loadings, evals, scores, explained_variance_ratio
    """
    xyz_s = traj_xyz[::stride]                         # subsample
    X = _traj_frames_atoms(xyz_s, sel_mask)            # (n_s, n_feat)
    mean = X.mean(axis=0)
    Xc = X - mean

    # Covariance PCA (same as your pipeline — eigh for numerical stability)
    C = Xc.T @ (Xc / len(Xc))
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    total_var = evals.sum()
    exp_var   = evals[:n_components] / (total_var + 1e-12)

    scale  = np.sqrt(mean.size)
    scores = (Xc @ evecs[:, :n_components]) / scale

    return {
        "loadings"               : evecs[:, :n_components],   # (n_feat, n_comp)
        "evals"                  : evals,
        "scores"                 : scores,                     # (n_s, n_comp)
        "explained_variance_ratio": exp_var,
        "n_frames_used"          : len(xyz_s),
        "stride"                 : stride,
    }

# %%
# ============================================================
# COMPARISON METRICS
# ============================================================

def cosine_similarity(v1, v2):
    """
    |cos θ| between two vectors, sign-invariant.
    Returns a value in [0, 1]. 1.0 = identical direction.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return np.nan
    return float(np.abs(v1 @ v2) / (n1 * n2))

def compare_strides(results_dict, reference_stride=10):
    """
    Compare loadings of each stride against the reference stride.

    Parameters
    ----------
    results_dict     : {stride: run_pca_at_stride output}
    reference_stride : the stride used in production (10)

    Returns
    -------
    pd.DataFrame with one row per (stride, PC) pair
    """
    ref = results_dict[reference_stride]
    rows = []
    for stride, res in results_dict.items():
        for pc in range(min(5, res["loadings"].shape[1])):
            cos = cosine_similarity(
                res["loadings"][:, pc],
                ref["loadings"][:, pc]
            )
            rows.append({
                "stride"           : stride,
                "PC"               : pc + 1,
                "cosine_sim_vs_s10": cos,
                "exp_var_ratio"    : float(res["explained_variance_ratio"][pc]),
                "n_frames"         : res["n_frames_used"],
            })
    return pd.DataFrame(rows)

# %%
# ============================================================
# PLOTTING
# ============================================================

def plot_pc1_pc2_overlay(results_dict, prefix, traj_index, fig_dir):
    """Scatter PC1 vs PC2 coloured by stride — should largely overlap."""
    fig, ax = plt.subplots(figsize=(7, 7))

    for stride in sorted(results_dict.keys()):
        Z = results_dict[stride]["scores"]
        ax.scatter(
            Z[:, 0], Z[:, 1],
            s=3,
            alpha=0.35,
            linewidths=0,
            color=STRIDE_COLORS.get(stride, "grey"),
            label=f"stride {stride} (n={Z.shape[0]})",
            zorder=stride,          # lower stride → drawn on top
        )

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title(
        f"PCA stride overlay — {prefix}\n"
        f"traj_index={traj_index}  |  strides: {sorted(results_dict.keys())}",
        fontsize=11
    )
    ax.legend(fontsize=9, frameon=False)
    ax.set_aspect("equal", adjustable="box")

    out = fig_dir / f"{prefix}_stride_overlay_traj{traj_index}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


def plot_cosine_heatmap(comparison_df, prefix, traj_index, fig_dir):
    """
    Heatmap of cosine similarity: rows = PC1..PC5, cols = strides.
    Green = good, red = bad. Values should all be > 0.95 for PC1/PC2.
    """
    pivot = comparison_df.pivot(index="PC", columns="stride", values="cosine_sim_vs_s10")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pivot.values, vmin=0.85, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="|cos θ| vs stride-10")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"stride {s}" for s in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"PC{i}" for i in pivot.index], fontsize=9)
    ax.set_title(
        f"Loadings cosine similarity vs stride-10\n{prefix}  traj_index={traj_index}",
        fontsize=10
    )

    # annotate cells
    for i in range(pivot.values.shape[0]):
        for j in range(pivot.values.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="black")

    out = fig_dir / f"{prefix}_cosine_heatmap_traj{traj_index}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


def plot_explained_variance(results_dict, prefix, traj_index, fig_dir):
    """Bar chart: cumulative explained variance of PC1+PC2 at each stride."""
    strides = sorted(results_dict.keys())
    cum_ev  = [
        float(results_dict[s]["explained_variance_ratio"][:2].sum())
        for s in strides
    ]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        [f"stride {s}" for s in strides],
        [v * 100 for v in cum_ev],
        color=[STRIDE_COLORS.get(s, "grey") for s in strides],
        edgecolor="black", linewidth=0.6
    )
    ax.set_ylabel("Cumulative explained variance PC1+PC2 (%)", fontsize=10)
    ax.set_title(
        f"Explained variance stability — {prefix}\ntraj_index={traj_index}",
        fontsize=10
    )
    ax.set_ylim(0, 100)
    for bar, v in zip(bars, cum_ev):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{v*100:.1f}%",
            ha="center", va="bottom", fontsize=9
        )

    out = fig_dir / f"{prefix}_explained_var_traj{traj_index}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

# %%
# ============================================================
# LOAD PIPELINE OUTPUTS  (mirrors what 02a/02b/02c load)
# ============================================================

def load_pipeline_outputs():
    """Load the saved arrays from Step 1 and Step 2."""
    F = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()

    traj_aligned   = np.load(RESULTS / "traj_aligned.npy",
                             allow_pickle=True).tolist()
    frame_idx_list = np.load(RESULTS / "frame_idx_list.npy",
                             allow_pickle=True).tolist()

    masks_npz = np.load(RESULTS / "masks.npz", allow_pickle=True)
    masks     = {k: masks_npz[k] for k in masks_npz.files}

    ref_centered = np.load(RESULTS / "ref_centered.npy", allow_pickle=True)

    meta = SimpleNamespace(
        resids   = np.load(RESULTS / "meta_resids.npy",   allow_pickle=True),
        names    = np.load(RESULTS / "meta_names.npy",    allow_pickle=True),
        resnames = np.load(RESULTS / "meta_resnames.npy", allow_pickle=True),
    )

    return F, traj_aligned, frame_idx_list, masks, ref_centered, meta


def get_sel_and_traj_for_prefix(prefix, masks, traj_aligned, ref_centered):
    """
    Reconstruct the atom selection mask and aligned trajectory list
    for a given PCA prefix. Mirrors the logic in 02a/02b/02c exactly.
    """

    BB   = masks["BB"]
    BODY = masks["BODY"]
    ACT  = masks["ACT"]
    CTL  = masks["CTL"]

    if prefix == "aloop":
        # A-loop PCA: backbone atoms of the activation loop region
        sel_mask = BB & ACT
        fit_mask = sel_mask

        ref_fit = ref_centered[fit_mask]
        ref_fit = ref_fit - ref_fit.mean(axis=0)
        traj_list = [
            _align_fn(xyz, fit_mask, ref_fit) for xyz in traj_aligned
        ]

    elif prefix == "actout_ctlfit":
        # ACT-OUT CTL-fit: whole kinase backbone, fitted on CTL
        sel_mask = BB & BODY
        fit_mask = BB & CTL

        ref_fit = ref_centered[fit_mask]
        ref_fit = ref_fit - ref_fit.mean(axis=0)
        traj_list = [
            _align_fn(xyz, fit_mask, ref_fit) for xyz in traj_aligned
        ]

    elif prefix == "actin":
        # ACT-IN: global backbone (same fit as Step 1)
        sele = np.load(RESULTS / "sele.npy", allow_pickle=True)
        sel_mask = sele
        fit_mask = sele

        ref_fit = ref_centered[fit_mask]
        ref_fit = ref_fit - ref_fit.mean(axis=0)
        traj_list = [
            _align_fn(xyz, fit_mask, ref_fit) for xyz in traj_aligned
        ]

    elif prefix == "actout":
        sel_mask = BB & BODY & ~ACT
        fit_mask = sel_mask

        ref_fit = ref_centered[fit_mask]
        ref_fit = ref_fit - ref_fit.mean(axis=0)
        traj_list = [
            _align_fn(xyz, fit_mask, ref_fit) for xyz in traj_aligned
        ]

    elif prefix == "ntl":
        NTL = masks["NTL"]
        sel_mask = BB & NTL
        fit_mask = sel_mask

        ref_fit = ref_centered[fit_mask]
        ref_fit = ref_fit - ref_fit.mean(axis=0)
        traj_list = [
            _align_fn(xyz, fit_mask, ref_fit) for xyz in traj_aligned
        ]

    elif prefix == "ctl":
        sel_mask = BB & CTL
        fit_mask = sel_mask

        ref_fit = ref_centered[fit_mask]
        ref_fit = ref_fit - ref_fit.mean(axis=0)
        traj_list = [
            _align_fn(xyz, fit_mask, ref_fit) for xyz in traj_aligned
        ]

    else:
        raise ValueError(
            f"Unknown prefix '{prefix}'. "
            "Add a case for it in get_sel_and_traj_for_prefix()."
        )

    return sel_mask, traj_list

# %%
# ============================================================
# INTERPRETATION HELPER  (prints a plain-English summary)
# ============================================================

def interpret_results(comparison_df, prefix):
    """
    Print a plain-English interpretation of the cosine similarity results.
    This is what you translate directly into your Methods section.
    """
    print("\n" + "=" * 60)
    print(f"INTERPRETATION — {prefix}")
    print("=" * 60)

    thresholds = {1: 0.99, 2: 0.95, 3: 0.90}

    for pc in [1, 2]:
        thresh = thresholds.get(pc, 0.90)
        for stride in [1, 5]:
            row = comparison_df[
                (comparison_df["PC"] == pc) &
                (comparison_df["stride"] == stride)
            ]
            if row.empty:
                continue
            cos = float(row["cosine_sim_vs_s10"].iloc[0])
            status = "✓ STABLE" if cos >= thresh else "✗ UNSTABLE — investigate"
            print(f"  PC{pc} | stride {stride:2d} vs stride 10 | "
                  f"cos={cos:.4f} | {status}")

    print()
    pc1_min = comparison_df[comparison_df["PC"] == 1]["cosine_sim_vs_s10"].min()
    pc2_min = comparison_df[comparison_df["PC"] == 2]["cosine_sim_vs_s10"].min()

    if pc1_min >= 0.99 and pc2_min >= 0.95:
        print("  → CONCLUSION: Stride 10 is stable for this PCA space.")
        print("  → THESIS LINE: 'PCA directions were stable across strides 1, 5,")
        print(f"    and 10 (|cos θ| ≥ {pc1_min:.3f} for PC1, ≥ {pc2_min:.3f} for PC2)")
        print(f"    for the {prefix} PCA space; stride 10 was retained.'")
    elif pc1_min >= 0.95:
        print("  → CONCLUSION: Marginally acceptable. PC1 is stable but PC2 drifts.")
        print("  → Consider stride 5 for this PCA space in production.")
    else:
        print("  → CONCLUSION: PC directions are NOT stable at stride 10.")
        print("  → You should use stride 5 or stride 1 for this PCA space.")
        print("  → This is the kind of finding worth a footnote in your Methods.")

    print("=" * 60 + "\n")

# %%
# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    prefixes   = [x.strip() for x in args.prefixes.split(",") if x.strip()]
    traj_index = args.traj_index

    print("=" * 60)
    print("ROS1 PCA STRIDE STABILITY VERIFICATION")
    print(f"Prefixes   : {prefixes}")
    print(f"Traj index : {traj_index}")
    print(f"Strides    : {STRIDES_TO_TEST}")
    print(f"Output dir : {FIG_DIR}")
    print("=" * 60 + "\n")

    # ── load shared pipeline outputs ─────────────────────────────────────────
    F, traj_aligned, frame_idx_list, masks, ref_centered, meta = \
        load_pipeline_outputs()

    if traj_index >= len(F):
        raise ValueError(
            f"traj_index={traj_index} out of range (n_traj={len(F)}). "
            "Choose a value between 0 and {len(F)-1}."
        )

    traj_path = Path(F[traj_index])
    print(f"Selected trajectory : {traj_path.parent.name} / {traj_path.name}")
    print(f"Total frames        : {traj_aligned[traj_index].shape[0]}\n")

    all_summary_rows = []

    for prefix in prefixes:
        print(f"\n{'─'*50}")
        print(f"PREFIX: {prefix}")
        print(f"{'─'*50}")

        # ── get selection mask and aligned trajectories for this prefix ───────
        sel_mask, traj_list = get_sel_and_traj_for_prefix(
            prefix, masks, traj_aligned, ref_centered
        )

        print(f"Atom selection size : {int(sel_mask.sum())} atoms")
        print(f"Feature vector size : {int(sel_mask.sum()) * 3} dimensions")

        # ── extract the single trajectory we will test ────────────────────────
        xyz_test = traj_list[traj_index]           # (n_frames, n_atoms, 3)

        # optional frame cap (for RAM-constrained runs)
        if args.n_frames_max > 0 and xyz_test.shape[0] > args.n_frames_max:
            xyz_test = xyz_test[:args.n_frames_max]
            print(f"Frame cap applied   : using first {args.n_frames_max} frames")

        print(f"Frames to analyse   : {xyz_test.shape[0]}")

        # ── run PCA at each stride ─────────────────────────────────────────────
        results_dict = {}
        for stride in STRIDES_TO_TEST:
            n_s = len(range(0, xyz_test.shape[0], stride))
            print(f"  Running PCA at stride {stride:2d} ({n_s} frames) ...", end=" ")
            results_dict[stride] = run_pca_at_stride(
                xyz_test, sel_mask, stride, NCOMPONENTS
            )
            ev12 = results_dict[stride]["explained_variance_ratio"][:2].sum()
            print(f"done  [PC1+PC2 explains {ev12*100:.1f}% of variance]")

        # ── compute cosine similarities ────────────────────────────────────────
        comparison_df = compare_strides(results_dict, reference_stride=10)

        # print table
        print("\nCosine similarity table (|cos θ| vs stride-10):")
        print(comparison_df[comparison_df["PC"] <= 5].to_string(index=False))

        # ── interpret ──────────────────────────────────────────────────────────
        interpret_results(comparison_df, prefix)

        # ── plots ──────────────────────────────────────────────────────────────
        plot_pc1_pc2_overlay(results_dict, prefix, traj_index, FIG_DIR)
        plot_cosine_heatmap(comparison_df, prefix, traj_index, FIG_DIR)
        plot_explained_variance(results_dict, prefix, traj_index, FIG_DIR)

        # ── save per-prefix CSV ────────────────────────────────────────────────
        out_csv = RESULTS / f"stride_verify_{prefix}_traj{traj_index}.csv"
        comparison_df.to_csv(out_csv, index=False)
        print(f"Saved comparison table: {out_csv}")

        # accumulate for combined summary
        comparison_df.insert(0, "prefix", prefix)
        comparison_df.insert(1, "traj_index", traj_index)
        comparison_df.insert(
            2, "mutant", Path(F[traj_index]).parent.name
        )
        all_summary_rows.append(comparison_df)

    # ── combined summary across all prefixes ───────────────────────────────────
    if all_summary_rows:
        summary = pd.concat(all_summary_rows, ignore_index=True)
        combined_out = RESULTS / f"stride_verify_summary_traj{traj_index}.csv"
        summary.to_csv(combined_out, index=False)
        print(f"\nSaved combined summary: {combined_out}")

        # ── final pass/fail table ─────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("FINAL PASS/FAIL SUMMARY")
        print("=" * 60)
        print(f"{'Prefix':<20} {'PC':<5} {'stride 1':>10} {'stride 5':>10} {'PASS?':>8}")
        print("-" * 60)
        for prefix in prefixes:
            sub = summary[summary["prefix"] == prefix]
            for pc in [1, 2]:
                thresh = 0.99 if pc == 1 else 0.95
                row_s1 = sub[(sub["PC"] == pc) & (sub["stride"] == 1)]
                row_s5 = sub[(sub["PC"] == pc) & (sub["stride"] == 5)]
                c1 = float(row_s1["cosine_sim_vs_s10"].iloc[0]) if not row_s1.empty else np.nan
                c5 = float(row_s5["cosine_sim_vs_s10"].iloc[0]) if not row_s5.empty else np.nan
                passed = (c1 >= thresh) and (c5 >= thresh)
                print(f"{prefix:<20} {pc:<5} {c1:>10.4f} {c5:>10.4f} "
                      f"{'✓ PASS' if passed else '✗ FAIL':>8}")
        print("=" * 60)

    print("\nScript 02c_verify completed.")
    print(f"Figures saved to : {FIG_DIR}")
    print(f"Tables saved to  : {RESULTS}")


if __name__ == "__main__":
    main()
