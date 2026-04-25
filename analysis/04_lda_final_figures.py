# %%
# ROS1 Analysis Pipeline
# Script 4: LDA and Final Figures
#
# CHANGES FROM ORIGINAL:
#   - Added argument parsing for --family and --k-use
#   - Removed hardcoded q_family = ["Q2022P", "Q2022P_S1986F", "Q2022P_S1986Y"]
#   - Removed hardcoded pairs list — now auto-generated from --family members
#   - Default values match original hardcoded values, so running with no
#     arguments produces identical output to the original script
#
# USAGE (identical to original — no arguments needed for default behaviour):
#   python 04_lda_final_figures.py
#
# USAGE (different mutant panel):
#   python 04_lda_final_figures.py \
#       --family Q2022P,Q2022P_S1986F,Q2022P_S1986Y,G2032R
#
#   python 04_lda_final_figures.py \
#       --family Q2022P,G2032R,D2033N \
#       --k-use 8

import argparse
import itertools
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.ros1_analysis_helpers import (
    summarize_pair,
    lda_multiclass,
    lda_2class_direction,
    pairwise_ld1_scatter,
)

# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="LDA final figures for a specified family of mutants."
    )
    p.add_argument(
        "--family",
        default="Q2022P,Q2022P_S1986F,Q2022P_S1986Y",
        help=(
            "Comma-separated mutant names for the multi-class and pairwise "
            "LDA analysis. Default: Q2022P,Q2022P_S1986F,Q2022P_S1986Y"
        ),
    )
    p.add_argument(
        "--k-use",
        type=int,
        default=6,
        help=(
            "Number of leading PCs to use from trajectory means. "
            "Default: 6 (capped at available PCs)."
        ),
    )
    p.add_argument(
        "--scores-prefix",
        default="actout_ctlfit",
        help=(
            "Preferred PCA scores prefix to load. Falls back to 'dist' if "
            "the preferred prefix is not found. Default: actout_ctlfit"
        ),
    )
    return p.parse_args()


args = parse_args()

# Parse family from arguments
q_family = [x.strip() for x in args.family.split(",") if x.strip()]

if len(q_family) < 2:
    raise ValueError(
        f"--family must contain at least 2 mutant names. Got: {q_family}"
    )

print("Running Script 4")
print(f"Family        : {q_family}")
print(f"k_use (max)   : {args.k_use}")
print(f"Scores prefix : {args.scores_prefix}")

# ============================================================
# PATHS
# ============================================================

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"

# figure directory includes family label so different runs don't overwrite each other
family_tag = "_".join(q_family)
FIG_DIR = (
    BASE / "figures" / "ROS1" / "ros1_prepared_final"
    / "step4_lda_final" / family_tag
)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Figure output folder:", FIG_DIR)


def save_current_figure(filename: str):
    out = FIG_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure:", out)


# ============================================================
# PART 1: Load PCA scores
# ============================================================

F = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
traj_mut = np.array([Path(p).parent.name for p in F], dtype=object)

preferred = args.scores_prefix
preferred_scores = RESULTS / f"{preferred}_scores.npy"
preferred_owner  = RESULTS / f"{preferred}_x_owner.npy"

if preferred_scores.exists() and preferred_owner.exists():
    scores_flat = np.load(preferred_scores)
    x_owner     = np.load(preferred_owner)
    ps_name     = f"{preferred}_scores"
elif (RESULTS / "actout_ctlfit_scores.npy").exists() and \
     (RESULTS / "actout_ctlfit_x_owner.npy").exists():
    scores_flat = np.load(RESULTS / "actout_ctlfit_scores.npy")
    x_owner     = np.load(RESULTS / "actout_ctlfit_x_owner.npy")
    ps_name     = "actout_ctlfit_scores"
elif (RESULTS / "dist_scores.npy").exists() and \
     (RESULTS / "dist_x_owner.npy").exists():
    scores_flat = np.load(RESULTS / "dist_scores.npy")
    x_owner     = np.load(RESULTS / "dist_x_owner.npy")
    ps_name     = "dist_scores"
else:
    raise FileNotFoundError(
        "Could not find PCA scores. Need actout_ctlfit_scores/x_owner "
        "or dist_scores/x_owner in results/ROS1."
    )

PS_list = [np.asarray(scores_flat[x_owner == i]) for i in range(len(F))]
ntraj   = len(PS_list)
nPC     = min(z.shape[1] for z in PS_list if z.shape[1] >= 1)
k_use   = min(args.k_use, nPC)

X_traj = np.vstack([z[:, :k_use].mean(axis=0, keepdims=True) for z in PS_list])

print(f"[OK] Using {ps_name} as PCA source, ntraj={ntraj}, k_use={k_use}")

# ============================================================
# PART 2: Subset to family and run multi-class LDA
# ============================================================

mask_q    = np.isin(traj_mut, q_family)
Xq        = X_traj[mask_q]
namesq    = traj_mut[mask_q]
uniq_sys  = np.unique(namesq)

print("\n[Family LDA] systems in subset:", uniq_sys.tolist())
print("[Family LDA] counts:", {u: int((namesq == u).sum()) for u in uniq_sys})

if uniq_sys.size < 2:
    raise RuntimeError(
        f"Family subset has <2 systems in the data. "
        f"Requested: {q_family}. Found: {uniq_sys.tolist()}"
    )

label_map = {m: i for i, m in enumerate(uniq_sys)}
yq        = np.array([label_map[m] for m in namesq], dtype=int)

W, evals = lda_multiclass(Xq, yq, eps=1e-8)
ndim = min(len(uniq_sys) - 1, k_use)
Z    = Xq @ W[:, :ndim]

plt.figure(figsize=(7, 3))
for m in uniq_sys:
    c  = label_map[m]
    zz = Z[yq == c, 0]
    plt.scatter(zz, np.zeros_like(zz), s=80, alpha=0.85,
                label=f"{m} (n={zz.size})")
plt.xlabel("LD1")
plt.yticks([])
plt.title("Multi-class LDA on family (per-trajectory means)\n"
          f"Family: {', '.join(q_family)}")
plt.legend(fontsize=8, frameon=False)
save_current_figure("step4_multiclass_lda_family.png")

weights_df = pd.DataFrame({
    "PC":         [f"PC{i+1:02d}" for i in range(k_use)],
    "LD1_weight": W[:k_use, 0],
})
weights_df.to_csv(RESULTS / "step4_multiclass_lda_weights.csv", index=False)
print("Saved multiclass LDA weights.")

# ============================================================
# PART 3: All pairwise LDA within family
# Auto-generates all pairs from --family — no hardcoding needed
# ============================================================

# All unique pairs from the family members that are actually present in data
present = set(uniq_sys.tolist())
pairs   = [
    (A, B)
    for A, B in itertools.combinations(q_family, 2)
    if A in present and B in present
]

print(f"\n[Pairwise LDA] {len(pairs)} pairs to test: {pairs}")

pairwise_results = {}

for A, B in pairs:
    pair_mask = np.isin(namesq, [A, B])
    Xpair     = Xq[pair_mask]
    npair     = namesq[pair_mask]
    y01       = (npair == B).astype(int)

    nA = int((y01 == 0).sum())
    nB = int((y01 == 1).sum())

    if nA < 2 or nB < 2:
        print(f"Skipping {A} vs {B} (<2 samples/class: n_{A}={nA}, n_{B}={nB})")
        continue

    w        = lda_2class_direction(Xpair, y01)
    ld1_traj = Xpair @ w

    pairwise_results[(A, B)] = {
        "w": w, "ld1_traj": ld1_traj, "y01": y01, "names": npair
    }

    plt.figure(figsize=(7, 3))
    plt.scatter(ld1_traj[y01 == 0], np.zeros(nA), s=80, alpha=0.85, label=A)
    plt.scatter(ld1_traj[y01 == 1], np.zeros(nB), s=80, alpha=0.85, label=B)
    plt.xlabel("LD1")
    plt.yticks([])
    plt.title(f"Pairwise LDA on trajectory means: {A} vs {B}")
    plt.legend(frameon=False, fontsize=8)
    save_current_figure(f"step4_pairwise_lda_means_{A}_vs_{B}.png")

# ============================================================
# PART 4: Project all frames onto learned LD1
# ============================================================

X_frames_q = [PS_list[i][:, :k_use] for i in range(len(F)) if mask_q[i]]
names_q    = traj_mut[mask_q]
uniq_sys   = np.unique(names_q)

summary_rows = []

for (A, B), R in pairwise_results.items():
    w = np.asarray(R["w"], float)
    if w.shape[0] != k_use:
        continue

    pair_mask  = np.isin(names_q, [A, B])
    Xp         = [X_frames_q[i] for i in range(len(X_frames_q)) if pair_mask[i]]
    npair      = names_q[pair_mask]
    ld1_frames = [z @ w for z in Xp]

    vals_A = [ld1_frames[i] for i in range(len(npair)) if npair[i] == A]
    vals_B = [ld1_frames[i] for i in range(len(npair)) if npair[i] == B]

    if not vals_A or not vals_B:
        print(f"Skipping frame histogram for {A} vs {B}: "
              "missing frame projections for one class")
        continue

    plt.figure(figsize=(8, 3))
    plt.hist(np.concatenate(vals_A), bins=60, alpha=0.55,
             label=A, density=True)
    plt.hist(np.concatenate(vals_B), bins=60, alpha=0.55,
             label=B, density=True)
    plt.xlabel("LD1 (frame-level projection)")
    plt.ylabel("density")
    plt.title(f"Frame-level overlap: {A} vs {B}")
    plt.legend()
    save_current_figure(f"step4_pairwise_lda_frames_{A}_vs_{B}.png")

    stats     = summarize_pair(A, B, ld1_frames, npair)
    stats_row = {"A": A, "B": B}
    stats_row.update(stats)
    summary_rows.append(stats_row)

# multi-class frame overlay
W_use = np.asarray(W, float)
if W_use.shape[0] >= k_use:
    w_mc         = W_use[:k_use, 0]
    w_mc         = w_mc / (np.linalg.norm(w_mc) + 1e-12)
    ld1_mc_frames = [z @ w_mc for z in X_frames_q]

    plt.figure(figsize=(9, 3))
    for sys in uniq_sys:
        vals = np.concatenate(
            [ld1_mc_frames[i] for i in range(len(names_q)) if names_q[i] == sys]
        )
        plt.hist(vals, bins=60, alpha=0.45, density=True, label=sys)
    plt.xlabel("LD1 (frame-level projection)")
    plt.ylabel("density")
    plt.title("Multi-class LDA LD1: frame-level overlap across family\n"
              f"Family: {', '.join(q_family)}")
    plt.legend(fontsize=8)
    save_current_figure("step4_multiclass_lda_frames_family.png")

if summary_rows:
    out = RESULTS / "step4_pairwise_lda_frame_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out, index=False)
    print("Saved:", out)

print("Script 4 completed.")
