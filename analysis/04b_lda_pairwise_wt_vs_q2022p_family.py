# ROS1 Analysis Pipeline
# Script 04b: Pairwise LDA — WT vs Q2022P Family
#
# Performs pairwise LDA between a reference system (default: WT) and each
# member of the Q2022P compound mutation family. For each comparison, generates:
#   - Scatter plot of LD1 trajectory means (one point per replica, with jitter)
#   - Frame-level LD1 histogram with mean lines
#
# Input: actout_ctlfit PCA scores (configurable via --scores-prefix)
# Output: figures and weight/summary CSVs to results/ROS1/

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.ros1_analysis_helpers import summarize_pair, lda_2class_direction

PALETTE = {
    "WT"            : "#1565C0",
    "Q2022P"        : "#E63946",
    "Q2022P_S1986F" : "#F4511E",
    "Q2022P_S1986Y" : "#F9A825",
    "S1986F"        : "#2E7D32",
    "S1986Y"        : "#66BB6A",
}
DEFAULT_C = "#78909C"

def get_colour(name):
    return PALETTE.get(name, DEFAULT_C)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ref",           default="WT")
    p.add_argument("--family",        default="Q2022P,Q2022P_S1986F,Q2022P_S1986Y")
    p.add_argument("--k-use",         type=int, default=6)
    p.add_argument("--scores-prefix", default="actout_ctlfit")
    return p.parse_args()


args     = parse_args()
wt_name  = args.ref.strip()
q_family = [x.strip() for x in args.family.split(",") if x.strip()]

BASE    = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"

family_tag = f"{wt_name}_vs_" + "_".join(q_family)
FIG_DIR    = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "step4b_lda_ref_vs_family" / family_tag
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Running Script 04b: {wt_name} vs Q2022P family pairwise LDA")
print(f"Reference     : {wt_name}")
print(f"Family        : {q_family}")
print(f"k_use         : {args.k_use}")
print(f"Scores prefix : {args.scores_prefix}")
print(f"Figure output : {FIG_DIR}")


def save_fig(filename):
    out = FIG_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


# ============================================================
# Load PCA scores
# ============================================================
F_all       = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
scores_path = RESULTS / f"{args.scores_prefix}_scores.npy"
owner_path  = RESULTS / f"{args.scores_prefix}_x_owner.npy"

if not scores_path.exists():
    raise FileNotFoundError(f"Scores not found: {scores_path}")

scores_flat = np.load(scores_path)
x_owner     = np.load(owner_path)

# ============================================================
# Apply F1994L exclusion consistent with PCA scripts.
# x_owner uses post-exclusion indices (0-113).
# ============================================================
EXCLUDE_MUTANTS = {"F1994L"}
keep         = [i for i, f in enumerate(F_all) if Path(f).parent.name not in EXCLUDE_MUTANTS]
F_excl       = [F_all[i] for i in keep]
traj_mut_all = np.array([Path(f).parent.name for f in F_excl], dtype=object)

ntraj   = len(F_excl)
PS_list = [np.asarray(scores_flat[x_owner == i]) for i in range(ntraj)]

nPC   = max(z.shape[1] for z in PS_list if z.shape[0] > 0)
k_use = min(args.k_use, nPC)

print(f"ntraj={ntraj} | k_use={k_use}")
print(f"Empty trajectories: {sum(1 for z in PS_list if z.shape[0] == 0)}")

# One row per replica (trajectory mean)
valid_idx = [i for i in range(ntraj) if PS_list[i].shape[0] > 0]
X_traj    = np.vstack([PS_list[i][:, :k_use].mean(axis=0) for i in valid_idx])
traj_mut  = traj_mut_all[valid_idx]

print(f"Valid trajectories: {len(valid_idx)} | X_traj shape: {X_traj.shape}")

# ============================================================
# Subset to reference + family
# ============================================================
keep_names = [wt_name] + q_family
mask_keep  = np.isin(traj_mut, keep_names)
X_keep     = X_traj[mask_keep]
names_keep = traj_mut[mask_keep]

present = set(np.unique(names_keep).tolist())
print(f"\nSystems present: {sorted(present)}")

if wt_name not in present:
    raise RuntimeError(f"Reference '{wt_name}' not found. Available: {sorted(present)}")

missing = [q for q in q_family if q not in present]
if missing:
    print(f"WARNING — family members not found, skipping: {missing}")

pairs = [(wt_name, q) for q in q_family if q in present]

# Frame arrays for reference + family
valid_keep_idx    = [i for i in valid_idx if traj_mut_all[i] in keep_names]
X_frames_keep     = [PS_list[i][:, :k_use] for i in valid_keep_idx]
names_frames_keep = traj_mut_all[valid_keep_idx]

# ============================================================
# Pairwise LDA: reference vs each family member
# ============================================================
pairwise_results = {}
weights_rows     = []
summary_rows     = []

for A, B in pairs:
    pair_mask = np.isin(names_keep, [A, B])
    Xpair     = X_keep[pair_mask]
    npair     = names_keep[pair_mask]
    y01       = (npair == B).astype(int)

    nA = int((y01 == 0).sum())
    nB = int((y01 == 1).sum())
    print(f"\n[{A} vs {B}] n_{A}={nA}, n_{B}={nB} replicas")

    if nA < 2 or nB < 2:
        print(f"  Skipping — <2 replicas per group")
        continue

    w        = lda_2class_direction(Xpair, y01)
    ld1_traj = Xpair @ w

    pairwise_results[(A, B)] = {"w": w, "ld1_traj": ld1_traj, "y01": y01, "names": npair}

    plt.figure(figsize=(7, 3))
    for label, val in [(A, 0), (B, 1)]:
        idx   = y01 == val
        y_jit = np.random.normal(0, 0.02, size=idx.sum())
        plt.scatter(ld1_traj[idx], y_jit, s=100, alpha=0.85,
                    color=get_colour(label), edgecolors="white", linewidths=0.5,
                    label=f"{label} (n={idx.sum()} replicas)")
    plt.xlabel("LD1 (trajectory means)")
    plt.yticks([])
    plt.title(f"LDA separation: {A} vs {B}\n(one point per replica)")
    plt.legend(frameon=False, fontsize=8)
    save_fig(f"step4b_pairwise_lda_means_{A}_vs_{B}.png")

    for j, val in enumerate(w[:k_use], start=1):
        weights_rows.append({"pair": f"{A}_vs_{B}", "PC": f"PC{j:02d}", "LD1_weight": float(val)})

# ============================================================
# Frame-level LD1 histograms for each pair
# ============================================================
for (A, B), R in pairwise_results.items():
    w = np.asarray(R["w"], float)

    pair_frame_idx = [i for i in range(len(names_frames_keep))
                      if names_frames_keep[i] in (A, B)]
    Xp             = [X_frames_keep[i] for i in pair_frame_idx]
    npair          = names_frames_keep[pair_frame_idx]
    ld1_frames     = [z @ w for z in Xp]

    vals_A = np.concatenate([ld1_frames[i] for i in range(len(npair)) if npair[i] == A])
    vals_B = np.concatenate([ld1_frames[i] for i in range(len(npair)) if npair[i] == B])

    if not np.isfinite(vals_A).any() or not np.isfinite(vals_B).any():
        print(f"  Skipping frame histogram {A} vs {B}: NaN values")
        continue

    plt.figure(figsize=(8, 3))
    plt.hist(vals_A, bins=60, alpha=0.55, density=True, label=A, color=get_colour(A))
    plt.hist(vals_B, bins=60, alpha=0.55, density=True, label=B, color=get_colour(B))
    plt.axvline(vals_A.mean(), color=get_colour(A), lw=2, ls="--", alpha=0.9, label=f"{A} mean")
    plt.axvline(vals_B.mean(), color=get_colour(B), lw=2, ls="--", alpha=0.9, label=f"{B} mean")
    plt.xlabel("LD1 (frame-level projection)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.title(f"Frame-level overlap: {A} vs {B}", fontsize=12, fontweight="bold")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.legend(frameon=False, fontsize=9)
    save_fig(f"step4b_pairwise_lda_frames_{A}_vs_{B}.png")

    stats     = summarize_pair(A, B, ld1_frames, npair)
    stats_row = {"A": A, "B": B}
    stats_row.update(stats)
    summary_rows.append(stats_row)

# ============================================================
# Save tables
# ============================================================
ref_tag = wt_name.lower()

if weights_rows:
    out = RESULTS / f"step4b_pairwise_lda_{ref_tag}_vs_family_weights.csv"
    pd.DataFrame(weights_rows).to_csv(out, index=False)
    print("\nSaved weights:", out)

if summary_rows:
    out = RESULTS / f"step4b_pairwise_lda_{ref_tag}_vs_family_frame_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out, index=False)
    print("Saved frame summary:", out)

print("\nScript 04b completed.")