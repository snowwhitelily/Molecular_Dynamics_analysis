# %%
# ROS1 Analysis Pipeline
# Script 4: LDA and Final Figures
#
# NOTEBOOK-MATCHED VERSION (May 2026):
#   Matches the approach in ROS1_pipeline.ipynb cells 27-36:
#   - Multi-class LDA on trajectory means (3 points per mutant = 3 replicas)
#   - Frame-level projection of ALL frames onto learned LD1 axis
#   - Pairwise scatter: one point per replica with y-axis jitter
#   - Pairwise frame histograms: all frames from all replicas per mutant
#
# F1994L EXCLUSION:
#   x_owner is saved in 117-space by PCA scripts (original F index).
#   F1994L entries in PS_list are empty (zero frames) because F1994L
#   was excluded from PCA computation. We filter empty entries before
#   building X_traj to avoid NaN means propagating into LDA.

import argparse
import itertools
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
)

# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="Q2022P,Q2022P_S1986F,Q2022P_S1986Y")
    p.add_argument("--k-use", type=int, default=6)
    p.add_argument("--scores-prefix", default="actout_ctlfit")
    return p.parse_args()

args     = parse_args()
q_family = [x.strip() for x in args.family.split(",") if x.strip()]

if len(q_family) < 2:
    raise ValueError("--family must contain at least 2 mutant names.")

print("Running Script 4 — LDA Final Figures")
print(f"Family        : {q_family}")
print(f"k_use (max)   : {args.k_use}")
print(f"Scores prefix : {args.scores_prefix}")

# ============================================================
# PATHS
# ============================================================

BASE    = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"

family_tag = "_".join(q_family)
FIG_DIR    = (BASE / "figures" / "ROS1" / "ros1_prepared_final"
              / "step4_lda_final" / family_tag)
FIG_DIR.mkdir(parents=True, exist_ok=True)
print("Figure output:", FIG_DIR)


def save_fig(filename):
    out = FIG_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


# ============================================================
# LOAD SCORES
# ============================================================

F_all = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()

scores_path = RESULTS / f"{args.scores_prefix}_scores.npy"
owner_path  = RESULTS / f"{args.scores_prefix}_x_owner.npy"

if not scores_path.exists():
    raise FileNotFoundError(f"Scores not found: {scores_path}")

scores_flat = np.load(scores_path)
x_owner     = np.load(owner_path)

# ============================================================
# APPLY SAME F1994L EXCLUSION AS PCA SCRIPTS
# x_owner was saved using post-exclusion indices (0-113).
# F_all has 117 entries. WT is at indices 114-116 in F_all
# but at indices 111-113 in the post-exclusion list.
# We must apply the same exclusion to get the correct mapping.
# ============================================================

EXCLUDE_MUTANTS = {"F1994L"}
keep         = [i for i, f in enumerate(F_all)
                if Path(f).parent.name not in EXCLUDE_MUTANTS]
F_excl       = [F_all[i] for i in keep]          # 114 entries
traj_mut_all = np.array([Path(f).parent.name for f in F_excl], dtype=object)

ntraj   = len(F_excl)
PS_list = [np.asarray(scores_flat[x_owner == i]) for i in range(ntraj)]

nPC   = max(z.shape[1] for z in PS_list if z.shape[0] > 0)
k_use = min(args.k_use, nPC)

print(f"\nLoaded: ntraj={ntraj}, k_use={k_use}")
print(f"Empty trajectories: {sum(1 for z in PS_list if z.shape[0] == 0)}")

# ============================================================
# BUILD TRAJECTORY-MEAN MATRIX
# Only include trajectories with at least 1 frame (skip F1994L).
# This gives one row per replica — 3 per mutant for 3 replicas.
# ============================================================

valid_idx  = [i for i in range(ntraj) if PS_list[i].shape[0] > 0]
X_traj     = np.vstack([PS_list[i][:, :k_use].mean(axis=0)
                        for i in valid_idx])
traj_mut   = traj_mut_all[valid_idx]
traj_F     = [F_all[i] for i in valid_idx]

print(f"Valid trajectories: {len(valid_idx)} (one row per replica in X_traj)")
print(f"X_traj shape: {X_traj.shape}")

# ============================================================
# SUBSET TO FAMILY
# ============================================================

mask_q   = np.isin(traj_mut, q_family)
Xq       = X_traj[mask_q]
namesq   = traj_mut[mask_q]
uniq_sys = np.unique(namesq)

present = set(uniq_sys.tolist())
print(f"\nFamily systems found: {sorted(present)}")
print(f"Counts per system: { {u: int((namesq==u).sum()) for u in sorted(present)} }")

missing_fam = [m for m in q_family if m not in present]
if missing_fam:
    print(f"WARNING — not found in data: {missing_fam}")

if len(present) < 2:
    raise RuntimeError(f"Need ≥2 family systems in data. Found: {sorted(present)}")

# ============================================================
# MULTI-CLASS LDA (trajectory means — 3 points per mutant)
# Matches notebook cell 27
# ============================================================

label_map = {m: i for i, m in enumerate(uniq_sys)}
yq        = np.array([label_map[m] for m in namesq], dtype=int)

W, evals = lda_multiclass(Xq, yq, eps=1e-8)
ndim     = min(len(uniq_sys) - 1, k_use)
Z        = Xq @ W[:, :ndim]

plt.figure(figsize=(7, 4))
for m in uniq_sys:
    c  = label_map[m]
    zz = Z[yq == c, 0]
    # y-axis jitter to separate overlapping replica points (matches notebook cell 32)
    y_jit = np.random.normal(0, 0.02, size=len(zz))
    plt.scatter(zz, y_jit, s=80, alpha=0.85, label=f"{m} (n={len(zz)} replicas)")

plt.xlabel("LD1")
plt.ylabel("")
plt.yticks([])
plt.title("Multi-class LDA on Q2022P family\n(trajectory means — one point per replica)")
plt.legend(fontsize=8, frameon=False)
save_fig("step4_multiclass_lda_family.png")

# ============================================================
# MULTI-CLASS FRAME-LEVEL PROJECTION
# Matches notebook cell 29-30
# ============================================================

# Get valid frame arrays for family members only
valid_family_idx   = [i for i in valid_idx if traj_mut_all[i] in q_family]
X_frames_family    = [PS_list[i][:, :k_use] for i in valid_family_idx]
names_family_frame = traj_mut_all[valid_family_idx]

w_mc      = W[:k_use, 0]
w_mc      = w_mc / (np.linalg.norm(w_mc) + 1e-12)

ld1_mc_frames = [z @ w_mc for z in X_frames_family]

plt.figure(figsize=(9, 4))
for sys_name in uniq_sys:
    vals = np.concatenate([ld1_mc_frames[i]
                           for i in range(len(names_family_frame))
                           if names_family_frame[i] == sys_name])
    if np.isfinite(vals).any():
        plt.hist(vals, bins=60, alpha=0.45, density=True, label=sys_name)

plt.xlabel("LD1 (frame-level projection)")
plt.ylabel("Density")
plt.title("Multi-class LDA LD1: frame-level overlap across Q2022P family")
plt.legend(fontsize=8, frameon=False)
save_fig("step4_multiclass_lda_frames_family.png")

# ============================================================
# ALL PAIRWISE LDA (trajectory means — 3 points per mutant)
# Matches notebook cells 31-32
# ============================================================

pairs = [(A, B)
         for A, B in itertools.combinations(q_family, 2)
         if A in present and B in present]

print(f"\nPairwise LDA: {len(pairs)} pairs")

pairwise_results = {}
weights_rows     = []
summary_rows     = []

for A, B in pairs:
    pair_mask = np.isin(namesq, [A, B])
    Xpair     = Xq[pair_mask]
    npair     = namesq[pair_mask]
    y01       = (npair == B).astype(int)

    nA = int((y01 == 0).sum())
    nB = int((y01 == 1).sum())
    print(f"\n  {A} vs {B}: n_{A}={nA} replicas, n_{B}={nB} replicas")

    if nA < 2 or nB < 2:
        print(f"  Skipping — <2 replicas per group")
        continue

    w        = lda_2class_direction(Xpair, y01)
    ld1_traj = Xpair @ w

    pairwise_results[(A, B)] = {
        "w": w, "ld1_traj": ld1_traj, "y01": y01, "names": npair
    }

    # Scatter — one point per replica with y-axis jitter (matches notebook cell 32)
    plt.figure(figsize=(7, 3))
    for label, val in [(A, 0), (B, 1)]:
        idx = y01 == val
        y_jit = np.random.normal(0, 0.02, size=idx.sum())
        plt.scatter(ld1_traj[idx], y_jit, s=100, alpha=0.85,
                    label=f"{label} (n={idx.sum()} replicas)")
    plt.xlabel("LD1 (trajectory means)")
    plt.yticks([])
    plt.title(f"LDA separation: {A} vs {B}\n(one point per replica)")
    plt.legend(frameon=False, fontsize=8)
    save_fig(f"step4_pairwise_lda_means_{A}_vs_{B}.png")

    for j, val in enumerate(w[:k_use], start=1):
        weights_rows.append({"pair": f"{A}_vs_{B}",
                              "PC": f"PC{j:02d}",
                              "LD1_weight": float(val)})

# ============================================================
# PAIRWISE FRAME-LEVEL PROJECTION
# Matches notebook cells 34-36
# ============================================================

for (A, B), R in pairwise_results.items():
    w = np.asarray(R["w"], float)

    pair_frame_idx   = [i for i in range(len(names_family_frame))
                        if names_family_frame[i] in (A, B)]
    Xp               = [X_frames_family[i] for i in pair_frame_idx]
    npair            = names_family_frame[pair_frame_idx]
    ld1_frames       = [z @ w for z in Xp]

    vals_A = np.concatenate([ld1_frames[i] for i in range(len(npair))
                             if npair[i] == A])
    vals_B = np.concatenate([ld1_frames[i] for i in range(len(npair))
                             if npair[i] == B])

    if not np.isfinite(vals_A).any() or not np.isfinite(vals_B).any():
        print(f"  Skipping frame histogram for {A} vs {B}: NaN values")
        continue

    plt.figure(figsize=(8, 3))
    plt.hist(vals_A, bins=60, alpha=0.55, density=True, label=A)
    plt.hist(vals_B, bins=60, alpha=0.55, density=True, label=B)
    plt.xlabel("LD1 (frame-level projection)")
    plt.ylabel("Density")
    plt.title(f"Frame-level overlap: {A} vs {B}")
    plt.legend(frameon=False)
    save_fig(f"step4_pairwise_lda_frames_{A}_vs_{B}.png")

    stats     = summarize_pair(A, B, ld1_frames, npair)
    stats_row = {"A": A, "B": B}
    stats_row.update(stats)
    summary_rows.append(stats_row)

# ============================================================
# SAVE TABLES
# ============================================================

if weights_rows:
    out = RESULTS / "step4_pairwise_lda_weights.csv"
    pd.DataFrame(weights_rows).to_csv(out, index=False)
    print("\nSaved weights:", out)

if summary_rows:
    out = RESULTS / "step4_pairwise_lda_frame_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out, index=False)
    print("Saved frame summary:", out)

print("\nScript 4 completed.")
