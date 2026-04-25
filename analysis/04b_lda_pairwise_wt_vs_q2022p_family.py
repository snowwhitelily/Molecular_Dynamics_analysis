# %%
# ROS1 Analysis Pipeline
# Script 4B: Pairwise LDA — reference system vs target family
#
# CHANGES FROM ORIGINAL:
#   - Added argument parsing for --ref, --family, --k-use, --scores-prefix
#   - Removed hardcoded wt_name = "WT"
#   - Removed hardcoded q_family = ["Q2022P", "Q2022P_S1986F", "Q2022P_S1986Y"]
#   - Default values match original hardcoded values, so running with no
#     arguments produces identical output to the original script
#
# USAGE (identical to original — no arguments needed for default behaviour):
#   python 04b_lda_pairwise_wt_vs_q2022p_family.py
#
# USAGE (different reference or panel):
#   python 04b_lda_pairwise_wt_vs_q2022p_family.py \
#       --ref WT \
#       --family Q2022P,Q2022P_S1986F,Q2022P_S1986Y,G2032R
#
#   python 04b_lda_pairwise_wt_vs_q2022p_family.py \
#       --ref Q2022P \
#       --family G2032R,D2033N,L2026M

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.ros1_analysis_helpers import summarize_pair, lda_2class_direction

# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Pairwise LDA: one reference system vs each member of a target family."
        )
    )
    p.add_argument(
        "--ref",
        default="WT",
        help=(
            "Name of the reference system (the left-hand side of every pair). "
            "Default: WT"
        ),
    )
    p.add_argument(
        "--family",
        default="Q2022P,Q2022P_S1986F,Q2022P_S1986Y",
        help=(
            "Comma-separated target mutant names (each is compared against --ref). "
            "Default: Q2022P,Q2022P_S1986F,Q2022P_S1986Y"
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


args    = parse_args()
wt_name = args.ref.strip()
q_family = [x.strip() for x in args.family.split(",") if x.strip()]

if len(q_family) < 1:
    raise ValueError("--family must contain at least one mutant name.")

# ============================================================
# PATHS
# ============================================================

BASE    = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"

# figure directory includes ref + family so different runs don't overwrite
family_tag = f"{wt_name}_vs_" + "_".join(q_family)
FIG_DIR = (
    BASE / "figures" / "ROS1" / "ros1_prepared_final"
    / "step4b_lda_ref_vs_family" / family_tag
)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Running Script 4B: {wt_name} vs family pairwise LDA")
print(f"Reference     : {wt_name}")
print(f"Family        : {q_family}")
print(f"k_use (max)   : {args.k_use}")
print(f"Scores prefix : {args.scores_prefix}")
print(f"Figure output : {FIG_DIR}")


def save_current_figure(filename: str):
    out = FIG_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure:", out)


# ============================================================
# PART 1: Load PCA scores
# ============================================================

F        = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
traj_mut = np.array([Path(p).parent.name for p in F], dtype=object)

preferred        = args.scores_prefix
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
# PART 2: Subset to ref + family and run pairwise LDA
# ============================================================

keep_names = [wt_name] + q_family
mask_keep  = np.isin(traj_mut, keep_names)
X_keep     = X_traj[mask_keep]
names_keep = traj_mut[mask_keep]

present = set(np.unique(names_keep).tolist())
print(f"\n[{wt_name} vs family] systems present in data : {sorted(present)}")
print(f"[{wt_name} vs family] counts : "
      f"{ {u: int((names_keep == u).sum()) for u in sorted(present)} }")

if wt_name not in present:
    raise RuntimeError(
        f"Reference '{wt_name}' not found in trajectory data. "
        f"Available mutants: {sorted(present)}"
    )

# one pair per family member that is actually present
pairs = [(wt_name, q) for q in q_family if q in present]
missing = [q for q in q_family if q not in present]
if missing:
    print(f"[WARN] These family members were not found in data and will be "
          f"skipped: {missing}")

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
    print(f"\n[{A} vs {B}] n_{A}={nA}, n_{B}={nB}")

    if nA < 2 or nB < 2:
        print(f"Skipping {A} vs {B} (<2 samples/class)")
        continue

    w        = lda_2class_direction(Xpair, y01)
    w        = np.asarray(w, float)
    ld1_traj = Xpair @ w

    pairwise_results[(A, B)] = {
        "w": w, "ld1_traj": ld1_traj, "y01": y01, "names": npair
    }

    for i, val in enumerate(w[:k_use], start=1):
        weights_rows.append({
            "pair":       f"{A}_vs_{B}",
            "PC":         f"PC{i:02d}",
            "LD1_weight": float(val),
        })

    plt.figure(figsize=(7, 3))
    plt.scatter(ld1_traj[y01 == 0], np.zeros(nA), s=80, alpha=0.85, label=A)
    plt.scatter(ld1_traj[y01 == 1], np.zeros(nB), s=80, alpha=0.85, label=B)
    plt.xlabel("LD1")
    plt.yticks([])
    plt.title(f"Pairwise LDA on trajectory means: {A} vs {B}")
    plt.legend(frameon=False, fontsize=8)
    save_current_figure(f"step4b_pairwise_lda_means_{A}_vs_{B}.png")


# ============================================================
# PART 3: Project all frames onto learned LD1
# ============================================================

X_frames_keep    = [PS_list[i][:, :k_use] for i in range(len(F)) if mask_keep[i]]
names_frames_keep = traj_mut[mask_keep]

for (A, B), R in pairwise_results.items():
    w = np.asarray(R["w"], float)
    if w.shape[0] != k_use:
        print(f"Skipping frame projection for {A} vs {B}: "
              f"w shape mismatch ({w.shape[0]} != {k_use})")
        continue

    pair_mask  = np.isin(names_frames_keep, [A, B])
    Xp         = [X_frames_keep[i]
                  for i in range(len(X_frames_keep)) if pair_mask[i]]
    npair      = names_frames_keep[pair_mask]
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
    save_current_figure(f"step4b_pairwise_lda_frames_{A}_vs_{B}.png")

    stats     = summarize_pair(A, B, ld1_frames, npair)
    stats_row = {"A": A, "B": B}
    stats_row.update(stats)
    summary_rows.append(stats_row)


# ============================================================
# PART 4: Save tables
# ============================================================

ref_tag = wt_name.lower()

if weights_rows:
    out = RESULTS / f"step4b_pairwise_lda_{ref_tag}_vs_family_weights.csv"
    pd.DataFrame(weights_rows).to_csv(out, index=False)
    print("Saved:", out)

if summary_rows:
    out = RESULTS / f"step4b_pairwise_lda_{ref_tag}_vs_family_frame_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out, index=False)
    print("Saved:", out)

print("Script 4B completed.")
