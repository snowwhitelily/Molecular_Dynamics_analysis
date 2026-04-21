# %%
# ROS1 Analysis Pipeline
# Script 4: LDA and Final Figures

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

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "step4_lda_final"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Running Script 4")
print("Figure output folder:", FIG_DIR)

def save_current_figure(filename: str):
    out = FIG_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure:", out)

# ============================================================
# PART 1: LDA on trajectory means (exact notebook logic, adapted)
# using ACT-OUT CTL-fit PCA source if available
# ============================================================
F = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
traj_mut = np.array([Path(p).parent.name for p in F], dtype=object)

if (RESULTS / "actout_ctlfit_scores.npy").exists() and (RESULTS / "actout_ctlfit_x_owner.npy").exists():
    scores_flat = np.load(RESULTS / "actout_ctlfit_scores.npy")
    x_owner = np.load(RESULTS / "actout_ctlfit_x_owner.npy")
    ps_name = "actout_ctlfit_scores"
elif (RESULTS / "dist_scores.npy").exists() and (RESULTS / "dist_x_owner.npy").exists():
    scores_flat = np.load(RESULTS / "dist_scores.npy")
    x_owner = np.load(RESULTS / "dist_x_owner.npy")
    ps_name = "dist_scores"
else:
    raise FileNotFoundError("Need actout_ctlfit_scores/x_owner or dist_scores/x_owner in results/ROS1.")

PS_list = [np.asarray(scores_flat[x_owner == i]) for i in range(len(F))]
ntraj = len(PS_list)
nPC = min(z.shape[1] for z in PS_list if z.shape[1] >= 1)
k_use = min(6, nPC)

X_traj = np.vstack([z[:, :k_use].mean(axis=0, keepdims=True) for z in PS_list])

print(f"[OK] Using {ps_name} as PCA source, ntraj={ntraj}, k_use={k_use}")

q_family = ["Q2022P", "Q2022P_S1986F", "Q2022P_S1986Y"]
mask_q = np.isin(traj_mut, q_family)

Xq = X_traj[mask_q]
namesq = traj_mut[mask_q]
uniq_sys = np.unique(namesq)

print("\n[Q2022P LDA] systems in subset:", uniq_sys.tolist())
print("[Q2022P LDA] counts:", {u: int((namesq == u).sum()) for u in uniq_sys})

if uniq_sys.size < 2:
    raise RuntimeError("Q2022P subset has <2 systems in the data.")

label_map = {m: i for i, m in enumerate(uniq_sys)}
yq = np.array([label_map[m] for m in namesq], dtype=int)

W, evals = lda_multiclass(Xq, yq, eps=1e-8)
ndim = min(len(uniq_sys) - 1, k_use)
Z = Xq @ W[:, :ndim]

plt.figure(figsize=(7, 3))
for m in uniq_sys:
    c = label_map[m]
    zz = Z[yq == c, 0]
    plt.scatter(zz, np.zeros_like(zz), s=80, alpha=0.85, label=f"{m} (n={zz.size})")
plt.xlabel("LD1")
plt.yticks([])
plt.title("Multi-class LDA on Q2022P family (per-trajectory means)")
plt.legend(fontsize=8, frameon=False)
save_current_figure("step4_multiclass_lda_q_family.png")

weights_df = pd.DataFrame({
    "PC": [f"PC{i+1:02d}" for i in range(k_use)],
    "LD1_weight": W[:k_use, 0],
})
weights_df.to_csv(RESULTS / "step4_multiclass_lda_weights.csv", index=False)

pairs = [
    ("Q2022P", "Q2022P_S1986F"),
    ("Q2022P", "Q2022P_S1986Y"),
    ("Q2022P_S1986F", "Q2022P_S1986Y"),
]

pairwise_results = {}

for A, B in pairs:
    pair_mask = np.isin(namesq, [A, B])
    Xpair = Xq[pair_mask]
    npair = namesq[pair_mask]
    y01 = (npair == B).astype(int)

    if (y01 == 0).sum() < 2 or (y01 == 1).sum() < 2:
        print(f"Skipping {A} vs {B} (<2 samples/class)")
        continue

    w = lda_2class_direction(Xpair, y01)
    ld1_traj = Xpair @ w

    pairwise_results[(A, B)] = {"w": w, "ld1_traj": ld1_traj, "y01": y01, "names": npair}

    plt.figure(figsize=(7, 3))
    plt.scatter(ld1_traj[y01 == 0], np.zeros(np.sum(y01 == 0)), s=80, alpha=0.85, label=A)
    plt.scatter(ld1_traj[y01 == 1], np.zeros(np.sum(y01 == 1)), s=80, alpha=0.85, label=B)
    plt.xlabel("LD1")
    plt.yticks([])
    plt.title(f"Pairwise LDA on trajectory means: {A} vs {B}")
    plt.legend(frameon=False, fontsize=8)
    save_current_figure(f"step4_pairwise_lda_means_{A}_vs_{B}.png")

# ============================================================
# PART 2: Project all frames onto learned LD1 (exact notebook logic)
# ============================================================
X_frames_q = [PS_list[i][:, :k_use] for i in range(len(F)) if mask_q[i]]
names_q = traj_mut[mask_q]
uniq_sys = np.unique(names_q)

summary_rows = []

for (A, B), R in pairwise_results.items():
    w = np.asarray(R["w"], float)
    if w.shape[0] != k_use:
        continue

    pair_mask = np.isin(names_q, [A, B])
    Xp = [X_frames_q[i] for i in range(len(X_frames_q)) if pair_mask[i]]
    npair = names_q[pair_mask]

    ld1_frames = [z @ w for z in Xp]

    plt.figure(figsize=(8, 3))
    plt.hist(np.concatenate([ld1_frames[i] for i in range(len(npair)) if npair[i] == A]),
             bins=60, alpha=0.55, label=A, density=True)
    plt.hist(np.concatenate([ld1_frames[i] for i in range(len(npair)) if npair[i] == B]),
             bins=60, alpha=0.55, label=B, density=True)
    plt.xlabel("LD1 (frame-level projection)")
    plt.ylabel("density")
    plt.title(f"Frame-level overlap: {A} vs {B}")
    plt.legend()
    save_current_figure(f"step4_pairwise_lda_frames_{A}_vs_{B}.png")

    stats = summarize_pair(A, B, ld1_frames, npair)
    stats_row = {"A": A, "B": B}
    stats_row.update(stats)
    summary_rows.append(stats_row)

if "W" in globals() or True:
    W_use = np.asarray(W, float)
    if W_use.shape[0] >= k_use:
        w_mc = W_use[:k_use, 0]
        w_mc = w_mc / (np.linalg.norm(w_mc) + 1e-12)

        ld1_mc_frames = [z @ w_mc for z in X_frames_q]

        plt.figure(figsize=(9, 3))
        for sys in uniq_sys:
            vals = np.concatenate([ld1_mc_frames[i] for i in range(len(names_q)) if names_q[i] == sys])
            plt.hist(vals, bins=60, alpha=0.45, density=True, label=sys)
        plt.xlabel("LD1 (frame-level projection)")
        plt.ylabel("density")
        plt.title("Multi-class LDA LD1: frame-level overlap across Q2022P family")
        plt.legend(fontsize=8)
        save_current_figure("step4_multiclass_lda_frames_q_family.png")

if summary_rows:
    pd.DataFrame(summary_rows).to_csv(RESULTS / "step4_pairwise_lda_frame_summary.csv", index=False)

print("Script 4 completed.")
