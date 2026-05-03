# %%
# ROS1 Analysis Pipeline
# Script 4C: Generalised Pairwise LDA
#
# NOTEBOOK-MATCHED VERSION (May 2026):
#   - Any reference vs any list of targets
#   - Trajectory means: one point per replica with y-axis jitter
#   - Frame-level histograms: ALL frames from all replicas
#   - F1994L handled by skipping empty PS_list entries
#   - NaN guard before every histogram
#
# USAGE:
#   python 04c_generalized_pairwise_lda.py \
#       --ref WT \
#       --targets Q2022P,Q2022P_S1986F,Q2022P_S1986Y

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE    = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--targets", required=True,
                   help="Comma-separated target mutant labels")
    p.add_argument("--scores-prefix", default="actout_ctlfit")
    p.add_argument("--k-use", type=int, default=6)
    p.add_argument("--ridge", type=float, default=1e-6)
    return p.parse_args()


def lda_direction_2class(X0, X1, ridge=1e-6):
    m0  = X0.mean(axis=0)
    m1  = X1.mean(axis=0)
    X0c = X0 - m0
    X1c = X1 - m1
    S0  = (X0c.T @ X0c) / max(len(X0) - 1, 1)
    S1  = (X1c.T @ X1c) / max(len(X1) - 1, 1)
    Sw  = 0.5 * (S0 + S1) + ridge * np.eye(S0.shape[0])
    w   = np.linalg.solve(Sw, (m1 - m0))
    norm = np.linalg.norm(w)
    if norm > 0:
        w = w / norm
    return w


def main():
    args    = parse_args()
    ref     = args.ref.strip()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    fig_dir = (BASE / "figures" / "ROS1" / "ros1_prepared_final"
               / "step4c_generalized_pairwise_lda")
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Running Script 4C — Generalised Pairwise LDA")
    print(f"Reference : {ref}")
    print(f"Targets   : {targets}")
    print(f"Prefix    : {args.scores_prefix}")

    # ── Load scores ──────────────────────────────────────────────
    F_all = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()

    scores_path = RESULTS / f"{args.scores_prefix}_scores.npy"
    owner_path  = RESULTS / f"{args.scores_prefix}_x_owner.npy"

    if not scores_path.exists():
        raise FileNotFoundError(f"Scores not found: {scores_path}")

    scores_flat = np.load(scores_path)
    x_owner     = np.load(owner_path)

    # ── Apply same F1994L exclusion as PCA scripts ───────────────
    # x_owner was saved using post-exclusion indices (0-113).
    # Must apply same exclusion to get correct mutant name mapping.
    EXCLUDE_MUTANTS = {"F1994L"}
    keep         = [i for i, f in enumerate(F_all)
                    if Path(f).parent.name not in EXCLUDE_MUTANTS]
    F_excl       = [F_all[i] for i in keep]       # 114 entries
    traj_mut_all = np.array([Path(f).parent.name for f in F_excl], dtype=object)

    # ── Build per-trajectory frame arrays ────────────────────────
    ntraj   = len(F_excl)
    PS_list = [np.asarray(scores_flat[x_owner == i]) for i in range(ntraj)]

    nPC   = max(z.shape[1] for z in PS_list if z.shape[0] > 0)
    k_use = min(args.k_use, nPC)

    print(f"\nntraj={ntraj}, k_use={k_use}")
    print(f"Empty trajectories: {sum(1 for z in PS_list if z.shape[0] == 0)}")

    # Trajectory means — one per replica, skip empty
    valid_idx = [i for i in range(ntraj) if PS_list[i].shape[0] > 0]
    X_traj    = np.vstack([PS_list[i][:, :k_use].mean(axis=0) for i in valid_idx])
    traj_mut  = traj_mut_all[valid_idx]

    print(f"Valid trajectories: {len(valid_idx)}, X_traj shape: {X_traj.shape}")

    # ── Check which systems are present ──────────────────────────
    present = set(np.unique(traj_mut).tolist())
    print(f"Mutants present: {sorted(present)}")

    # ── Pairwise LDA: ref vs each target ─────────────────────────
    weights_rows = []
    frame_rows   = []

    for target in targets:
        if ref not in present:
            print(f"Skipping {ref} vs {target}: {ref} not found in data")
            continue
        if target not in present:
            print(f"Skipping {ref} vs {target}: {target} not found in data")
            continue

        pair_mask = np.isin(traj_mut, [ref, target])
        Xpair     = X_traj[pair_mask]
        npair     = traj_mut[pair_mask]
        y01       = (npair == target).astype(int)

        nA = int((y01 == 0).sum())
        nB = int((y01 == 1).sum())
        print(f"\n[{ref} vs {target}] n_{ref}={nA}, n_{target}={nB} replicas")

        if nA < 2 or nB < 2:
            print(f"  Skipping — <2 replicas per group")
            continue

        w        = lda_direction_2class(Xpair[y01==0], Xpair[y01==1], args.ridge)
        ld1_traj = Xpair @ w

        # Orient: target > ref on LD1
        if ld1_traj[y01==1].mean() < ld1_traj[y01==0].mean():
            w        = -w
            ld1_traj = -ld1_traj

        # Scatter with y-axis jitter — one point per replica
        plt.figure(figsize=(7, 3))
        for label, val in [(ref, 0), (target, 1)]:
            idx   = y01 == val
            y_jit = np.random.normal(0, 0.02, size=idx.sum())
            plt.scatter(ld1_traj[idx], y_jit, s=100, alpha=0.85,
                        label=f"{label} (n={idx.sum()} replicas)")
        plt.xlabel("LD1 (trajectory means)")
        plt.yticks([])
        plt.title(f"LDA separation: {ref} vs {target}\n(one point per replica)")
        plt.legend(frameon=False, fontsize=8)
        out_png = fig_dir / f"{ref}_vs_{target}_{args.scores_prefix}_ld1_traj.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved:", out_png)

        # Frame-level arrays for this pair
        pair_frame_idx = [i for i in valid_idx
                          if traj_mut_all[i] in (ref, target)]
        Xp             = [PS_list[i][:, :k_use] for i in pair_frame_idx]
        npair_frame    = traj_mut_all[pair_frame_idx]
        ld1_frames     = [z @ w for z in Xp]

        vals_ref = np.concatenate([ld1_frames[i]
                                   for i in range(len(npair_frame))
                                   if npair_frame[i] == ref])
        vals_tar = np.concatenate([ld1_frames[i]
                                   for i in range(len(npair_frame))
                                   if npair_frame[i] == target])

        if not np.isfinite(vals_ref).any() or not np.isfinite(vals_tar).any():
            print(f"  Skipping frame histogram: NaN values")
            continue

        plt.figure(figsize=(8, 4))
        plt.hist(vals_ref, bins=80, alpha=0.45, density=True, label=ref)
        plt.hist(vals_tar, bins=80, alpha=0.45, density=True, label=target)
        plt.xlabel("LD1 (frame-level projection)")
        plt.ylabel("Density")
        plt.title(f"Frame-level overlap: {ref} vs {target}")
        plt.legend(frameon=False)
        out_png = fig_dir / f"{ref}_vs_{target}_{args.scores_prefix}_ld1_hist.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved:", out_png)

        row = {"comparison": f"{ref}_vs_{target}", "ref": ref,
               "target": target, "scores_prefix": args.scores_prefix,
               "k_use": k_use}
        for j, val in enumerate(w, start=1):
            row[f"w_feat{j}"] = float(val)
        weights_rows.append(row)

        # Collect frame rows
        for i in range(len(npair_frame)):
            frame_rows.append({
                "mutant":     npair_frame[i],
                "LD1":        float(ld1_frames[i].mean()),
                "comparison": f"{ref}_vs_{target}"
            })

    if weights_rows:
        out = RESULTS / f"step4c_pairwise_lda_{ref}_weights.csv"
        pd.DataFrame(weights_rows).to_csv(out, index=False)
        print("\nSaved weights:", out)

    if frame_rows:
        out = RESULTS / f"step4c_pairwise_lda_{ref}_frame_summary.csv"
        pd.DataFrame(frame_rows).to_csv(out, index=False)
        print("Saved frame summary:", out)

    print("\nStep 4C done.")


if __name__ == "__main__":
    main()
