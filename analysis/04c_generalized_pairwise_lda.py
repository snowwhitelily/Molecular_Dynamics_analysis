# %%
# ROS1 Analysis Pipeline
# Script 4C: Generalized Pairwise LDA

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="Reference mutant label, e.g. WT")
    p.add_argument(
        "--targets",
        required=True,
        help="Comma-separated target mutant labels, e.g. Q2022P,Q2022P_S1986F,Q2022P_S1986Y",
    )
    p.add_argument(
        "--scores-prefix",
        default="actout_ctlfit",
        help="Prefix for PCA score files, e.g. actout_ctlfit or dist",
    )
    p.add_argument(
        "--k-use",
        type=int,
        default=6,
        help="Number of leading PCs/features to use from trajectory means",
    )
    p.add_argument(
        "--ridge",
        type=float,
        default=1e-6,
        help="Small ridge regularization for pooled covariance inverse",
    )
    return p.parse_args()

def mutant_replica_from_F(F):
    rows = []
    for i, path in enumerate(F):
        p = Path(path)
        rows.append({
            "traj_index": i,
            "mutant": p.parent.name,
            "replica": p.name,
            "path": str(p),
        })
    return pd.DataFrame(rows)

def lda_direction_2class(X0, X1, ridge=1e-6):
    m0 = X0.mean(axis=0)
    m1 = X1.mean(axis=0)

    X0c = X0 - m0
    X1c = X1 - m1

    S0 = (X0c.T @ X0c) / max(len(X0) - 1, 1)
    S1 = (X1c.T @ X1c) / max(len(X1) - 1, 1)
    Sw = 0.5 * (S0 + S1)

    Sw = Sw + ridge * np.eye(Sw.shape[0])
    w = np.linalg.solve(Sw, (m1 - m0))
    norm = np.linalg.norm(w)
    if norm > 0:
        w = w / norm
    return w, m0, m1

def main():
    args = parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    ref = args.ref.strip()

    fig_dir = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "step4c_generalized_pairwise_lda"
    fig_dir.mkdir(parents=True, exist_ok=True)

    scores = np.load(RESULTS / f"{args.scores_prefix}_scores.npy")
    x_owner = np.load(RESULTS / f"{args.scores_prefix}_x_owner.npy")
    F = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()

    meta_df = mutant_replica_from_F(F)

    n_pc = scores.shape[1]
    k_use = min(args.k_use, n_pc)
    print("Using scores prefix:", args.scores_prefix)
    print("Using k_use:", k_use)
    print("Reference group:", ref)
    print("Targets:", targets)

    # trajectory mean feature vectors
    traj_mean_rows = []
    for ti in range(len(F)):
        Xi = scores[x_owner == ti, :k_use]
        if Xi.shape[0] == 0:
            continue
        row = {"traj_index": ti}
        for j, v in enumerate(Xi.mean(axis=0), start=1):
            row[f"feat{j}"] = float(v)
        traj_mean_rows.append(row)

    traj_mean_df = pd.DataFrame(traj_mean_rows).merge(meta_df, on="traj_index", how="left")

    weights_rows = []
    frame_rows = []

    for target in targets:
        df_pair = traj_mean_df[traj_mean_df["mutant"].isin([ref, target])].copy()

        if ref not in set(df_pair["mutant"]) or target not in set(df_pair["mutant"]):
            print(f"Skipping {ref} vs {target}: missing one group in data.")
            continue

        feat_cols = [c for c in df_pair.columns if c.startswith("feat")]
        Xref = df_pair.loc[df_pair["mutant"] == ref, feat_cols].to_numpy(dtype=float)
        Xtar = df_pair.loc[df_pair["mutant"] == target, feat_cols].to_numpy(dtype=float)

        if len(Xref) < 1 or len(Xtar) < 1:
            print(f"Skipping {ref} vs {target}: insufficient trajectories.")
            continue

        w, m0, m1 = lda_direction_2class(Xref, Xtar, ridge=args.ridge)

        # orient LD1 so target is positive on average
        if (m1 @ w) < (m0 @ w):
            w = -w

        # frame-level projections
        pair_traj_idx = set(df_pair["traj_index"].tolist())
        keep = np.isin(x_owner, list(pair_traj_idx))
        scores_pair = scores[keep, :k_use]
        owners_pair = x_owner[keep]

        ld1 = scores_pair @ w

        frame_df = pd.DataFrame({
            "traj_index": owners_pair,
            "LD1": ld1,
        }).merge(meta_df, on="traj_index", how="left")

        frame_df["comparison"] = f"{ref}_vs_{target}"
        frame_rows.append(frame_df)

        # save histogram
        plt.figure(figsize=(8, 4))
        for g in [ref, target]:
            vals = frame_df.loc[frame_df["mutant"] == g, "LD1"].to_numpy()
            if len(vals) == 0:
                continue
            plt.hist(vals, bins=80, alpha=0.45, density=True, label=g)
        plt.xlabel("LD1")
        plt.ylabel("Density")
        plt.title(f"{ref} vs {target} (scores prefix: {args.scores_prefix})")
        plt.legend()
        out_png = fig_dir / f"{ref}_vs_{target}_{args.scores_prefix}_ld1_hist.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved figure:", out_png)

        # save weights
        row = {
            "comparison": f"{ref}_vs_{target}",
            "ref": ref,
            "target": target,
            "scores_prefix": args.scores_prefix,
            "k_use": k_use,
        }
        for j, val in enumerate(w, start=1):
            row[f"w_feat{j}"] = float(val)
        weights_rows.append(row)

    if weights_rows:
        weights_df = pd.DataFrame(weights_rows)
        weights_out = RESULTS / f"step4c_pairwise_lda_{ref}_weights.csv"
        weights_df.to_csv(weights_out, index=False)
        print("Saved weights:", weights_out)

    if frame_rows:
        frame_summary_df = pd.concat(frame_rows, ignore_index=True)
        frame_out = RESULTS / f"step4c_pairwise_lda_{ref}_frame_summary.csv"
        frame_summary_df.to_csv(frame_out, index=False)
        print("Saved frame summary:", frame_out)

    print("Step 4C done.")

if __name__ == "__main__":
    main()
