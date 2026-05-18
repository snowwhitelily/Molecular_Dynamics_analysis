# ROS1 Analysis Pipeline
# Script 02f: Cluster Transition Analysis
#
# Computes per-trajectory transition statistics from cluster label sequences:
# switch counts, mean dwell times, dwell distributions, and transition
# count matrices. Summaries are produced at trajectory, replica, and
# mutant level and saved to results/ROS1/.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE    = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prefixes", default="actout_ctlfit,aloop",
                   help="Comma-separated clustered PCA prefixes.")
    p.add_argument("--cluster-col", default="cluster_hdb",
                   choices=["cluster_hdb", "cluster_db"],
                   help="Cluster label column to use.")
    p.add_argument("--drop-noise", action="store_true",
                   help="Drop noise frames (label -1) before transition analysis.")
    return p.parse_args()


def switch_count(labels):
    """Count the number of state transitions in a label sequence."""
    x = np.asarray(labels, dtype=int)
    if x.size <= 1:
        return 0
    return int(np.sum(x[1:] != x[:-1]))


def mean_dwell_frames(labels):
    """Mean number of consecutive frames spent in a single state."""
    x = np.asarray(labels, dtype=int)
    if x.size == 0:
        return np.nan
    edges = np.where(np.diff(x) != 0)[0] + 1
    runs  = np.diff(np.r_[0, edges, len(x)])
    return float(np.mean(runs)) if len(runs) else float(len(x))


def dwell_rows(labels):
    """
    Return a list of dicts describing each contiguous dwell period.
    Each dict has keys: state, dwell_frames.
    """
    x = np.asarray(labels, dtype=int)
    if x.size == 0:
        return []
    edges  = np.where(np.diff(x) != 0)[0] + 1
    starts = np.r_[0, edges]
    ends   = np.r_[edges, len(x)]
    return [{"state": int(x[s]), "dwell_frames": int(e - s)} for s, e in zip(starts, ends)]


def transition_count_matrix(labels):
    """
    Count all pairwise state transitions (i -> j where i != j).
    Returns a DataFrame with columns: from_state, to_state, count.
    """
    x = np.asarray(labels, dtype=int)
    if x.size <= 1:
        return pd.DataFrame(columns=["from_state", "to_state", "count"])

    pairs = [(int(a), int(b)) for a, b in zip(x[:-1], x[1:]) if a != b]
    if not pairs:
        return pd.DataFrame(columns=["from_state", "to_state", "count"])

    df = pd.DataFrame(pairs, columns=["from_state", "to_state"])
    return df.groupby(["from_state", "to_state"]).size().reset_index(name="count")


def main():
    args     = parse_args()
    prefixes = [x.strip() for x in args.prefixes.split(",") if x.strip()]

    for prefix in prefixes:
        print(f"\n=== Transition analysis: {prefix} ===")

        cluster_f = RESULTS / f"{prefix}_cluster_labels.csv"
        if not cluster_f.exists():
            raise FileNotFoundError(f"Missing cluster label file for '{prefix}'. Run 02d first.")

        df = pd.read_csv(cluster_f)
        df = df[["traj_index", "mutant", "replica", "frame_index_downsampled",
                 "frame_index_original", args.cluster_col]].copy()
        df = df.sort_values(["traj_index", "frame_index_original"]).reset_index(drop=True)
        df = df.rename(columns={args.cluster_col: "state"})

        if args.drop_noise:
            before = len(df)
            df     = df[df["state"] >= 0].copy()
            print(f"Noise frames dropped: {before - len(df)}")

        # Save frame-level state trace
        state_trace_out = RESULTS / f"{prefix}_state_trace_{args.cluster_col}.csv"
        df.to_csv(state_trace_out, index=False)
        print("Saved state trace:", state_trace_out)

        traj_rows              = []
        dwell_detail_rows      = []
        transition_detail_rows = []

        for ti, g in df.groupby("traj_index", sort=True):
            g      = g.sort_values("frame_index_original")
            labels = g["state"].to_numpy(dtype=int)
            mutant = g["mutant"].iloc[0]
            replica = g["replica"].iloc[0]

            traj_rows.append({
                "traj_index"       : int(ti),
                "mutant"           : mutant,
                "replica"          : replica,
                "n_frames_used"    : int(len(labels)),
                "n_states_observed": int(pd.unique(labels).size),
                "switch_count"     : switch_count(labels),
                "mean_dwell_frames": mean_dwell_frames(labels),
            })

            for rec in dwell_rows(labels):
                rec.update({"traj_index": int(ti), "mutant": mutant, "replica": replica})
                dwell_detail_rows.append(rec)

            trans_df = transition_count_matrix(labels)
            if len(trans_df):
                trans_df.insert(0, "traj_index", int(ti))
                trans_df.insert(1, "mutant", mutant)
                trans_df.insert(2, "replica", replica)
                transition_detail_rows.append(trans_df)

        # Per-trajectory summary
        traj_df  = pd.DataFrame(traj_rows)
        traj_out = RESULTS / f"{prefix}_transition_summary_{args.cluster_col}.csv"
        traj_df.to_csv(traj_out, index=False)
        print("Saved transition summary:", traj_out)

        # Dwell detail and per-mutant aggregates
        if dwell_detail_rows:
            dwell_df  = pd.DataFrame(dwell_detail_rows)
            dwell_out = RESULTS / f"{prefix}_dwell_detail_{args.cluster_col}.csv"
            dwell_df.to_csv(dwell_out, index=False)
            print("Saved dwell detail:", dwell_out)

            dwell_mut = (
                dwell_df.groupby(["mutant", "state"])["dwell_frames"]
                .agg(["mean", "median", "count"])
                .reset_index()
            )
            dwell_mut.to_csv(RESULTS / f"{prefix}_dwell_by_mutant_{args.cluster_col}.csv", index=False)
            print("Saved dwell by mutant.")

        # Transition count matrices
        if transition_detail_rows:
            trans_all = pd.concat(transition_detail_rows, ignore_index=True)
            trans_all.to_csv(RESULTS / f"{prefix}_transition_counts_{args.cluster_col}.csv", index=False)
            print("Saved transition counts.")

            trans_mut = (
                trans_all.groupby(["mutant", "from_state", "to_state"])["count"]
                .sum().reset_index()
            )
            trans_mut.to_csv(
                RESULTS / f"{prefix}_transition_counts_by_mutant_{args.cluster_col}.csv",
                index=False,
            )
            print("Saved transition counts by mutant.")

        # Per-mutant transition summary (mean ± std across replicas)
        mutant_summary = (
            traj_df.groupby("mutant")[["switch_count", "mean_dwell_frames", "n_states_observed"]]
            .agg(["mean", "std", "count"])
        )
        mutant_summary.columns = ["__".join(col) for col in mutant_summary.columns.to_flat_index()]
        mutant_summary = mutant_summary.reset_index()
        mutant_summary.to_csv(
            RESULTS / f"{prefix}_transition_summary_by_mutant_{args.cluster_col}.csv",
            index=False,
        )
        print("Saved transition summary by mutant.")

    print("Script 02f completed.")


if __name__ == "__main__":
    main()