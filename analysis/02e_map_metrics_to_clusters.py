# ROS1 Analysis Pipeline
# Script 02e: Map Frame Metrics to Cluster Labels
#
# Merges the per-frame structural metrics from Script 03a with the
# cluster label assignments from Script 02d. Produces per-cluster
# metric summaries and per-mutant cluster occupancy tables.

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
    p.add_argument("--metrics", default="step3a_frame_metrics_downsampled.csv",
                   help="Frame metrics CSV filename in results/ROS1/.")
    return p.parse_args()


def main():
    args     = parse_args()
    prefixes = [x.strip() for x in args.prefixes.split(",") if x.strip()]

    metrics_df = pd.read_csv(RESULTS / args.metrics)

    for prefix in prefixes:
        print(f"\n=== Mapping metrics: {prefix} ===")

        cluster_f = RESULTS / f"{prefix}_cluster_labels.csv"
        if not cluster_f.exists():
            raise FileNotFoundError(f"Missing cluster label file for '{prefix}'. Run 02d first.")

        cluster_df = pd.read_csv(cluster_f)

        # Merge cluster labels with per-frame structural metrics
        merged = cluster_df.merge(
            metrics_df,
            on=["traj_index", "mutant", "replica", "frame_index_original"],
            how="left",
            validate="one_to_one",
        )

        merged_out = RESULTS / f"{prefix}_cluster_metric_frames.csv"
        merged.to_csv(merged_out, index=False)
        print("Saved merged frame table:", merged_out)

        # Per-cluster mean and std for each structural metric
        exclude_cols = {
            "traj_index", "mutant", "replica",
            "frame_index_downsampled_x", "frame_index_downsampled_y",
            "frame_index_original", "cluster_hdb", "cluster_db", "trajectory",
        }
        numeric_cols = [
            c for c in merged.columns
            if c not in exclude_cols
            and pd.api.types.is_numeric_dtype(merged[c])
            and c not in {"cluster_hdb", "cluster_db"}
        ]

        summary     = merged.groupby("cluster_hdb")[numeric_cols].agg(["mean", "std", "count"])
        summary.columns = ["__".join(col) for col in summary.columns.to_flat_index()]
        summary     = summary.reset_index()
        summary_out = RESULTS / f"{prefix}_cluster_metric_summary_hdb.csv"
        summary.to_csv(summary_out, index=False)
        print("Saved cluster metric summary:", summary_out)

        # Per-mutant cluster occupancy fractions
        occ      = merged.groupby(["mutant", "cluster_hdb"]).size().unstack(fill_value=0).sort_index()
        occ_frac = occ.div(occ.sum(axis=1), axis=0)
        occ_out  = RESULTS / f"{prefix}_cluster_occupancy_by_mutant_hdb.csv"
        occ_frac.to_csv(occ_out)
        print("Saved occupancy by mutant:", occ_out)

        # Per-replica cluster occupancy fractions
        occ_rep = merged.groupby(["mutant", "replica", "cluster_hdb"]).size().reset_index(name="count")
        occ_rep["fraction"] = occ_rep.groupby(["mutant", "replica"])["count"].transform(
            lambda s: s / s.sum()
        )
        occ_rep_out = RESULTS / f"{prefix}_cluster_occupancy_by_replica_hdb.csv"
        occ_rep.to_csv(occ_rep_out, index=False)
        print("Saved occupancy by replica:", occ_rep_out)

    print("Script 02e completed.")


if __name__ == "__main__":
    main()