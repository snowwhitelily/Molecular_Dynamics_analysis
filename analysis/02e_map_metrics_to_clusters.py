# %%
# ROS1 Analysis Pipeline
# Script 2E: Map Frame Metrics to Selected PCA Clusters

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prefixes", default="actout_ctlfit,aloop", help="Comma-separated clustered PCA prefixes")
    p.add_argument("--metrics", default="step3a_frame_metrics_downsampled.csv", help="Frame metrics CSV in results/ROS1")
    return p.parse_args()


def main():
    args = parse_args()
    prefixes = [x.strip() for x in args.prefixes.split(",") if x.strip()]
    metrics_df = pd.read_csv(RESULTS / args.metrics)

    for prefix in prefixes:
        print("\n=== Mapping metrics for prefix:", prefix, "===")
        cluster_f = RESULTS / f"{prefix}_cluster_labels.csv"
        if not cluster_f.exists():
            raise FileNotFoundError(f"Missing cluster label file for '{prefix}'")
        cluster_df = pd.read_csv(cluster_f)

        merged = cluster_df.merge(
            metrics_df,
            on=["traj_index", "mutant", "replica", "frame_index_original"],
            how="left",
            validate="one_to_one",
        )

        merged_out = RESULTS / f"{prefix}_cluster_metric_frames.csv"
        merged.to_csv(merged_out, index=False)
        print("Saved merged frame table:", merged_out)

        numeric_cols = [
            c for c in merged.columns
            if c not in {"traj_index", "mutant", "replica", "frame_index_downsampled_x", "frame_index_downsampled_y", "frame_index_original", "cluster_hdb", "cluster_db", "trajectory"}
            and pd.api.types.is_numeric_dtype(merged[c])
        ]
        numeric_cols = [c for c in numeric_cols if c not in ["cluster_hdb", "cluster_db"]]

        summary = merged.groupby("cluster_hdb")[numeric_cols].agg(["mean", "std", "count"])
        summary.columns = ["__".join(col).strip() for col in summary.columns.to_flat_index()]
        summary = summary.reset_index()
        summary_out = RESULTS / f"{prefix}_cluster_metric_summary_hdb.csv"
        summary.to_csv(summary_out, index=False)
        print("Saved cluster metric summary:", summary_out)

        occ = (
            merged.groupby(["mutant", "cluster_hdb"]).size().unstack(fill_value=0).sort_index()
        )
        occ_frac = occ.div(occ.sum(axis=1), axis=0)
        occ_out = RESULTS / f"{prefix}_cluster_occupancy_by_mutant_hdb.csv"
        occ_frac.to_csv(occ_out)
        print("Saved occupancy by mutant:", occ_out)

        occ_rep = (
            merged.groupby(["mutant", "replica", "cluster_hdb"]).size().reset_index(name="count")
        )
        occ_rep["fraction"] = occ_rep.groupby(["mutant", "replica"])["count"].transform(lambda s: s / s.sum())
        occ_rep_out = RESULTS / f"{prefix}_cluster_occupancy_by_replica_hdb.csv"
        occ_rep.to_csv(occ_rep_out, index=False)
        print("Saved occupancy by replica:", occ_rep_out)

    print("Script 2E completed.")


if __name__ == "__main__":
    main()
