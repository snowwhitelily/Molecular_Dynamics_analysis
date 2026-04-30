# %%
# ROS1 Analysis Pipeline
# Script 2D: Cluster Selected PCA Spaces
#
# FIX: Added F1994L exclusion — filters F and frame_idx_list
# to match the PCA that was run without F1994L.

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.clustering import run_hdbscan, run_dbscan
from scripts.plotting import plot_occupancy_heatmap, plot_cluster_population, plot_pca_clusters
from scripts.ros1_clustering import run_cluster_panel_varlen, save_cluster_outputs

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_ROOT = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "step2d_selected_clustering"
FIG_ROOT.mkdir(parents=True, exist_ok=True)
LOCAL_FIG_DIR = Path("figures")
LOCAL_FIG_DIR.mkdir(exist_ok=True)

# ============================================================
# EXCLUDE MUTANTS NOT IN ANALYSIS
# F1994L excluded per supervisor instruction:
# started in inactive form, not comparable to active-form panel
# ============================================================
EXCLUDE_MUTANTS = {"F1994L"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prefixes", default="actout_ctlfit,aloop", help="Comma-separated PCA prefixes to cluster")
    p.add_argument("--n-pcs", type=int, default=5)
    p.add_argument("--min-cluster-size", type=int, default=200)
    p.add_argument("--eps", type=float, default=0.9)
    p.add_argument("--min-samples", type=int, default=60)
    return p.parse_args()


def move_local_figures(prefix, fig_dir):
    for suffix in ["occupancy_heatmap", "cluster_population", "pca_clusters"]:
        src = LOCAL_FIG_DIR / f"{prefix}_{suffix}.png"
        if src.exists():
            dst = fig_dir / src.name
            shutil.move(str(src), str(dst))
            print("Saved figure:", dst)


def main():
    args = parse_args()
    F = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
    frame_idx_list = np.load(RESULTS / "frame_idx_list.npy", allow_pickle=True).tolist()

    # Apply exclusion
    keep = [i for i, f in enumerate(F)
            if Path(f).parent.name not in EXCLUDE_MUTANTS]
    F              = [F[i] for i in keep]
    frame_idx_list = [frame_idx_list[i] for i in keep]
    print(f"Excluded mutants: {EXCLUDE_MUTANTS}")
    print(f"Remaining trajectories: {len(F)}")

    prefixes = [x.strip() for x in args.prefixes.split(",") if x.strip()]
    print("Selected PCA prefixes:", prefixes)

    for prefix in prefixes:
        print("\n=== Clustering prefix:", prefix, "===")
        scores_f = RESULTS / f"{prefix}_scores.npy"
        x_owner_f = RESULTS / f"{prefix}_x_owner.npy"
        scores_csv_f = RESULTS / f"{prefix}_scores.csv"
        if not scores_f.exists() or not x_owner_f.exists() or not scores_csv_f.exists():
            raise FileNotFoundError(f"Missing PCA outputs for prefix '{prefix}'")

        scores_flat = np.load(scores_f)
        x_owner = np.load(x_owner_f)
        scores_df = pd.read_csv(scores_csv_f)

        title_prefix = prefix.replace("_", " ").upper()
        fig_dir = FIG_ROOT / prefix
        fig_dir.mkdir(parents=True, exist_ok=True)

        cluster_result = run_cluster_panel_varlen(
            scores_flat=scores_flat,
            x_owner=x_owner,
            F=F,
            prefix=prefix,
            title_prefix=title_prefix,
            n_pcs=args.n_pcs,
            min_cluster_size=args.min_cluster_size,
            eps=args.eps,
            min_samples=args.min_samples,
            run_hdbscan=run_hdbscan,
            run_dbscan=run_dbscan,
            plot_occupancy_heatmap=plot_occupancy_heatmap,
            plot_cluster_population=plot_cluster_population,
            plot_pca_clusters=plot_pca_clusters,
            display_fn=print,
        )

        save_cluster_outputs(
            cluster_result=cluster_result,
            F=F,
            x_owner=x_owner,
            prefix=prefix,
            results_dir=RESULTS,
            frame_index_downsampled=scores_df["frame_index_downsampled"].to_numpy(dtype=int),
            frame_index_original=scores_df["frame_index_original"].to_numpy(dtype=int),
            n_pcs=args.n_pcs,
            min_cluster_size=args.min_cluster_size,
            eps=args.eps,
            min_samples=args.min_samples,
        )
        move_local_figures(prefix, fig_dir)

    print("Script 2D completed.")


if __name__ == "__main__":
    main()
