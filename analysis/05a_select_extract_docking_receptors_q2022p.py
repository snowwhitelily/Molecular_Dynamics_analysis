# ROS1 Analysis Pipeline
# Script 05a: Select and Extract Docking Receptors — Q2022P Family
#
# Extracts representative receptor PDB files from the Q2022P compound
# mutation family MD ensembles for use in ensemble docking. For each
# system and PCA prefix, the medoid frame of the dominant cluster is
# extracted as the primary receptor. Mutant-enriched alternative clusters
# (if present) are also extracted up to --max-alt-per-mutant.
#
# Medoid frame: the frame closest to the cluster centroid in PCA score space.
# This minimises structural bias from arbitrary frame selection.

import argparse
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd

BASE     = Path.home() / "Molecular_Dynamics_analysis"
RESULTS  = BASE / "results" / "ROS1"
OUT_BASE = BASE / "dock" / "ROS1" / "q2022p_subset_receptors"

Q_FAMILY = ["WT", "Q2022P", "Q2022P_S1986F", "Q2022P_S1986Y", "S1986F"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prefixes",    required=True,
                   help="Comma-separated PCA prefixes, e.g. actout_ctlfit")
    p.add_argument("--cluster-col", default="cluster_hdb",
                   choices=["cluster_hdb", "cluster_db"])
    p.add_argument("--k-use",       type=int,   default=5,
                   help="Number of PCs used to find the medoid frame.")
    p.add_argument("--min-alt-frac", type=float, default=0.15,
                   help="Minimum cluster occupancy fraction to keep an alternative cluster.")
    p.add_argument("--min-enrichment-vs-wt", type=float, default=0.10,
                   help="Minimum occupancy excess over WT to keep an alternative cluster.")
    p.add_argument("--max-alt-per-mutant", type=int, default=1,
                   help="Maximum alternative clusters per mutant per prefix.")
    p.add_argument("--atom-selection", default="protein",
                   help="MDAnalysis atom selection for writing receptor PDB files.")
    return p.parse_args()


def mutant_replica_from_F(F):
    """Build a DataFrame of trajectory metadata from the F folder list."""
    rows = []
    for i, path in enumerate(F):
        p = Path(path)
        rows.append({
            "traj_index": i,
            "trajectory": str(p),
            "mutant"    : p.parent.name,
            "replica"   : int(p.name),
            "pdb"       : str(p / f"{p.parent.name}-MD-prot.pdb"),
            "xtc"       : str(p / f"{p.parent.name}-MD-prot.xtc"),
        })
    return pd.DataFrame(rows)


def choose_medoid_frame(df_cluster_scores: pd.DataFrame, pc_cols):
    """
    Select the medoid frame from a cluster — the frame closest to the
    cluster centroid in PC score space.
    """
    X      = df_cluster_scores[pc_cols].to_numpy(dtype=float)
    center = X.mean(axis=0, keepdims=True)
    d2     = np.sum((X - center) ** 2, axis=1)
    return df_cluster_scores.iloc[int(np.argmin(d2))]


def write_receptor_pdb(top_path: str, xtc_path: str, frame_idx: int,
                       atom_selection: str, out_pdb: Path):
    """Extract a single frame from a trajectory and write it as a PDB file."""
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    u  = mda.Universe(top_path, xtc_path)
    u.trajectory[int(frame_idx)]
    ag = u.select_atoms(atom_selection)
    ag.write(str(out_pdb))


def main():
    args     = parse_args()
    prefixes = [x.strip() for x in args.prefixes.split(",") if x.strip()]
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    F       = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
    meta_df = mutant_replica_from_F(F)

    overall_rows = []

    for prefix in prefixes:
        print(f"\n=== Prefix: {prefix} ===")

        cluster_csv = RESULTS / f"{prefix}_cluster_labels.csv"
        scores_npy  = RESULTS / f"{prefix}_scores.npy"
        owner_npy   = RESULTS / f"{prefix}_x_owner.npy"

        if not cluster_csv.exists():
            raise FileNotFoundError(f"Missing cluster labels: {cluster_csv}")
        if not scores_npy.exists():
            raise FileNotFoundError(f"Missing scores: {scores_npy}")
        if not owner_npy.exists():
            raise FileNotFoundError(f"Missing owner: {owner_npy}")

        dfc = pd.read_csv(cluster_csv)
        if args.cluster_col not in dfc.columns:
            raise ValueError(f"{args.cluster_col} not found in {cluster_csv.name}")

        # Build scores DataFrame from npy files
        scores_arr = np.load(scores_npy)
        owner_arr  = np.load(owner_npy)
        n_pcs      = scores_arr.shape[1]
        pc_names   = [f"PC{i+1}" for i in range(n_pcs)]
        dfs        = pd.DataFrame(scores_arr, columns=pc_names)
        dfs["traj_index"]              = owner_arr.astype(int)
        dfs["frame_index_downsampled"] = dfs.groupby("traj_index").cumcount()

        # Merge scores with cluster labels on traj_index + frame_index_downsampled
        df = dfs.merge(
            dfc[["traj_index", "mutant", "replica",
                 "frame_index_downsampled", "frame_index_original",
                 args.cluster_col]],
            on=["traj_index", "frame_index_downsampled"],
            how="inner",
        )

        # Merge pdb/xtc paths on mutant + replica
        # (traj_index differs between F_paths and cluster_labels after F1994L exclusion)
        df = df.merge(
            meta_df[["mutant", "replica", "traj_index", "pdb", "xtc"]]
                .rename(columns={"traj_index": "traj_index_fpaths"}),
            on=["mutant", "replica"],
            how="left",
        )

        df = df[df["mutant"].isin(Q_FAMILY)].copy()
        df = df[df[args.cluster_col] >= 0].copy()

        if df.empty:
            print(f"No non-noise frames for {prefix} in Q2022P subset.")
            continue

        pc_cols = [c for c in df.columns if c.startswith("PC")][:max(1, args.k_use)]

        # Compute cluster occupancy fractions and enrichment vs WT
        occ        = df.groupby(["mutant", args.cluster_col]).size().rename("count").reset_index()
        totals     = occ.groupby("mutant")["count"].transform("sum")
        occ["fraction"] = occ["count"] / totals

        wt_occ     = (occ[occ["mutant"] == "WT"][[args.cluster_col, "fraction"]]
                      .rename(columns={"fraction": "wt_fraction"}))
        occ        = occ.merge(wt_occ, on=args.cluster_col, how="left")
        occ["wt_fraction"]        = occ["wt_fraction"].fillna(0.0)
        occ["enrichment_vs_wt"]   = occ["fraction"] - occ["wt_fraction"]

        prefix_rows = []

        for mutant in Q_FAMILY:
            occ_mut = occ[occ["mutant"] == mutant].sort_values("fraction", ascending=False).copy()
            if occ_mut.empty:
                continue

            # Dominant cluster (highest occupancy)
            dom_row     = occ_mut.iloc[0]
            dom_cluster = int(dom_row[args.cluster_col])
            sub_dom     = df[(df["mutant"] == mutant) & (df[args.cluster_col] == dom_cluster)].copy()
            rep_dom     = choose_medoid_frame(sub_dom, pc_cols)

            dom_out = OUT_BASE / prefix / mutant / f"{mutant}_{prefix}_dominant_cluster{dom_cluster}_rep.pdb"
            write_receptor_pdb(rep_dom["pdb"], rep_dom["xtc"],
                               int(rep_dom["frame_index_original"]),
                               args.atom_selection, dom_out)

            prefix_rows.append({
                "prefix"                   : prefix,
                "mutant"                   : mutant,
                "selection_type"           : "dominant_cluster_representative",
                "cluster_id"               : dom_cluster,
                "cluster_fraction_mutant"  : float(dom_row["fraction"]),
                "cluster_fraction_wt"      : float(dom_row["wt_fraction"]),
                "enrichment_vs_wt"         : float(dom_row["enrichment_vs_wt"]),
                "traj_index"               : int(rep_dom["traj_index"]),
                "replica"                  : rep_dom["replica"],
                "frame_index_downsampled"  : int(rep_dom["frame_index_downsampled"]),
                "frame_index_original"     : int(rep_dom["frame_index_original"]),
                "pdb_source"               : rep_dom["pdb"],
                "xtc_source"               : rep_dom["xtc"],
                "output_pdb"               : str(dom_out),
            })

            # Alternative enriched clusters
            alt_occ = occ_mut[
                (occ_mut[args.cluster_col] != dom_cluster)
                & (occ_mut["fraction"]          >= args.min_alt_frac)
                & (occ_mut["enrichment_vs_wt"]  >= args.min_enrichment_vs_wt)
            ].sort_values(["enrichment_vs_wt", "fraction"], ascending=False)

            for _, alt_row in alt_occ.head(args.max_alt_per_mutant).iterrows():
                alt_cluster = int(alt_row[args.cluster_col])
                sub_alt     = df[(df["mutant"] == mutant) & (df[args.cluster_col] == alt_cluster)].copy()
                rep_alt     = choose_medoid_frame(sub_alt, pc_cols)

                alt_out = OUT_BASE / prefix / mutant / f"{mutant}_{prefix}_alt_cluster{alt_cluster}_rep.pdb"
                write_receptor_pdb(rep_alt["pdb"], rep_alt["xtc"],
                                   int(rep_alt["frame_index_original"]),
                                   args.atom_selection, alt_out)

                prefix_rows.append({
                    "prefix"                   : prefix,
                    "mutant"                   : mutant,
                    "selection_type"           : "mutant_specific_alternative_cluster_representative",
                    "cluster_id"               : alt_cluster,
                    "cluster_fraction_mutant"  : float(alt_row["fraction"]),
                    "cluster_fraction_wt"      : float(alt_row["wt_fraction"]),
                    "enrichment_vs_wt"         : float(alt_row["enrichment_vs_wt"]),
                    "traj_index"               : int(rep_alt["traj_index"]),
                    "replica"                  : rep_alt["replica"],
                    "frame_index_downsampled"  : int(rep_alt["frame_index_downsampled"]),
                    "frame_index_original"     : int(rep_alt["frame_index_original"]),
                    "pdb_source"               : rep_alt["pdb"],
                    "xtc_source"               : rep_alt["xtc"],
                    "output_pdb"               : str(alt_out),
                })

        prefix_manifest     = pd.DataFrame(prefix_rows)
        prefix_manifest_out = RESULTS / f"{prefix}_q2022p_docking_receptors.csv"
        prefix_manifest.to_csv(prefix_manifest_out, index=False)
        print("Saved receptor manifest:", prefix_manifest_out)

        occ_out = RESULTS / f"{prefix}_q2022p_cluster_occupancy_for_docking.csv"
        occ.to_csv(occ_out, index=False)
        print("Saved occupancy summary:", occ_out)

        overall_rows.extend(prefix_rows)

    if overall_rows:
        overall_df  = pd.DataFrame(overall_rows)
        overall_out = RESULTS / "step5a_q2022p_docking_receptors_all_prefixes.csv"
        overall_df.to_csv(overall_out, index=False)
        print("Saved overall receptor manifest:", overall_out)

    print("Script 05a completed.")


if __name__ == "__main__":
    main()