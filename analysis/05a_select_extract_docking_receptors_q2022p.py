
# %%
# ROS1 Analysis Pipeline
# Script 5A: Select and Extract Docking Receptors for the Q2022P subset

import argparse
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
OUT_BASE = BASE / "dock" / "ROS1" / "q2022p_subset_receptors"

Q_FAMILY = ["WT", "Q2022P", "Q2022P_S1986F", "Q2022P_S1986Y"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prefixes",
        required=True,
        help="Comma-separated selected PCA prefixes, e.g. actout_ctlfit,aloop",
    )
    p.add_argument(
        "--cluster-col",
        default="cluster_hdb",
        choices=["cluster_hdb", "cluster_db"],
        help="Cluster label column to use from *_cluster_labels.csv",
    )
    p.add_argument(
        "--k-use",
        type=int,
        default=5,
        help="Number of PCs to use when finding a medoid-like representative frame.",
    )
    p.add_argument(
        "--min-alt-frac",
        type=float,
        default=0.15,
        help="Minimum occupancy fraction for keeping an alternative cluster for a mutant.",
    )
    p.add_argument(
        "--min-enrichment-vs-wt",
        type=float,
        default=0.10,
        help="Require mutant alternative cluster occupancy to exceed WT by at least this much.",
    )
    p.add_argument(
        "--max-alt-per-mutant",
        type=int,
        default=1,
        help="Maximum number of alternative enriched clusters to keep per mutant/prefix.",
    )
    p.add_argument(
        "--atom-selection",
        default="protein",
        help="MDAnalysis selection used when writing receptor PDB files.",
    )
    return p.parse_args()


def mutant_replica_from_F(F):
    rows = []
    for i, path in enumerate(F):
        p = Path(path)
        rows.append(
            {
                "traj_index": i,
                "trajectory": str(p),
                "mutant": p.parent.name,
                "replica": p.name,
                "pdb": str(p / f"{p.parent.name}-MD-prot.pdb"),
                "xtc": str(p / f"{p.parent.name}-MD-prot.xtc"),
            }
        )
    return pd.DataFrame(rows)


def choose_medoid_frame(df_cluster_scores: pd.DataFrame, pc_cols):
    X = df_cluster_scores[pc_cols].to_numpy(dtype=float)
    center = X.mean(axis=0, keepdims=True)
    d2 = np.sum((X - center) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return df_cluster_scores.iloc[idx]


def write_receptor_pdb(top_path: str, xtc_path: str, frame_idx: int, atom_selection: str, out_pdb: Path):
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    u = mda.Universe(top_path, xtc_path)
    u.trajectory[int(frame_idx)]
    ag = u.select_atoms(atom_selection)
    ag.write(str(out_pdb))


def main():
    args = parse_args()
    prefixes = [x.strip() for x in args.prefixes.split(",") if x.strip()]
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    F = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
    meta_df = mutant_replica_from_F(F)

    overall_rows = []

    for prefix in prefixes:
        print(f"\n=== Prefix: {prefix} ===")
        cluster_csv = RESULTS / f"{prefix}_cluster_labels.csv"
        scores_csv = RESULTS / f"{prefix}_scores.csv"

        if not cluster_csv.exists():
            raise FileNotFoundError(f"Missing cluster labels: {cluster_csv}")
        if not scores_csv.exists():
            raise FileNotFoundError(f"Missing scores table: {scores_csv}")

        dfc = pd.read_csv(cluster_csv)
        dfs = pd.read_csv(scores_csv)

        if args.cluster_col not in dfc.columns:
            raise ValueError(f"{args.cluster_col} not found in {cluster_csv.name}")

        df = dfs.merge(
            dfc[
                [
                    "traj_index",
                    "mutant",
                    "replica",
                    "frame_index_downsampled",
                    "frame_index_original",
                    args.cluster_col,
                ]
            ],
            on=["traj_index", "mutant", "replica", "frame_index_downsampled", "frame_index_original"],
            how="inner",
        ).merge(meta_df, on=["traj_index", "trajectory", "mutant", "replica"], how="left")

        df = df[df["mutant"].isin(Q_FAMILY)].copy()
        df = df[df[args.cluster_col] >= 0].copy()

        if df.empty:
            print(f"No non-noise frames for {prefix} in Q2022P subset.")
            continue

        pc_cols = [c for c in df.columns if c.startswith("PC")]
        pc_cols = pc_cols[: max(1, args.k_use)]

        occ = (
            df.groupby(["mutant", args.cluster_col]).size().rename("count").reset_index()
        )
        totals = occ.groupby("mutant")["count"].transform("sum")
        occ["fraction"] = occ["count"] / totals

        wt_occ = (
            occ[occ["mutant"] == "WT"][[args.cluster_col, "fraction"]]
            .rename(columns={"fraction": "wt_fraction"})
        )
        occ = occ.merge(wt_occ, on=args.cluster_col, how="left")
        occ["wt_fraction"] = occ["wt_fraction"].fillna(0.0)
        occ["enrichment_vs_wt"] = occ["fraction"] - occ["wt_fraction"]

        prefix_rows = []

        for mutant in Q_FAMILY:
            occ_mut = occ[occ["mutant"] == mutant].sort_values("fraction", ascending=False).copy()
            if occ_mut.empty:
                continue

            dom_row = occ_mut.iloc[0]
            dom_cluster = int(dom_row[args.cluster_col])

            sub_dom = df[(df["mutant"] == mutant) & (df[args.cluster_col] == dom_cluster)].copy()
            rep_dom = choose_medoid_frame(sub_dom, pc_cols)

            dom_out = OUT_BASE / prefix / mutant / f"{mutant}_{prefix}_dominant_cluster{dom_cluster}_rep.pdb"
            write_receptor_pdb(
                top_path=rep_dom["pdb"],
                xtc_path=rep_dom["xtc"],
                frame_idx=int(rep_dom["frame_index_original"]),
                atom_selection=args.atom_selection,
                out_pdb=dom_out,
            )

            prefix_rows.append(
                {
                    "prefix": prefix,
                    "mutant": mutant,
                    "selection_type": "dominant_cluster_representative",
                    "cluster_id": dom_cluster,
                    "cluster_fraction_mutant": float(dom_row["fraction"]),
                    "cluster_fraction_wt": float(dom_row["wt_fraction"]),
                    "enrichment_vs_wt": float(dom_row["enrichment_vs_wt"]),
                    "traj_index": int(rep_dom["traj_index"]),
                    "replica": rep_dom["replica"],
                    "frame_index_downsampled": int(rep_dom["frame_index_downsampled"]),
                    "frame_index_original": int(rep_dom["frame_index_original"]),
                    "trajectory": rep_dom["trajectory"],
                    "pdb_source": rep_dom["pdb"],
                    "xtc_source": rep_dom["xtc"],
                    "output_pdb": str(dom_out),
                }
            )

            alt_occ = occ_mut[
                (occ_mut[args.cluster_col] != dom_cluster)
                & (occ_mut["fraction"] >= args.min_alt_frac)
                & (occ_mut["enrichment_vs_wt"] >= args.min_enrichment_vs_wt)
            ].sort_values(["enrichment_vs_wt", "fraction"], ascending=False)

            for _, alt_row in alt_occ.head(args.max_alt_per_mutant).iterrows():
                alt_cluster = int(alt_row[args.cluster_col])
                sub_alt = df[(df["mutant"] == mutant) & (df[args.cluster_col] == alt_cluster)].copy()
                rep_alt = choose_medoid_frame(sub_alt, pc_cols)

                alt_out = OUT_BASE / prefix / mutant / f"{mutant}_{prefix}_alt_cluster{alt_cluster}_rep.pdb"
                write_receptor_pdb(
                    top_path=rep_alt["pdb"],
                    xtc_path=rep_alt["xtc"],
                    frame_idx=int(rep_alt["frame_index_original"]),
                    atom_selection=args.atom_selection,
                    out_pdb=alt_out,
                )

                prefix_rows.append(
                    {
                        "prefix": prefix,
                        "mutant": mutant,
                        "selection_type": "mutant_specific_alternative_cluster_representative",
                        "cluster_id": alt_cluster,
                        "cluster_fraction_mutant": float(alt_row["fraction"]),
                        "cluster_fraction_wt": float(alt_row["wt_fraction"]),
                        "enrichment_vs_wt": float(alt_row["enrichment_vs_wt"]),
                        "traj_index": int(rep_alt["traj_index"]),
                        "replica": rep_alt["replica"],
                        "frame_index_downsampled": int(rep_alt["frame_index_downsampled"]),
                        "frame_index_original": int(rep_alt["frame_index_original"]),
                        "trajectory": rep_alt["trajectory"],
                        "pdb_source": rep_alt["pdb"],
                        "xtc_source": rep_alt["xtc"],
                        "output_pdb": str(alt_out),
                    }
                )

        prefix_manifest = pd.DataFrame(prefix_rows)
        prefix_manifest_out = RESULTS / f"{prefix}_q2022p_docking_receptors.csv"
        prefix_manifest.to_csv(prefix_manifest_out, index=False)
        print("Saved receptor manifest:", prefix_manifest_out)

        occ_out = RESULTS / f"{prefix}_q2022p_cluster_occupancy_for_docking.csv"
        occ.to_csv(occ_out, index=False)
        print("Saved occupancy summary:", occ_out)

        overall_rows.extend(prefix_rows)

    if overall_rows:
        overall_df = pd.DataFrame(overall_rows)
        overall_out = RESULTS / "step5a_q2022p_docking_receptors_all_prefixes.csv"
        overall_df.to_csv(overall_out, index=False)
        print("Saved overall receptor manifest:", overall_out)

    print("Step 5A complete.")


if __name__ == "__main__":
    main()
