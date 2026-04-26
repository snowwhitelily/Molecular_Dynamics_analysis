
# %%
# ROS1 Analysis Pipeline
# Script 5C: Rank AutoDock Vina results for the Q2022P subset

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--jobs-manifest",
        default=str(RESULTS / "step5b_q2022p_vina_jobs.csv"),
        help="CSV produced by 05b_prepare_vina_q2022p.py",
    )
    return p.parse_args()


def parse_vina_log(log_path: Path):
    if not log_path.exists():
        return []

    rows = []
    pose_section = False
    pattern = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)")

    for line in log_path.read_text(errors="ignore").splitlines():
        if "mode |   affinity" in line:
            pose_section = True
            continue
        if pose_section:
            m = pattern.match(line)
            if m:
                rows.append(
                    {
                        "mode": int(m.group(1)),
                        "affinity_kcal_mol": float(m.group(2)),
                        "rmsd_lb": float(m.group(3)),
                        "rmsd_ub": float(m.group(4)),
                    }
                )
    return rows


def main():
    args = parse_args()
    jobs = pd.read_csv(args.jobs_manifest)

    all_rows = []
    best_rows = []

    for _, row in jobs.iterrows():
        log_path = Path(row["vina_log"])
        poses = parse_vina_log(log_path)
        if not poses:
            continue

        pose_df = pd.DataFrame(poses)
        for _, pose in pose_df.iterrows():
            rec = row.to_dict()
            rec.update(pose.to_dict())
            all_rows.append(rec)

        best = pose_df.sort_values("affinity_kcal_mol", ascending=True).iloc[0]
        rec_best = row.to_dict()
        rec_best.update(best.to_dict())
        best_rows.append(rec_best)

    if all_rows:
        all_df = pd.DataFrame(all_rows)
        all_out = RESULTS / "step5c_q2022p_vina_all_poses.csv"
        all_df.to_csv(all_out, index=False)
        print("Saved all poses:", all_out)

    if best_rows:
        best_df = pd.DataFrame(best_rows).sort_values("affinity_kcal_mol", ascending=True)
        best_out = RESULTS / "step5c_q2022p_vina_best_pose_summary.csv"
        best_df.to_csv(best_out, index=False)
        print("Saved best-pose summary:", best_out)

        by_mut_lig = (
            best_df.groupby(["mutant", "ligand"])["affinity_kcal_mol"]
            .min()
            .reset_index()
            .sort_values(["ligand", "affinity_kcal_mol"], ascending=[True, True])
        )
        by_mut_lig_out = RESULTS / "step5c_q2022p_vina_best_by_mutant_ligand.csv"
        by_mut_lig.to_csv(by_mut_lig_out, index=False)
        print("Saved ligand-by-mutant summary:", by_mut_lig_out)

    print("Step 5C complete.")


if __name__ == "__main__":
    main()
