"""
Prepare AutoDock Vina docking jobs for the Q2022P subset receptors.

Reads the receptor manifest from step 5a, pairs each receptor with the
specified ligands, writes per-job Vina config files, generates a bash
run script, and saves a job manifest CSV for downstream ranking (step 5c).

Usage:
    python 05b_prepare_vina_q2022p.py \\
        --ligands lorlatinib,crizotinib \\
        --center-x X --center-y Y --center-z Z \\
        --size-x SX --size-y SY --size-z SZ
"""

import argparse
from pathlib import Path

import pandas as pd

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
DOCK_BASE = BASE / "dock" / "ROS1" / "q2022p_vina"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--receptor-manifest",
        default=str(RESULTS / "step5a_q2022p_docking_receptors_all_prefixes.csv"),
        help="CSV created by 05a_select_extract_docking_receptors_q2022p.py",
    )
    p.add_argument(
        "--ligands",
        required=True,
        help="Comma-separated ligand base names (without extension), e.g. lorlatinib,crizotinib",
    )
    p.add_argument(
        "--receptor-pdbqt-dir",
        default=str(DOCK_BASE / "receptor_pdbqt"),
        help="Directory where receptor PDBQT files are stored.",
    )
    p.add_argument(
        "--ligand-pdbqt-dir",
        default=str(DOCK_BASE / "ligand_pdbqt"),
        help="Directory where ligand PDBQT files are stored.",
    )
    p.add_argument("--center-x", type=float, required=True)
    p.add_argument("--center-y", type=float, required=True)
    p.add_argument("--center-z", type=float, required=True)
    p.add_argument("--size-x",   type=float, required=True)
    p.add_argument("--size-y",   type=float, required=True)
    p.add_argument("--size-z",   type=float, required=True)
    p.add_argument("--exhaustiveness", type=int, default=16)
    p.add_argument("--num-modes",      type=int, default=20)
    p.add_argument("--energy-range",   type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    DOCK_BASE.mkdir(parents=True, exist_ok=True)

    receptors = pd.read_csv(args.receptor_manifest)
    ligands   = [x.strip() for x in args.ligands.split(",") if x.strip()]

    receptor_pdbqt_dir = Path(args.receptor_pdbqt_dir)
    ligand_pdbqt_dir   = Path(args.ligand_pdbqt_dir)
    config_dir         = DOCK_BASE / "configs"
    out_dir            = DOCK_BASE / "vina_outputs"
    log_dir            = DOCK_BASE / "vina_logs"

    for d in (receptor_pdbqt_dir, ligand_pdbqt_dir, config_dir, out_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows     = []
    sh_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]

    for _, rec in receptors.iterrows():
        receptor_stem  = Path(rec["output_pdb"]).stem
        receptor_pdbqt = receptor_pdbqt_dir / f"{receptor_stem}.pdbqt"

        for ligand in ligands:
            ligand_pdbqt = ligand_pdbqt_dir / f"{ligand}.pdbqt"
            cfg_name     = f"{receptor_stem}__{ligand}.txt"
            out_name     = f"{receptor_stem}__{ligand}.pdbqt"
            log_name     = f"{receptor_stem}__{ligand}.log"

            cfg_path = config_dir / cfg_name
            out_path = out_dir    / out_name
            log_path = log_dir    / log_name

            cfg_text = (
                f"receptor = {receptor_pdbqt}\n"
                f"ligand = {ligand_pdbqt}\n"
                f"center_x = {args.center_x}\n"
                f"center_y = {args.center_y}\n"
                f"center_z = {args.center_z}\n"
                f"size_x = {args.size_x}\n"
                f"size_y = {args.size_y}\n"
                f"size_z = {args.size_z}\n"
                f"exhaustiveness = {args.exhaustiveness}\n"
                f"num_modes = {args.num_modes}\n"
                f"energy_range = {args.energy_range}\n"
                f"out = {out_path}\n"
                f"log = {log_path}\n"
            )
            cfg_path.write_text(cfg_text)
            sh_lines.append(f"vina --config {cfg_path}")

            rows.append({
                "prefix":               rec["prefix"],
                "mutant":               rec["mutant"],
                "selection_type":       rec["selection_type"],
                "cluster_id":           rec["cluster_id"],
                "replica":              rec["replica"],
                "frame_index_original": rec["frame_index_original"],
                "receptor_pdb":         rec["output_pdb"],
                "receptor_pdbqt":       str(receptor_pdbqt),
                "ligand":               ligand,
                "ligand_pdbqt":         str(ligand_pdbqt),
                "vina_config":          str(cfg_path),
                "vina_out":             str(out_path),
                "vina_log":             str(log_path),
            })

    manifest     = pd.DataFrame(rows)
    manifest_out = RESULTS / "step5b_q2022p_vina_jobs.csv"
    manifest.to_csv(manifest_out, index=False)

    sh_path = DOCK_BASE / "run_vina_q2022p.sh"
    sh_path.write_text("\n".join(sh_lines) + "\n")

    print("Saved docking job manifest:", manifest_out)
    print("Saved Vina run script:", sh_path)
    print("NOTE: receptor and ligand PDBQT files must be prepared externally before running Vina.")


if __name__ == "__main__":
    main()