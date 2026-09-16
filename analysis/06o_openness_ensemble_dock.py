#!/usr/bin/env python3
"""
06o_openness_ensemble_dock.py

Test whether pocket openness drives binding, and whether the double mutant
really binds worse than Q2022P or the single-frame difference was noise.

For Q2022P and Q2022P_S1986F:
  - use the per-frame pocket_open_state (step3a_frame_metrics.csv) to pick
    N_OPEN clearly-open + N_CLOSED clearly-closed frames per variant
    (open/closed by the flag; within each, spread across the front-distance
     range so we sample the extremes, not one micro-state)
  - extract each frame as a receptor (same as 05a/06i: protein selection)
  - Meeko -> pdbqt
  - dock cabozantinib AND lorlatinib into each, shared box, locked params+seed
  - write a manifest that records, per frame: variant, replica, frame,
    pocket_open_state, pocket_front_dist_nm  -> so scores can be joined to
    openness afterwards.

Prepares configs + run script; does not run vina. Run in venv.

Usage:
    python analysis/06o_openness_ensemble_dock.py
    nohup bash dock/ROS1/q2022p_vina/openness_ensemble/run_vina_openness.sh \
        > /tmp/openness_dock.log 2>&1 &
"""
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import MDAnalysis as mda

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
DOCK = BASE / "dock" / "ROS1"
VINA = DOCK / "q2022p_vina"
LIG_DIR = VINA / "ligand_pdbqt"

FRAME_METRICS = RESULTS / "step3a_frame_metrics.csv"
F_PATHS = RESULTS / "F_paths.csv"

VARIANTS = ["Q2022P", "Q2022P_S1986F"]
LIGANDS = ["cabozantinib", "lorlatinib"]
N_OPEN = 8
N_CLOSED = 8

SITE_REAL = [2026, 2102, 2103, 2004, 2075, 1980]
OFFSET = 1933
BOX_SIZE = (28.7, 30.3, 30.2)     # same shared box as the main runs
SEED = 42
EXH, NMODES, ERANGE = 16, 20, 4
ATOM_SEL = "protein"

OUT = VINA / "openness_ensemble"
REC_PDB = OUT / "receptor_pdb"
REC_PQT = OUT / "receptor_pdbqt"
CFG = OUT / "configs"
VOUT = OUT / "vina_outputs"
VLOG = OUT / "vina_logs"
for d in (REC_PDB, REC_PQT, CFG, VOUT, VLOG):
    d.mkdir(parents=True, exist_ok=True)


def pick_frames(sub):
    """Pick N_OPEN open + N_CLOSED closed frames, spread across front-distance."""
    picks = []
    for state, n in [(1, N_OPEN), (0, N_CLOSED)]:
        s = sub[sub["pocket_open_state"] == state]
        if len(s) == 0:
            print(f"    WARNING: no frames with open_state={state}")
            continue
        # spread across the front-distance range so we sample the spectrum
        s = s.sort_values("pocket_front_dist_nm")
        idx = np.linspace(0, len(s) - 1, min(n, len(s))).astype(int)
        picks.append(s.iloc[idx])
    return pd.concat(picks) if picks else pd.DataFrame()


def main():
    fm = pd.read_csv(FRAME_METRICS)
    F = pd.read_csv(F_PATHS)["folder"].tolist()

    # ensure ligands present
    import shutil
    for lig in LIGANDS:
        dst = LIG_DIR / f"{lig}.pdbqt"
        if not dst.exists():
            src = DOCK / "ligands_pdbqt" / f"{lig}.pdbqt"
            if not src.exists():
                raise SystemExit(f"missing ligand {src}")
            shutil.copy2(src, dst)

    run = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    manifest = []

    for v in VARIANTS:
        sub = fm[fm["mutant"] == v]
        if sub.empty:
            print(f"{v}: no frames in metrics, skipping"); continue
        chosen = pick_frames(sub)
        print(f"\n{v}: picked {len(chosen)} frames "
              f"({int((chosen['pocket_open_state']==1).sum())} open, "
              f"{int((chosen['pocket_open_state']==0).sum())} closed)")

        for _, row in chosen.iterrows():
            ti = int(row["traj_index"]); frame = int(row["frame_index_original"])
            rep = int(row["replica"]); openst = int(row["pocket_open_state"])
            front = float(row["pocket_front_dist_nm"])
            top = str(Path(F[ti]) / f"{Path(F[ti]).parent.name}-MD-prot.pdb")
            xtc = str(Path(F[ti]) / f"{Path(F[ti]).parent.name}-MD-prot.xtc")
            state_tag = "open" if openst == 1 else "closed"
            tag = f"{v}_rep{rep}_f{frame}_{state_tag}"
            pdb = REC_PDB / f"{tag}.pdb"

            u = mda.Universe(top, xtc)
            u.trajectory[frame]
            u.select_atoms(ATOM_SEL).write(str(pdb))

            pqt = REC_PQT / f"{tag}.pdbqt"
            subprocess.run(["mk_prepare_receptor.py", "--read_pdb", str(pdb),
                            "-o", str(pqt.with_suffix("")), "--write_pdbqt", str(pqt)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # box centre from this frame's pocket residues
            uu = mda.Universe(str(pdb))
            sel = "resid " + " ".join(str(r) for r in SITE_REAL) + " and not name H*"
            ag = uu.select_atoms(sel)
            if ag.n_atoms == 0:
                sel = "resid " + " ".join(str(r-OFFSET) for r in SITE_REAL) + " and not name H*"
                ag = uu.select_atoms(sel)
            c = (ag.positions.min(0) + ag.positions.max(0)) / 2.0

            for lig in LIGANDS:
                cfg = CFG / f"{tag}__{lig}.txt"
                out = VOUT / f"{tag}__{lig}.pdbqt"
                log = VLOG / f"{tag}__{lig}.log"
                cfg.write_text(
                    f"receptor = {pqt}\nligand = {LIG_DIR/f'{lig}.pdbqt'}\n"
                    f"center_x = {c[0]:.2f}\ncenter_y = {c[1]:.2f}\ncenter_z = {c[2]:.2f}\n"
                    f"size_x = {BOX_SIZE[0]}\nsize_y = {BOX_SIZE[1]}\nsize_z = {BOX_SIZE[2]}\n"
                    f"out = {out}\nseed = {SEED}\nexhaustiveness = {EXH}\n"
                    f"num_modes = {NMODES}\nenergy_range = {ERANGE}\n")
                run.append(f"vina --config {cfg} 2>&1 | tee {log}")
            manifest.append(dict(tag=tag, variant=v, replica=rep, frame=frame,
                                 pocket_open_state=openst, pocket_front_dist_nm=front))

    run_sh = OUT / "run_vina_openness.sh"
    run_sh.write_text("\n".join(run) + "\n")
    pd.DataFrame(manifest).to_csv(OUT / "manifest.csv", index=False)
    print(f"\nwrote {len(manifest)} receptors x {len(LIGANDS)} ligands = "
          f"{len(manifest)*len(LIGANDS)} docks")
    print(f"manifest: {OUT/'manifest.csv'}")
    print(f"run script: {run_sh}")


if __name__ == "__main__":
    main()