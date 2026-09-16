#!/usr/bin/env python3
"""
06r_openness_ensemble_all.py

Generalizes 06o from (Q2022P vs double) to ALL FOUR focal variants x the chosen
ligands, so every active-form docking number can be made ensemble-based instead
of single-medoid.

For each variant in VARIANTS:
  - use per-frame pocket_open_state (step3a_frame_metrics.csv) to pick
    N_OPEN clearly-open + N_CLOSED clearly-closed frames, spread across the
    front-distance range within each state (sample the spectrum, not one
    micro-state)
  - extract each frame as a receptor (protein selection, as 05a/06i/06o)
  - Meeko -> pdbqt
  - dock every ligand in LIGANDS into each frame, SHARED box, locked params+seed
  - manifest records per frame: variant, replica, frame, pocket_open_state,
    pocket_front_dist_nm  -> scores join to openness afterwards (06s).

Prepares configs + run script + manifest; DOES NOT run vina. Run in venv, check
the manifest counts, THEN nohup the run script.

Usage:
    python analysis/06r_openness_ensemble_all.py
    # inspect: dock/ROS1/q2022p_vina/ensemble_all/manifest.csv
    nohup bash dock/ROS1/q2022p_vina/ensemble_all/run_vina_ensemble.sh \
        > /tmp/ensemble_all_dock.log 2>&1 &
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import MDAnalysis as mda

BASE    = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
DOCK    = BASE / "dock" / "ROS1"
VINA    = DOCK / "q2022p_vina"
LIG_DIR = VINA / "ligand_pdbqt"

FRAME_METRICS = RESULTS / "step3a_frame_metrics.csv"
# NOTE: F_paths.csv is deliberately NOT used for folder lookup. It lists 117 rows
# (39 variants x 3 replicas) but the metrics use 114 trajectories (one variant
# dropped -> 38), so positional F[traj_index] is misaligned and returns the wrong
# folder for shifted indices. The correct per-frame folder is read straight from
# the metrics 'trajectory' column in the loop below.

# ── What to dock ───────────────────────────────────────────────────────────────
VARIANTS = ["WT", "S1986F", "Q2022P", "Q2022P_S1986F"]

# All six ligands from the step6k heatmap -> every active docking number becomes
# ensemble-based. For a quick ~2 h dry run first, swap in the two-ligand line.
LIGANDS = ["ceritinib", "crizotinib", "lorlatinib",
           "repotrectinib", "zidesamtinib", "cabozantinib"]
# LIGANDS = ["cabozantinib", "lorlatinib"]   # paper's two, fast sanity run

N_OPEN   = 8      # clearly-open frames per variant  (bump both to 10 for 20/var)
N_CLOSED = 8      # clearly-closed frames per variant

# ── Docking box + params (LOCKED — identical to 06o / cabozantinib shared box) ──
SITE_REAL = [2026, 2102, 2103, 2004, 2075, 1980]   # gatekeeper,DFG,type-II,cat-K
OFFSET    = 1933
# NOTE box lineage: 28.7 = the cabozantinib/06o shared box. The step6k heatmap
# used 25.7 (sized over active receptors only). Kept at 28.7 here so 06o's
# Q2022P-vs-double ensemble numbers are directly comparable; 06t flags the 25.7
# vs 28.7 difference when comparing back to step6k.
BOX_SIZE  = (28.7, 30.3, 30.2)
SEED      = 42
EXH, NMODES, ERANGE = 16, 20, 4
ATOM_SEL  = "protein"

OUT     = VINA / "ensemble_all"
REC_PDB = OUT / "receptor_pdb"
REC_PQT = OUT / "receptor_pdbqt"
CFG     = OUT / "configs"
VOUT    = OUT / "vina_outputs"
VLOG    = OUT / "vina_logs"
for d in (REC_PDB, REC_PQT, CFG, VOUT, VLOG):
    d.mkdir(parents=True, exist_ok=True)


def pick_frames(sub):
    """N_OPEN open + N_CLOSED closed frames, spread across front-distance."""
    picks = []
    for state, n in [(1, N_OPEN), (0, N_CLOSED)]:
        s = sub[sub["pocket_open_state"] == state]
        if len(s) == 0:
            print(f"    WARNING: no frames with open_state={state}")
            continue
        s = s.sort_values("pocket_front_dist_nm")
        idx = np.linspace(0, len(s) - 1, min(n, len(s))).astype(int)
        picks.append(s.iloc[idx])
    return pd.concat(picks) if picks else pd.DataFrame()


def main():
    print(">>> 06r v2 — folder-from-metrics + per-frame in-range assert <<<")
    fm = pd.read_csv(FRAME_METRICS)

    # ensure every ligand pdbqt is present; fail loudly and early if not
    missing = []
    for lig in LIGANDS:
        dst = LIG_DIR / f"{lig}.pdbqt"
        if not dst.exists():
            src = DOCK / "ligands_pdbqt" / f"{lig}.pdbqt"
            if src.exists():
                shutil.copy2(src, dst)
            else:
                missing.append(lig)
    if missing:
        raise SystemExit(
            "missing ligand pdbqt for: " + ", ".join(missing) +
            f"\n  looked in {LIG_DIR} and {DOCK/'ligands_pdbqt'}")

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
            # folder straight from the metrics row — never positional F_paths.
            fol    = Path(row["trajectory"])
            frame  = int(row["frame_index_local"])   # == frame_index_original here
            rep    = int(row["replica"])
            openst = int(row["pocket_open_state"])
            front  = float(row["pocket_front_dist_nm"])
            top = str(fol / f"{fol.parent.name}-MD-prot.pdb")
            xtc = str(fol / f"{fol.parent.name}-MD-prot.xtc")
            state_tag = "open" if openst == 1 else "closed"
            tag = f"{v}_rep{rep}_f{frame}_{state_tag}"
            pdb = REC_PDB / f"{tag}.pdb"

            u = mda.Universe(top, xtc)
            # hard guard: a wrong/out-of-range frame can never slip through silently
            assert frame < len(u.trajectory), (
                f"{tag}: frame {frame} out of range (traj len "
                f"{len(u.trajectory)}) for {fol}")
            u.trajectory[frame]
            u.select_atoms(ATOM_SEL).write(str(pdb))

            pqt = REC_PQT / f"{tag}.pdbqt"
            subprocess.run(["mk_prepare_receptor.py", "--read_pdb", str(pdb),
                            "-o", str(pqt.with_suffix("")), "--write_pdbqt", str(pqt)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # box centre from THIS frame's pocket residues (scheme auto-detect)
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

    run_sh = OUT / "run_vina_ensemble.sh"
    run_sh.write_text("\n".join(run) + "\n")
    pd.DataFrame(manifest).to_csv(OUT / "manifest.csv", index=False)
    n_dock = len(manifest) * len(LIGANDS)
    print(f"\nwrote {len(manifest)} receptors x {len(LIGANDS)} ligands = {n_dock} docks")
    print(f"  variants: {', '.join(VARIANTS)}")
    print(f"  ligands : {', '.join(LIGANDS)}")
    print(f"manifest  : {OUT/'manifest.csv'}")
    print(f"run script: {run_sh}")
    print("\nInspect the manifest, then launch:")
    print(f"  nohup bash {run_sh} > /tmp/ensemble_all_dock.log 2>&1 &")


if __name__ == "__main__":
    main()