#!/usr/bin/env python3
"""
06i_s1986f_multiframe.py

Robustness check for the S1986F ACTIVE cabozantinib score (-8.74, the weakest
active binder). S1986F's active ensemble is conformationally diffuse -- one
broad DBSCAN cluster 0 (2845 frames), and HDBSCAN finds no structure at all --
so the single dominant-cluster medoid may not be representative. This docks N
frames spanning that cluster into the SAME shared box and reports the spread.

Mirrors 05a exactly for frame extraction (MDAnalysis, protein selection,
frame_index_original), so frames are comparable to the original medoid receptor.

Steps:
  1. read cluster labels; take S1986F, DBSCAN cluster 0 (the column 05a uses).
  2. pick N frames spread across the cluster in PC-score space (not by time),
     so they sample the conformational range; include the medoid as reference.
  3. write each as a receptor PDB (protein selection, like 05a).
  4. Meeko -> pdbqt.
  5. write a Vina config per frame with the SHARED box + locked params + fixed
     seed, and a run script.

This script PREPARES; it does not run Vina (do that with the printed run script,
same as the main pipeline). Run in the venv.

Usage:
    python analysis/06i_s1986f_multiframe.py            # default N=10
    N_FRAMES=12 python analysis/06i_s1986f_multiframe.py
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import MDAnalysis as mda

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
DOCK = BASE / "dock" / "ROS1"
VINA = DOCK / "q2022p_vina"
LIG = VINA / "ligand_pdbqt" / "cabozantinib.pdbqt"

LABELS = RESULTS / "actout_ctlfit_cluster_labels.csv"
CLUSTER_COL = "cluster_db"          # the column 05a uses by default
MUTANT = "S1986F"
PREFIX = "actout_ctlfit"

N_FRAMES = int(os.environ.get("N_FRAMES", 10))
ATOM_SEL = "protein"                # same as 05a default

# shared box (from 06g preview: identical size for all; centre re-derived per
# receptor from the six pocket residues -- we recompute per frame below)
SITE_REAL = [2026, 2102, 2103, 2004, 2075, 1980]
OFFSET = 1933
BOX_SIZE = (28.7, 30.3, 30.2)       # matches 06g common box
MARGIN = 4.0
SEED = 42
EXH, NMODES, ERANGE = 16, 20, 4

OUT_REC = VINA / "s1986f_multiframe" / "receptor_pdb"
OUT_PQT = VINA / "s1986f_multiframe" / "receptor_pdbqt"
CFG_DIR = VINA / "s1986f_multiframe" / "configs"
OUT_DIR = VINA / "s1986f_multiframe" / "vina_outputs"
LOG_DIR = VINA / "s1986f_multiframe" / "vina_logs"
for d in (OUT_REC, OUT_PQT, CFG_DIR, OUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_scores():
    """Find PC score columns for spreading frames across the cluster.

    02c/02d saved scores as actout_ctlfit_scores.npy aligned to the label rows.
    If unavailable, fall back to even sampling by row order (time)."""
    scores_f = RESULTS / f"{PREFIX}_scores.npy"
    if scores_f.exists():
        return np.load(scores_f, allow_pickle=True)
    return None


def pick_frames(sub, scores, n):
    """Pick n frames spread across the cluster in PC space (or by order)."""
    idx = sub.index.to_numpy()
    if scores is not None and scores.shape[0] >= (idx.max() + 1):
        X = np.asarray(scores)[idx][:, :2]        # first 2 PCs
        centre = X.mean(0)
        # order by distance from centre, then take evenly-spaced ranks so we
        # sample from core out to edge of the diffuse cluster
        order = np.argsort(np.sum((X - centre) ** 2, axis=1))
        picks = order[np.linspace(0, len(order) - 1, n).astype(int)]
        medoid_local = int(order[0])
        return idx[picks], idx[medoid_local]
    # fallback: evenly spaced by row order
    picks = np.linspace(0, len(idx) - 1, n).astype(int)
    return idx[picks], idx[0]


def pocket_centre(pdb):
    """Recompute the box centre from this frame's six pocket residues (REAL)."""
    u = mda.Universe(str(pdb))
    sel = "resid " + " ".join(str(r) for r in SITE_REAL) + " and not name H*"
    ag = u.select_atoms(sel)
    if ag.n_atoms == 0:
        # frames use GRO numbering? try real-1933
        sel = "resid " + " ".join(str(r - OFFSET) for r in SITE_REAL) + " and not name H*"
        ag = u.select_atoms(sel)
    pos = ag.positions
    return (pos.min(0) + pos.max(0)) / 2.0


def main():
    if not LABELS.exists():
        raise SystemExit(f"missing {LABELS}")
    df = pd.read_csv(LABELS)
    sub = df[(df["mutant"] == MUTANT) & (df[CLUSTER_COL] == 0)].copy()
    if sub.empty:
        raise SystemExit(f"no {MUTANT} frames in {CLUSTER_COL}==0")
    print(f"{MUTANT}: {len(sub)} frames in {CLUSTER_COL} cluster 0 "
          f"(replicas {sorted(sub['replica'].unique())})")

    scores = load_scores()
    print("PC scores available for spread:" , scores is not None)

    picks, medoid_row = pick_frames(sub, scores, N_FRAMES)
    print(f"picked {len(picks)} frames; medoid row = {medoid_row}\n")

    # need pdb/xtc per frame; labels file lacks paths, so join via F_paths meta
    F = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
    meta = {}
    for i, folder in enumerate(F):
        p = Path(folder)
        meta[i] = (str(p / f"{p.parent.name}-MD-prot.pdb"),
                   str(p / f"{p.parent.name}-MD-prot.xtc"))

    run_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    manifest = []
    for k, row_idx in enumerate(picks):
        r = df.loc[row_idx]
        ti = int(r["traj_index"]); frame = int(r["frame_index_original"])
        top, xtc = meta[ti]
        tag = f"S1986F_frame{k:02d}_traj{ti}_f{frame}"
        is_medoid = (row_idx == medoid_row)
        if is_medoid:
            tag += "_MEDOID"
        pdb = OUT_REC / f"{tag}.pdb"

        u = mda.Universe(top, xtc)
        u.trajectory[frame]
        u.select_atoms(ATOM_SEL).write(str(pdb))

        pqt = OUT_PQT / f"{tag}.pdbqt"
        subprocess.run(["mk_prepare_receptor.py", "--read_pdb", str(pdb),
                        "-o", str(pqt.with_suffix("")), "--write_pdbqt", str(pqt)],
                       check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

        cx, cy, cz = pocket_centre(pdb)
        cfg = CFG_DIR / f"{tag}__cabozantinib.txt"
        out = OUT_DIR / f"{tag}__cabozantinib.pdbqt"
        log = LOG_DIR / f"{tag}__cabozantinib.log"
        cfg.write_text(
            f"receptor = {pqt}\nligand = {LIG}\n"
            f"center_x = {cx:.2f}\ncenter_y = {cy:.2f}\ncenter_z = {cz:.2f}\n"
            f"size_x = {BOX_SIZE[0]}\nsize_y = {BOX_SIZE[1]}\nsize_z = {BOX_SIZE[2]}\n"
            f"out = {out}\nseed = {SEED}\nexhaustiveness = {EXH}\n"
            f"num_modes = {NMODES}\nenergy_range = {ERANGE}\n")
        run_lines.append(f"vina --config {cfg} 2>&1 | tee {log}")
        manifest.append(dict(tag=tag, traj=ti, frame=frame,
                             replica=int(r["replica"]), medoid=is_medoid))
        print(f"  {tag:40s} centre ({cx:.1f} {cy:.1f} {cz:.1f})")

    run_sh = VINA / "s1986f_multiframe" / "run_vina_s1986f_multiframe.sh"
    run_sh.write_text("\n".join(run_lines) + "\n")
    pd.DataFrame(manifest).to_csv(
        VINA / "s1986f_multiframe" / "manifest.csv", index=False)
    print(f"\nwrote {len(picks)} configs + receptors")
    print(f"run script: {run_sh}")
    print("\nAfter running, collect scores with:")
    print(f"  for f in {OUT_DIR}/*.pdbqt; do "
          f"grep -m1 'VINA RESULT' $f | awk '{{print FILENAME, $4}}'; done")


if __name__ == "__main__":
    main()