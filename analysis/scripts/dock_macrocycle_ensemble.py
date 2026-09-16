#!/usr/bin/env python3
"""
dock_macrocycle_ensemble.py

Rigid conformer-ENSEMBLE docking for the two macrocycles Meeko cannot open
(lorlatinib, zidesamtinib). Their verbose prep shows every ring bond as
breakable:False (aromatic / chorded by fused aromatics), so AutoDock's flexible
ring-opening is impossible and a single rigid conformer can miss the bound-state
ring shape. The documented fix for un-openable rings (Glide MCS; D3R GC4;
Meeko/AutoDock macrocycle papers) is to dock a pre-generated ENSEMBLE of ring
conformers rigidly and keep the best-scoring one per receptor frame.

Per ligand:
  1. ETKDGv3 (macrocycle-aware) -> up to N diverse ring conformers, MMFF-optimised,
     lowest-energy kept (RMSD-pruned so they are genuinely different shapes).
  2. prep each conformer rigid with Meeko (these rings can't be opened anyway).
  3. for each of the 64 receptor frames, dock ALL conformers (rigid, identical
     box/seed/params to the main run) and keep the best (most negative) as that
     frame's pose/score.
  4. write the winning pose into vina_outputs_all3d/<tag>__<lig>.pdbqt, i.e. it
     OVERWRITES the single-conformer result in place. Everything downstream
     (16-frame medians, heatmap, figures) reads the same paths, unchanged.

Resumable: a frame whose final output already carries the ENSEMBLE marker is
skipped, so you can re-run after an interruption. Winning conformer + score per
frame is logged to results/ROS1/macrocycle_ensemble_log.csv.

Usage (venv on; run under nohup - it is long):
    nohup python analysis/scripts/dock_macrocycle_ensemble.py > macro_ensemble.out 2>&1 &
    # one ligand only:
    python analysis/scripts/dock_macrocycle_ensemble.py lorlatinib
"""
import os
import sys
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

BASE    = Path.home() / "Molecular_Dynamics_analysis"
ENS     = BASE / "dock" / "ROS1" / "q2022p_vina" / "ensemble_all"
CFGDIR  = ENS / "configs_all3d"
OUTDIR  = ENS / "vina_outputs_all3d"
RAW3D   = BASE / "dock" / "ROS1" / "ligands_raw_3d"
CONFDIR = BASE / "dock" / "ROS1" / "q2022p_vina" / "ligand_pdbqt" / "ensemble_confs"
LOGCSV  = BASE / "results" / "ROS1" / "macrocycle_ensemble_log.csv"
VINA    = os.path.expanduser("~/bin/vina")

N_CONF  = 15                       # target; fewer kept if the ring yields fewer shapes
MARKER  = "REMARK ENSEMBLE_BEST"
LIGANDS = sys.argv[1:] or ["lorlatinib", "zidesamtinib"]


def gen_conformers(lig):
    """Up to N_CONF diverse, MMFF-optimised ring conformers -> rigid pdbqt files."""
    CONFDIR.mkdir(parents=True, exist_ok=True)
    m = Chem.AddHs(Chem.MolFromMolFile(str(RAW3D / f"{lig}_3d.sdf")))
    p = AllChem.ETKDGv3()
    p.randomSeed = 42
    try:
        p.useMacrocycleTorsions = True
    except Exception:
        pass
    p.pruneRmsThresh = 0.5
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=N_CONF * 3, params=p))
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(m)
        energies = [e for _conv, e in res]
    except Exception:
        AllChem.UFFOptimizeMoleculeConfs(m)
        energies = [0.0] * len(cids)
    order = sorted(range(len(cids)), key=lambda i: energies[i])[:N_CONF]

    paths = []
    for k, idx in enumerate(order):
        csdf = CONFDIR / f"{lig}_conf{k:02d}.sdf"
        csdf.write_text(Chem.MolToMolBlock(m, confId=cids[idx]))
        cpdbqt = CONFDIR / f"{lig}_conf{k:02d}.pdbqt"
        subprocess.run(["mk_prepare_ligand.py", "-i", str(csdf), "-o", str(cpdbqt)],
                       capture_output=True)
        if cpdbqt.exists():
            paths.append(cpdbqt)
    return paths


def parse_cfg(path):
    d = {}
    for ln in open(path):
        if "=" in ln:
            k, v = ln.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def best_score(pdbqt):
    for ln in open(pdbqt):
        if ln.startswith("REMARK VINA RESULT"):
            return float(ln.split()[3])
    return None


def dock(receptor, ligand, c, out):
    subprocess.run([VINA,
                    "--receptor", receptor, "--ligand", str(ligand),
                    "--center_x", c["center_x"], "--center_y", c["center_y"],
                    "--center_z", c["center_z"], "--size_x", c["size_x"],
                    "--size_y", c["size_y"], "--size_z", c["size_z"],
                    "--seed", c.get("seed", "42"),
                    "--exhaustiveness", c.get("exhaustiveness", "16"),
                    "--num_modes", c.get("num_modes", "20"),
                    "--energy_range", c.get("energy_range", "4"),
                    "--out", str(out)], capture_output=True)


def main():
    LOGCSV.parent.mkdir(parents=True, exist_ok=True)
    newlog = not LOGCSV.exists()
    logf = open(LOGCSV, "a")
    if newlog:
        logf.write("ligand,tag,best_conf,best_score,n_conf\n")

    for lig in LIGANDS:
        print(f"=== {lig}: generating conformers ===", flush=True)
        confs = gen_conformers(lig)
        cfgs = sorted(CFGDIR.glob(f"*__{lig}.txt"))
        print(f"    {len(confs)} rigid conformers x {len(cfgs)} frames "
              f"= {len(confs) * len(cfgs)} docks", flush=True)
        if not confs:
            print(f"    !! no conformers prepped for {lig} - skipping", flush=True)
            continue

        for i, cfg in enumerate(cfgs, 1):
            tag = cfg.name[:-4]
            final = OUTDIR / f"{tag}.pdbqt"
            if final.exists() and any(l.startswith(MARKER) for l in open(final)):
                continue                      # already upgraded -> resume-safe
            c = parse_cfg(cfg)
            best = (None, None, None)         # (score, pose_path, conf_stem)
            with tempfile.TemporaryDirectory() as td:
                for cp in confs:
                    outp = Path(td) / f"{cp.stem}.pdbqt"
                    dock(c["receptor"], cp, c, outp)
                    s = best_score(outp) if outp.exists() else None
                    if s is not None and (best[0] is None or s < best[0]):
                        best = (s, outp.read_text(), cp.stem)
                if best[0] is not None:
                    with open(final, "w") as fh:
                        fh.write(f"{MARKER}: {best[2]} score {best[0]:.3f} "
                                 f"over {len(confs)} confs\n")
                        fh.write(best[1])
                    logf.write(f"{lig},{tag},{best[2]},{best[0]:.3f},{len(confs)}\n")
                    logf.flush()
            if i % 8 == 0:
                print(f"    {lig}: {i}/{len(cfgs)} frames done", flush=True)
        print(f"=== {lig}: done ===", flush=True)

    logf.close()
    print("all ligands complete", flush=True)


if __name__ == "__main__":
    main()