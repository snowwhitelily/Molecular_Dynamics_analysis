#!/usr/bin/env python3
"""
06k_prepare_heatmap_sharedbox.py

Re-dock ALL heatmap ligands into the 4 focus active receptors with the SAME
shared box, so the whole active-form heatmap is one consistent method (no old
per-ligand box, no N/A rescue asterisk).

Focus variants (publication set): WT, Q2022P, Q2022P_S1986F, S1986F
                                  (Q2022P_S1986Y dropped)
Ligands: cabozantinib + the 5 originals (ceritinib, crizotinib, lorlatinib,
         repotrectinib, zidesamtinib)
Receptors: ACTIVE only (the heatmap is active-form). Inactive stays
           cabozantinib-only elsewhere.

Shared box: identical size for all, centred per receptor on the six pocket
residues (same rule as 06g). Locked params + fixed seed.

Writes configs + a run script. Does NOT run Vina. Run in venv.

Usage:
    python analysis/06k_prepare_heatmap_sharedbox.py
    nohup bash dock/ROS1/q2022p_vina/run_vina_heatmap_sharedbox.sh \
        > /tmp/heatmap_dock.log 2>&1 &
"""
from pathlib import Path

BASE = Path.home() / "Molecular_Dynamics_analysis"
DOCK = BASE / "dock" / "ROS1"
VINA = DOCK / "q2022p_vina"
ACT_REC_DIR = DOCK / "receptors_pdbqt"
LIG_DIR = VINA / "ligand_pdbqt"
LIG_SRC = DOCK / "ligands_pdbqt"

VARIANTS = ["WT", "Q2022P", "Q2022P_S1986F", "S1986F"]
LIGANDS = ["cabozantinib", "ceritinib", "crizotinib", "lorlatinib",
           "repotrectinib", "zidesamtinib"]

SITE_REAL = [2026, 2102, 2103, 2004, 2075, 1980]
OFFSET = 1933
MARGIN = 4.0
SEED = 42
EXH, NMODES, ERANGE = 16, 20, 4

CFG_DIR = VINA / "configs_heatmap_sharedbox"
OUT_DIR = VINA / "vina_outputs_heatmap_sharedbox"
LOG_DIR = VINA / "vina_logs_heatmap_sharedbox"
for d in (CFG_DIR, OUT_DIR, LOG_DIR, LIG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def read_atoms(pdbqt):
    with open(pdbqt) as fh:
        for ln in fh:
            if ln[:6] not in ("ATOM  ", "HETATM"):
                continue
            try:
                resid = int(ln[22:26])
                x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
            except ValueError:
                continue
            yield resid, x, y, z


def detect_scheme(pdbqt):
    for resid, *_ in read_atoms(pdbqt):
        return "GRO" if resid < 1000 else "REAL"
    raise SystemExit(f"no atoms in {pdbqt}")


def pocket_span_centre(pdbqt):
    scheme = detect_scheme(pdbqt)
    wanted = ([r - OFFSET for r in SITE_REAL] if scheme == "GRO"
              else list(SITE_REAL))
    pts = [(x, y, z) for resid, x, y, z in read_atoms(pdbqt) if resid in wanted]
    if not pts:
        raise SystemExit(f"no pocket atoms in {pdbqt} ({scheme})")
    xs, ys, zs = zip(*pts)
    lo = (min(xs), min(ys), min(zs)); hi = (max(xs), max(ys), max(zs))
    centre = tuple((l + h) / 2 for l, h in zip(lo, hi))
    span = tuple(h - l for l, h in zip(lo, hi))
    return centre, span


def main():
    # ensure ligand pdbqt are in the dir configs reference
    import shutil
    for lig in LIGANDS:
        dst = LIG_DIR / f"{lig}.pdbqt"
        if not dst.exists():
            src = LIG_SRC / f"{lig}.pdbqt"
            if not src.exists():
                raise SystemExit(f"missing ligand {src}")
            shutil.copy2(src, dst)

    receptors = {}
    for v in VARIANTS:
        p = ACT_REC_DIR / f"{v}_actout_ctlfit_dominant_cluster0_rep.pdbqt"
        if not p.exists():
            raise SystemExit(f"missing active receptor {p}")
        receptors[v] = p

    # pass 1: common box size across the 4 active receptors
    info = {}
    max_span = [0.0, 0.0, 0.0]
    for v, p in receptors.items():
        centre, span = pocket_span_centre(p)
        info[v] = (p, centre, span)
        max_span = [max(m, s) for m, s in zip(max_span, span)]
    box = tuple(round(m + 2 * MARGIN, 1) for m in max_span)
    print(f"common box size = {box[0]} {box[1]} {box[2]}  (4 active receptors)")
    print(f"{len(VARIANTS)} variants x {len(LIGANDS)} ligands = "
          f"{len(VARIANTS)*len(LIGANDS)} docks  seed={SEED}\n")

    run = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for v, (p, centre, span) in info.items():
        for lig in LIGANDS:
            stem = f"{v}_actout_ctlfit_dominant_cluster0_rep"
            cfg = CFG_DIR / f"{stem}__{lig}.txt"
            out = OUT_DIR / f"{stem}__{lig}.pdbqt"
            log = LOG_DIR / f"{stem}__{lig}.log"
            cfg.write_text(
                f"receptor = {p}\nligand = {LIG_DIR / f'{lig}.pdbqt'}\n"
                f"center_x = {centre[0]:.2f}\ncenter_y = {centre[1]:.2f}\n"
                f"center_z = {centre[2]:.2f}\n"
                f"size_x = {box[0]}\nsize_y = {box[1]}\nsize_z = {box[2]}\n"
                f"out = {out}\nseed = {SEED}\nexhaustiveness = {EXH}\n"
                f"num_modes = {NMODES}\nenergy_range = {ERANGE}\n")
            run.append(f"vina --config {cfg} 2>&1 | tee {log}")
        print(f"  {v:16s} centre ({centre[0]:.1f} {centre[1]:.1f} {centre[2]:.1f})")

    run_sh = VINA / "run_vina_heatmap_sharedbox.sh"
    run_sh.write_text("\n".join(run) + "\n")
    print(f"\nwrote {len(VARIANTS)*len(LIGANDS)} configs to {CFG_DIR}")
    print(f"run script: {run_sh}")


if __name__ == "__main__":
    main()