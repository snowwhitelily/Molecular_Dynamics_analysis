#!/usr/bin/env python3
"""
06g_prepare_cabozantinib_sharedbox.py

Re-dock cabozantinib into ALL variants in BOTH conformations (active DFG-out
and inactive) with ONE comparable search box, so active-vs-inactive affinities
can be compared fairly. This replaces the mismatched-box situation where 06a
used a fixed ATP box (GRO-numbered active receptors) and 06b computed a
per-structure box (REAL-numbered inactive receptors).

WHAT IT DOES
  - loops over the 5 active + 5 inactive receptors
  - detects each receptor's numbering scheme automatically:
        chain 'X' and first resid 1  -> GRO numbering  (active)
        first resid ~1934            -> REAL numbering  (inactive)
  - selects the SAME six physical pocket residues in whichever scheme the file
    uses (gatekeeper, DFG D/F, F2004, F2075, catalytic K)
  - box CENTRE = midpoint of those residues' heavy atoms (per receptor, so the
    box always sits on the pocket regardless of coordinate frame)
  - box SIZE  = ONE identical size for every receptor = the largest pocket span
    seen across all 10 receptors + 2*MARGIN  (policy (a): fully comparable size)
  - writes a Vina config per receptor with IDENTICAL search params + fixed seed
  - DOES NOT TOUCH ANY .pdbqt  (configs only; atom typing preserved)

It does NOT run Vina. It writes configs and prints a run script. Preview first.

Usage (venv on):
    python 06g_prepare_cabozantinib_sharedbox.py            # all 10
    python 06g_prepare_cabozantinib_sharedbox.py --preview  # WT active+inactive only, print box, stop
"""
import sys
from pathlib import Path

BASE = Path.home() / "Molecular_Dynamics_analysis"
DOCK = BASE / "dock" / "ROS1"
VINA = DOCK / "q2022p_vina"
ACT_REC_DIR = DOCK / "receptors_pdbqt"                 # active pdbqt (GRO, chain X)
INA_REC_DIR = VINA / "receptor_pdbqt"                  # inactive pdbqt (REAL)
LIG = VINA / "ligand_pdbqt" / "cabozantinib.pdbqt"
CFG_DIR = VINA / "configs_sharedbox"
OUT_DIR = VINA / "vina_outputs_sharedbox"
LOG_DIR = VINA / "vina_logs_sharedbox"

OFFSET = 1933                        # real = GRO + 1933
MARGIN = 4.0                         # A headroom around pocket-lining atoms
                                     # (pocket already spans ATP + type-II crevice;
                                     #  4 A covers side chains without trawling surface)
MIN_SIDE = 22.0                      # floor: never smaller than this on any side
SEED = 42                            # fixed seed -> reproducible, fair
EXHAUSTIVENESS = 16
NUM_MODES = 20
ENERGY_RANGE = 4

# the six physical pocket residues, in REAL numbering
SITE_REAL = [2026, 2102, 2103, 2004, 2075, 1980]

PREVIEW = "--preview" in sys.argv

# receptor list: (tag, pdbqt path, conformation)
RECEPTORS = []
for v in ["WT", "Q2022P", "S1986F", "Q2022P_S1986F", "Q2022P_S1986Y"]:
    RECEPTORS.append(
        (f"{v}_actout_ctlfit_dominant_cluster0_rep",
         ACT_REC_DIR / f"{v}_actout_ctlfit_dominant_cluster0_rep.pdbqt",
         "active"))
    RECEPTORS.append(
        (f"ROS1_{v}_i",
         INA_REC_DIR / f"ROS1_{v}_i.pdbqt",
         "inactive"))

if PREVIEW:
    RECEPTORS = [r for r in RECEPTORS if r[0] in
                 ("WT_actout_ctlfit_dominant_cluster0_rep", "ROS1_WT_i")]


def read_atoms(pdbqt):
    """yield (resid_int, x, y, z) for ATOM/HETATM lines, fixed PDB columns."""
    with open(pdbqt) as fh:
        for ln in fh:
            if ln[:6] not in ("ATOM  ", "HETATM"):
                continue
            try:
                resid = int(ln[22:26])
                x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            except ValueError:
                continue
            yield resid, x, y, z


def detect_scheme(pdbqt):
    """GRO if first residue is 1, REAL if it's ~1934."""
    for resid, *_ in read_atoms(pdbqt):
        return "GRO" if resid < 1000 else "REAL"
    raise SystemExit(f"no atoms in {pdbqt}")


def pocket_span_and_centre(pdbqt):
    scheme = detect_scheme(pdbqt)
    wanted = ([r - OFFSET for r in SITE_REAL] if scheme == "GRO"
              else list(SITE_REAL))
    pts = [(x, y, z) for resid, x, y, z in read_atoms(pdbqt) if resid in wanted]
    if not pts:
        raise SystemExit(f"no pocket atoms found in {pdbqt} (scheme {scheme})")
    xs, ys, zs = zip(*pts)
    lo = (min(xs), min(ys), min(zs))
    hi = (max(xs), max(ys), max(zs))
    centre = tuple((l + h) / 2 for l, h in zip(lo, hi))
    span = tuple(h - l for l, h in zip(lo, hi))
    return scheme, centre, span, len(pts)


def main():
    for p in (LIG,):
        if not p.exists():
            raise SystemExit(f"missing ligand {p}")
    for _, path, _ in RECEPTORS:
        if not path.exists():
            raise SystemExit(f"missing receptor {path}")

    # pass 1: measure every pocket span to fix one common box size
    info = {}
    max_span = [0.0, 0.0, 0.0]
    for tag, path, conf in RECEPTORS:
        scheme, centre, span, natom = pocket_span_and_centre(path)
        info[tag] = (path, conf, scheme, centre, span, natom)
        max_span = [max(m, s) for m, s in zip(max_span, span)]
    box_size = tuple(round(m + 2 * MARGIN, 1) for m in max_span)

    print(f"common box size (max pocket span + 2x{MARGIN:.0f} A margin) = "
          f"{box_size[0]} {box_size[1]} {box_size[2]}")
    print(f"receptors: {len(info)}   seed={SEED}  exh={EXHAUSTIVENESS}  "
          f"modes={NUM_MODES}  erange={ENERGY_RANGE}\n")

    for d in (CFG_DIR, OUT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    run_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for tag, (path, conf, scheme, centre, span, natom) in info.items():
        cfg = CFG_DIR / f"{tag}__cabozantinib.txt"
        out = OUT_DIR / f"{tag}__cabozantinib.pdbqt"
        log = LOG_DIR / f"{tag}__cabozantinib.log"
        cfg.write_text(
            f"receptor = {path}\n"
            f"ligand = {LIG}\n"
            f"center_x = {centre[0]:.2f}\n"
            f"center_y = {centre[1]:.2f}\n"
            f"center_z = {centre[2]:.2f}\n"
            f"size_x = {box_size[0]}\n"
            f"size_y = {box_size[1]}\n"
            f"size_z = {box_size[2]}\n"
            f"out = {out}\n"
            f"seed = {SEED}\n"
            f"exhaustiveness = {EXHAUSTIVENESS}\n"
            f"num_modes = {NUM_MODES}\n"
            f"energy_range = {ENERGY_RANGE}\n")
        print(f"  {tag:52s} [{conf:8s} {scheme:4s}] "
              f"centre ({centre[0]:6.1f} {centre[1]:6.1f} {centre[2]:6.1f}) "
              f"{natom} pocket atoms")
        run_lines.append(
            f"vina --config {cfg} 2>&1 | tee {log}")

    run_sh = VINA / ("run_vina_cabozantinib_sharedbox"
                     + ("_preview" if PREVIEW else "") + ".sh")
    run_sh.write_text("\n".join(run_lines) + "\n")
    print(f"\nwrote {len(info)} configs to {CFG_DIR}")
    print(f"run script: {run_sh}")
    if PREVIEW:
        print("\nPREVIEW: only WT active + inactive. Check the two centres sit "
              "in the pocket and the box size looks sane, then run without "
              "--preview for all 10.")


if __name__ == "__main__":
    main()