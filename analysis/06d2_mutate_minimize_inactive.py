#!/usr/bin/env python3
"""
06d2_mutate_minimize_inactive.py

REPLACEMENT for the broken 06d. Builds inactive-form mutant receptors from
ROS1_WT_i.pdb, correctly, with verification. The old 06d used PyMOL's
mutagenesis wizard with a chain selector "/obj//A/resid" -- but ROS1_WT_i.pdb
has a BLANK chain id, so the selection was empty, no mutation was applied, and
four identical unmutated copies were saved (all passing a DFG check that never
looked at residue identity). This script fixes that.

Pipeline per variant:
  1. Mutate with PDBFixer (blank-chain-safe; verified to build full side chains).
  2. VERIFY the residue identity actually changed -> hard fail if not.
  3. Restrained local minimization (OpenMM, amber14):
       - freeze ALL backbone atoms (N, CA, C, O) everywhere
       - freeze every atom whose residue lies >5 A from the mutation site
       - relax only the local side chains -> relieves the rotamer clash without
         letting the DFG-out scaffold drift.
  4. VERIFY DFG stayed out: K1980-E1997-F2103 angle at vertex K1980 in ~55-90,
     F2103 chi1 trans (|chi1| > 140). Hard fail otherwise.
  5. Save clean PDB.

Run in the venv (has pdbfixer + openmm):
    source ~/.venvs/ros1_analysis_env/bin/activate
    cd ~/Molecular_Dynamics_analysis
    python analysis/06d2_mutate_minimize_inactive.py

Old 06d is left untouched. This writes the SAME output filenames into
inactive_ref/, overwriting the phantom mutants -- WT_i itself is never touched.
"""
import io
import sys
import math
from pathlib import Path

import numpy as np
from pdbfixer import PDBFixer
from openmm.app import (ForceField, Modeller, PDBFile, NoCutoff, HBonds,
                        Simulation)
from openmm import (LangevinIntegrator, CustomExternalForce, Platform, unit)

BASE = Path.home() / "Molecular_Dynamics_analysis"
INACTIVE_REF = BASE / "dock" / "ROS1" / "inactive_ref"
SRC_PDB = INACTIVE_REF / "ROS1_WT_i.pdb"
OUT_DIR = INACTIVE_REF

# variant -> list of (resid, wt_3letter, new_3letter)
VARIANTS = {
    "ROS1_Q2022P_i":        [(2022, "GLN", "PRO")],
    "ROS1_S1986F_i":        [(1986, "SER", "PHE")],
    "ROS1_Q2022P_S1986F_i": [(2022, "GLN", "PRO"), (1986, "SER", "PHE")],
    "ROS1_Q2022P_S1986Y_i": [(2022, "GLN", "PRO"), (1986, "SER", "TYR")],
}

MIN_RADIUS_NM = 0.5        # 5 A: atoms farther than this are frozen
RESTRAINT_K = 5000.0       # kJ/mol/nm^2 positional restraint (stiff = frozen)
MAX_MIN_ITERS = 500

# DFG-out verification (from old 06d)
DFG_ANGLE_MIN, DFG_ANGLE_MAX = 55.0, 90.0
CHI1_TRANS_MIN = 140.0


def load_fixer(path):
    with open(path) as fh:
        fixer = PDBFixer(pdbfile=io.StringIO(fh.read()))
    return fixer


def chain_id(fixer):
    return list(fixer.topology.chains())[0].id


def resname_at(topology, resid):
    for r in topology.residues():
        if r.id == str(resid):
            return r.name
    return None


def mutate(fixer, mutations):
    cid = chain_id(fixer)
    specs = [f"{wt}-{resid}-{new}" for resid, wt, new in mutations]
    fixer.applyMutations(specs, cid)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    return fixer


def restrained_minimize(topology, positions, mut_resids):
    ff = ForceField('amber14-all.xml')
    mod = Modeller(topology, positions)
    system = ff.createSystem(mod.topology, nonbondedMethod=NoCutoff,
                             constraints=HBonds)
    pos = np.array(mod.positions.value_in_unit(unit.nanometer))

    mut_idx = [a.index for a in mod.topology.atoms()
               if a.residue.id in {str(r) for r in mut_resids}]
    if not mut_idx:
        raise SystemExit("no atoms for mutated residues during minimization")
    centre = pos[mut_idx].mean(0)

    restraint = CustomExternalForce(
        "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", RESTRAINT_K)
    for p in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(p)

    n_free = 0
    for a in mod.topology.atoms():
        d = np.linalg.norm(pos[a.index] - centre)
        is_backbone = a.name in ("N", "CA", "C", "O")
        far = d > MIN_RADIUS_NM
        # FREEZE backbone (everywhere) and anything far from the mutation;
        # leave only nearby side-chain atoms free to relax
        if is_backbone or far:
            restraint.addParticle(a.index, [float(c) for c in pos[a.index]])
        else:
            n_free += 1
    system.addForce(restraint)

    integ = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                               0.002 * unit.picoseconds)
    sim = Simulation(mod.topology, system, integ,
                     Platform.getPlatformByName('CPU'))
    sim.context.setPositions(mod.positions)
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()
    sim.minimizeEnergy(maxIterations=MAX_MIN_ITERS)
    e1 = sim.context.getState(getEnergy=True).getPotentialEnergy()
    state = sim.context.getState(getPositions=True)
    return mod.topology, state.getPositions(), n_free, e0, e1


def measure_dfg(topology, positions):
    pos = np.array(positions.value_in_unit(unit.angstrom))
    idx = {}
    for a in topology.atoms():
        if a.name == "CA" and a.residue.id in ("1980", "1997", "2103"):
            idx[a.residue.id] = pos[a.index]
    if not all(k in idx for k in ("1980", "1997", "2103")):
        return None, None
    v1 = idx["1997"] - idx["1980"]
    v2 = idx["2103"] - idx["1980"]
    cos_a = (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    angle = math.degrees(math.acos(max(-1, min(1, cos_a))))

    # F2103 chi1 = N-CA-CB-CG
    names = {}
    for a in topology.atoms():
        if a.residue.id == "2103" and a.name in ("N", "CA", "CB", "CG"):
            names[a.name] = pos[a.index]
    chi1 = None
    if all(k in names for k in ("N", "CA", "CB", "CG")):
        chi1 = dihedral(names["N"], names["CA"], names["CB"], names["CG"])
    return angle, chi1


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    x = v @ w
    y = np.cross(b1, v) @ w
    return math.degrees(math.atan2(y, x))


def main():
    if not SRC_PDB.exists():
        sys.exit(f"missing {SRC_PDB}")

    # sanity: WT scaffold is DFG-out before we start
    wt = load_fixer(SRC_PDB)
    wt.findMissingResidues(); wt.findMissingAtoms(); wt.addMissingAtoms()
    wt.addMissingHydrogens(7.0)
    a0, c0 = measure_dfg(wt.topology, wt.positions)
    print(f"WT inactive scaffold: DFG angle {a0:.1f} deg, F2103 chi1 "
          f"{c0:.1f} deg" if a0 else "WT DFG unmeasurable")
    if a0 is None or not (DFG_ANGLE_MIN <= a0 <= DFG_ANGLE_MAX):
        sys.exit("WT scaffold fails DFG-out check; aborting")

    results = []
    for name, muts in VARIANTS.items():
        print(f"\n{'-'*60}\nBuilding {name}: "
              + ", ".join(f"{wt3}{rid}{new3}" for rid, wt3, new3 in muts))
        fixer = load_fixer(SRC_PDB)

        # confirm WT residue identity BEFORE mutating (catch numbering drift)
        for rid, wt3, new3 in muts:
            have = resname_at(fixer.topology, rid)
            if have != wt3:
                sys.exit(f"  expected {wt3} at {rid} but found {have}; "
                         f"numbering mismatch, aborting {name}")

        fixer = mutate(fixer, muts)

        # VERIFY mutation took (the guard the old script lacked)
        for rid, wt3, new3 in muts:
            got = resname_at(fixer.topology, rid)
            if got != new3:
                sys.exit(f"  MUTATION FAILED: {rid} is {got}, expected {new3}; "
                         f"refusing to save {name}")
            print(f"  verified: resid {rid} is now {got}")

        # restrained local minimization
        topo, newpos, n_free, e0, e1 = restrained_minimize(
            fixer.topology, fixer.positions, [rid for rid, _, _ in muts])
        print(f"  minimized {n_free} local side-chain atoms: "
              f"E {e0.value_in_unit(unit.kilojoule_per_mole):.0f} -> "
              f"{e1.value_in_unit(unit.kilojoule_per_mole):.0f} kJ/mol")

        # VERIFY DFG-out preserved after minimization
        angle, chi1 = measure_dfg(topo, newpos)
        dfg_ok = angle is not None and DFG_ANGLE_MIN <= angle <= DFG_ANGLE_MAX
        chi1_ok = chi1 is not None and abs(chi1) > CHI1_TRANS_MIN
        print(f"  DFG angle {angle:.1f} deg {'OK' if dfg_ok else 'FAIL'}; "
              f"F2103 chi1 {chi1:.1f} deg {'OK' if chi1_ok else 'FAIL'}")
        if not (dfg_ok and chi1_ok):
            sys.exit(f"  {name} lost DFG-out after minimization; not saving. "
                     f"Inspect manually (rotamer near DFG may need a different "
                     f"starting rotamer).")

        out = OUT_DIR / f"{name}.pdb"
        with open(out, "w") as fh:
            PDBFile.writeFile(topo, newpos, fh, keepIds=True)
        print(f"  saved {out}")
        results.append((name, angle, chi1))

    print(f"\n{'='*60}\nSUMMARY")
    for name, angle, chi1 in results:
        print(f"  {name:28s} DFG {angle:5.1f}  chi1 {chi1:7.1f}  OK")
    print("\nAll mutants built, verified, and minimized. Next: convert each to "
          "PDBQT (Meeko) and dock with the shared box (06g).")


if __name__ == "__main__":
    main()