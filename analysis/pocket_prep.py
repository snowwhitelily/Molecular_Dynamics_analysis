"""
pocket_prep.py -- MDAnalysis prep for the Task 10 pocket figure (runs in venv).

Writes plain files the PyMOL render step reads without MDAnalysis:
    prep_<SEL>_<DRUG>_drug.pdb        frame-corrected drug pose (NTL frame)
    prep_<SEL>_<DRUG>_resids.txt      pocket residue list (GRO), comma-sep
    prep_avg_backbone_<VARIANT>.pdb   averaged whole-protein backbone (cartoon)
    prep_pocket_atoms_<VARIANT>.pdb   averaged ALL-HEAVY-ATOM pocket positions   <-- NEW

Christa's Figure 2 plots the *average atomic positions* of every heavy atom of
the pocket residues (not just Ci), so each residue's atoms cluster into a small
blob and the mutated side chain shows up as a lump in the pocket. We reproduce
that with prep_pocket_atoms_<VARIANT>.pdb. The mutation sites (1986, 2022 ->
GRO 53, 89) are force-included so the "round thing" always appears.

Usage (venv on):
    python pocket_prep.py SELECTION DRUG [CUTOFF]
"""
import os, sys
sys.path.insert(0, os.path.join(os.environ['HOME'], 'Molecular_Dynamics_analysis', 'analysis'))
import pocket_lib as PL
import MDAnalysis as mda
from MDAnalysis.analysis import align as _mdaalign

SELECTION = sys.argv[1]; DRUG = sys.argv[2]
CUTOFF = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

BASE = os.path.join(os.environ['HOME'], 'Molecular_Dynamics_analysis')
TRAJ = os.path.join(BASE, 'trajectories/ros1_prepared_final')
DOCK = os.path.join(BASE, 'dock/ROS1/q2022p_vina/vina_outputs')
RECEPTOR = os.path.join(BASE, 'dock/ROS1/q2022p_subset_receptors/actout_ctlfit/WT/'
                              'WT_actout_ctlfit_dominant_cluster0_rep.pdb')
FIGS = os.path.join(BASE, 'figures/ROS1/ros1_prepared_final/pocket_box')
os.makedirs(FIGS, exist_ok=True)
VARIANTS = ['WT', 'S1986F', 'Q2022P', 'Q2022P_S1986F']

# mutation sites (GRO = real - 1933): 1986 -> 53, 2022 -> 89. Always shown.
MUT_GRO = [53, 89]

DRUG_PDBQT = os.path.join(DOCK, f'WT_actout_ctlfit_dominant_cluster0_rep__{DRUG}.pdbqt')


def _ad_to_element(adtype):
    """Map an AutoDock atom type (e.g. 'A','C','OA','NA','HD','Cl') to element."""
    t = adtype.strip().upper()
    if t in ('CL', 'BR'):
        return t.capitalize()
    first = t[0] if t else 'C'
    return {'A': 'C'}.get(first, first.capitalize())


def read_pose1_elements(path):
    """Elements for pose 1, in the SAME atom order as PL.read_pdbqt_pose1."""
    elems, recording = [], False
    for line in open(path):
        s = line.rstrip('\n')
        if s.startswith('MODEL'):
            recording = s.split()[-1] == '1'; continue
        if s.startswith('ENDMDL'):
            if recording:
                break
            continue
        if recording and s.startswith(('ATOM', 'HETATM')):
            tok = s.split()
            elems.append(_ad_to_element(tok[-1]) if tok else 'C')
    return elems


def averaged_pocket_atoms(traj_dir, variant, wt_ref, resids):
    """Average heavy-atom positions of the pocket residues, each frame aligned
    to the WT N-lobe (same frame as the cartoon and the drug)."""
    rsel = '(resid ' + ' '.join(str(r) for r in resids) + ') and not name H*'
    csum, n, wu = None, 0, None
    for top, trj in PL.replica_paths(traj_dir, variant):
        u = mda.Universe(top, trj); wu = u
        ag = u.select_atoms(rsel)
        for _ in u.trajectory:
            _mdaalign.alignto(u, wt_ref, select=PL.NTL_SEL, weights='mass')
            p = ag.positions
            csum = p.copy() if csum is None else csum + p
            n += 1
    wu.trajectory[0]
    ag = wu.select_atoms(rsel)
    ag.positions = csum / n
    return ag, n


# ---- drug: pose, elements, frame-correction (geometry unchanged) -------------
wt = PL.ntl_reference(TRAJ, 'WT')
pose = PL.read_pdbqt_pose1(DRUG_PDBQT)
elems = read_pose1_elements(DRUG_PDBQT)
if len(elems) != len(pose):
    print(f'WARNING: {len(elems)} elements vs {len(pose)} atoms; defaulting to C')
    elems = ['C'] * len(pose)
drug_ntl, rmsd = PL.correct_drug_to_ntl(pose, RECEPTOR, wt)
print(f'NTL-only frame-correction RMSD {rmsd:.3f} A, {len(drug_ntl)} drug atoms')

# ---- pocket residue list (force-include the mutation sites) ------------------
if SELECTION == 'motif':
    resids = PL.pocket_resids('motif')
elif SELECTION == 'distance':
    resids = PL.pocket_resids('distance', wt, drug_ntl, cutoff=CUTOFF)
    print(f'distance pocket ({CUTOFF} A): {len(resids)} residues (GRO) {resids}')
else:
    raise SystemExit("SELECTION must be motif or distance")
resids = sorted(set(resids) | set(MUT_GRO))
print(f'pocket residue count (with mutation sites) = {len(resids)}')

# ---- write drug pdb (real elements -> CPK heteroatom colours) ----------------
with open(os.path.join(FIGS, f'prep_{SELECTION}_{DRUG}_drug.pdb'), 'w') as fh:
    for i, ((x, y, z), el) in enumerate(zip(drug_ntl, elems), 1):
        aname = f'{el}{i}'
        fh.write(f'HETATM{i:>5} {aname:<4} LIG A   1    '
                 f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2}\n')
    fh.write('END\n')
open(os.path.join(FIGS, f'prep_{SELECTION}_{DRUG}_resids.txt'), 'w').write(
    ','.join(str(r) for r in resids) + '\n')

# ---- averaged backbone cartoon (unchanged) -----------------------------------
for v in VARIANTS:
    out = os.path.join(FIGS, f'prep_avg_backbone_{v}.pdb')
    if os.path.exists(out):
        print(f'cartoon {v}: exists, skip'); continue
    bb, n = PL.averaged_backbone(TRAJ, v, wt)
    bb.write(out)
    print(f'cartoon {v}: averaged {n} frames -> {os.path.basename(out)}')

# ---- averaged all-heavy-atom pocket (NEW: the Christa-style spheres) ----------
for v in VARIANTS:
    outp = os.path.join(FIGS, f'prep_pocket_atoms_{v}.pdb')
    if os.path.exists(outp):
        print(f'pocket-atoms {v}: exists, skip'); continue
    ag, n = averaged_pocket_atoms(TRAJ, v, wt, resids)
    ag.write(outp)
    print(f'pocket-atoms {v}: averaged {n} frames, {ag.n_atoms} atoms -> {os.path.basename(outp)}')

print('prep done')