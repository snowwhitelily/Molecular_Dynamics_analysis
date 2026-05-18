# ROS1 Analysis Pipeline
# Script 01: Data Loading and Preprocessing
#
# Loads all trajectory files, computes the common atom selection across
# all systems, aligns all trajectories to a shared reference, computes
# RMSD and RMSF, and saves all preprocessed outputs to results/ROS1/
# for use by downstream pipeline scripts.

import os
import pandas as pd
import numpy as np
import MDAnalysis as mda
from pathlib import Path
from MDAnalysis.analysis.dssp import DSSP
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from scripts.ros1_utils import (
    build_standard_masks,
    combine_masks,
    traj_frames_atoms,
)
from scripts.ros1_align import align_traj_to_ref
from scripts.ros1_io import (
    subsampled_len,
    get_frame_idx,
    xtc2array_varlen_by_indices,
    selection_keys_and_indices,
)
from scripts.timer import Timer

tim = Timer()
os.makedirs("figures", exist_ok=True)

# ============================================================
# Paths
# ============================================================
DATA_ROOT = Path.home() / "Molecular_Dynamics_analysis" / "trajectories" / "ros1_prepared_final"
BASE      = Path.home() / "Molecular_Dynamics_analysis"
RESULTS   = BASE / "results" / "ROS1"
RESULTS.mkdir(parents=True, exist_ok=True)

print("DATA_ROOT =", DATA_ROOT)

# ============================================================
# Atom selection
# Backbone heavy atoms used for alignment and PCA throughout
# ============================================================
selection = "protein and name N CA C O CB CG and not element H"

# ============================================================
# Auto-detect mutants and build simulation list F
# Directory structure: trajectories/<MUT>/<REP>/
# ============================================================
IGNORE = {"__pycache__", "orientation_cluster_reps", "figures", "plots", "tmp"}

mutants = []
for d in sorted(DATA_ROOT.iterdir()):
    if not d.is_dir() or d.name in IGNORE:
        continue
    if (d / "0" / f"{d.name}-MD-prot.pdb").exists():
        mutants.append(d.name)

print("Detected mutants:", mutants)
nrep = 3

F = []
for mut in mutants:
    for run in ("0", "1", "2"):
        run_dir = DATA_ROOT / mut / run
        pdb = run_dir / f"{mut}-MD-prot.pdb"
        xtc = run_dir / f"{mut}-MD-prot.xtc"
        if pdb.exists() and xtc.exists():
            F.append(str(run_dir))

print("Total simulations:", len(F))
print("Example:", F[0] if F else "None")

# Lock F so it cannot be overwritten downstream
F_paths = F.copy()
F = F_paths.copy()
print("[OK] F locked. len(F) =", len(F))

# ============================================================
# Frame index list
# Each trajectory may have a different number of frames.
# frame_idx_list stores the valid frame indices per trajectory.
# ============================================================
stride = 1
nframes_sub = min(subsampled_len(d) for d in F)
print("nframes_sub =", nframes_sub)

frame_idx_list = [get_frame_idx(d) for d in F]
print("frame_idx_list built for", len(frame_idx_list), "trajectories.")

# ============================================================
# Common atom intersection
# Find the set of (resid, atomname) keys present in all systems.
# This handles any minor topology differences between mutants.
# ============================================================
with tim("Computing common atom selection across all simulations"):
    all_keysets = []
    all_maps    = []

    for d in F:
        keys, k2i = selection_keys_and_indices(d, selection)
        all_keysets.append(keys)
        all_maps.append(k2i)

    common_keys = set.intersection(*all_keysets)
    print("Selection sizes (min/max):", min(len(s) for s in all_keysets), max(len(s) for s in all_keysets))
    print("Common atoms kept:", len(common_keys))

    if len(common_keys) == 0:
        raise ValueError("Common atom set is empty — check selection string or topology consistency.")

    common_keys_sorted = sorted(common_keys, key=lambda x: (x[0], x[1]))
    per_sim_indices    = [[k2i[k] for k in common_keys_sorted] for k2i in all_maps]

# Build metadata from the first simulation
first_run  = Path(F[0])
first_mut  = first_run.parent.name
u0         = mda.Universe(str(first_run / f"{first_mut}-MD-prot.pdb"))
meta       = u0.atoms[per_sim_indices[0]]
natoms     = len(meta)
print("Authoritative natoms (common):", natoms)

# ============================================================
# Load all trajectories (common atoms only, variable length)
# ============================================================
with tim("Reading all trajectories"):
    traj_xyz = []
    traj_pbc = []
    for d, idx in zip(F, per_sim_indices):
        fi = frame_idx_list[F.index(d)].tolist()
        xyz, pbc = xtc2array_varlen_by_indices(d, idx, frame_selection=fi)
        traj_xyz.append(xyz)
        traj_pbc.append(pbc)

traj_len = [x.shape[0] for x in traj_xyz]
print("Frames min/max:", min(traj_len), max(traj_len))

# ============================================================
# Standard masks (backbone, CA, NTL, CTL, body regions)
# ============================================================
masks       = build_standard_masks(meta)
CA          = masks["CA"]
BB          = masks["BB"]
BODY        = masks["BODY"]
NTL         = masks["NTL"]
CTL         = masks["CTL"]
resids_atom = masks["resids_atom"]

# Additional domain masks used in RMSF plotting
CHX = (resids_atom >= 1960) & (resids_atom <= 1975)
PLP = (resids_atom >= 1980) & (resids_atom <= 1990)

# Atom selection for PCA: backbone atoms in the body region
mode = "BB"
sele = combine_masks(BB, BODY)
fit  = combine_masks(BB, BODY)
print("mode:", mode, "| sele atoms:", int(sele.sum()), "| fit atoms:", int(fit.sum()))

# ============================================================
# Secondary structure (DSSP on WT reference, used for RMSF plot)
# ============================================================
wt_ref = DATA_ROOT / "WT" / "0" / "WT-MD-prot.pdb"
r = mda.Universe(str(wt_ref), str(wt_ref)).select_atoms("resid 1-291")
with tim("Secondary structure (DSSP)"):
    secondary_structure = "".join(DSSP(r).run().results.dssp[0]) + "-"

resids_ca = resids_atom[CA]

# ============================================================
# Reference frame and alignment
# Use the first frame of the last WT replica as the global reference.
# All trajectories are aligned to this reference by backbone fit.
# ============================================================
ref_sim_index = -3
print("Reference simulation:", F[ref_sim_index])

ref         = traj_xyz[ref_sim_index][0].copy()
ref_centered = ref - ref[fit].mean(axis=0, keepdims=True)
ref_fit      = ref_centered[fit] - ref_centered[fit].mean(axis=0, keepdims=True)

with tim("Aligning all trajectories to reference"):
    traj_aligned = [align_traj_to_ref(xyz, fit, ref_fit) for xyz in traj_xyz]

# ============================================================
# RMSD vs own first frame (per trajectory)
# ============================================================
colors = ("#ee5500", "#55ee00", "#0066ff", "#ff0066")

with tim("RMSD per trajectory"):
    rmsd_list = []
    for xyz in traj_aligned:
        x0  = xyz[0, sele]
        dif = xyz[:, sele] - x0[None, :, :]
        r   = np.sqrt((dif * dif).sum(axis=(1, 2)) / sele.sum()) / 10.0
        rmsd_list.append(r)

for r, c in zip(rmsd_list, np.repeat(colors, 3)):
    t = np.arange(len(r)) * 0.05
    plt.plot(t, r, c=c, linewidth=0.1)
plt.title("RMSD vs own first frame (nm)")
plt.xlabel("Time (ns)")
plt.show()

# ============================================================
# Aligned coordinate density (YZ plane, sanity check)
# ============================================================
stride_density = 10
YZ_all = []
for xyz in traj_aligned:
    A    = xyz[::stride_density, sele, 1:].reshape((-1, 2))
    mask = ~((np.abs(A[:, 0]) < 1e-6) & (np.abs(A[:, 1]) < 1e-6))
    YZ_all.append(A[mask])
YZ_all = np.vstack(YZ_all)

plt.figure(figsize=(7, 6))
plt.hist2d(YZ_all[:, 0], YZ_all[:, 1], bins=256)
plt.gca().set_aspect("equal")
plt.title("Aligned coordinate density (YZ)")
plt.xlabel("Y")
plt.ylabel("Z")
plt.show()

# ============================================================
# RMSF (per trajectory and overall mean)
# ============================================================
with tim("RMSF per trajectory"):
    rmsf_list = []
    for xyz in traj_aligned:
        mean_i  = xyz.mean(axis=0, keepdims=True)
        rmsf_i  = np.sqrt(((xyz - mean_i) ** 2).sum(axis=(0, 2)) / xyz.shape[0])
        rmsf_list.append(rmsf_i)
    rmsf   = np.vstack(rmsf_list)
    overall = rmsf.mean(axis=0)

# RMSF per residue (CA atoms) with secondary structure annotation
fig, ax = plt.subplots(figsize=(10, 10))
for s, c in zip(rmsf, np.repeat(colors, 3)):
    ax.plot(resids_atom[CA], s[CA], c=c, linewidth=0.5)

for x, ss in zip(resids_ca, secondary_structure):
    fc = "#ffddee" if ss == "H" else ("#ffeedd" if ss == "E" else "white")
    ax.add_patch(patches.Rectangle((x - 0.5, 0), 1, 1, facecolor=fc, zorder=0))

ax.set_ylim((0, 0.65))
ymin, ymax = ax.get_ylim()
bar_h = 0.03 * (ymax - ymin)
bar_y = ymin + 0.01 * (ymax - ymin)

for dom_mask, col in ((NTL, "#ffcc00"), (CTL, "#000077"), (CHX, "#ff9900"), (PLP, "#ff6600")):
    dom_ca = np.where(dom_mask & CA)[0]
    if len(dom_ca) == 0:
        continue
    x0 = resids_atom[dom_ca[0]]
    x1 = resids_atom[dom_ca[-1]]
    ax.add_patch(patches.Rectangle((x0 - 0.5, bar_y), (x1 - x0) + 1, bar_h, facecolor=col, zorder=10))

plt.title("RMSF (CA residues)")
plt.show()

# RMSF ratio (log2 per-trajectory / overall) highlighting variable regions
fig, ax = plt.subplots(figsize=(10, 10))
ratio = rmsf / (overall + 1e-12)
ymin, ymax = -2.5, 2.5
ax.set_ylim((ymin, ymax))

for x, ss in zip(resids_ca, secondary_structure):
    fc = "#fde" if ss == "H" else ("#fed" if ss == "E" else "white")
    ax.add_patch(patches.Rectangle((x - 0.5, ymin), 1, ymax - ymin, facecolor=fc, zorder=0))

ax.add_patch(patches.Rectangle((2103, ymin), 20, ymax - ymin, facecolor="#eee", zorder=1))

bar_h = 0.06
bar_y = ymin + 0.02
for dom_mask, col in ((NTL, "#ffcc00"), (CTL, "#000077"), (CHX, "#ff9900"), (PLP, "#ff6600")):
    dom_ca = np.where(dom_mask & CA)[0]
    if len(dom_ca) == 0:
        continue
    x0 = resids_atom[dom_ca[0]]
    x1 = resids_atom[dom_ca[-1]]
    ax.add_patch(patches.Rectangle((x0 - 0.5, bar_y), (x1 - x0) + 1, bar_h, facecolor=col, zorder=2))

for s, c in zip(ratio, np.repeat(colors, 3)):
    ax.plot(resids_atom[CA], np.log2(s[CA] + 1e-12), c=c, linewidth=0.5, zorder=5)

ax.axvline(2026, linestyle=":", zorder=6)
ax.axvline(2032, linestyle=":", zorder=6)
ax.axhline(0, color="k", linestyle="--", linewidth=0.8, zorder=6)
plt.title("RMSF ratio log2 (per trajectory / overall)")
plt.show()

# ============================================================
# Save all preprocessing outputs
# ============================================================
pd.DataFrame({"folder": F}).to_csv(RESULTS / "F_paths.csv", index=False)
np.save(RESULTS / "traj_aligned.npy", np.array(traj_aligned, dtype=object), allow_pickle=True)

masks_to_save = {k: np.asarray(v) for k, v in masks.items()}
masks_to_save["CHX"] = np.asarray(CHX)
masks_to_save["PLP"] = np.asarray(PLP)
np.savez(RESULTS / "masks.npz", **masks_to_save)

np.save(RESULTS / "sele.npy", sele)
np.save(RESULTS / "fit.npy", fit)
np.save(RESULTS / "ref_centered.npy", ref_centered)
np.save(RESULTS / "meta_resids.npy", meta.resids)
np.save(RESULTS / "meta_names.npy", meta.names.astype(str))
np.save(RESULTS / "meta_resnames.npy", meta.resnames.astype(str))
np.save(RESULTS / "frame_idx_list.npy", np.array(frame_idx_list, dtype=object), allow_pickle=True)

with open(RESULTS / "selection.txt", "w") as f:
    f.write(selection)
with open(RESULTS / "colors.txt", "w") as f:
    f.writelines(f"{c}\n" for c in colors)
with open(RESULTS / "nrep.txt", "w") as f:
    f.write(str(nrep))
with open(RESULTS / "nframes_sub.txt", "w") as f:
    f.write(str(nframes_sub))

print("Preprocessing complete. Outputs saved to:", RESULTS)