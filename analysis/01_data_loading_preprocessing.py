# %% [markdown]
# # Molecular Dynamics Simulations of ROS1 Mutations in Non-Small-Cell Lung Cancer (NSCLC)
#
# ## Overview
#
# This notebook presents a comparative molecular dynamics (MD) analysis of ROS1 kinase domain mutants.
#
# The goal is to investigate how specific mutations affect the conformational dynamics of the kinase and potentially influence functional behavior such as inhibitor resistance.
#
# The workflow includes:
# - Trajectory preprocessing and alignment
# - Principal Component Analysis (PCA)
# - Comparative analysis of conformational landscapes
# - Structural interpretation of dominant motions

# %% [markdown]
# ## Objective
#
# The goal of this analysis is to compare the conformational dynamics of ROS1 kinase mutants using molecular dynamics simulations.
#
# To ensure a fair comparison between mutants, trajectories are processed in a consistent and unbiased manner.
#
# Key steps include:
# - Removing global motion through alignment
# - Selecting relevant atoms (backbone)
# - Normalizing trajectory lengths
# - Applying PCA to identify dominant motions
#
# This enables meaningful comparison of structural dynamics across mutants.

# %%
# importing necessary libraries
import os
import pandas as pd
import numpy as np
import MDAnalysis as mda
import molly
from pathlib import Path
from MDAnalysis.analysis.dssp import DSSP
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import calc_dihedrals

from scripts.clustering import run_hdbscan, run_dbscan
from scripts.occupancy import occupancy_from_traj_labels
from scripts.plotting import (
    plot_occupancy_heatmap,
    plot_cluster_population,
    plot_pca_clusters
)

from scripts.ros1_utils import (
    ensure_meta,
    ensure_exists_all,
    build_standard_masks,
    combine_masks,
    traj_frames_atoms,
    principal_axis,
    angle,
)

from scripts.ros1_plots import (
    plot_scree,
    plot_pca_scatter,
    plot_pca_scatter_zoom,
)

from scripts.ros1_align import (
    align_traj_to_ref_by_fit,
    kabsch_R,
)

from scripts.ros1_analysis_helpers import (
    switch_count,
    mean_dwell_frames,
    cohens_d,
    summarize_pair,
    lda_multiclass,
    lda_2class_direction,
    lda_2class_w,
    write_mode_movie_pdb,
    pairwise_ld1_scatter,
    _as_mean_xyz,
    _infer_n_atoms_from_loadings,
    _mean_atom_count,
    _get_sel_idx_from_mask,
    _resolve_sel_idx,
)

from scripts.ros1_clustering import (
    mutant_names_from_F,
    occ_table_from_owner,
    run_cluster_panel_varlen,
    run_cluster_panel_equal,
)

from scripts.ros1_io import (
    subsampled_len,
    get_frame_idx,
    xtc2array_varlen_by_indices,
    selection_keys_and_indices,
    align_traj_to_ref,
)

from scripts.ros1_dihedrals import (
    get_atom_indices_for_resid,
    compute_chi1_deg_for_traj,
)

from scripts.timer import Timer
tim = Timer()

eps = 1e-6
os.makedirs("figures", exist_ok=True)

# %%
# ============================================
# dataset root
# ============================================
DATA_ROOT = Path.home() / "Molecular_Dynamics_analysis" / "trajectories" / "ros1_prepared_final"
print("DATA_ROOT =", DATA_ROOT)

# %%
# ============================================
# 1) ONE authoritative selection 
# ============================================
selection = "protein and name N CA C O CB CG and not element H"

# %% [markdown]
# ## Loading Trajectories
#
# Each mutant contains multiple simulation replicas.
#
# All trajectories are loaded and organized into a consistent structure to allow direct comparison.
#
# Each trajectory is represented as:
#
# (n_frames, n_atoms, 3)
#
# This format enables efficient numerical processing for alignment and PCA.

# %%
# ============================================
# 2) Auto-detect mutants + build simulation list F 
#    UPDATED FOR ros1_prepared_final/<MUT>/<REP>/
# ============================================

IGNORE = {
    "__pycache__",
    "orientation_cluster_reps",
    "figures",
    "plots",
    "tmp",
}

mutants = []
for d in sorted(DATA_ROOT.iterdir()):
    if not d.is_dir():
        continue
    if d.name in IGNORE:
        continue
    rep0 = d / "0" / f"{d.name}-MD-prot.pdb"
    if rep0.exists():
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
print("Example:", F[0] if len(F) else "None")

# %%
# =========================
# locking trajectory paths so F never gets overwritten
# =========================
if "F_paths" not in globals():
    assert "F" in globals(), "Define F (list of traj folder strings) first."
    assert isinstance(F, list) and isinstance(F[0], str), "F must be list[str] when saving it."
    F_paths = F.copy()

F = F_paths.copy()

print("[OK] F restored:", type(F), "len(F)=", len(F), "example:", F[0])

# %% [markdown]
# ## Frame Normalization
#
# Different simulations may contain different numbers of frames.
#
# To ensure fair comparison:
# - All trajectories are truncated to the same number of frames
#
# This prevents bias in statistical analysis and PCA.
#
# The minimum trajectory length across all simulations is used.

# %%
# selecting the number of frames based on the minimum across trajectories
stride = 1

nframes_sub = min(subsampled_len(d) for d in F)
print("[frame_idx_list] Using nframes_sub =", nframes_sub)

frame_idx_list = [get_frame_idx(d) for d in F]

print("[frame_idx_list] built for", len(frame_idx_list), "trajectories.")
print("[frame_idx_list] example first/last:", frame_idx_list[0][0], frame_idx_list[0][-1])

# %%
# ============================================
# 3b) COMMON ATOMS intersection 
#     Building keys by (resid, atomname) within the selection
# ============================================

with tim("Computing COMMON selection atoms across all simulations"):
    all_keysets = []
    all_maps = []

    for d in F:
        keys, k2i = selection_keys_and_indices(d, selection)
        all_keysets.append(keys)
        all_maps.append(k2i)

    common_keys = set.intersection(*all_keysets)
    print("Selection sizes (min/max):", min(len(s) for s in all_keysets), max(len(s) for s in all_keysets))
    print("COMMON atoms kept:", len(common_keys))

    if len(common_keys) == 0:
        raise ValueError("COMMON atom set is empty — selection too strict or inconsistent topologies.")

    common_keys_sorted = sorted(list(common_keys), key=lambda x: (x[0], x[1]))

    per_sim_indices = []
    for k2i in all_maps:
        idx = [k2i[k] for k in common_keys_sorted]
        per_sim_indices.append(idx)

first_run = Path(F[0])
first_mut = first_run.parent.name
u0 = mda.Universe(str(first_run / f"{first_mut}-MD-prot.pdb"))
meta = u0.atoms[per_sim_indices[0]]
natoms = len(meta)
print("Authoritative natoms (common):", natoms)

# %%
# ============================================
# 3c) Read all trajectories (variable length, COMMON atoms only)
# ============================================
with tim("Reading all trajectories (true variable length, common atoms only)"):
    traj_xyz = []
    traj_pbc = []
    for d, idx in zip(F, per_sim_indices):
        fi = frame_idx_list[F.index(d)].tolist()
        xyz, pbc = xtc2array_varlen_by_indices(d, idx, frame_selection=fi)
        traj_xyz.append(xyz)
        traj_pbc.append(pbc)

traj_len = [x.shape[0] for x in traj_xyz]
print("Frames min/max:", min(traj_len), max(traj_len))

# %%
# ============================================
# 4) Standard masks (computed on COMMON meta)
# ============================================
masks = build_standard_masks(meta)

CA = masks["CA"]
BB = masks["BB"]
BODY = masks["BODY"]
resids_atom = masks["resids_atom"]
exists_all = masks["exists_all"]

mode = "BB"
if mode == "CA":
    sele = combine_masks(CA, BODY)
    fit = combine_masks(CA, BODY)
elif mode == "BB":
    sele = combine_masks(BB, BODY)
    fit = combine_masks(BB, BODY)
else:
    raise ValueError("mode must be 'CA' or 'BB'")

print("mode:", mode, "sele atoms:", int(sele.sum()), "fit atoms:", int(fit.sum()))

# %%
# ============================================
# 5) DSSP background (WT reference)
# ============================================
wt_ref = DATA_ROOT / "WT" / "0" / "WT-MD-prot.pdb"
r = mda.Universe(str(wt_ref), str(wt_ref)).select_atoms('resid 1-291')
with tim("Secondary structure (DSSP)"):
    secondary_structure = ''.join(DSSP(r).run().results.dssp[0]) + '-'
print(secondary_structure)

resids_ca = resids_atom[CA]

# %%
# ============================================
# 6) Plot reference axes cube (E) 
# ============================================
E = np.array((
    0,0,0, 1,0,0,
    0,0,0, 0,1,0,
    0,0,0, 0,0,1,
    1,0,0, 1,1,0,
    1,0,0, 1,0,1,
    0,1,0, 1,1,0,
    0,1,0, 0,1,1,
    0,0,1, 1,0,1,
    0,0,1, 0,1,1,
    1,1,0, 1,1,1,
    1,0,1, 1,1,1,
    0,1,1, 1,1,1,
)).reshape((12, 2, 3)) @ traj_pbc[0][0]
print("E shape:", E.shape)

# %%
# ============================================
# 7) Reference choice 
# ============================================
ref_sim_index = -3
print("Reference simulation:", F[ref_sim_index])

ref = traj_xyz[ref_sim_index][0].copy()

for k in ([0,1], [0,2], [1,2]):
    plt.scatter(*ref[:, k].T, c=-np.arange(len(ref)), cmap='turbo', s=2)
    for e in E:
        plt.plot(*e[:, k].T, c='#ccc', linewidth=1)
    plt.gca().set_aspect('equal')
    plt.show()

# %%
# ============================================
# 8) Reference views + alignment
# ============================================
ref_centered = ref - ref[fit].mean(axis=0, keepdims=True)
ref_fit = ref_centered[fit]
ref_fit = ref_fit - ref_fit.mean(axis=0, keepdims=True)

with tim("Aligning ALL trajectories to ONE reference (variable length)"):
    traj_aligned = []
    for xyz in traj_xyz:
        traj_aligned.append(align_traj_to_ref(xyz, fit, ref_fit))

# %%
# ============================================
# 9) RMSD vs own first frame (per trajectory)
# ============================================
colors = ('#ee5500', '#55ee00', '#0066ff', '#ff0066')

with tim("RMSD per trajectory (vs own first frame)"):
    rmsd_list = []
    for xyz in traj_aligned:
        x0 = xyz[0, sele]
        dif = xyz[:, sele] - x0[None, :, :]
        r = np.sqrt((dif * dif).sum(axis=(1,2)) / sele.sum()) / 10.0
        rmsd_list.append(r)

for r, c in zip(rmsd_list, np.repeat(colors, 3)):
    t = np.arange(len(r)) * 0.05
    plt.plot(t, r, c=c, linewidth=0.1)
plt.title("RMSD vs own first frame (nm)")
plt.xlabel("Time (ns)")
plt.show()

# %%
# ============================================
# 10) Plot of all structures aligned (YZ density)
# ============================================
stride_density = 10
YZ_all = []
for xyz in traj_aligned:
    A = xyz[::stride_density, sele, 1:].reshape((-1, 2))
    mask = ~((np.abs(A[:, 0]) < 1e-6) & (np.abs(A[:, 1]) < 1e-6))
    YZ_all.append(A[mask])
YZ_all = np.vstack(YZ_all)

plt.figure(figsize=(7, 6))
plt.hist2d(YZ_all[:, 0], YZ_all[:, 1], bins=256)
plt.gca().set_aspect('equal')
plt.title("Aligned coordinate density (YZ)")
plt.xlabel("Y")
plt.ylabel("Z")
plt.show()

# %%
# ============================================
# 11) RMSF (per traj + overall)
# ============================================
with tim("RMSF per trajectory (variable length)"):
    rmsf_list = []
    for xyz in traj_aligned:
        mean_i = xyz.mean(axis=0, keepdims=True)
        rmsf_i = np.sqrt(((xyz - mean_i) ** 2).sum(axis=(0, 2)) / xyz.shape[0])
        rmsf_list.append(rmsf_i)
    rmsf = np.vstack(rmsf_list)
    overall = rmsf.mean(axis=0)

# ============================================
# 11.5) Domain masks
# ============================================
NTL = masks["NTL"]
CTL = masks["CTL"]
CHX = (resids_atom >= 1960) & (resids_atom <= 1975)
PLP = (resids_atom >= 1980) & (resids_atom <= 1990)

# %%
# ============================================
# 12) RMSF plot (CA) + domain band
# ============================================
fig, ax = plt.subplots(figsize=(10, 10))

for s, c in zip(rmsf, np.repeat(colors, 3)):
    ax.plot(resids_atom[CA], s[CA], c=c, linewidth=0.5)

for x, ss in zip(resids_ca, secondary_structure):
    if ss == 'H':
        fc = '#ffddee'
    elif ss == 'E':
        fc = '#ffeedd'
    else:
        fc = 'white'
    ax.add_patch(patches.Rectangle((x - 0.5, 0), 1, 1, facecolor=fc, zorder=0))

ax.set_ylim((0, 0.65))

ymin, ymax = ax.get_ylim()
bar_h = 0.03 * (ymax - ymin)
bar_y = ymin + 0.01 * (ymax - ymin)

for dom_mask, col in ((NTL, '#ffcc00'), (CTL, '#000077'), (CHX, '#ff9900'), (PLP, '#ff6600')):
    dom_ca = np.where(dom_mask & CA)[0]
    if len(dom_ca) == 0:
        continue
    x0 = resids_atom[dom_ca[0]]
    x1 = resids_atom[dom_ca[-1]]
    ax.add_patch(
        patches.Rectangle(
            (x0 - 0.5, bar_y),
            (x1 - x0) + 1,
            bar_h,
            facecolor=col,
            zorder=10
        )
    )

plt.title("RMSF (CA residues)")
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))

ratio = rmsf / (overall + 1e-12)

ymin, ymax = -2.5, 2.5
ax.set_ylim((ymin, ymax))

for x, ss in zip(resids_ca, secondary_structure):
    fc = '#fde' if ss == 'H' else ('#fed' if ss == 'E' else 'white')
    ax.add_patch(patches.Rectangle((x - 0.5, ymin), 1, ymax - ymin, facecolor=fc, zorder=0))

ax.add_patch(patches.Rectangle((2103, ymin), 20, ymax - ymin, facecolor='#eee', zorder=1))

bar_h = 0.06
bar_y = ymin + 0.02
for dom_mask, col in ((NTL, '#ffcc00'), (CTL, '#000077'), (CHX, '#ff9900'), (PLP, '#ff6600')):
    dom_ca = np.where(dom_mask & CA)[0]
    if len(dom_ca) == 0:
        continue
    x0 = resids_atom[dom_ca[0]]
    x1 = resids_atom[dom_ca[-1]]
    ax.add_patch(patches.Rectangle((x0 - 0.5, bar_y), (x1 - x0) + 1, bar_h, facecolor=col, zorder=2))

for s, c in zip(ratio, np.repeat(colors, 3)):
    ax.plot(resids_atom[CA], np.log2(s[CA] + 1e-12), c=c, linewidth=0.5, zorder=5)

ax.axvline(2026, linestyle=':', zorder=6)
ax.axvline(2032, linestyle=':', zorder=6)
ax.axhline(0, color='k', linestyle='--', linewidth=0.8, zorder=6)

plt.title("RMSF ratio log2 (per traj / overall)")
plt.show()

# %%
BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
RESULTS.mkdir(parents=True, exist_ok=True)

# Save trajectory folder list
pd.DataFrame({"folder": F}).to_csv(RESULTS / "F_paths.csv", index=False)

# Save aligned trajectories
np.save(RESULTS / "traj_aligned.npy", np.array(traj_aligned, dtype=object), allow_pickle=True)

# Save masks
masks_to_save = {k: np.asarray(v) for k, v in masks.items()}
masks_to_save["CHX"] = np.asarray(CHX)
masks_to_save["PLP"] = np.asarray(PLP)

np.savez(
    RESULTS / "masks.npz",
    **masks_to_save,
)

# Save additional objects needed by Step 2
np.save(RESULTS / "sele.npy", sele)
np.save(RESULTS / "fit.npy", fit)
np.save(RESULTS / "ref_centered.npy", ref_centered)
np.save(RESULTS / "meta_resids.npy", meta.resids)
np.save(RESULTS / "meta_names.npy", meta.names.astype(str))
np.save(RESULTS / "meta_resnames.npy", meta.resnames.astype(str))

with open(RESULTS / "selection.txt", "w") as f:
    f.write(selection)

with open(RESULTS / "colors.txt", "w") as f:
    for c in colors:
        f.write(f"{c}\n")

with open(RESULTS / "nrep.txt", "w") as f:
    f.write(str(nrep))

print("Saved preprocessing outputs to:", RESULTS)