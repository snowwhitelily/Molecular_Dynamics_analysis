# %%
# ROS1 Analysis Pipeline
# Script 02b2: PCA (Active Site)
#
# PURPOSE:
#   Computes PCA specifically on the backbone atoms of the ATP binding
#   site residues. This captures how the shape of the active site
#   changes across mutants — directly addressing the supervisor's
#   request to focus on "the conformation of the active site itself."
#
# ACTIVE SITE REGIONS (real residue numbers, real = internal + 1933):
#   P-loop       : 1957-1962  — grips ATP phosphate groups
#   alphaC-helix : 1983-1993  — contains E1993 (salt bridge partner)
#   Catalytic    : 2019-2025  — contains catalytic aspartate
#   Hinge        : 2031-2038  — hydrogen bonds with ATP adenine ring
#   DFG motif    : 2042-2044  — coordinates Mg2+ and positions gamma-phosphate
#
# OUTPUT PREFIX: activesite
#   Saves: activesite_scores.npy, activesite_scores.csv,
#          activesite_x_owner.npy, activesite_loadings.npy,
#          activesite_mean.npy, activesite_pca_metadata.json
#   These plug directly into 02d, 03c, and the rest of the pipeline.
#
# ALIGNMENT STRATEGY:
#   Fit on NTL backbone (internal 5-97, real 1938-2030) to remove
#   global rigid-body rotation. Then run PCA on active site backbone
#   atoms only. This way PC1/PC2 reflect genuine active site shape
#   changes, not just the protein tumbling in space.
#
# EXCLUSION:
#   F1994L excluded per supervisor instruction — started in inactive
#   form, not comparable to active-form panel.
#
# USAGE:
#   python 02b2_pca_activesite.py

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from types import SimpleNamespace

from scripts.ros1_utils import combine_masks
from scripts.ros1_align import align_traj_to_ref_by_fit
from scripts.ros1_pca import run_pca_block, make_traj_colors, save_current_figure
from scripts.timer import Timer

tim = Timer()

PCA_STRIDE = 10
NCOMPONENTS = 10
PLOT_PAIRS = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = (BASE / "figures" / "ROS1" / "ros1_prepared_final"
           / "step2_fast" / "02b2_activesite")
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Running Script 02b2: Active Site PCA")
print("=" * 60)
print("PCA_STRIDE =", PCA_STRIDE)
print("Figure output folder:", FIG_DIR)

# ============================================================
# Load preprocessed data
# ============================================================
F = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
traj_aligned = np.load(RESULTS / "traj_aligned.npy", allow_pickle=True).tolist()
frame_idx_list = np.load(RESULTS / "frame_idx_list.npy", allow_pickle=True).tolist()

# ============================================================
# EXCLUDE MUTANTS NOT IN ANALYSIS
# F1994L excluded per supervisor instruction:
# started in inactive form, not comparable to active-form panel
# ============================================================
EXCLUDE_MUTANTS = {"F1994L"}
keep = [i for i, f in enumerate(F)
        if Path(f).parent.name not in EXCLUDE_MUTANTS]
F              = [F[i] for i in keep]
traj_aligned   = [traj_aligned[i] for i in keep]
frame_idx_list = [frame_idx_list[i] for i in keep]
print(f"Excluded mutants: {EXCLUDE_MUTANTS}")
print(f"Remaining trajectories: {len(F)}")

# ============================================================
# Load masks and metadata
# ============================================================
masks_npz = np.load(RESULTS / "masks.npz", allow_pickle=True)
masks = {k: masks_npz[k] for k in masks_npz.files}

BB   = masks["BB"]    # backbone atoms
NTL  = masks["NTL"]   # N-terminal lobe mask

ref_centered = np.load(RESULTS / "ref_centered.npy", allow_pickle=True)

meta = SimpleNamespace(
    resids=np.load(RESULTS / "meta_resids.npy", allow_pickle=True),
    names=np.load(RESULTS / "meta_names.npy", allow_pickle=True),
    resnames=np.load(RESULTS / "meta_resnames.npy", allow_pickle=True),
)

with open(RESULTS / "colors.txt") as f:
    colors = tuple(line.strip() for line in f if line.strip())

with open(RESULTS / "nrep.txt") as f:
    nrep = int(f.read().strip())

print("Setup ok.")
print("n trajectories:", len(traj_aligned))
print("example F:", F[0])

traj_colors = make_traj_colors(F, colors, nrep)

# ============================================================
# Define active site atom selection mask
#
# We select backbone atoms (BB) from the five active site regions.
# Real residue = internal + 1933.
# We build the mask by selecting residues in the real residue ranges.
# ============================================================
resids_abs = np.asarray(meta.resids).astype(int) + 1933  # real residue numbers
names_arr  = np.asarray(meta.names)

# Active site regions (real residue ranges)
ACTIVE_SITE_REGIONS = {
    "P_loop"      : (1957, 1962),   # glycine-rich loop, grips ATP phosphates
    "alphaC_helix": (1983, 1993),   # contains E1993, key for salt bridge
    "catalytic"   : (2019, 2025),   # catalytic aspartate loop
    "hinge"       : (2031, 2038),   # H-bonds with ATP adenine ring
    "DFG"         : (2042, 2044),   # DFG motif, coordinates Mg2+
}

print("\nActive site regions:")
for region, (start, end) in ACTIVE_SITE_REGIONS.items():
    n_res = end - start + 1
    print(f"  {region:<15}: real {start}-{end} ({n_res} residues)")

# Build combined active site mask: backbone atoms in any active site region
mask_activesite = np.zeros(len(resids_abs), dtype=bool)
for region, (start, end) in ACTIVE_SITE_REGIONS.items():
    region_mask = (resids_abs >= start) & (resids_abs <= end)
    mask_activesite |= region_mask

# Intersect with backbone atoms
mask_activesite = mask_activesite & BB

n_activesite_atoms = int(mask_activesite.sum())
print(f"\nTotal active site backbone atoms selected: {n_activesite_atoms}")

if n_activesite_atoms == 0:
    raise ValueError(
        "Active site mask is empty! "
        "Check that ACTIVE_SITE_REGIONS real residue numbers match "
        "meta_resids.npy. Real residue = internal + 1933."
    )

# Print which residues were actually selected (sanity check)
selected_resids = np.unique(resids_abs[mask_activesite])
print(f"Selected real residues: {selected_resids[:10]}...{selected_resids[-5:]}")
print(f"Total unique residues selected: {len(selected_resids)}")

# ============================================================
# Alignment: fit on NTL backbone to remove global rigid body motion
#
# We align on the NTL (N-terminal lobe, real 1938-2030) because:
# - NTL is relatively rigid across mutants
# - Fitting on NTL means PC1/PC2 of active site PCA reflect
#   genuine active site shape changes, not global tumbling
# - This is consistent with how actout_ctlfit aligns on CTL
# ============================================================
fit_mask = BB & NTL   # fit on NTL backbone atoms

print(f"\nAlignment: fitting on NTL backbone")
print(f"NTL fit atoms: {int(fit_mask.sum())}")

if fit_mask.sum() == 0:
    raise ValueError(
        "NTL fit mask is empty! Check that NTL mask is correctly loaded."
    )

ref_fit = ref_centered[fit_mask]
ref_fit = ref_fit - ref_fit.mean(axis=0, keepdims=True)

print("Aligning all trajectories on NTL backbone...")
with tim("Active site: aligning all trajectories on NTL"):
    traj_activesite = [
        align_traj_to_ref_by_fit(xyz, fit_mask, ref_fit)
        for xyz in traj_aligned
    ]
print("Alignment done.")

# ============================================================
# Coordinate density plot (sanity check — same as 02a/02b)
# ============================================================
plt.figure(figsize=(7, 6))
A = []
for xyz in traj_activesite:
    a = xyz[:, mask_activesite, 1:].reshape((-1, 2))
    mask_nonzero = ~((np.abs(a[:, 0]) < 1e-6) & (np.abs(a[:, 1]) < 1e-6))
    A.append(a[mask_nonzero])
A = np.vstack(A)
plt.hist2d(A[:, 0], A[:, 1], bins=256)
plt.gca().set_aspect("equal")
plt.title("Active site aligned coordinate density (YZ)\n"
          "P-loop + alphaC + catalytic + hinge + DFG")
plt.xlabel("Y")
plt.ylabel("Z")
save_current_figure(FIG_DIR, "activesite_density_yz.png")

# ============================================================
# Run PCA using the same run_pca_block as all other PCA scripts
# This ensures outputs are in exactly the same format and plug
# directly into 02d, 02e, 02f, 03c, 04, 04b, 04c
# ============================================================
print("\nRunning active site PCA...")
PCA_ACTIVESITE = run_pca_block(
    traj_list=traj_activesite,
    sel_mask=mask_activesite,
    title_prefix="Active Site",
    prefix="activesite",
    traj_colors=traj_colors,
    F=F,
    frame_idx_list=frame_idx_list,
    meta=meta,
    results_dir=RESULTS,
    fig_dir=FIG_DIR,
    pca_stride=PCA_STRIDE,
    ncomponents=NCOMPONENTS,
    plot_pairs=PLOT_PAIRS,
    timer=tim,
)

print("\nScript 02b2 completed.")
print("Output prefix: activesite")
print("Saved PCA outputs to:", RESULTS)
print("Saved figures to:", FIG_DIR)
print("\nNext steps:")
print("  1. Review scree plot — check how many PCs explain meaningful variance")
print("  2. Review loadings — confirm P-loop/hinge/DFG dominate PC1")
print("  3. Run 02d with --prefixes actout_ctlfit,activesite")
print("  4. Run 03c with --prefixes actout_ctlfit,ctl,activesite --vmax 2.5 --per-mutant")
print("\nMethods note for thesis:")
print(
    "Active site PCA was performed on backbone atoms of five functional "
    "regions: the P-loop (real 1957-1962), alphaC-helix (real 1983-1993), "
    "catalytic loop (real 2019-2025), hinge region (real 2031-2038), and "
    "DFG motif (real 2042-2044). Trajectories were aligned on the NTL "
    "backbone (real 1938-2030) prior to PCA to remove global rigid-body "
    "motion, ensuring that PC1 and PC2 reflect genuine active site "
    "conformational changes. The prefix 'activesite' is used for all "
    "downstream outputs."
)
