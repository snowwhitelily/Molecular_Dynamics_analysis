"""
ROS1 Analysis Pipeline — Script 02b2: Active Site PCA

Computes PCA on backbone atoms of six ATP binding site regions, verified
by sequence search and DSSP secondary structure assignment on WT-MD-prot.pdb:

    G-loop        (real 1951-1959, GRO 18-26)  — phosphate-binding loop (GxGxxG)
    beta3 Lys     (real 1980,      GRO 47)      — catalytic lysine (K1980)
    alphaC-helix  (real 1988-2003, GRO 55-70)  — contains E1997 salt bridge partner
    Hinge         (real 2026-2033, GRO 93-100) — H-bonds with ATP adenine ring
    Catalytic     (real 2077-2084, GRO 144-151)— HRD catalytic loop
    DFG motif     (real 2102-2104, GRO 169-171)— ASP-PHE-GLY, coordinates Mg2+ ion

Trajectories are aligned on NTL backbone (real 1938-2030) before PCA
so that PC1/PC2 reflect genuine active site shape changes rather than
global rigid-body motion. Output prefix: 'activesite'.
"""

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

PCA_STRIDE  = 10
NCOMPONENTS = 10
PLOT_PAIRS  = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]

BASE    = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "step2_fast" / "02b2_activesite"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("PCA_STRIDE =", PCA_STRIDE)
print("Figure output:", FIG_DIR)

# ── Load preprocessed data from Script 01 ────────────────────────────────────
F              = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
traj_aligned   = np.load(RESULTS / "traj_aligned.npy", allow_pickle=True).tolist()
frame_idx_list = np.load(RESULTS / "frame_idx_list.npy", allow_pickle=True).tolist()

# ── Exclude F1994L ────────────────────────────────────────────────────────────
# AlphaFold2 predicted this variant in a distinct inactive conformation
# not comparable to the rest of the panel.
EXCLUDE_MUTANTS = {"F1994L"}
keep           = [i for i, f in enumerate(F) if Path(f).parent.name not in EXCLUDE_MUTANTS]
F              = [F[i] for i in keep]
traj_aligned   = [traj_aligned[i] for i in keep]
frame_idx_list = [frame_idx_list[i] for i in keep]
print(f"Trajectories after exclusion: {len(F)}")

# ── Load masks and metadata ───────────────────────────────────────────────────
masks_npz = np.load(RESULTS / "masks.npz", allow_pickle=True)
masks     = {k: masks_npz[k] for k in masks_npz.files}

BB  = masks["BB"]
NTL = masks["NTL"]

ref_centered = np.load(RESULTS / "ref_centered.npy", allow_pickle=True)

meta = SimpleNamespace(
    resids   = np.load(RESULTS / "meta_resids.npy", allow_pickle=True),
    names    = np.load(RESULTS / "meta_names.npy", allow_pickle=True),
    resnames = np.load(RESULTS / "meta_resnames.npy", allow_pickle=True),
)

with open(RESULTS / "colors.txt") as f:
    colors = tuple(line.strip() for line in f if line.strip())
with open(RESULTS / "nrep.txt") as f:
    nrep = int(f.read().strip())

traj_colors = make_traj_colors(F, colors, nrep)

# ── Build active site atom selection mask ─────────────────────────────────────
# Real residue number = internal GRO index + 1933.
# All ranges verified by sequence search on WT-MD-prot.pdb and DSSP
# secondary structure assignment.
resids_abs = np.asarray(meta.resids).astype(int) + 1933

ACTIVE_SITE_REGIONS = {
    "G_loop"      : (1951, 1959),   # GxGxxG phosphate-binding loop
    "beta3_K1980" : (1980, 1980),   # catalytic lysine
    "alphaC_helix": (1988, 2003),   # contains E1997; DSSP verified GRO 55-70
    "hinge"       : (2026, 2033),   # L2026 gatekeeper to G2032 solvent front
    "catalytic"   : (2077, 2084),   # HRD catalytic loop
    "DFG"         : (2102, 2104),   # ASP-PHE-GLY; sequence verified GRO 169-171
}

mask_activesite = np.zeros(len(resids_abs), dtype=bool)
for start, end in ACTIVE_SITE_REGIONS.values():
    mask_activesite |= (resids_abs >= start) & (resids_abs <= end)
mask_activesite &= BB

n_activesite_atoms = int(mask_activesite.sum())
print(f"Active site backbone atoms selected: {n_activesite_atoms}")

if n_activesite_atoms == 0:
    raise ValueError(
        "Active site mask is empty. Check that ACTIVE_SITE_REGIONS real residue "
        "numbers match meta_resids.npy (real residue = internal + 1933)."
    )

# ── Align on NTL backbone before active site PCA ─────────────────────────────
fit_mask = BB & NTL
print(f"NTL fit atoms: {int(fit_mask.sum())}")

ref_fit = ref_centered[fit_mask] - ref_centered[fit_mask].mean(axis=0, keepdims=True)

with tim("Active site: aligning trajectories on NTL"):
    traj_activesite = [align_traj_to_ref_by_fit(xyz, fit_mask, ref_fit) for xyz in traj_aligned]

# Coordinate density sanity check
plt.figure(figsize=(7, 6))
A = []
for xyz in traj_activesite:
    a    = xyz[:, mask_activesite, 1:].reshape((-1, 2))
    mask = ~((np.abs(a[:, 0]) < 1e-6) & (np.abs(a[:, 1]) < 1e-6))
    A.append(a[mask])
A = np.vstack(A)
plt.hist2d(A[:, 0], A[:, 1], bins=256)
plt.gca().set_aspect("equal")
plt.title("Active site aligned coordinate density (YZ)")
plt.xlabel("Y")
plt.ylabel("Z")
save_current_figure(FIG_DIR, "activesite_density_yz.png")

# ── Run active site PCA ───────────────────────────────────────────────────────
PCA_ACTIVESITE = run_pca_block(
    traj_list      = traj_activesite,
    sel_mask       = mask_activesite,
    title_prefix   = "Active Site",
    prefix         = "activesite",
    traj_colors    = traj_colors,
    F              = F,
    frame_idx_list = frame_idx_list,
    meta           = meta,
    results_dir    = RESULTS,
    fig_dir        = FIG_DIR,
    pca_stride     = PCA_STRIDE,
    ncomponents    = NCOMPONENTS,
    plot_pairs     = PLOT_PAIRS,
    timer          = tim,
)

print("Script 02b2 completed. Output prefix: activesite. Saved to:", RESULTS)