# %%
# ROS1 Analysis Pipeline
# Script 2C: PCA (A-loop + CTL-fit whole kinase)
#
# FIXES:
# 1. Removed duplicate 02a code block that was copy-pasted in error
# 2. Added F1994L exclusion block after loading

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from types import SimpleNamespace

from scripts.ros1_utils import combine_masks
from scripts.ros1_align import align_traj_to_ref_by_fit
from scripts.ros1_pca import run_pca_block, make_traj_colors
from scripts.timer import Timer

tim = Timer()

PCA_STRIDE = 10
NCOMPONENTS = 10
PLOT_PAIRS = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "step2_fast" / "02c_aloop_ctlfit"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Running exploratory PCA mode")
print("PCA_STRIDE =", PCA_STRIDE)
print("Figure output folder:", FIG_DIR)

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

masks_npz = np.load(RESULTS / "masks.npz", allow_pickle=True)
masks = {k: masks_npz[k] for k in masks_npz.files}

BB = masks["BB"]
BODY = masks["BODY"]
CTL = masks["CTL"]
ACT = masks["ACT"]

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

traj_colors = make_traj_colors(F, colors, nrep)

# A-loop PCA
sele_aloop = combine_masks(BB, ACT)
fit_aloop = sele_aloop
print("A-loop sele atoms:", int(sele_aloop.sum()))
print("A-loop fit atoms:", int(fit_aloop.sum()))

ref_fit_aloop = ref_centered[fit_aloop]
ref_fit_aloop = ref_fit_aloop - ref_fit_aloop.mean(axis=0, keepdims=True)

with tim("A-loop: aligning all trajectories on fit"):
    traj_aloop = [align_traj_to_ref_by_fit(xyz, fit_aloop, ref_fit_aloop) for xyz in traj_aligned]

PCA_ALOOP = run_pca_block(
    traj_list=traj_aloop,
    sel_mask=sele_aloop,
    title_prefix="A-loop",
    prefix="aloop",
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

# Whole-kinase PCA after CTL fit
sele_ctlfit = combine_masks(BB, BODY)
fit_ctlfit = combine_masks(BB, CTL)
print("CTL-fit sele atoms:", int(sele_ctlfit.sum()))
print("CTL-fit fit atoms:", int(fit_ctlfit.sum()))

ref_fit_ctlfit = ref_centered[fit_ctlfit]
ref_fit_ctlfit = ref_fit_ctlfit - ref_fit_ctlfit.mean(axis=0, keepdims=True)

with tim("ACT-OUT CTL-fit: aligning all trajectories on CTL"):
    traj_ctlfit = [align_traj_to_ref_by_fit(xyz, fit_ctlfit, ref_fit_ctlfit) for xyz in traj_aligned]

PCA_CTLFIT = run_pca_block(
    traj_list=traj_ctlfit,
    sel_mask=sele_ctlfit,
    title_prefix="ACT-OUT CTL-fit",
    prefix="actout_ctlfit",
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

print("Script 2C completed:")
print("- A-loop")
print("- ACT-OUT CTL-fit")
print("Flow used: Scree -> Scores -> Loadings -> first 10% / last 10%")
print("Mode used: exploratory downsampled PCA, stride =", PCA_STRIDE)
print("Saved figures to:", FIG_DIR)
print("Saved PCA outputs to:", RESULTS)
