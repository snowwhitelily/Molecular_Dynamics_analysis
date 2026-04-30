# %%
# ROS1 Analysis Pipeline
# Script 2B: PCA (NTL + CTL)
#
# FIXES:
# 1. Removed duplicate 02a code block that was copy-pasted in error
# 2. Added F1994L exclusion block after loading

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
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "step2_fast" / "02b_ntl_ctl"
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
NTL = masks["NTL"]
CTL = masks["CTL"]

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

print("Step 2B setup ok.")
print("n trajectories:", len(traj_aligned))
print("example F:", F[0])

traj_colors = make_traj_colors(F, colors, nrep)

# NTL PCA
sele_ntl = combine_masks(BB, NTL)
fit_ntl = sele_ntl
print("NTL sele atoms:", int(sele_ntl.sum()))
print("NTL fit atoms:", int(fit_ntl.sum()))

ref_fit_ntl = ref_centered[fit_ntl]
ref_fit_ntl = ref_fit_ntl - ref_fit_ntl.mean(axis=0, keepdims=True)

with tim("NTL: aligning all trajectories on fit"):
    traj_ntl = [align_traj_to_ref_by_fit(xyz, fit_ntl, ref_fit_ntl) for xyz in traj_aligned]

plt.figure(figsize=(7, 6))
A = []
for xyz in traj_ntl:
    a = xyz[:, sele_ntl, 1:].reshape((-1, 2))
    mask_nonzero = ~((np.abs(a[:, 0]) < 1e-6) & (np.abs(a[:, 1]) < 1e-6))
    A.append(a[mask_nonzero])
A = np.vstack(A)
plt.hist2d(A[:, 0], A[:, 1], bins=256)
plt.gca().set_aspect("equal")
plt.title("NTL aligned coordinate density (YZ)")
plt.xlabel("Y")
plt.ylabel("Z")
save_current_figure(FIG_DIR, "ntl_density_yz.png")

PCA_NTL = run_pca_block(
    traj_list=traj_ntl,
    sel_mask=sele_ntl,
    title_prefix="NTL",
    prefix="ntl",
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

# CTL PCA
sele_ctl = combine_masks(BB, CTL)
fit_ctl = sele_ctl
print("CTL sele atoms:", int(sele_ctl.sum()))
print("CTL fit atoms:", int(fit_ctl.sum()))

ref_fit_ctl = ref_centered[fit_ctl]
ref_fit_ctl = ref_fit_ctl - ref_fit_ctl.mean(axis=0, keepdims=True)

with tim("CTL: aligning all trajectories on fit"):
    traj_ctl = [align_traj_to_ref_by_fit(xyz, fit_ctl, ref_fit_ctl) for xyz in traj_aligned]

plt.figure(figsize=(7, 6))
A = []
for xyz in traj_ctl:
    a = xyz[:, sele_ctl, 1:].reshape((-1, 2))
    mask_nonzero = ~((np.abs(a[:, 0]) < 1e-6) & (np.abs(a[:, 1]) < 1e-6))
    A.append(a[mask_nonzero])
A = np.vstack(A)
plt.hist2d(A[:, 0], A[:, 1], bins=256)
plt.gca().set_aspect("equal")
plt.title("CTL aligned coordinate density (YZ)")
plt.xlabel("Y")
plt.ylabel("Z")
save_current_figure(FIG_DIR, "ctl_density_yz.png")

PCA_CTL = run_pca_block(
    traj_list=traj_ctl,
    sel_mask=sele_ctl,
    title_prefix="CTL",
    prefix="ctl",
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

print("Script 2B completed:")
print("- NTL")
print("- CTL")
print("Flow used: Scree -> Scores -> Loadings -> first 10% / last 10%")
print("Mode used: exploratory downsampled PCA, stride =", PCA_STRIDE)
print("Saved figures to:", FIG_DIR)
print("Saved PCA outputs to:", RESULTS)
