# %%
# ROS1 Analysis Pipeline
# Script 2A: PCA (ACT-IN + ACT-OUT)

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
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "step2_fast" / "02a_actin_actout"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Running exploratory PCA mode")
print("PCA_STRIDE =", PCA_STRIDE)
print("Figure output folder:", FIG_DIR)

F = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
traj_aligned = np.load(RESULTS / "traj_aligned.npy", allow_pickle=True).tolist()
frame_idx_list = np.load(RESULTS / "frame_idx_list.npy", allow_pickle=True).tolist()

# ============================================================
# EXCLUDE MUTANTS NOT IN ANALYSIS
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
ACT = masks["ACT"]

sele = np.load(RESULTS / "sele.npy", allow_pickle=True)
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

print("Step 2A setup ok.")
print("n trajectories:", len(traj_aligned))
print("example F:", F[0])
print("sele atoms:", int(sele.sum()))

traj_colors = make_traj_colors(F, colors, nrep)

PCA_ACT_IN = run_pca_block(
    traj_list=traj_aligned,
    sel_mask=sele,
    title_prefix="ACT-IN",
    prefix="actin",
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

sele_actout = combine_masks(BB, BODY, ~ACT)
fit_actout = sele_actout

print("ACT-OUT sele atoms:", int(sele_actout.sum()))
print("ACT-OUT fit atoms:", int(fit_actout.sum()))

ref_fit_actout = ref_centered[fit_actout]
ref_fit_actout = ref_fit_actout - ref_fit_actout.mean(axis=0, keepdims=True)

with tim("ACT-OUT: aligning all trajectories on fit"):
    traj_actout = [align_traj_to_ref_by_fit(xyz, fit_actout, ref_fit_actout) for xyz in traj_aligned]

plt.figure(figsize=(7, 6))
A = []
for xyz in traj_actout:
    a = xyz[:, sele_actout, 1:].reshape((-1, 2))
    mask_nonzero = ~((np.abs(a[:, 0]) < 1e-6) & (np.abs(a[:, 1]) < 1e-6))
    A.append(a[mask_nonzero])
A = np.vstack(A)
plt.hist2d(A[:, 0], A[:, 1], bins=256)
plt.gca().set_aspect("equal")
plt.title("ACT-OUT aligned coordinate density (YZ)")
plt.xlabel("Y")
plt.ylabel("Z")
save_current_figure(FIG_DIR, "actout_density_yz.png")

PCA_ACT_OUT = run_pca_block(
    traj_list=traj_actout,
    sel_mask=sele_actout,
    title_prefix="ACT-OUT",
    prefix="actout",
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

print("Script 2A completed:")
print("- ACT-IN")
print("- ACT-OUT")
print("Flow used: Scree -> Scores -> Loadings -> first 10% / last 10%")
print("Mode used: exploratory downsampled PCA, stride =", PCA_STRIDE)
print("Saved figures to:", FIG_DIR)
print("Saved PCA outputs to:", RESULTS)
