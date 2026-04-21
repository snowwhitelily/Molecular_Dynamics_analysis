# %%
# ROS1 Analysis Pipeline
# Script 3B: Distance PCA

import os
from pathlib import Path
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.ros1_utils import ensure_meta, combine_masks
from scripts.ros1_plots import plot_scree
from scripts.timer import Timer

tim = Timer()

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "step3_distance_pca"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Running Script 3B")
print("Figure output folder:", FIG_DIR)

def save_current_figure(filename: str):
    out = FIG_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure:", out)

F = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
traj_aligned = np.load(RESULTS / "traj_aligned.npy", allow_pickle=True).tolist()
masks_npz = np.load(RESULTS / "masks.npz", allow_pickle=True)
masks = {k: masks_npz[k] for k in masks_npz.files}
selection = (RESULTS / "selection.txt").read_text().strip()

meta = SimpleNamespace(
    resids=np.load(RESULTS / "meta_resids.npy", allow_pickle=True),
    names=np.load(RESULTS / "meta_names.npy", allow_pickle=True),
    resnames=np.load(RESULTS / "meta_resnames.npy", allow_pickle=True),
)

# ============================================================
# DISTANCE-PCA (NTL–CTL CA–CA distances) exact notebook logic
# ============================================================
nmut = len(traj_aligned)

meta_obj = ensure_meta(
    meta=meta,
    top=f"{F[0]}/{Path(F[0]).parent.name}-MD-prot.pdb",
    selection=selection,
    natoms=traj_aligned[0].shape[1]
)

CA = masks["CA"]
BODY = masks["BODY"]
ACT = masks["ACT"]
NTL = masks["NTL"]
CTL = masks["CTL"]

mask_ctl = combine_masks(CTL, BODY, CA, ~ACT)
mask_ntl = combine_masks(NTL, BODY, CA)

ctl_idx = np.where(mask_ctl)[0]
ntl_idx = np.where(mask_ntl)[0]

print("CTL CA:", len(ctl_idx), "NTL CA:", len(ntl_idx))
if len(ctl_idx) == 0 or len(ntl_idx) == 0:
    raise ValueError("CTL/NTL CA masks are empty.")

# optional A-loop labels from 02c, if present
labels_act = None
aloop_owner_f = RESULTS / "aloop_x_owner.npy"
aloop_scores_f = RESULTS / "aloop_scores.npy"
if aloop_owner_f.exists():
    # this uses cluster labels if they were saved separately; otherwise left None
    pass

D_list = []
traj_id_list = []
fform_list = []
xmask_list = []

for ti, xyz in enumerate(traj_aligned):
    MCTL = xyz[:, ctl_idx, :]
    MNTL = xyz[:, ntl_idx, :]
    Dij = np.sqrt(((MCTL[:, :, None, :] - MNTL[:, None, :, :]) ** 2).sum(axis=-1))
    D_i = Dij.reshape(xyz.shape[0], -1).astype(np.float32)
    D_list.append(D_i)
    traj_id_list.append(np.full(xyz.shape[0], ti, dtype=int))
    fform_i = np.zeros(xyz.shape[0], dtype=int)
    fform_list.append(fform_i)

    split_idx = min(1, xyz.shape[0])
    xmask_i = np.zeros(xyz.shape[0], dtype=bool)
    xmask_i[split_idx:] = True
    xmask_list.append(xmask_i)

D = np.vstack(D_list)
traj_id_dist = np.concatenate(traj_id_list)
fform_flat_dist = np.concatenate(fform_list)
xmask_dist = np.concatenate(xmask_list)

print("D:", D.shape)

m = D.mean(axis=0, keepdims=True)
Dc = D - m
C = (Dc.T @ Dc) / len(Dc)

evals, evecs = np.linalg.eigh(C)
order = np.argsort(evals)[::-1]
evals = evals[order]
evecs = evecs[:, order]

plot_scree(evals, n=25, title="Scree (Distance PCA)")
save_current_figure("step3b_distance_pca_scree.png")

yscores_dist = Dc @ evecs[:, :3]
print("yscores_dist:", yscores_dist.shape)

scores_flat_dist = yscores_dist.copy()
X_owner_dist = traj_id_dist.copy()

yscores_dist_x = yscores_dist[xmask_dist]
traj_id_dist_x = traj_id_dist[xmask_dist]
fform_flat_dist_x = fform_flat_dist[xmask_dist]

print("yscores_dist_x:", yscores_dist_x.shape)

d_dist = np.exp(-5 * D).sum(axis=1)
d_dist -= d_dist.min() - 1e-6
d_dist /= d_dist.max()
d_dist_x = d_dist[xmask_dist]

np.save(RESULTS / "dist_scores.npy", scores_flat_dist)
np.save(RESULTS / "dist_x_owner.npy", X_owner_dist)
np.save(RESULTS / "dist_scores_x.npy", yscores_dist_x)
np.save(RESULTS / "dist_x_owner_x.npy", traj_id_dist_x)
np.save(RESULTS / "dist_density_scale.npy", d_dist)
np.save(RESULTS / "dist_density_scale_x.npy", d_dist_x)

plt.figure(figsize=(7, 7))
plt.scatter(
    yscores_dist[:, 0],
    yscores_dist[:, 1],
    c=fform_flat_dist,
    s=80 * d_dist,
    alpha=0.5,
    linewidths=0
)
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Distance-PCA (variable-length trajectories)")
save_current_figure("step3b_distance_pca_pc1_pc2.png")

scores_df = pd.DataFrame({
    "traj_index": traj_id_dist,
    "PC1": yscores_dist[:, 0],
    "PC2": yscores_dist[:, 1],
    "PC3": yscores_dist[:, 2],
    "fform": fform_flat_dist,
    "size_scale": d_dist,
})
scores_df.to_csv(RESULTS / "step3b_distance_pca_scores.csv", index=False)

print("Script 3B completed.")
