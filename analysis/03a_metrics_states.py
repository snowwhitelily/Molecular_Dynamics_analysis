# ROS1 Analysis Pipeline
# Script 03a: Active Site Structural Metrics
#
# Computes per-frame and per-trajectory structural metrics for all systems:
#   - A-loop RMSD and binary state (extended vs folded)
#   - DFG-Phe chi1 dihedral angle and DFG-in/out state
#   - alphaC-helix to hinge centre-of-mass distance
#   - K1980-E1993 salt bridge distance
#   - ATP pocket geometry: P-loop/hinge distance, P-loop/DFG distance,
#     gatekeeper/DFG-Phe distance
#
# Residue mapping notes (real residue = internal + 1933):
#   alphaC-helix : real 1983-1993 (internal 50-60)
#   hinge        : real 2031-2038 (internal 98-105)
#   P-loop       : real 1957-1962 (internal 24-29)
#   DFG motif    : real 2042-2044 (internal 109-111)
#   DFG-Phe      : real 2043      (internal 110)
#   K1980        : real 1980
#   E1993        : real 1993

import os
from pathlib import Path
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.ros1_utils import combine_masks
from scripts.ros1_io import get_frame_idx
from scripts.ros1_dihedrals import compute_chi1_deg_for_traj
from scripts.timer import Timer

tim = Timer()

BASE    = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "step3_metrics_states"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Running Script 3A")
print("Figure output folder:", FIG_DIR)


def save_current_figure(filename: str):
    out = FIG_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure:", out)


def save_table(df: pd.DataFrame, filename: str):
    out = RESULTS / filename
    df.to_csv(out, index=False)
    print("Saved table:", out)


def traj_mutant(p):
    return Path(p).parent.name


def traj_replica(p):
    return Path(p).name


def get_frame_index_original(frame_idx_list, ti, local_i):
    return int(np.asarray(frame_idx_list[ti]).astype(int)[local_i])


# ============================================================
# Load preprocessed data from Script 01
# ============================================================
F              = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
traj_aligned   = np.load(RESULTS / "traj_aligned.npy", allow_pickle=True).tolist()
frame_idx_list = np.load(RESULTS / "frame_idx_list.npy", allow_pickle=True).tolist()

# ============================================================
# Exclude F1994L: AlphaFold2 predicted this variant in a distinct
# inactive conformation not comparable to the rest of the panel.
# ============================================================
EXCLUDE_MUTANTS = {"F1994L"}
keep           = [i for i, f in enumerate(F) if Path(f).parent.name not in EXCLUDE_MUTANTS]
F              = [F[i] for i in keep]
traj_aligned   = [traj_aligned[i] for i in keep]
frame_idx_list = [frame_idx_list[i] for i in keep]
print(f"Trajectories after exclusion: {len(F)}")

masks_npz = np.load(RESULTS / "masks.npz", allow_pickle=True)
masks     = {k: masks_npz[k] for k in masks_npz.files}
selection = (RESULTS / "selection.txt").read_text().strip()

meta = SimpleNamespace(
    resids   = np.load(RESULTS / "meta_resids.npy", allow_pickle=True),
    names    = np.load(RESULTS / "meta_names.npy", allow_pickle=True),
    resnames = np.load(RESULTS / "meta_resnames.npy", allow_pickle=True),
)

sele         = np.load(RESULTS / "sele.npy", allow_pickle=True)
ref_centered = np.load(RESULTS / "ref_centered.npy", allow_pickle=True)

print("Loaded trajectories:", len(traj_aligned))
print("Loaded masks:", sorted(masks.keys()))

mut_names = [traj_mutant(p) for p in F]

# ============================================================
# A-loop state classification
# Binary state: 0 = compact (folded), 1 = extended
# Threshold: RMSD > 0.25 nm relative to first frame
# ============================================================
ACT    = masks["ACT"]
BB     = masks["BB"]
sele_act = combine_masks(ACT, BB)
ACT_idx  = np.where(sele_act)[0].astype(int)
thr_act_nm = 0.25

ACT_state        = []
ACT_rmsd_nm_list = []
for xyz in traj_aligned:
    act_xyz  = xyz[:, ACT_idx, :]
    ref0     = act_xyz[0]
    diff     = act_xyz - ref0[None, :, :]
    rmsd_nm  = np.sqrt((diff ** 2).sum(axis=(1, 2)) / ACT_idx.size) / 10.0
    ACT_rmsd_nm_list.append(rmsd_nm)
    ACT_state.append((rmsd_nm > thr_act_nm).astype(np.int8))

act_rows = []
for i, st in enumerate(ACT_state):
    act_rows.append({
        "trajectory"      : F[i],
        "mutant"          : traj_mutant(F[i]),
        "replica"         : traj_replica(F[i]),
        "act_state_frac1" : float(np.mean(st)),
        "act_rmsd_nm_mean": float(np.mean(ACT_rmsd_nm_list[i])),
    })
df_act = pd.DataFrame(act_rows)
save_table(df_act, "step3a_act_state_summary.csv")

plt.figure(figsize=(7, 4))
plt.hist(np.concatenate(ACT_state), bins=2)
plt.xlabel("A-loop state")
plt.ylabel("Count")
plt.title("A-loop state distribution (0/1)")
save_current_figure("step3a_act_state_distribution.png")

# ============================================================
# DFG-Phe chi1 dihedral and DFG-in/out state classification
# DFG-Phe = internal residue 110 = real residue 2043
# DFG-in  ~ chi1 = -70 deg; DFG-out ~ chi1 = +60 or ±170 deg
# ============================================================
DFG_F = 170   # internal residue index for DFG-Phe (real 2043)
manual_thr_deg = None

frame_idx_local_list = [np.asarray(get_frame_idx(d)).astype(int) for d in F]
nframes_sub          = min(len(fi) for fi in frame_idx_local_list)

chi1_deg_list = []
with tim("DFG chi1 per-trajectory"):
    for d in F:
        p   = Path(d)
        mut = p.parent.name
        pdb = str(p / f"{mut}-MD-prot.pdb")
        xtc = str(p / f"{mut}-MD-prot.xtc")
        chi1_deg_list.append(compute_chi1_deg_for_traj(pdb, xtc, DFG_F))

all_vals = np.concatenate([v[np.isfinite(v)] for v in chi1_deg_list])
plt.figure(figsize=(7, 4))
plt.hist(all_vals, bins=120)
plt.title(f"DFG-Phe chi1 distribution (internal {DFG_F}, real {DFG_F + 1933})")
plt.xlabel("chi1 (deg)")
plt.ylabel("count")
save_current_figure("step3a_dfg_chi1_distribution.png")

thr_dfg = float(manual_thr_deg) if manual_thr_deg is not None else float(np.nanmedian(all_vals))
dfg_state_list = [(v > thr_dfg).astype(np.int8) for v in chi1_deg_list]

print("DFG chi1 threshold (deg):", thr_dfg)
print("Overall fraction state=1:", float(np.mean(np.concatenate(dfg_state_list))))

DFG_state    = []
DFG_chi1_sub = []
for ti in range(len(F)):
    fi        = frame_idx_local_list[ti]
    full      = np.asarray(dfg_state_list[ti], dtype=np.int8)
    full_chi1 = np.asarray(chi1_deg_list[ti], dtype=float)
    if fi.max() >= full.size:
        raise ValueError(
            f"Traj {ti} ({F[ti]}): max(fi)={int(fi.max())} "
            f"but dfg_state length={full.size}."
        )
    DFG_state.append(full[fi][:nframes_sub])
    DFG_chi1_sub.append(full_chi1[fi][:nframes_sub])

df_dfg = pd.DataFrame({
    "trajectory"      : F,
    "mutant"          : [traj_mutant(p) for p in F],
    "replica"         : [traj_replica(p) for p in F],
    "dfg_state_frac1" : [float(np.mean(x)) for x in DFG_state],
    "dfg_chi1_deg_mean": [float(np.nanmean(x)) for x in DFG_chi1_sub],
})
save_table(df_dfg, "step3a_dfg_state_summary.csv")

# ============================================================
# alphaC-helix displacement and K1980-E1993 salt bridge distance
#
# alphaC-helix region : real 1983-1993
# hinge region        : real 2031-2038
# K1980               : real 1980 (salt bridge donor)
# E1993               : real 1993 (salt bridge acceptor)
# Salt bridge threshold: 0.4 nm (never met in apo simulations — see patch script)
# ============================================================
resids_abs = np.asarray(meta.resids).astype(int) + 1933

REGIONS = {
    "alphaC": (1983, 1993),
    "hinge" : (2031, 2038),
    "P_loop": (1957, 1962),
    "DFG"   : (2102, 2104),
}

def region_mask(resids, start, end):
    return (resids >= start) & (resids <= end)

mask_alphaC = region_mask(resids_abs, *REGIONS["alphaC"])
mask_hinge  = region_mask(resids_abs, *REGIONS["hinge"])

thr_alphaC = 0.4   # nm salt bridge threshold

# K1980 and E1993 atom indices (Calpha used as proxy)
CA = masks["CA"]
resids_ca = resids_abs[CA]
ca_idx    = np.where(CA)[0]

def find_ca_idx(target_real):
    hits = np.where(resids_ca == target_real)[0]
    return int(ca_idx[hits[0]]) if len(hits) else None

idx_lys = find_ca_idx(1980)
idx_glu = find_ca_idx(1993)

print(f"K1980 CA index: {idx_lys}")
print(f"E1993 CA index: {idx_glu}")

alphaC_hinge_dist_per_traj = []
lys_glu_dist_per_traj      = []
alphaC_rows                = []

for ti, xyz in enumerate(traj_aligned):
    alphaC_com = xyz[:, mask_alphaC, :].mean(axis=1)
    hinge_com  = xyz[:, mask_hinge,  :].mean(axis=1)
    d_alphaC   = np.linalg.norm(alphaC_com - hinge_com, axis=1)
    alphaC_hinge_dist_per_traj.append(d_alphaC)

    if idx_lys is not None and idx_glu is not None:
        d_sb = np.linalg.norm(xyz[:, idx_lys, :] - xyz[:, idx_glu, :], axis=1)
        lys_glu_dist_per_traj.append(d_sb)
        compact_frac = float((d_sb <= thr_alphaC).mean())
    else:
        lys_glu_dist_per_traj.append(None)
        d_sb         = np.full(xyz.shape[0], np.nan)
        compact_frac = np.nan

    alphaC_rows.append({
        "trajectory"           : F[ti],
        "mutant"               : traj_mutant(F[ti]),
        "replica"              : traj_replica(F[ti]),
        "alphaC_hinge_mean_dist": float(d_alphaC.mean()),
        "alphaC_hinge_std_dist" : float(d_alphaC.std()),
        "alphaC_rmsf_mean"     : float(d_alphaC.std()),
        "alphaC_compact_frac"  : compact_frac,
        "lys_glu_mean_dist_nm" : float(np.nanmean(d_sb)),
    })

df_alphaC = pd.DataFrame(alphaC_rows)
save_table(df_alphaC, "step3a_alphaC_metrics.csv")

# ============================================================
# ATP pocket geometry
# Front pocket: P-loop COM to hinge COM distance
# Back pocket:  gatekeeper (L2026) CA to DFG-Phe (F2043) CA distance
# Open-state: frame is open if front pocket OR back pocket distance
#             exceeds the median across all frames
# ============================================================
mask_ploop = region_mask(resids_abs, *REGIONS["P_loop"])
mask_dfg   = region_mask(resids_abs, *REGIONS["DFG"])

idx_gate = find_ca_idx(2026)
idx_dfgF = find_ca_idx(2103)
print(f"Gatekeeper L2026 CA index: {idx_gate}")
print(f"DFG-Phe F2043 CA index: {idx_dfgF}")

pocket_front_mean       = []
pocket_front_std        = []
pocket_dfg_mean         = []
pocket_dfg_std          = []
pocket_open_frac        = []
gate_dfgF_mean          = []
gate_dfgF_std           = []
per_traj_front          = []
per_traj_back           = []
all_front               = []
all_back                = []

for ti, xyz in enumerate(traj_aligned):
    ploop_com = xyz[:, mask_ploop, :].mean(axis=1)
    hinge_com = xyz[:, mask_hinge, :].mean(axis=1)
    dfg_com   = xyz[:, mask_dfg,   :].mean(axis=1)

    d_ploop_hinge_i = np.linalg.norm(ploop_com - hinge_com, axis=1)
    d_ploop_dfg_i   = np.linalg.norm(ploop_com - dfg_com,   axis=1)

    per_traj_front.append(d_ploop_hinge_i)
    pocket_front_mean.append(float(d_ploop_hinge_i.mean()))
    pocket_front_std.append(float(d_ploop_hinge_i.std()))
    pocket_dfg_mean.append(float(d_ploop_dfg_i.mean()))
    pocket_dfg_std.append(float(d_ploop_dfg_i.std()))
    all_front.append(d_ploop_hinge_i)

    if idx_gate is not None and idx_dfgF is not None:
        d_gate_dfgF_i = np.linalg.norm(
            xyz[:, idx_gate, :] - xyz[:, idx_dfgF, :], axis=1
        )
        per_traj_back.append(d_gate_dfgF_i)
        all_back.append(d_gate_dfgF_i)
    else:
        per_traj_back.append(None)

all_front_flat = np.concatenate(all_front)
thr_front      = np.nanmedian(all_front_flat)

if idx_gate is not None and idx_dfgF is not None and all_back:
    all_back_flat = np.concatenate(all_back)
    thr_back      = np.nanmedian(all_back_flat)
    print(f"Pocket front threshold (P-loop/hinge): {thr_front:.3f} nm")
    print(f"Gatekeeper/DFG-Phe threshold: {thr_back:.3f} nm")
else:
    thr_back = np.nan
    print(f"Pocket front threshold: {thr_front:.3f} nm")
    print("Gatekeeper/DFG-Phe indices not found; skipping back pocket metric.")

pocket_open_state_per_traj = []
for i in range(len(traj_aligned)):
    front_open_i = (per_traj_front[i] >= thr_front).astype(int)
    if np.isfinite(thr_back) and per_traj_back[i] is not None:
        back_open_i  = (per_traj_back[i] >= thr_back).astype(int)
        pocket_open_i = ((front_open_i + back_open_i) >= 1).astype(int)
        gate_dfgF_mean.append(float(np.nanmean(per_traj_back[i])))
        gate_dfgF_std.append(float(np.nanstd(per_traj_back[i])))
    else:
        pocket_open_i = front_open_i.copy()
    pocket_open_state_per_traj.append(pocket_open_i)
    pocket_open_frac.append(float(pocket_open_i.mean()))

df_pocket = pd.DataFrame({
    "trajectory"       : F,
    "mutant"           : mut_names,
    "replica"          : [traj_replica(p) for p in F],
    "pocket_front_mean": pocket_front_mean,
    "pocket_front_std" : pocket_front_std,
    "pocket_dfg_mean"  : pocket_dfg_mean,
    "pocket_dfg_std"   : pocket_dfg_std,
    "pocket_open_frac" : pocket_open_frac,
})
if np.isfinite(thr_back):
    df_pocket["gate_dfgF_mean"] = gate_dfgF_mean
    df_pocket["gate_dfgF_std"]  = gate_dfgF_std
save_table(df_pocket, "step3a_atp_pocket_metrics.csv")

df_pocket_mut = (
    df_pocket.groupby("mutant")
    .mean(numeric_only=True)
    .sort_values("pocket_open_frac", ascending=False)
)
save_table(df_pocket_mut.reset_index(), "step3a_atp_pocket_metrics_by_mutant.csv")

plt.figure(figsize=(11, 4))
plt.bar(df_pocket_mut.index, df_pocket_mut["pocket_front_mean"])
plt.ylabel("P-loop to hinge distance (nm)")
plt.title("Per-mutant front-pocket opening")
plt.xticks(rotation=90)
save_current_figure("step3a_pocket_front_opening.png")

plt.figure(figsize=(11, 4))
plt.bar(df_pocket_mut.index, df_pocket_mut["pocket_dfg_mean"])
plt.ylabel("P-loop to DFG distance (nm)")
plt.title("Per-mutant pocket-to-activation-segment coupling")
plt.xticks(rotation=90)
save_current_figure("step3a_pocket_dfg_coupling.png")

plt.figure(figsize=(11, 4))
plt.bar(df_pocket_mut.index, df_pocket_mut["pocket_open_frac"])
plt.ylabel("fraction pocket-open frames")
plt.title("Per-mutant ATP-pocket open-state occupancy")
plt.xticks(rotation=90)
save_current_figure("step3a_pocket_open_frac.png")

# ============================================================
# Frame-level metrics table (full resolution, for cluster mapping)
# ============================================================
frame_rows = []
for ti, xyz in enumerate(traj_aligned):
    nframes_i = xyz.shape[0]
    for local_i in range(nframes_i):
        row = {
            "traj_index"          : ti,
            "trajectory"          : F[ti],
            "mutant"              : traj_mutant(F[ti]),
            "replica"             : traj_replica(F[ti]),
            "frame_index_local"   : int(local_i),
            "frame_index_original": get_frame_index_original(frame_idx_list, ti, local_i),
            "aloop_rmsd_nm"       : float(ACT_rmsd_nm_list[ti][local_i]),
            "aloop_state"         : int(ACT_state[ti][local_i]),
            "alphaC_hinge_dist_nm": float(alphaC_hinge_dist_per_traj[ti][local_i]),
            "pocket_front_dist_nm": float(per_traj_front[ti][local_i]),
            "pocket_dfg_dist_nm"  : 0.0,  # filled in below
            "pocket_open_state"   : int(pocket_open_state_per_traj[ti][local_i]),
        }
        if local_i < len(DFG_chi1_sub[ti]):
            row["dfg_chi1_deg"] = float(DFG_chi1_sub[ti][local_i])
            row["dfg_state"]    = int(DFG_state[ti][local_i])
        else:
            row["dfg_chi1_deg"] = np.nan
            row["dfg_state"]    = np.nan

        if (idx_lys is not None and idx_glu is not None
                and lys_glu_dist_per_traj[ti] is not None):
            row["lys_glu_dist_nm"]      = float(lys_glu_dist_per_traj[ti][local_i])
            row["alphaC_compact_state"] = int(lys_glu_dist_per_traj[ti][local_i] <= thr_alphaC)
        else:
            row["lys_glu_dist_nm"]      = np.nan
            row["alphaC_compact_state"] = np.nan

        frame_rows.append(row)

frame_df = pd.DataFrame(frame_rows)

# Fill in true pocket_dfg_dist_nm (P-loop COM to DFG COM per frame)
true_pocket_dfg = []
for ti in range(len(traj_aligned)):
    for local_i in range(traj_aligned[ti].shape[0]):
        true_pocket_dfg.append(float(np.linalg.norm(
            traj_aligned[ti][local_i, mask_ploop, :].mean(axis=0) -
            traj_aligned[ti][local_i, mask_dfg,   :].mean(axis=0)
        )))
frame_df["pocket_dfg_dist_nm"] = true_pocket_dfg
save_table(frame_df, "step3a_frame_metrics.csv")

# Downsampled metrics table for PCA cluster mapping (stride 10 matches PCA stride)
pca_stride   = 10
frame_df_down = (
    frame_df
    .groupby("traj_index", group_keys=False)
    .apply(lambda x: x.iloc[::pca_stride].copy())
    .reset_index(drop=True)
)
frame_df_down["frame_index_downsampled"] = (
    frame_df_down.groupby("traj_index").cumcount().astype(int)
)
save_table(frame_df_down, "step3a_frame_metrics_downsampled.csv")

# ============================================================
# State transition analysis
# ============================================================
def switch_count(x01):
    """Count the number of state transitions in a binary label sequence."""
    x = np.asarray(x01, dtype=np.int8)
    if x.size <= 1:
        return 0
    return int(np.sum(x[1:] != x[:-1]))


def mean_dwell_frames(x01):
    """Mean number of consecutive frames spent in a single state."""
    x = np.asarray(x01, dtype=np.int8)
    if x.size == 0:
        return np.nan
    edges = np.where(np.diff(x) != 0)[0] + 1
    runs  = np.diff(np.r_[0, edges, len(x)])
    return float(np.mean(runs)) if len(runs) else float(len(x))


df_trans = pd.DataFrame({
    "trajectory"            : F,
    "mutant"                : mut_names,
    "replica"               : [traj_replica(p) for p in F],
    "act_switches"          : [switch_count(x) for x in ACT_state],
    "act_mean_dwell_frames" : [mean_dwell_frames(x) for x in ACT_state],
    "dfg_switches"          : [switch_count(x) for x in DFG_state],
    "dfg_mean_dwell_frames" : [mean_dwell_frames(x) for x in DFG_state],
    "pocket_switches"       : [switch_count(x) for x in pocket_open_state_per_traj],
    "pocket_mean_dwell_frames": [mean_dwell_frames(x) for x in pocket_open_state_per_traj],
})
save_table(df_trans, "step3a_transition_summary.csv")

print("\nScript 3A completed. Outputs saved to:", RESULTS)