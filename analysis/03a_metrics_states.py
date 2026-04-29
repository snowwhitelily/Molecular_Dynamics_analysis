# %%
# ROS1 Analysis Pipeline
# Script 3A: Metrics and State Analysis
#
# CORRECTIONS FROM ORIGINAL:
# 1. REGIONS boundaries corrected (all four were wrong):
#       alphaC: (1960,1975) -> (1983,1993)
#       hinge:  (2011,2016) -> (2031,2038)
#       P_loop: (1978,1987) -> (1957,1962)
#       DFG:    (2103,2105) -> (2042,2044)
# 2. Salt bridge glutamate corrected: resid 1967 -> 1993
#       Salt bridge is K1980-E1993, not K1980-E1967
# 3. DFG_F (DFG-Phe internal residue) corrected: 170 -> 110
#       Internal 170 = real 2103 (wrong)
#       Internal 110 = real 2043 = DFG-Phe (correct)
# 4. idx_dfgF atom selection corrected: resid 2104 -> 2043
# 5. Placeholder pocket_dfg_dist_nm cleaned up (was confusing duplicate)
#
# Metrics computed:
#   - A-loop RMSD state (0/1 binary per frame)
#   - DFG chi1 angle and DFG-in/DFG-out state
#   - alphaC-helix to hinge distance (alphaC displacement)
#   - K1980-E1993 salt bridge distance (alphaC-in/out indicator)
#   - ATP pocket geometry: P-loop<->hinge, P-loop<->DFG, gatekeeper<->DFG-Phe

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
from scripts.ros1_io import get_frame_idx
from scripts.ros1_dihedrals import compute_chi1_deg_for_traj
from scripts.timer import Timer

tim = Timer()

BASE = Path.home() / "Molecular_Dynamics_analysis"
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


# ============================================
# Load outputs from Step 1
# ============================================
F = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
traj_aligned = np.load(RESULTS / "traj_aligned.npy", allow_pickle=True).tolist()
frame_idx_list = np.load(RESULTS / "frame_idx_list.npy", allow_pickle=True).tolist()
masks_npz = np.load(RESULTS / "masks.npz", allow_pickle=True)
masks = {k: masks_npz[k] for k in masks_npz.files}
selection = (RESULTS / "selection.txt").read_text().strip()

meta = SimpleNamespace(
    resids=np.load(RESULTS / "meta_resids.npy", allow_pickle=True),
    names=np.load(RESULTS / "meta_names.npy", allow_pickle=True),
    resnames=np.load(RESULTS / "meta_resnames.npy", allow_pickle=True),
)

sele = np.load(RESULTS / "sele.npy", allow_pickle=True)
ref_centered = np.load(RESULTS / "ref_centered.npy", allow_pickle=True)

print("Loaded trajectories:", len(traj_aligned))
print("Loaded masks:", sorted(masks.keys()))


# ============================================
# A-loop state classification
# ============================================
ACT = masks["ACT"]
BB = masks["BB"]

sele_act = combine_masks(ACT, BB)
ACT_idx = np.where(sele_act)[0].astype(int)

thr_act_nm = 0.25
ACT_state = []
ACT_rmsd_nm_list = []
for xyz in traj_aligned:
    act_xyz = xyz[:, ACT_idx, :]
    ref0 = act_xyz[0]
    diff = act_xyz - ref0[None, :, :]
    rmsd_nm = np.sqrt((diff ** 2).sum(axis=(1, 2)) / ACT_idx.size) / 10.0
    ACT_rmsd_nm_list.append(rmsd_nm)
    ACT_state.append((rmsd_nm > thr_act_nm).astype(np.int8))

act_rows = []
for i, st in enumerate(ACT_state):
    act_rows.append({
        "trajectory": F[i],
        "mutant": traj_mutant(F[i]),
        "replica": traj_replica(F[i]),
        "act_state_frac1": float(np.mean(st)),
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


# ============================================
# DFG chi1 labeling
# CORRECTION: DFG_F changed from 170 to 110
#   Original: internal 170 = real 2103 (wrong — that is in substrate binding loop)
#   Correct:  internal 110 = real 2043 = DFG-Phe (the F in the DFG motif)
#   DFG motif is at real 2042-2044 (internal 109-111)
# ============================================
DFG_F = 110   # CORRECTED from 170 — internal residue for DFG-Phe (real 2043)
manual_thr_deg = None

frame_idx_local_list = [get_frame_idx(d) for d in F]
frame_idx_local_list = [np.asarray(x).astype(int) for x in frame_idx_local_list]
nframes_sub = min(len(fi) for fi in frame_idx_local_list)

chi1_deg_list = []
with tim("DFG chi1 (REAL) per-trajectory computation"):
    for d in F:
        p = Path(d)
        mut = p.parent.name
        pdb = str(p / f"{mut}-MD-prot.pdb")
        xtc = str(p / f"{mut}-MD-prot.xtc")
        chi1_deg_list.append(compute_chi1_deg_for_traj(pdb, xtc, DFG_F))

all_vals = np.concatenate([v[np.isfinite(v)] for v in chi1_deg_list])
plt.figure(figsize=(7, 4))
plt.hist(all_vals, bins=120)
plt.title(f"DFG-Phe chi1 distribution (internal resid {DFG_F}, real {DFG_F + 1933})")
plt.xlabel("chi1 (deg)")
plt.ylabel("count")
save_current_figure("step3a_dfg_chi1_distribution.png")

if manual_thr_deg is not None:
    thr_dfg = float(manual_thr_deg)
else:
    thr_dfg = float(np.nanmedian(all_vals))

dfg_state_list = [(v > thr_dfg).astype(np.int8) for v in chi1_deg_list]

print("DFG chi1 threshold (deg):", thr_dfg)
print("Overall fraction state=1:", float(np.mean(np.concatenate(dfg_state_list))))

DFG_state = []
DFG_chi1_sub = []
for ti in range(len(F)):
    fi = frame_idx_local_list[ti]
    full = np.asarray(dfg_state_list[ti], dtype=np.int8)
    full_chi1 = np.asarray(chi1_deg_list[ti], dtype=float)
    if fi.max() >= full.size:
        raise ValueError(
            f"[DFG BRIDGE] Traj {ti} ({F[ti]}): max(fi)={int(fi.max())} "
            f"but dfg_state_list length={full.size}."
        )
    DFG_state.append(full[fi][:nframes_sub])
    DFG_chi1_sub.append(full_chi1[fi][:nframes_sub])

df_dfg = pd.DataFrame({
    "trajectory": F,
    "mutant": [traj_mutant(p) for p in F],
    "replica": [traj_replica(p) for p in F],
    "dfg_state_frac1": [float(np.mean(x)) for x in DFG_state],
    "dfg_chi1_deg_mean": [float(np.nanmean(x)) for x in DFG_chi1_sub],
})
save_table(df_dfg, "step3a_dfg_state_summary.csv")


# ============================================
# Structural-region helpers
# CORRECTION: All four REGIONS boundaries corrected.
#   Real residue = internal + 1933.
#   Previous values were all incorrect.
# ============================================
def get_abs_resids(meta):
    return np.asarray(meta.resids).astype(int) + 1933

REGIONS = {
    # CORRECTED from (1960, 1975)
    # alphaC-helix: internal ~50-60, real 1983-1993
    "alphaC": (1983, 1993),

    # CORRECTED from (2011, 2016)
    # hinge region: connects NTL to CTL, real 2031-2038
    "hinge":  (2031, 2038),

    # CORRECTED from (1978, 1987)
    # P-loop (glycine-rich loop): real 1957-1962
    "P_loop": (1957, 1962),

    # CORRECTED from (2103, 2105)
    # DFG motif: Asp-Phe-Gly at real 2042-2044
    "DFG":    (2042, 2044),
}
print("Defined REGIONS (corrected):", REGIONS)


# ============================================
# alphaC-HELIX METRICS
# ============================================
resids_abs = get_abs_resids(meta)
names = np.asarray(meta.names)
mut_names = np.array([traj_mutant(p) for p in F])

mask_alphaC = (
    (resids_abs >= REGIONS["alphaC"][0]) &
    (resids_abs <= REGIONS["alphaC"][1]) &
    (names == "CA")
)
mask_pocket_anchor = (
    (resids_abs >= REGIONS["hinge"][0]) &
    (resids_abs <= REGIONS["hinge"][1]) &
    (names == "CA")
)

print("alphaC CA atoms:", int(mask_alphaC.sum()),
      "| hinge CA atoms:", int(mask_pocket_anchor.sum()))

if mask_alphaC.sum() == 0:
    raise ValueError(
        "alphaC mask is empty. Check REGIONS['alphaC'] boundaries "
        f"(currently {REGIONS['alphaC']}) against meta_resids."
    )
if mask_pocket_anchor.sum() == 0:
    raise ValueError(
        "hinge mask is empty. Check REGIONS['hinge'] boundaries "
        f"(currently {REGIONS['hinge']}) against meta_resids."
    )


def first_existing(mask):
    idx = np.where(mask)[0]
    return None if len(idx) == 0 else idx[0]


# CORRECTION: salt bridge is K1980 NZ -- E1993 OE (confirmed in structural analysis)
# Original had E1967 which is wrong — E1967 is not the alphaC-helix glutamate
idx_lys = first_existing((resids_abs == 1980) & np.isin(names, ["NZ", "CA"]))
idx_glu = first_existing((resids_abs == 1993) & np.isin(names, ["OE1", "OE2", "CA"]))
# CORRECTED: was resid 1967, now 1993

if idx_lys is None:
    print("WARNING: K1980 (NZ/CA) not found in selection. Salt bridge metric skipped.")
if idx_glu is None:
    print("WARNING: E1993 (OE1/OE2/CA) not found in selection. Salt bridge metric skipped.")

alphaC_rmsf_mean_list = []
alphaC_hinge_mean_dist_list = []
alphaC_hinge_std_dist_list = []
alphaC_compact_frac_list = []
all_d_lys_glu = []
alphaC_hinge_dist_per_traj = []
lys_glu_dist_per_traj = []

# Threshold for alphaC-compact state (salt bridge formed = alphaC-in = active-like)
# Salt bridge K1980-E1993 is considered formed when distance <= 0.4 nm
thr_alphaC = 0.4  # nm

for xyz in traj_aligned:
    # alphaC centre of mass per frame
    alphaC_com = xyz[:, mask_alphaC, :].mean(axis=1)
    # hinge centre of mass per frame
    hinge_com = xyz[:, mask_pocket_anchor, :].mean(axis=1)
    # distance between alphaC COM and hinge COM per frame
    d_alphaC_hinge = np.linalg.norm(alphaC_com - hinge_com, axis=1)

    alphaC_hinge_dist_per_traj.append(d_alphaC_hinge)
    alphaC_hinge_mean_dist_list.append(float(d_alphaC_hinge.mean()))
    alphaC_hinge_std_dist_list.append(float(d_alphaC_hinge.std()))

    # alphaC RMSF (fluctuation within trajectory)
    alphaC_xyz = xyz[:, mask_alphaC, :]
    alphaC_mean = alphaC_xyz.mean(axis=0)
    rmsf = float(np.sqrt(((alphaC_xyz - alphaC_mean[None]) ** 2).sum(axis=2).mean()))
    alphaC_rmsf_mean_list.append(rmsf)

    # Salt bridge K1980-E1993 distance per frame
    if idx_lys is not None and idx_glu is not None:
        d_salt = np.linalg.norm(
            xyz[:, idx_lys, :] - xyz[:, idx_glu, :], axis=1
        )
        lys_glu_dist_per_traj.append(d_salt)
        all_d_lys_glu.append(d_salt)
        compact_frac = float(np.mean(d_salt <= thr_alphaC))
        alphaC_compact_frac_list.append(compact_frac)
    else:
        lys_glu_dist_per_traj.append(None)
        alphaC_compact_frac_list.append(np.nan)

df_alphaC = pd.DataFrame({
    "trajectory": F,
    "mutant": mut_names,
    "replica": [traj_replica(p) for p in F],
    "alphaC_hinge_mean_dist": alphaC_hinge_mean_dist_list,
    "alphaC_hinge_std_dist": alphaC_hinge_std_dist_list,
    "alphaC_rmsf_mean": alphaC_rmsf_mean_list,
    "alphaC_compact_frac": alphaC_compact_frac_list,
})
save_table(df_alphaC, "step3a_alphaC_metrics.csv")

df_alphaC_mut = df_alphaC.groupby("mutant").mean(numeric_only=True)

plt.figure(figsize=(11, 4))
plt.bar(df_alphaC_mut.index, df_alphaC_mut["alphaC_hinge_mean_dist"])
plt.ylabel("mean alphaC-to-hinge COM distance (nm)")
plt.title("Per-mutant alphaC displacement relative to hinge\n"
          "(alphaC real 1983-1993, hinge real 2031-2038)")
plt.xticks(rotation=90)
save_current_figure("step3a_alphaC_hinge_distance.png")

if "alphaC_compact_frac" in df_alphaC_mut.columns:
    plt.figure(figsize=(11, 4))
    plt.bar(df_alphaC_mut.index, df_alphaC_mut["alphaC_compact_frac"])
    plt.ylabel("fraction alphaC-compact frames\n(K1980-E1993 salt bridge <= 0.4 nm)")
    plt.title("Per-mutant alphaC-in (active-like) occupancy\n"
              "Salt bridge: K1980 NZ -- E1993 OE (CORRECTED from K1980-E1967)")
    plt.xticks(rotation=90)
    save_current_figure("step3a_alphaC_compact_frac.png")

    # Also plot the raw salt bridge distance distribution
    if all_d_lys_glu:
        plt.figure(figsize=(7, 4))
        plt.hist(np.concatenate(all_d_lys_glu), bins=100)
        plt.axvline(thr_alphaC, color="red", linestyle="--",
                    label=f"threshold {thr_alphaC} nm")
        plt.xlabel("K1980-E1993 distance (nm)")
        plt.ylabel("count")
        plt.title("Salt bridge distance distribution (all trajectories)\n"
                  "alphaC-in = distance <= 0.4 nm")
        plt.legend()
        save_current_figure("step3a_salt_bridge_distribution.png")


# ============================================
# ATP-POCKET GEOMETRY METRICS
# CORRECTION: mask_ploop, mask_hinge, mask_dfg now use corrected REGIONS
# CORRECTION: idx_dfgF corrected from resid 2104 to resid 2043
# ============================================
mask_ploop = (
    (resids_abs >= REGIONS["P_loop"][0]) &
    (resids_abs <= REGIONS["P_loop"][1]) &
    (names == "CA")
)
mask_hinge = (
    (resids_abs >= REGIONS["hinge"][0]) &
    (resids_abs <= REGIONS["hinge"][1]) &
    (names == "CA")
)
mask_dfg = (
    (resids_abs >= REGIONS["DFG"][0]) &
    (resids_abs <= REGIONS["DFG"][1]) &
    (names == "CA")
)

print("P-loop CA atoms:", int(mask_ploop.sum()))
print("hinge CA atoms:", int(mask_hinge.sum()))
print("DFG CA atoms:", int(mask_dfg.sum()))

if mask_ploop.sum() == 0:
    raise ValueError(
        f"P-loop mask is empty. Check REGIONS['P_loop'] = {REGIONS['P_loop']}"
    )
if mask_dfg.sum() == 0:
    raise ValueError(
        f"DFG mask is empty. Check REGIONS['DFG'] = {REGIONS['DFG']}"
    )

# Gatekeeper residue: L2026 (sits at entrance to ATP pocket)
idx_gate = first_existing((resids_abs == 2026) & (names == "CA"))

# CORRECTION: DFG-Phe is at real 2043 (not 2104)
# Original had resid 2104 which is in the substrate binding loop — wrong
idx_dfgF = first_existing(
    (resids_abs == 2043) & np.isin(names, ["CZ", "CE1", "CE2", "CA"])
)

if idx_gate is None:
    print("WARNING: Gatekeeper L2026 CA not found. Gate-DFG distance metric skipped.")
if idx_dfgF is None:
    print("WARNING: DFG-Phe F2043 aromatic/CA atom not found. "
          "Gate-DFG distance metric skipped.")

pocket_front_mean = []
pocket_front_std = []
pocket_dfg_mean = []
pocket_dfg_std = []
pocket_open_frac = []
gate_dfgF_mean = []
gate_dfgF_std = []

all_front = []
all_back = []
per_traj_front = []
per_traj_back = []

for xyz in traj_aligned:
    ploop_com = xyz[:, mask_ploop, :].mean(axis=1)
    hinge_com = xyz[:, mask_hinge, :].mean(axis=1)
    dfg_com = xyz[:, mask_dfg, :].mean(axis=1)

    # Front pocket: P-loop to hinge distance
    # When this increases, the front of the ATP pocket is more open
    d_ploop_hinge_i = np.linalg.norm(ploop_com - hinge_com, axis=1)

    # P-loop to DFG distance: coupling between P-loop and activation segment
    d_ploop_dfg_i = np.linalg.norm(ploop_com - dfg_com, axis=1)

    per_traj_front.append(d_ploop_hinge_i)
    pocket_front_mean.append(float(d_ploop_hinge_i.mean()))
    pocket_front_std.append(float(d_ploop_hinge_i.std()))
    pocket_dfg_mean.append(float(d_ploop_dfg_i.mean()))
    pocket_dfg_std.append(float(d_ploop_dfg_i.std()))
    all_front.append(d_ploop_hinge_i)

    # Back pocket: gatekeeper to DFG-Phe distance
    if idx_gate is not None and idx_dfgF is not None:
        d_gate_dfgF_i = np.linalg.norm(
            xyz[:, idx_gate, :] - xyz[:, idx_dfgF, :], axis=1
        )
        per_traj_back.append(d_gate_dfgF_i)
        all_back.append(d_gate_dfgF_i)
    else:
        per_traj_back.append(None)

all_front_flat = np.concatenate(all_front)
thr_front = np.nanmedian(all_front_flat)

if idx_gate is not None and idx_dfgF is not None and len(all_back):
    all_back_flat = np.concatenate(all_back)
    thr_back = np.nanmedian(all_back_flat)
    print(f"Pocket front threshold (P-loop<->hinge): {thr_front:.3f} nm")
    print(f"Gatekeeper(L2026)--DFG-Phe(F2043) threshold: {thr_back:.3f} nm")
else:
    thr_back = np.nan
    print(f"Pocket front threshold: {thr_front:.3f} nm")
    print("Gatekeeper/DFG-Phe atom selection missing; skipping that metric.")

pocket_open_state_per_traj = []
for i in range(len(traj_aligned)):
    front_open_i = (per_traj_front[i] >= thr_front).astype(int)
    if np.isfinite(thr_back) and per_traj_back[i] is not None:
        back_open_i = (per_traj_back[i] >= thr_back).astype(int)
        pocket_open_i = ((front_open_i + back_open_i) >= 1).astype(int)
        gate_dfgF_mean.append(float(np.nanmean(per_traj_back[i])))
        gate_dfgF_std.append(float(np.nanstd(per_traj_back[i])))
    else:
        pocket_open_i = front_open_i.copy()
    pocket_open_state_per_traj.append(pocket_open_i)
    pocket_open_frac.append(float(pocket_open_i.mean()))

df_pocket = pd.DataFrame({
    "trajectory": F,
    "mutant": mut_names,
    "replica": [traj_replica(p) for p in F],
    "pocket_front_mean": pocket_front_mean,
    "pocket_front_std": pocket_front_std,
    "pocket_dfg_mean": pocket_dfg_mean,
    "pocket_dfg_std": pocket_dfg_std,
    "pocket_open_frac": pocket_open_frac,
})
if np.isfinite(thr_back):
    df_pocket["gate_dfgF_mean"] = gate_dfgF_mean
    df_pocket["gate_dfgF_std"] = gate_dfgF_std
save_table(df_pocket, "step3a_atp_pocket_metrics.csv")

df_pocket_mut = (
    df_pocket.groupby("mutant")
    .mean(numeric_only=True)
    .sort_values("pocket_open_frac", ascending=False)
)
save_table(df_pocket_mut.reset_index(), "step3a_atp_pocket_metrics_by_mutant.csv")

plt.figure(figsize=(11, 4))
plt.bar(df_pocket_mut.index, df_pocket_mut["pocket_front_mean"])
plt.ylabel("P-loop to hinge distance (nm)\n(real 1957-1962 to real 2031-2038)")
plt.title("Per-mutant front-pocket opening\n(larger = more open ATP pocket front)")
plt.xticks(rotation=90)
save_current_figure("step3a_pocket_front_opening.png")

plt.figure(figsize=(11, 4))
plt.bar(df_pocket_mut.index, df_pocket_mut["pocket_dfg_mean"])
plt.ylabel("P-loop to DFG distance (nm)\n(real 1957-1962 to real 2042-2044)")
plt.title("Per-mutant pocket-to-activation-segment coupling")
plt.xticks(rotation=90)
save_current_figure("step3a_pocket_dfg_coupling.png")

plt.figure(figsize=(11, 4))
plt.bar(df_pocket_mut.index, df_pocket_mut["pocket_open_frac"])
plt.ylabel("fraction pocket-open frames")
plt.title("Per-mutant ATP-pocket open-state occupancy")
plt.xticks(rotation=90)
save_current_figure("step3a_pocket_open_frac.png")


# ============================================
# Frame-level metrics table for later cluster mapping
# ============================================
frame_rows = []
for ti, xyz in enumerate(traj_aligned):
    nframes_i = xyz.shape[0]
    for local_i in range(nframes_i):
        row = {
            "traj_index": ti,
            "trajectory": F[ti],
            "mutant": traj_mutant(F[ti]),
            "replica": traj_replica(F[ti]),
            "frame_index_local": int(local_i),
            "frame_index_original": get_frame_index_original(
                frame_idx_list, ti, local_i
            ),
            "aloop_rmsd_nm": float(ACT_rmsd_nm_list[ti][local_i]),
            "aloop_state": int(ACT_state[ti][local_i]),
            "alphaC_hinge_dist_nm": float(
                alphaC_hinge_dist_per_traj[ti][local_i]
            ),
            "pocket_front_dist_nm": float(per_traj_front[ti][local_i]),
            # CORRECTION: placeholder removed — true value filled in below
            "pocket_dfg_dist_nm": 0.0,
            "pocket_open_state": int(pocket_open_state_per_traj[ti][local_i]),
        }
        if local_i < len(DFG_chi1_sub[ti]):
            row["dfg_chi1_deg"] = float(DFG_chi1_sub[ti][local_i])
            row["dfg_state"] = int(DFG_state[ti][local_i])
        else:
            row["dfg_chi1_deg"] = np.nan
            row["dfg_state"] = np.nan

        if (idx_lys is not None and idx_glu is not None
                and lys_glu_dist_per_traj[ti] is not None):
            row["lys_glu_dist_nm"] = float(
                lys_glu_dist_per_traj[ti][local_i]
            )
            row["alphaC_compact_state"] = int(
                lys_glu_dist_per_traj[ti][local_i] <= thr_alphaC
            )
        else:
            row["lys_glu_dist_nm"] = np.nan
            row["alphaC_compact_state"] = np.nan

        frame_rows.append(row)

frame_df = pd.DataFrame(frame_rows)

# Fill in true pocket_dfg_dist_nm (P-loop COM to DFG COM per frame)
true_pocket_dfg = []
for ti in range(len(traj_aligned)):
    for local_i in range(traj_aligned[ti].shape[0]):
        true_pocket_dfg.append(float(np.linalg.norm(
            traj_aligned[ti][local_i, mask_ploop, :].mean(axis=0) -
            traj_aligned[ti][local_i, mask_dfg, :].mean(axis=0)
        )))
frame_df["pocket_dfg_dist_nm"] = true_pocket_dfg
save_table(frame_df, "step3a_frame_metrics.csv")

# Downsampled metrics table for PCA mapping (stride 10 to match PCA stride)
pca_stride = 10
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


# ============================================
# Transition analysis
# ============================================
def switch_count(x01):
    x = np.asarray(x01, dtype=np.int8)
    if x.size <= 1:
        return 0
    return int(np.sum(x[1:] != x[:-1]))


def mean_dwell_frames(x01):
    x = np.asarray(x01, dtype=np.int8)
    if x.size == 0:
        return np.nan
    edges = np.where(np.diff(x) != 0)[0] + 1
    runs = np.diff(np.r_[0, edges, len(x)])
    return float(np.mean(runs)) if len(runs) else float(len(x))


df_trans = pd.DataFrame({
    "trajectory": F,
    "mutant": mut_names,
    "replica": [traj_replica(p) for p in F],
    "act_switches": [switch_count(x) for x in ACT_state],
    "act_mean_dwell_frames": [mean_dwell_frames(x) for x in ACT_state],
    "dfg_switches": [switch_count(x) for x in DFG_state],
    "dfg_mean_dwell_frames": [mean_dwell_frames(x) for x in DFG_state],
    "pocket_switches": [switch_count(x) for x in pocket_open_state_per_traj],
    "pocket_mean_dwell_frames": [
        mean_dwell_frames(x) for x in pocket_open_state_per_traj
    ],
})
save_table(df_trans, "step3a_transition_summary.csv")

print("\nScript 3A completed.")
print("\nOutputs saved:")
print("  step3a_act_state_summary.csv       — A-loop state per trajectory")
print("  step3a_dfg_state_summary.csv       — DFG chi1 state per trajectory")
print("  step3a_alphaC_metrics.csv          — alphaC displacement and salt bridge")
print("  step3a_atp_pocket_metrics.csv      — ATP pocket geometry per trajectory")
print("  step3a_atp_pocket_metrics_by_mutant.csv  — averaged per mutant")
print("  step3a_frame_metrics.csv           — all metrics per frame (full)")
print("  step3a_frame_metrics_downsampled.csv     — stride-10 for PCA mapping")
print("  step3a_transition_summary.csv      — state switching counts")
print("\nKey corrections applied vs original script:")
print("  REGIONS: all four boundaries corrected")
print("  Salt bridge: E1967 -> E1993 (K1980-E1993)")
print("  DFG_F: internal 170 (real 2103) -> internal 110 (real 2043)")
print("  idx_dfgF: resid 2104 -> resid 2043")
print("  pocket_dfg_dist_nm placeholder: cleaned up")
