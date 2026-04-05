import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.scripts.ros1_utils import ensure_meta

STAGE1 = Path("analysis/outputs/stage1_common_align")
OUTDIR = Path("analysis/outputs/stage2_atp_pocket_metrics")
OUTDIR.mkdir(parents=True, exist_ok=True)

REGIONS = {
    "P_loop": (1978, 1987),
    "hinge": (2011, 2016),
    "DFG": (2103, 2105),
}

def load_list_npz(path):
    z = np.load(path, allow_pickle=True)
    keys = sorted(z.files, key=lambda x: int(x.split("_")[1]))
    return [z[k] for k in keys]

def get_abs_resids(meta):
    return np.asarray(meta.resids).astype(int) + 1933

def first_existing(mask):
    idx = np.where(mask)[0]
    return None if len(idx) == 0 else idx[0]

def main():
    print("=" * 80)
    print("STAGE 2: ATP pocket metrics")
    print("=" * 80)

    F = np.load(STAGE1 / "F.npy", allow_pickle=True).tolist()
    per_sim_indices = np.load(STAGE1 / "per_sim_indices.npy", allow_pickle=True)
    traj_aligned = load_list_npz(STAGE1 / "traj_aligned_list.npz")

    natoms = traj_aligned[0].shape[1]
    u0 = mda.Universe(f"{F[0]}/md-prot.pdb")
    meta = u0.atoms[np.asarray(per_sim_indices[0], dtype=int)]
    meta = ensure_meta(meta=meta, natoms=natoms)

    resids_abs = get_abs_resids(meta)
    names = np.asarray(meta.names)
    mut_names = np.array([p.split("/")[0] for p in F], dtype=object)

    mask_ploop = (resids_abs >= REGIONS["P_loop"][0]) & (resids_abs <= REGIONS["P_loop"][1]) & (names == "CA")
    mask_hinge = (resids_abs >= REGIONS["hinge"][0]) & (resids_abs <= REGIONS["hinge"][1]) & (names == "CA")
    mask_dfg = (resids_abs >= REGIONS["DFG"][0]) & (resids_abs <= REGIONS["DFG"][1]) & (names == "CA")

    print("P-loop CA atoms:", int(mask_ploop.sum()))
    print("hinge CA atoms:", int(mask_hinge.sum()))
    print("DFG CA atoms:", int(mask_dfg.sum()))

    idx_gate = first_existing((resids_abs == 2026) & (names == "CA"))
    idx_dfgF = first_existing((resids_abs == 2104) & np.isin(names, ["CZ", "CE1", "CE2", "CA"]))

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

        d_ploop_hinge_i = np.linalg.norm(ploop_com - hinge_com, axis=1) / 10.0
        d_ploop_dfg_i = np.linalg.norm(ploop_com - dfg_com, axis=1) / 10.0

        per_traj_front.append(d_ploop_hinge_i)
        pocket_front_mean.append(d_ploop_hinge_i.mean())
        pocket_front_std.append(d_ploop_hinge_i.std())
        pocket_dfg_mean.append(d_ploop_dfg_i.mean())
        pocket_dfg_std.append(d_ploop_dfg_i.std())
        all_front.append(d_ploop_hinge_i)

        if idx_gate is not None and idx_dfgF is not None:
            d_gate_dfgF_i = np.linalg.norm(xyz[:, idx_gate, :] - xyz[:, idx_dfgF, :], axis=1) / 10.0
            per_traj_back.append(d_gate_dfgF_i)
            all_back.append(d_gate_dfgF_i)
        else:
            per_traj_back.append(None)

    thr_front = np.nanmedian(np.concatenate(all_front))
    print(f"Pocket front threshold: {thr_front:.3f} nm")

    if idx_gate is not None and idx_dfgF is not None and len(all_back):
        thr_back = np.nanmedian(np.concatenate(all_back))
        print(f"Gatekeeper-DFG-Phe threshold: {thr_back:.3f} nm")
    else:
        thr_back = np.nan

    for i in range(len(traj_aligned)):
        front_open_i = (per_traj_front[i] >= thr_front).astype(int)
        if np.isfinite(thr_back) and per_traj_back[i] is not None:
            back_open_i = (per_traj_back[i] >= thr_back).astype(int)
            pocket_open_i = ((front_open_i + back_open_i) >= 1).astype(int)
            gate_dfgF_mean.append(np.nanmean(per_traj_back[i]))
            gate_dfgF_std.append(np.nanstd(per_traj_back[i]))
        else:
            pocket_open_i = front_open_i.copy()
        pocket_open_frac.append(pocket_open_i.mean())

    df = pd.DataFrame({
        "trajectory": F,
        "mutant": mut_names,
        "pocket_front_mean": pocket_front_mean,
        "pocket_front_std": pocket_front_std,
        "pocket_dfg_mean": pocket_dfg_mean,
        "pocket_dfg_std": pocket_dfg_std,
        "pocket_open_frac": pocket_open_frac,
    })
    if np.isfinite(thr_back):
        df["gate_dfgF_mean"] = gate_dfgF_mean
        df["gate_dfgF_std"] = gate_dfgF_std

    df_mut = df.groupby("mutant").mean(numeric_only=True).sort_values("pocket_open_frac", ascending=False)
    print(df_mut.head())

    df.to_csv(OUTDIR / "per_trajectory.csv", index=False)
    df_mut.to_csv(OUTDIR / "per_mutant.csv")

    plt.figure(figsize=(11, 4))
    plt.bar(df_mut.index, df_mut["pocket_front_mean"])
    plt.ylabel("P-loop ↔ hinge distance (nm)")
    plt.title("Per-mutant front-pocket opening")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("figures/atp_front_opening.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(11, 4))
    plt.bar(df_mut.index, df_mut["pocket_open_frac"])
    plt.ylabel("fraction pocket-open frames")
    plt.title("Per-mutant ATP-pocket open-state occupancy")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("figures/atp_open_frac.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSaved outputs to:", OUTDIR)
    print("DONE.")

if __name__ == "__main__":
    main()
