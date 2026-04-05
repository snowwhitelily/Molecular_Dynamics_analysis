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
OUTDIR = Path("analysis/outputs/stage2_alphaC_metrics")
OUTDIR.mkdir(parents=True, exist_ok=True)

REGIONS = {
    "alphaC": (1960, 1975),
    "hinge": (2011, 2016),
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
    print("STAGE 2: alphaC helix metrics")
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

    mask_alphaC = (resids_abs >= REGIONS["alphaC"][0]) & (resids_abs <= REGIONS["alphaC"][1]) & (names == "CA")
    mask_hinge = (resids_abs >= REGIONS["hinge"][0]) & (resids_abs <= REGIONS["hinge"][1]) & (names == "CA")

    print("alphaC CA atoms:", int(mask_alphaC.sum()), "| hinge CA atoms:", int(mask_hinge.sum()))

    idx_lys = first_existing((resids_abs == 1980) & np.isin(names, ["NZ", "CA"]))
    idx_glu = first_existing((resids_abs == 1967) & np.isin(names, ["OE1", "OE2", "CA"]))

    alphaC_rmsf_mean_list = []
    alphaC_hinge_mean_dist_list = []
    alphaC_hinge_std_dist_list = []
    alphaC_compact_frac_list = []
    all_d_lys_glu = []

    for xyz in traj_aligned:
        alphaC_xyz = xyz[:, mask_alphaC, :]
        alphaC_mean = alphaC_xyz.mean(axis=0, keepdims=True)
        alphaC_rmsf = np.sqrt(((alphaC_xyz - alphaC_mean) ** 2).sum(axis=(0, 2)) / alphaC_xyz.shape[0])
        alphaC_rmsf_mean_list.append(alphaC_rmsf.mean())

        alphaC_com = alphaC_xyz.mean(axis=1)
        hinge_com = xyz[:, mask_hinge, :].mean(axis=1)
        alphaC_hinge_dist = np.linalg.norm(alphaC_com - hinge_com, axis=1)

        alphaC_hinge_mean_dist_list.append(alphaC_hinge_dist.mean())
        alphaC_hinge_std_dist_list.append(alphaC_hinge_dist.std())

        if idx_lys is not None and idx_glu is not None:
            d_i = np.linalg.norm(xyz[:, idx_lys, :] - xyz[:, idx_glu, :], axis=1)
            all_d_lys_glu.append(d_i)

    if idx_lys is not None and idx_glu is not None:
        thr_alphaC = np.nanmedian(np.concatenate(all_d_lys_glu))
        for d_i in all_d_lys_glu:
            alphaC_state_i = (d_i <= thr_alphaC).astype(int)
            alphaC_compact_frac_list.append(alphaC_state_i.mean())
        print(f"Lys1980-Glu1967 proxy threshold: {thr_alphaC/10.0:.3f} nm")

    df = pd.DataFrame({
        "trajectory": F,
        "mutant": mut_names,
        "alphaC_mean_RMSF": alphaC_rmsf_mean_list,
        "alphaC_hinge_mean_dist": alphaC_hinge_mean_dist_list,
        "alphaC_hinge_std_dist": alphaC_hinge_std_dist_list,
    })
    if idx_lys is not None and idx_glu is not None:
        df["alphaC_compact_frac"] = alphaC_compact_frac_list

    df_mut = df.groupby("mutant").mean(numeric_only=True).sort_values("alphaC_mean_RMSF", ascending=False)
    print(df_mut.head())

    df.to_csv(OUTDIR / "per_trajectory.csv", index=False)
    df_mut.to_csv(OUTDIR / "per_mutant.csv")

    plt.figure(figsize=(11, 4))
    plt.bar(df_mut.index, df_mut["alphaC_mean_RMSF"])
    plt.ylabel("mean αC RMSF-like value (Å)")
    plt.title("Per-mutant αC-helix mobility")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("figures/alphaC_mobility.png", dpi=300, bbox_inches="tight")
    plt.close()

    if "alphaC_compact_frac" in df_mut.columns:
        plt.figure(figsize=(11, 4))
        plt.bar(df_mut.index, df_mut["alphaC_compact_frac"])
        plt.ylabel("fraction αC-compact frames")
        plt.title("Per-mutant αC compact-state occupancy")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("figures/alphaC_compact_frac.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("\nSaved outputs to:", OUTDIR)
    print("DONE.")

if __name__ == "__main__":
    main()
