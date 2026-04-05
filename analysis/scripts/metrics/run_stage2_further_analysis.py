# 6) analysis/scripts/metrics/run_stage2_further_analysis.py
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.scripts.ros1_clustering import switch_count, mean_dwell_frames

STAGE1 = Path("analysis/outputs/stage1_common_align")
ORIENT = Path("analysis/outputs/stage2_orientation_pca")
OUTDIR = Path("analysis/outputs/stage2_further_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_list_npz(path):
    z = np.load(path, allow_pickle=True)
    keys = sorted(z.files, key=lambda x: int(x.split("_")[1]))
    return [z[k] for k in keys]


def main():
    print("=" * 80)
    print("STAGE 2: Further analysis")
    print("=" * 80)

    F = np.load(STAGE1 / "F.npy", allow_pickle=True).tolist()
    traj_aligned = load_list_npz(STAGE1 / "traj_aligned_list.npz")

    scores_flat = np.load(ORIENT / "scores_orient_flat.npy")
    x_owner = np.load(ORIENT / "X_owner_orient.npy").astype(int)
    ctl_idx = np.load(ORIENT / "ctl_idx.npy").astype(int)
    ntl_idx = np.load(ORIENT / "ntl_idx.npy").astype(int)

    ntraj = len(traj_aligned)
    natoms = traj_aligned[0].shape[1]
    print("ntraj:", ntraj, "| natoms:", natoms)
    print("len(F):", len(F))

    pc1 = scores_flat[:, 0].astype(float)
    thr = float(np.median(pc1))
    basin_flat = (pc1 > thr).astype(np.int8)
    basin = [basin_flat[x_owner == ti] for ti in range(len(F))]

    print("Using basins from: ORIENTATION PCA | scores_flat shape:", scores_flat.shape)
    print("[PC1 BASIN] threshold (median):", thr)
    print("[PC1 BASIN] overall basin=1 fraction:", float(np.mean(basin_flat)))

    wt_idx = [i for i, p in enumerate(F) if p.startswith("WT/")]
    if wt_idx:
        print("\nWT PC1 basin occupancies:")
        for i in wt_idx:
            print(F[i], "basin1 frac:", round(float(np.mean(basin[i])), 3))

    mutant_of = np.array([p.split("/")[0] for p in F], dtype=object)
    uniq_mut = np.unique(mutant_of)

    print("\nPC1 basin occupancy per mutant (avg over replicas):")
    basin_means = {}
    for mut in uniq_mut:
        idx = np.where(mutant_of == mut)[0]
        vals = [float(np.mean(basin[i])) for i in idx]
        basin_means[mut] = float(np.mean(vals))
        print(mut, round(basin_means[mut], 3))

    sw = np.array([switch_count(basin[i]) for i in range(len(F))], dtype=float)
    dw = np.array([mean_dwell_frames(basin[i]) for i in range(len(F))], dtype=float)

    print("\nTransitions (PC1 basins):")
    print("WT switch counts:", [int(sw[i]) for i in wt_idx] if wt_idx else "WT not found")
    print("WT mean dwell (frames):", [round(dw[i], 1) for i in wt_idx] if wt_idx else "WT not found")

    print("\nPer-mutant transition summary (mean over replicas):")
    for mut in uniq_mut:
        idx = np.where(mutant_of == mut)[0]
        print(f"{mut:16s}  switches={sw[idx].mean():6.1f} ± {sw[idx].std():5.1f}   dwell(fr)={dw[idx].mean():6.1f} ± {dw[idx].std():5.1f}")

    print("\nComputing NTL–CTL COM distance per basin...")
    comdist_list = []
    for ti in range(len(F)):
        xyz = traj_aligned[ti]
        ctl_com = xyz[:, ctl_idx, :].mean(axis=1)
        ntl_com = xyz[:, ntl_idx, :].mean(axis=1)
        d_i = np.linalg.norm(ntl_com - ctl_com, axis=1) / 10.0
        comdist_list.append(d_i)

    dA = np.concatenate([comdist_list[i][:len(basin[i])][basin[i] == 0] for i in range(len(F)) if len(basin[i])])
    dB = np.concatenate([comdist_list[i][:len(basin[i])][basin[i] == 1] for i in range(len(F)) if len(basin[i])])
    print("COMdist nm: mean basin0 =", round(float(dA.mean()), 4), "| mean basin1 =", round(float(dB.mean()), 4))

    np.save(OUTDIR / "basin_flat.npy", basin_flat.astype(np.int8))
    np.save(OUTDIR / "x_owner.npy", x_owner.astype(int))
    np.save(OUTDIR / "sw.npy", sw.astype(np.float32))
    np.save(OUTDIR / "dw.npy", dw.astype(np.float32))
    np.save(OUTDIR / "pc1_threshold.npy", np.array([thr], dtype=np.float32))

    with open(OUTDIR / "summary.txt", "w") as fh:
        fh.write(f"pc1_threshold={thr}\n")
        fh.write(f"overall_basin1_fraction={float(np.mean(basin_flat))}\n")
        fh.write(f"mean_comdist_basin0={float(dA.mean())}\n")
        fh.write(f"mean_comdist_basin1={float(dB.mean())}\n")

    print("\nSaved outputs to:", OUTDIR)
    print("DONE.")


if __name__ == "__main__":
    main()
