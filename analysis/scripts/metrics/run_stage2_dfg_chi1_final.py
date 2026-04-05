# 5) analysis/scripts/metrics/run_stage2_dfg_chi1_final.py
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.scripts.ros1_dihedrals import compute_chi1_deg_for_traj

STAGE1 = Path("analysis/outputs/stage1_common_align")
OUTDIR = Path("analysis/outputs/stage2_dfg_chi1_final")
OUTDIR.mkdir(parents=True, exist_ok=True)

DFG_F = 170
manual_thr_deg = None


def save_list_npz(path, arr_list):
    np.savez_compressed(path, **{f"traj_{i}": np.asarray(arr) for i, arr in enumerate(arr_list)})


def main():
    print("=" * 80)
    print("STAGE 2: DFG chi1 analysis (final, DFG-valid common length)")
    print("=" * 80)

    F = np.load(STAGE1 / "F.npy", allow_pickle=True).tolist()

    print("Number of trajectories:", len(F))
    print("DFG residue:", DFG_F)

    chi1_deg_list_full = []
    print("DFG chi1 (REAL) full-trajectory computation")
    for d in F:
        pdb = f"{d}/md-prot.pdb"
        xtc = f"{d}/md-prot.xtc"
        chi1_deg_list_full.append(compute_chi1_deg_for_traj(pdb, xtc, DFG_F))

    all_vals_full = np.concatenate([v[np.isfinite(v)] for v in chi1_deg_list_full])

    plt.figure(figsize=(7, 4))
    plt.hist(all_vals_full, bins=120)
    plt.title(f"DFG-Phe chi1 distribution (resid {DFG_F})")
    plt.xlabel("chi1 (deg)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("figures/dfg_chi1_hist_final.png", dpi=300, bbox_inches="tight")
    plt.close()

    thr = float(manual_thr_deg) if manual_thr_deg is not None else float(np.nanmedian(all_vals_full))
    print("DFG chi1 threshold (deg):", thr)

    dfg_state_list_full = [(v > thr).astype(np.int8) for v in chi1_deg_list_full]

    lengths = [len(x) for x in dfg_state_list_full]
    n_common = min(lengths)

    for ti, (d, arr) in enumerate(zip(F, dfg_state_list_full)):
        print(f"[DFG MAP] traj {ti:3d} {d:28s} full={len(arr):5d} mapped={len(arr):5d}")

    print("DFG-valid common length:", n_common)

    chi1_deg_list_sub = [arr[:n_common] for arr in chi1_deg_list_full]
    dfg_state_list_sub = [arr[:n_common] for arr in dfg_state_list_full]
    DFG_state_sub = np.vstack(dfg_state_list_sub)

    print("[DFG FINAL] DFG_state_sub shape:", DFG_state_sub.shape)
    print("[DFG FINAL] mean state=1:", float(DFG_state_sub.mean()))

    np.save(OUTDIR / "F.npy", np.array(F, dtype=object), allow_pickle=True)
    np.save(OUTDIR / "threshold_deg.npy", np.array([thr], dtype=np.float32))
    np.save(OUTDIR / "chi1_all_vals_full.npy", all_vals_full.astype(np.float32))
    np.save(OUTDIR / "DFG_state_sub.npy", DFG_state_sub.astype(np.int8))

    save_list_npz(OUTDIR / "chi1_deg_list_full.npz", [arr.astype(np.float32) for arr in chi1_deg_list_full])
    save_list_npz(OUTDIR / "chi1_deg_list_sub.npz", [arr.astype(np.float32) for arr in chi1_deg_list_sub])
    save_list_npz(OUTDIR / "dfg_state_list_sub.npz", [arr.astype(np.int8) for arr in dfg_state_list_sub])
    save_list_npz(OUTDIR / "DFG_state_list.npz", [arr.astype(np.int8) for arr in dfg_state_list_sub])

    with open(OUTDIR / "summary.txt", "w") as fh:
        fh.write(f"DFG_F={DFG_F}\n")
        fh.write(f"threshold_deg={thr}\n")
        fh.write(f"ntraj={len(F)}\n")
        fh.write(f"n_common={n_common}\n")
        fh.write(f"DFG_state_shape={DFG_state_sub.shape}\n")
        fh.write(f"mean_state1={float(DFG_state_sub.mean())}\n")

    print("\nSaved outputs to:", OUTDIR)
    print("DONE.")


if __name__ == "__main__":
    main()
