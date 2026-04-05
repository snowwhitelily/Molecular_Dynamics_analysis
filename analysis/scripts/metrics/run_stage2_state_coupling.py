# 7) analysis/scripts/metrics/run_stage2_state_coupling.py
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

STAGE1 = Path("analysis/outputs/stage1_common_align")
FURTHER = Path("analysis/outputs/stage2_further_analysis")
ALOOP = Path("analysis/outputs/stage2_aloop_state_labels")
DFG = Path("analysis/outputs/stage2_dfg_chi1_final")
OUTDIR = Path("analysis/outputs/stage2_state_coupling")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_list_npz(path):
    z = np.load(path, allow_pickle=True)
    keys = sorted(z.files, key=lambda x: int(x.split("_")[1]))
    return [z[k] for k in keys]


def main():
    print("=" * 80)
    print("STATE COUPLING ANALYSIS")
    print("=" * 80)

    F = np.load(STAGE1 / "F.npy", allow_pickle=True).tolist()
    basin_flat = np.load(FURTHER / "basin_flat.npy")
    x_owner = np.load(FURTHER / "x_owner.npy").astype(int)
    basin = [basin_flat[x_owner == ti] for ti in range(len(F))]

    ACT_state = load_list_npz(ALOOP / "ACT_state_list.npz")
    DFG_state = load_list_npz(DFG / "DFG_state_list.npz")

    print("\n--- Global A-loop coupling ---")
    all_basin = np.concatenate(basin)
    all_act = np.concatenate(ACT_state)
    m = min(len(all_basin), len(all_act))
    all_basin_a = all_basin[:m]
    all_act = all_act[:m]

    for b in [0, 1]:
        frac = np.mean(all_act[all_basin_a == b])
        print(f"P(A-loop inactive | basin={b}) = {frac:.3f}")

    print("\n--- Global DFG coupling ---")
    basin_trim = []
    dfg_trim = []
    for ti in range(len(F)):
        b = np.asarray(basin[ti], dtype=np.int8)
        d = np.asarray(DFG_state[ti], dtype=np.int8)
        m = min(len(b), len(d))
        basin_trim.append(b[:m])
        dfg_trim.append(d[:m])

    all_basin_d = np.concatenate(basin_trim)
    all_dfg = np.concatenate(dfg_trim)
    for b in [0, 1]:
        frac = np.mean(all_dfg[all_basin_d == b])
        print(f"P(DFG state=1 | basin={b}) = {frac:.3f}")

    print("\n--- Per mutant (Q2022P family + WT) ---")
    rows = []
    for mut in ["Q2022P", "Q2022P_S1986F", "Q2022P_S1986Y", "WT"]:
        idx = [i for i, p in enumerate(F) if p.startswith(mut)]
        if not idx:
            continue

        b = np.concatenate([basin[i] for i in idx])
        a = np.concatenate([ACT_state[i] for i in idx])
        m = min(len(b), len(a))
        b_a = b[:m]
        a = a[:m]

        d_b = np.concatenate([basin_trim[i] for i in idx])
        d = np.concatenate([dfg_trim[i] for i in idx])

        print(f"\n{mut}:")
        for state in [0, 1]:
            a_frac = np.mean(a[b_a == state])
            d_frac = np.mean(d[d_b == state])
            print(f"  P(A-loop inactive | basin={state}) = {a_frac:.3f}")
            print(f"  P(DFG state=1 | basin={state}) = {d_frac:.3f}")
            rows.append((mut, state, a_frac, d_frac))

    np.save(OUTDIR / "rows.npy", np.array(rows, dtype=object), allow_pickle=True)

    with open(OUTDIR / "summary.txt", "w") as fh:
        fh.write("mutant\tbasin\tp_aloop_inactive\tp_dfg_state1\n")
        for row in rows:
            fh.write("\t".join(map(str, row)) + "\n")

    print("\nSaved outputs to:", OUTDIR)
    print("DONE.")


if __name__ == "__main__":
    main()
