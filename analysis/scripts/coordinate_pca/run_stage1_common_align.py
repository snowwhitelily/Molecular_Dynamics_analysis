import sys
from pathlib import Path
import numpy as np
import MDAnalysis as mda

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.scripts.ros1_io import (
    selection_keys_and_indices,
    xtc2array_varlen_by_indices,
)
from analysis.scripts.ros1_align import align_traj_to_ref_by_fit
from analysis.scripts.timer import Timer

tim = Timer()

OUTDIR = Path("analysis/outputs/stage1_common_align")
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("STAGE 1: COMMON ALIGNMENT")
    print("=" * 80)

    # Find trajectories
    F = sorted([str(p) for p in Path(".").glob("*/Simulation_*")])
    print("Number of trajectories:", len(F))

    # Common atoms
    with tim("Finding common atoms"):
        common_keys = None
        key_maps = []

        for d in F:
            keys, key_to_idx = selection_keys_and_indices(d, "protein")
            key_maps.append(key_to_idx)

            if common_keys is None:
                common_keys = keys
            else:
                common_keys &= keys

        common_keys = sorted(common_keys)
        print("Common atoms:", len(common_keys))

    # Map indices
    per_sim_indices = []
    for km in key_maps:
        idx = [km[k] for k in common_keys]
        per_sim_indices.append(np.array(idx))

    # Load trajectories
    with tim("Loading trajectories"):
        traj_list = []
        for d, idx in zip(F, per_sim_indices):
            xyz, _ = xtc2array_varlen_by_indices(d, idx)
            traj_list.append(xyz)

    # Reference
    ref = traj_list[0][0]
    ref_centered = ref - ref.mean(axis=0, keepdims=True)
    fit_mask = np.ones(ref.shape[0], dtype=bool)

    # Align
    with tim("Aligning"):
        traj_aligned = [
            align_traj_to_ref_by_fit(xyz, fit_mask, ref_centered)
            for xyz in traj_list
        ]

    # Save
    np.save(OUTDIR / "F.npy", np.array(F, dtype=object), allow_pickle=True)
    np.save(OUTDIR / "per_sim_indices.npy", np.array(per_sim_indices, dtype=object), allow_pickle=True)
    np.save(OUTDIR / "ref_centered.npy", ref_centered.astype(np.float32))

    np.savez_compressed(
        OUTDIR / "traj_aligned_list.npz",
        **{f"traj_{i}": xyz.astype(np.float32) for i, xyz in enumerate(traj_aligned)}
    )

    print("\nSaved to:", OUTDIR)
    print("DONE.")


if __name__ == "__main__":
    main()
