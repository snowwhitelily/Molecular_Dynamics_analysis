import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.scripts.ros1_clustering import run_dbscan
from analysis.scripts.occupancy import occupancy_from_frame_labels
from analysis.scripts.plotting import plot_occupancy_heatmap, plot_cluster_population, plot_pca_clusters

STAGE1 = Path("analysis/outputs/stage1_common_align")
STAGE2 = Path("analysis/outputs/stage2_orientation_pca")
OUTDIR = Path("analysis/outputs/stage3_orientation_clustering")
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("STAGE 3: Orientation clustering (last-20% frames, DBSCAN)")
    print("=" * 80)

    F = np.load(STAGE1 / "F.npy", allow_pickle=True).tolist()
    scores = np.load(STAGE2 / "scores_orient_flat.npy")
    owners = np.load(STAGE2 / "X_owner_orient.npy").astype(int)

    print("Original scores shape:", scores.shape)

    keep = np.zeros(len(scores), dtype=bool)
    for ti in range(len(F)):
        idx = np.where(owners == ti)[0]
        if len(idx) == 0:
            continue
        start = int(0.8 * len(idx))
        keep[idx[start:]] = True

    scores_use = scores[keep]
    owners_use = owners[keep]

    print("Filtered scores shape (last 20% only):", scores_use.shape)

    labels = run_dbscan(scores_use[:, :2], n_pcs=2, eps=0.3, min_samples=100)

    print("Orientation DBSCAN clusters:", len(set(labels)) - (1 if -1 in labels else 0))
    print("Orientation DBSCAN noise fraction:", float(np.mean(labels == -1)))

    traj_mutant = np.array([p.split("/")[0] for p in F], dtype=object)
    frame_mutant = traj_mutant[owners_use]
    counts, frac = occupancy_from_frame_labels(frame_mutant, labels)

    print("\nOrientation occupancy (DBSCAN, last 20% only):")
    print(frac)

    plot_occupancy_heatmap(
        frac,
        title="Orientation occupancy (last 20% frames)",
        outpath="figures/orientation_occupancy_heatmap.png"
    )
    plot_cluster_population(
        labels,
        title="Orientation cluster population",
        outpath="figures/orientation_cluster_population.png"
    )
    plot_pca_clusters(
        scores_use[:, 0], scores_use[:, 1], labels,
        title="Orientation PCA clusters",
        outpath="figures/orientation_pca_clusters.png"
    )

    np.save(OUTDIR / "labels.npy", labels.astype(int))
    np.save(OUTDIR / "owners.npy", owners_use.astype(int))
    counts.to_csv(OUTDIR / "counts.csv")
    frac.to_csv(OUTDIR / "fractions.csv")

    print("\nSaved outputs to:", OUTDIR)
    print("DONE.")


if __name__ == "__main__":
    main()
