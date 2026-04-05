import sys
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.scripts.ros1_clustering import run_cluster_panel_varlen
from analysis.scripts.plotting import plot_occupancy_heatmap, plot_cluster_population, plot_pca_clusters

try:
    import hdbscan
except Exception:
    hdbscan = None

STAGE1 = Path("analysis/outputs/stage1_common_align")


def run_hdbscan(Z, n_pcs=5, min_cluster_size=150):
    if hdbscan is None:
        return np.full(Z.shape[0], -1, dtype=int)
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    return model.fit_predict(Z[:, :n_pcs])

def run_dbscan(Z, n_pcs=5, eps=0.8, min_samples=60):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(Z[:, :n_pcs])


CFG = {
    "actin":         dict(prefix="actin",         title="ACT-IN",        n_pcs=5, min_cluster_size=200, eps=0.9, min_samples=60),
    "actout_plain":  dict(prefix="actout",        title="ACT-OUT",       n_pcs=5, min_cluster_size=200, eps=0.9, min_samples=60),
    "actout_ctlfit": dict(prefix="actout_ctlfit", title="ACT-OUT CTL-fit", n_pcs=5, min_cluster_size=200, eps=0.9, min_samples=60),
    "ntl":           dict(prefix="ntl",           title="NTL",           n_pcs=5, min_cluster_size=150, eps=0.8, min_samples=60),
    "ctl":           dict(prefix="ctl",           title="CTL",           n_pcs=5, min_cluster_size=200, eps=0.9, min_samples=60),
    "aloop":         dict(prefix="aloop",         title="A-loop",        n_pcs=3, min_cluster_size=120, eps=0.6, min_samples=50),
    "tyrloop":       dict(prefix="tyrloop",       title="Tyrosine loop", n_pcs=3, min_cluster_size=120, eps=0.6, min_samples=50),
}


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_stage3_coord_clustering_branch.py <branch>")

    branch = sys.argv[1]
    if branch not in CFG:
        raise ValueError(branch)

    cfg = CFG[branch]
    STAGE2 = Path(f"analysis/outputs/stage2_{cfg['prefix']}_pca")
    OUTDIR = Path(f"analysis/outputs/stage3_{cfg['prefix']}_clustering")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"STAGE 3: {cfg['title']} clustering")
    print("=" * 80)

    F = np.load(STAGE1 / "F.npy", allow_pickle=True).tolist()
    scores_flat = np.load(STAGE2 / "scores_flat.npy")
    x_owner = np.load(STAGE2 / "X_owner.npy").astype(int)

    res = run_cluster_panel_varlen(
        scores_flat=scores_flat,
        x_owner=x_owner,
        F=F,
        prefix=cfg["prefix"],
        title_prefix=cfg["title"],
        n_pcs=cfg["n_pcs"],
        min_cluster_size=cfg["min_cluster_size"],
        eps=cfg["eps"],
        min_samples=cfg["min_samples"],
        run_hdbscan=run_hdbscan,
        run_dbscan=run_dbscan,
        plot_occupancy_heatmap=plot_occupancy_heatmap,
        plot_cluster_population=plot_cluster_population,
        plot_pca_clusters=plot_pca_clusters,
        display_fn=print,
    )

    np.save(OUTDIR / "labels_hdb.npy", res["labels_hdb"].astype(int))
    np.save(OUTDIR / "labels_db.npy", res["labels_db"].astype(int))
    res["counts_hdb"].to_csv(OUTDIR / "counts_hdb.csv")
    res["frac_hdb"].to_csv(OUTDIR / "frac_hdb.csv")
    res["counts_db"].to_csv(OUTDIR / "counts_db.csv")
    res["frac_db"].to_csv(OUTDIR / "frac_db.csv")

    print("\nSaved outputs to:", OUTDIR)
    print("DONE.")


if __name__ == "__main__":
    main()
