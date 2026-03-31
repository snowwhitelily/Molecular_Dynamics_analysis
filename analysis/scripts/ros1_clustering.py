import numpy as np
import pandas as pd


def mutant_names_from_F(F):
    return np.array([p.split("/")[0] for p in F], dtype=object)


def occ_table_from_owner(F, x_owner, labels):
    """
    For variable-length PCA sections where x_owner maps each frame to trajectory index.
    """
    traj_mutant = mutant_names_from_F(F)
    frame_mutant = traj_mutant[np.asarray(x_owner).astype(int)]

    df = pd.DataFrame({
        "mutant": frame_mutant,
        "state": np.asarray(labels).astype(int)
    })

    counts = df.groupby(["mutant", "state"]).size().unstack(fill_value=0).sort_index()
    fracs = counts.div(counts.sum(axis=1), axis=0)
    return counts, fracs


def run_cluster_panel_varlen(
    scores_flat,
    x_owner,
    F,
    prefix,
    title_prefix,
    n_pcs=5,
    min_cluster_size=150,
    eps=0.8,
    min_samples=60,
    run_hdbscan=None,
    run_dbscan=None,
    plot_occupancy_heatmap=None,
    plot_cluster_population=None,
    plot_pca_clusters=None,
    display_fn=None,
):
    """
    For variable-length PCA sections:
    - scores_flat shape: (n_frames_total, n_components)
    - x_owner shape: (n_frames_total,) trajectory index per frame

    External dependencies are injected so the module stays notebook-agnostic.
    """
    if run_hdbscan is None or run_dbscan is None:
        raise ValueError("run_hdbscan and run_dbscan must be provided.")
    if plot_occupancy_heatmap is None or plot_cluster_population is None or plot_pca_clusters is None:
        raise ValueError("Plot helper functions must be provided.")
    if display_fn is None:
        display_fn = print

    Z = np.asarray(scores_flat)[:, :n_pcs]

    labels_hdb = run_hdbscan(Z, n_pcs=n_pcs, min_cluster_size=min_cluster_size)
    labels_db = run_dbscan(Z, n_pcs=n_pcs, eps=eps, min_samples=min_samples)

    counts_hdb, frac_hdb = occ_table_from_owner(F, x_owner, labels_hdb)
    counts_db, frac_db = occ_table_from_owner(F, x_owner, labels_db)

    print(f"{title_prefix} HDBSCAN clusters:", len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0))
    print(f"{title_prefix} HDBSCAN noise fraction:", float(np.mean(labels_hdb == -1)))
    print(f"{title_prefix} DBSCAN clusters:", len(set(labels_db)) - (1 if -1 in labels_db else 0))
    print(f"{title_prefix} DBSCAN noise fraction:", float(np.mean(labels_db == -1)))

    print(f"{title_prefix} occupancy (HDBSCAN primary):")
    display_fn(frac_hdb)

    print(f"{title_prefix} occupancy (DBSCAN baseline):")
    display_fn(frac_db)

    plot_occupancy_heatmap(
        frac_hdb,
        title=f"State Occupancy per Mutant ({title_prefix} Clusters)",
        outpath=f"figures/{prefix}_occupancy_heatmap.png"
    )

    plot_cluster_population(
        labels_hdb,
        title=f"Cluster Population ({title_prefix} Representation)",
        outpath=f"figures/{prefix}_cluster_population.png"
    )

    plot_pca_clusters(
        np.asarray(scores_flat)[:, 0],
        np.asarray(scores_flat)[:, 1],
        labels_hdb,
        title=f"{title_prefix} PCA — HDBSCAN Clusters",
        outpath=f"figures/{prefix}_pca_clusters.png"
    )

    return {
        "labels_hdb": labels_hdb,
        "labels_db": labels_db,
        "counts_hdb": counts_hdb,
        "frac_hdb": frac_hdb,
        "counts_db": counts_db,
        "frac_db": frac_db,
    }


def run_cluster_panel_equal(
    scores_3d,
    F,
    prefix,
    title_prefix,
    n_pcs=5,
    min_cluster_size=150,
    eps=0.8,
    min_samples=60,
    run_hdbscan=None,
    run_dbscan=None,
    occupancy_from_traj_labels=None,
    plot_occupancy_heatmap=None,
    plot_cluster_population=None,
    plot_pca_clusters=None,
    display_fn=None,
):
    """
    For equal-length PCA sections:
    - scores_3d shape: (nmut, nframes_each, n_components)

    External dependencies are injected so the module stays notebook-agnostic.
    """
    if run_hdbscan is None or run_dbscan is None:
        raise ValueError("run_hdbscan and run_dbscan must be provided.")
    if occupancy_from_traj_labels is None:
        raise ValueError("occupancy_from_traj_labels must be provided.")
    if plot_occupancy_heatmap is None or plot_cluster_population is None or plot_pca_clusters is None:
        raise ValueError("Plot helper functions must be provided.")
    if display_fn is None:
        display_fn = print

    scores_3d = np.asarray(scores_3d)
    nmut, nframes_each, ncomp = scores_3d.shape

    Z = scores_3d[:, :, :n_pcs].reshape(-1, n_pcs)

    labels_hdb = run_hdbscan(Z, n_pcs=n_pcs, min_cluster_size=min_cluster_size)
    labels_db = run_dbscan(Z, n_pcs=n_pcs, eps=eps, min_samples=min_samples)

    counts_hdb, frac_hdb = occupancy_from_traj_labels(F, nframes_each, labels_hdb)
    counts_db, frac_db = occupancy_from_traj_labels(F, nframes_each, labels_db)

    print(f"{title_prefix} HDBSCAN clusters:", len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0))
    print(f"{title_prefix} HDBSCAN noise fraction:", float(np.mean(labels_hdb == -1)))
    print(f"{title_prefix} DBSCAN clusters:", len(set(labels_db)) - (1 if -1 in labels_db else 0))
    print(f"{title_prefix} DBSCAN noise fraction:", float(np.mean(labels_db == -1)))

    print(f"{title_prefix} occupancy (HDBSCAN primary):")
    display_fn(frac_hdb)

    print(f"{title_prefix} occupancy (DBSCAN baseline):")
    display_fn(frac_db)

    plot_occupancy_heatmap(
        frac_hdb,
        title=f"State Occupancy per Mutant ({title_prefix} Clusters)",
        outpath=f"figures/{prefix}_occupancy_heatmap.png"
    )

    plot_cluster_population(
        labels_hdb,
        title=f"Cluster Population ({title_prefix} Representation)",
        outpath=f"figures/{prefix}_cluster_population.png"
    )

    flat = scores_3d.reshape(-1, ncomp)
    plot_pca_clusters(
        flat[:, 0],
        flat[:, 1],
        labels_hdb,
        title=f"{title_prefix} PCA — HDBSCAN Clusters",
        outpath=f"figures/{prefix}_pca_clusters.png"
    )

    return {
        "labels_hdb": labels_hdb,
        "labels_db": labels_db,
        "counts_hdb": counts_hdb,
        "frac_hdb": frac_hdb,
        "counts_db": counts_db,
        "frac_db": frac_db,
    }
