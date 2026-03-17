import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import hdbscan


def run_hdbscan(scores, n_pcs=2, min_cluster_size=150, min_samples=None):
    """
    Run HDBSCAN on the first n_pcs principal components.

    Parameters
    ----------
    scores : np.ndarray
        PCA scores of shape (n_samples, n_components).
    n_pcs : int
        Number of PCs to use.
    min_cluster_size : int
        HDBSCAN min_cluster_size parameter.
    min_samples : int or None
        HDBSCAN min_samples parameter.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each sample.
    """
    Z = scores[:, :n_pcs]
    Zs = StandardScaler().fit_transform(Z)

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    labels = model.fit_predict(Zs)
    return labels


def run_dbscan(scores, n_pcs=2, eps=0.3, min_samples=100):
    """
    Run DBSCAN on the first n_pcs principal components.
    """
    Z = scores[:, :n_pcs]
    Zs = StandardScaler().fit_transform(Z)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(Zs)
    return labels


def cluster_summary(labels):
    """
    Return summary information for a clustering result.
    """
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)

    summary = {int(k): int(v) for k, v in zip(unique, counts)}
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_fraction = float(np.mean(labels == -1))

    return {
        "cluster_sizes": summary,
        "n_clusters": n_clusters,
        "noise_fraction": noise_fraction
    }
