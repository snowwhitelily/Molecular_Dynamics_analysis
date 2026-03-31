# plotting.py

# importing necssary libraries
import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path="figures"):
    os.makedirs(path, exist_ok=True)


def plot_pca_clusters(x, y, labels, title, xlabel="PC1", ylabel="PC2",
                      outpath=None, s=4, alpha=0.5, cmap="tab20"):
    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, c=labels, s=s, alpha=alpha, cmap=cmap)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if outpath is not None:
        ensure_dir(os.path.dirname(outpath) or ".")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()


def plot_cluster_population(labels, title="Cluster Population", outpath=None):
    labels = np.asarray(labels)
    labels = labels[labels != -1]

    clusters, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(6, 4))
    plt.bar(clusters, counts)
    plt.xlabel("Cluster / State")
    plt.ylabel("Number of Frames")
    plt.title(title)
    plt.tight_layout()

    if outpath is not None:
        ensure_dir(os.path.dirname(outpath) or ".")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()


def plot_occupancy_heatmap(occupancy_df, title="State Occupancy per Mutant", outpath=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(occupancy_df.values, aspect="auto")

    ax.set_xticks(np.arange(occupancy_df.shape[1]))
    ax.set_yticks(np.arange(occupancy_df.shape[0]))
    ax.set_xticklabels(occupancy_df.columns)
    ax.set_yticklabels(occupancy_df.index)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_xlabel("Cluster / State")
    ax.set_ylabel("Mutant")
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Occupancy fraction")

    plt.tight_layout()

    if outpath is not None:
        ensure_dir(os.path.dirname(outpath) or ".")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()
