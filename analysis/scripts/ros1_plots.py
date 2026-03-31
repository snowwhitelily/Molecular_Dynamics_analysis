import numpy as np
import matplotlib.pyplot as plt


def plot_scree(evals, n=25, title="Scree plot"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(np.arange(1, min(n, len(evals)) + 1), evals[:n])
    if len(evals[:n]) > 0:
        ax.set_ylim((0, float(1.1 * np.max(evals[:n]))))
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(title)
    plt.show()


def plot_pca_scatter(scores, pcx=0, pcy=1, stride=20, title="", colors=None, alpha=0.25, size=3):
    fig, ax = plt.subplots(figsize=(8, 6))
    ntraj = scores.shape[0]

    for i in range(ntraj):
        x = scores[i, ::stride, pcx]
        y = scores[i, ::stride, pcy]

        if colors is None:
            ax.scatter(x, y, s=size, alpha=alpha, linewidths=0)
        else:
            ax.scatter(x, y, s=size, c=[colors[i]], alpha=alpha, linewidths=0)

    ax.set_aspect("auto")
    ax.set_xlabel(f"PC{pcx+1}")
    ax.set_ylabel(f"PC{pcy+1}")
    ax.set_title(title)
    plt.show()


def plot_pca_scatter_zoom(scores, pcx=0, pcy=1, stride=20, title="", colors=None,
                          alpha=0.25, size=3, qlow=1, qhigh=99):
    fig, ax = plt.subplots(figsize=(8, 6))
    ntraj = scores.shape[0]

    all_x = scores[:, ::stride, pcx].reshape(-1)
    all_y = scores[:, ::stride, pcy].reshape(-1)

    x0, x1 = np.percentile(all_x[np.isfinite(all_x)], [qlow, qhigh])
    y0, y1 = np.percentile(all_y[np.isfinite(all_y)], [qlow, qhigh])

    for i in range(ntraj):
        x = scores[i, ::stride, pcx]
        y = scores[i, ::stride, pcy]

        if colors is None:
            ax.scatter(x, y, s=size, alpha=alpha, linewidths=0)
        else:
            ax.scatter(x, y, s=size, c=[colors[i]], alpha=alpha, linewidths=0)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("auto")
    ax.set_xlabel(f"PC{pcx+1}")
    ax.set_ylabel(f"PC{pcy+1}")
    ax.set_title(title)
    plt.show()
