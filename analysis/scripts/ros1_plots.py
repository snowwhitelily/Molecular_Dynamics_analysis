import matplotlib.pyplot as plt
import numpy as np


def plot_scree(evals, n=25, title=None):
    """Plots a scree plot with variance percentage labels and no in-canvas title."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    n_comp = min(n, len(evals))
    components = np.arange(1, n_comp + 1)
    eval_sub = evals[:n_comp]

    # Calculate percentage of total variance explained
    total_var = np.sum(evals)
    var_perc = (
        (eval_sub / total_var) * 100
        if total_var > 0
        else np.zeros_like(eval_sub)
    )

    # Plot Scree line/points
    ax.plot(
        components,
        eval_sub,
        marker="o",
        color="#2b5c8f",
        linewidth=1.5,
        markersize=5,
    )

    # Annotate variance percentages for top 3 PCs
    for i in range(min(3, n_comp)):
        ax.annotate(
            f"{var_perc[i]:.1f}%",
            (components[i], eval_sub[i]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#2b5c8f",
        )

    if len(eval_sub) > 0:
        ax.set_ylim((0, float(1.15 * np.max(eval_sub))))

    ax.set_xlabel("Principal Component", fontsize=11)
    ax.set_ylabel("Eigenvalue", fontsize=11)
    ax.set_xticks(np.arange(1, n_comp + 1, max(1, n_comp // 10)))

    # Hide top title for academic figure compliance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_pca_scatter(
    scores,
    pcx=0,
    pcy=1,
    stride=20,
    title="",
    colors=None,
    alpha=0.25,
    size=3,
    equal_aspect=True,
):
    """Plots 2D PCA scatter plot enforcing strict 1:1 aspect ratio."""
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ntraj = scores.shape[0]

    for i in range(ntraj):
        x = scores[i, ::stride, pcx]
        y = scores[i, ::stride, pcy]
        if colors is None:
            ax.scatter(x, y, s=size, alpha=alpha, linewidths=0)
        else:
            ax.scatter(x, y, s=size, c=[colors[i]], alpha=alpha, linewidths=0)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(f"PC{pcx+1}", fontsize=11)
    ax.set_ylabel(f"PC{pcy+1}", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_pca_scatter_zoom(
    scores,
    pcx=0,
    pcy=1,
    stride=20,
    title="",
    colors=None,
    alpha=0.25,
    size=3,
    qlow=1,
    qhigh=99,
    equal_aspect=True,
):
    """Plots zoomed 2D PCA scatter plot with 1:1 aspect ratio."""
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
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

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(f"PC{pcx+1}", fontsize=11)
    ax.set_ylabel(f"PC{pcy+1}", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()