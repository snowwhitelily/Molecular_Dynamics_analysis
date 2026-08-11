# ROS1 Analysis Pipeline
# Script 03c: Free Energy Landscapes (KDE-Smoothed & 1:1 Aspect Ratio)

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_BASE = (
    BASE
    / "figures"
    / "ROS1"
    / "ros1_prepared_final"
    / "step3_free_energy_landscapes"
)


def compute_kde_fel(
    x, y, grid_bins=100, x_range=None, y_range=None, temperature=300.0
):
    """Computes a 2D Free Energy Landscape using Gaussian Kernel Density Estimation (KDE)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]

    if x_range is None:
        x_range = (x_clean.min(), x_clean.max())
    if y_range is None:
        y_range = (y_clean.min(), y_clean.max())

    # Build grid
    xi = np.linspace(x_range[0], x_range[1], grid_bins)
    yi = np.linspace(y_range[0], y_range[1], grid_bins)
    Xi, Yi = np.meshgrid(xi, yi)

    # Evaluate Gaussian KDE
    positions = np.vstack([Xi.ravel(), Yi.ravel()])
    values = np.vstack([x_clean, y_clean])
    kernel = gaussian_kde(values)
    Zi = kernel(positions).reshape(Xi.shape)

    # Convert probability density to Free Energy (delta G = -k_B * T * ln(P))
    # Normalize density so max is 1 (min energy = 0 kJ/mol)
    P_max = Zi.max()
    if P_max > 0:
        Zi_norm = Zi / P_max
    else:
        Zi_norm = Zi

    # Avoid log(0) warnings by clipping low probability densities
    Zi_norm = np.clip(Zi_norm, 1e-10, 1.0)

    # kB = 0.0083144626 kJ/(mol*K)
    kB = 0.0083144626
    delta_G = -kB * temperature * np.log(Zi_norm)

    return Xi, Yi, delta_G


def plot_fel_panel(
    Xi, Yi, delta_G, x_range, y_range, out_path, vmax=2.5, cbar_label="ΔG (kJ/mol)"
):
    """Plots a single FEL surface enforcing a 1:1 aspect ratio and clean spines."""
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    # Contour levels from 0 to vmax kJ/mol
    levels = np.linspace(0, vmax, 21)

    cf = ax.contourf(
        Xi, Yi, delta_G, levels=levels, cmap="viridis_r", extend="max"
    )
    ax.contour(
        Xi, Yi, delta_G, levels=levels[::4], colors="black", linewidths=0.3, alpha=0.5
    )

    # Enforce strict 1:1 physical aspect ratio
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel("PC1 (nm)", fontsize=11)
    ax.set_ylabel("PC2 (nm)", fontsize=11)

    cbar = fig.colorbar(cf, ax=ax, shrink=0.82)
    cbar.set_label(cbar_label, fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved FEL:", out_path)


def process_prefix(prefix, vmax, per_mutant):
    out_dir = FIG_BASE / prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS / f"step2_{prefix}_pca_projection.csv"
    if not csv_path.exists():
        print(f"Warning: CSV not found for prefix {prefix} at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Global coordinate bounds for comparative panel consistency
    x_range = (df["PC1"].min() - 0.2, df["PC1"].max() + 0.2)
    y_range = (df["PC2"].min() - 0.2, df["PC2"].max() + 0.2)

    # 1. Global Landscape
    Xi, Yi, dG = compute_kde_fel(
        df["PC1"].values, df["PC2"].values, x_range=x_range, y_range=y_range
    )
    plot_fel_panel(
        Xi, Yi, dG, x_range, y_range, out_dir / f"{prefix}_global_fel.png", vmax=vmax
    )

    # 2. Per-Mutant Landscapes (Using shared global bounds)
    if per_mutant and "mutant" in df.columns:
        mut_dir = out_dir / "per_mutant"
        mut_dir.mkdir(exist_ok=True)
        for mut, group in df.groupby("mutant"):
            Xi_m, Yi_m, dG_m = compute_kde_fel(
                group["PC1"].values,
                group["PC2"].values,
                x_range=x_range,
                y_range=y_range,
            )
            plot_fel_panel(
                Xi_m,
                Yi_m,
                dG_m,
                x_range,
                y_range,
                mut_dir / f"{prefix}_{mut}_fel.png",
                vmax=vmax,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate KDE-smoothed FEL plots with 1:1 aspect ratio."
    )
    parser.add_argument(
        "--prefixes",
        default="actout_ctlfit,ctl",
        help="Comma-separated prefixes",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=2.5,
        help="Max energy cutoff for colorbar",
    )
    parser.add_argument(
        "--per-mutant", action="store_true", help="Generate per-mutant landscapes"
    )
    args = parser.parse_args()

    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]
    for p in prefixes:
        process_prefix(p, args.vmax, args.per_mutant)


if __name__ == "__main__":
    main()