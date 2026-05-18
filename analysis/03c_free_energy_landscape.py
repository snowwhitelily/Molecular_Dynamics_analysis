# ROS1 Analysis Pipeline
# Script 03c: Free Energy Landscapes
#
# Computes and plots 2D free energy landscapes from PCA score matrices.
# Free energy is computed as G = -kT ln(P), where P is the normalised
# frame density per bin of a 2D histogram over PC1 and PC2.
# Values are in units of kT at 300 K (1 kT ~ 0.593 kcal/mol).
#
# The colour scale is fixed at 0-2.5 kT across all plots (all-systems
# and per-mutant) to allow direct visual comparison between systems.
# Per-mutant plots reuse the all-systems axis limits so that basin
# positions are directly comparable.
#
# Usage:
#   python 03c_free_energy_landscape.py --prefixes actout_ctlfit,ctl --vmax 2.5 --per-mutant

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE     = Path.home() / "Molecular_Dynamics_analysis"
RESULTS  = BASE / "results" / "ROS1"
FIG_BASE = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "step3c_free_energy_landscapes"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prefixes", default="actout_ctlfit,ctl",
                   help="Comma-separated PCA prefixes to process.")
    p.add_argument("--bins",  type=int,   default=120,
                   help="Number of 2D histogram bins per axis.")
    p.add_argument("--kbt",   type=float, default=1.0,
                   help="kBT scale factor. With kbt=1.0, output is in units of kT.")
    p.add_argument("--vmax",  type=float, default=2.5,
                   help="Fixed colorbar maximum in kT. Applied to ALL plots.")
    p.add_argument("--per-mutant", action="store_true",
                   help="Also generate one FEL per mutant for each prefix.")
    return p.parse_args()


def save_current_figure(fig_dir: Path, filename: str):
    out = fig_dir / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure:", out)


def load_score_table(prefix: str):
    """
    Load PCA scores for a given prefix.

    Tries the CSV first (preferred). Falls back to loading the npy files
    directly if the CSV is not available.

    Returns a DataFrame with columns: traj_index, mutant, replica, PC1, PC2.
    """
    csv_f   = RESULTS / f"{prefix}_scores.csv"
    npy_f   = RESULTS / f"{prefix}_scores.npy"
    owner_f = RESULTS / f"{prefix}_x_owner.npy"
    F_f     = RESULTS / "F_paths.csv"

    if csv_f.exists():
        df      = pd.read_csv(csv_f)
        needed  = {"traj_index", "mutant", "replica", "PC1", "PC2"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{csv_f} missing required columns: {sorted(missing)}")
        return df

    if not (npy_f.exists() and owner_f.exists() and F_f.exists()):
        raise FileNotFoundError(
            f"Could not find {csv_f.name} or the trio "
            f"{npy_f.name}, {owner_f.name}, F_paths.csv"
        )

    scores  = np.load(npy_f)
    x_owner = np.load(owner_f).astype(int)
    F       = pd.read_csv(F_f)["folder"].tolist()

    if scores.shape[0] != x_owner.shape[0]:
        raise ValueError(f"{prefix}: scores/x_owner row mismatch.")
    if scores.shape[1] < 2:
        raise ValueError(f"{prefix}: need at least 2 PCs for FEL.")

    counts_seen = np.zeros(len(F), dtype=int)
    rows = []
    for i, ti in enumerate(x_owner):
        p = Path(F[ti])
        rows.append({
            "traj_index"              : int(ti),
            "mutant"                  : p.parent.name,
            "replica"                 : p.name,
            "frame_index_downsampled" : int(counts_seen[ti]),
            "PC1"                     : float(scores[i, 0]),
            "PC2"                     : float(scores[i, 1]),
        })
        counts_seen[ti] += 1
    return pd.DataFrame(rows)


def compute_fel(pc1, pc2, bins=120, kbt=1.0):
    """
    Compute a 2D free energy landscape from PC1 and PC2 coordinates.

    Builds a 2D histogram, normalises to probability, and applies the
    Boltzmann relation G = -kbt * ln(P). The global minimum is shifted
    to G = 0. Bins with no frames are returned as NaN.

    Returns
    -------
    G      : 2D array (transposed for imshow), shape (bins, bins)
    xedges : bin edges along PC1
    yedges : bin edges along PC2
    H      : raw count histogram (transposed)
    """
    H, xedges, yedges = np.histogram2d(pc1, pc2, bins=bins)
    P    = H / np.sum(H)
    G    = np.full_like(P, np.nan, dtype=float)
    mask = P > 0
    G[mask] = -kbt * np.log(P[mask])
    if np.any(mask):
        G[mask] -= np.nanmin(G[mask])
    return G.T, xedges, yedges, H.T


def plot_fel(pc1, pc2, G, xedges, yedges, title, fig_dir, filename, vmax=2.5):
    """
    Plot a 2D free energy landscape with a fixed colour scale.

    The colour scale is fixed between vmin=0 and vmax=vmax so that
    all plots in the analysis are directly comparable. Bins with no
    frames are shown as white (NaN). Contour lines are overlaid for
    readability.
    """
    plt.figure(figsize=(7.5, 6.5))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    im = plt.imshow(
        G,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=vmax,
        cmap="viridis",
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Relative free energy (kT)")

    finite = np.isfinite(G)
    if np.any(finite):
        xc, yc = (0.5 * (xedges[:-1] + xedges[1:])), (0.5 * (yedges[:-1] + yedges[1:]))
        Xc, Yc = np.meshgrid(xc, yc)
        nlev    = min(8, max(4, int(np.ceil(vmax)) + 1))
        levels  = np.linspace(0.0, vmax, nlev)
        plt.contour(Xc, Yc, G, levels=levels, linewidths=0.6, alpha=0.8)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title, fontsize=12, fontweight="bold")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    save_current_figure(fig_dir, filename)


def save_fel_table(prefix, group_name, G, xedges, yedges):
    """Save the FEL grid as a CSV for downstream use."""
    xc   = 0.5 * (xedges[:-1] + xedges[1:])
    yc   = 0.5 * (yedges[:-1] + yedges[1:])
    rows = [
        {"prefix": prefix, "group": group_name,
         "x_center": float(x), "y_center": float(y),
         "free_energy": float(G[iy, ix]) if np.isfinite(G[iy, ix]) else np.nan}
        for iy, y in enumerate(yc)
        for ix, x in enumerate(xc)
    ]
    out = RESULTS / f"{prefix}_fel_{group_name}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved FEL table:", out)


def main():
    args     = parse_args()
    prefixes = [x.strip() for x in args.prefixes.split(",") if x.strip()]
    if not prefixes:
        raise ValueError("No prefixes provided.")

    FIG_BASE.mkdir(parents=True, exist_ok=True)

    print(f"Prefixes: {prefixes} | bins={args.bins} | kbt={args.kbt} | vmax={args.vmax} kT")

    for prefix in prefixes:
        print(f"\nProcessing: {prefix}")

        df = load_score_table(prefix)
        if len(df) == 0:
            print(f"  Skipping {prefix}: empty score table.")
            continue

        print(f"  {len(df)} frames | mutants: {sorted(df['mutant'].unique())}")

        fig_dir = FIG_BASE / prefix
        fig_dir.mkdir(parents=True, exist_ok=True)

        pc1 = df["PC1"].to_numpy(dtype=float)
        pc2 = df["PC2"].to_numpy(dtype=float)

        # All-systems FEL
        G, xedges, yedges, H = compute_fel(pc1, pc2, bins=args.bins, kbt=args.kbt)
        plot_fel(pc1, pc2, G, xedges, yedges,
                 title=f"Free-Energy Landscape: {prefix} (all systems)",
                 fig_dir=fig_dir, filename=f"{prefix}_fel_all.png", vmax=args.vmax)
        save_fel_table(prefix, "all", G, xedges, yedges)

        score_out = RESULTS / f"{prefix}_fel_frame_points.csv"
        df.to_csv(score_out, index=False)
        print(f"  Saved frame points: {score_out}")

        # Per-mutant FELs
        # xedges and yedges from the all-systems histogram are reused so that
        # all per-mutant plots share identical axis limits and colour scale.
        if args.per_mutant:
            mutants = sorted(df["mutant"].unique())
            print(f"  Generating per-mutant FELs for {len(mutants)} mutants...")
            for mut in mutants:
                dfi = df[df["mutant"] == mut]
                if len(dfi) < 20:
                    print(f"  Skipping {prefix}/{mut}: too few frames ({len(dfi)}).")
                    continue
                pc1m = dfi["PC1"].to_numpy(dtype=float)
                pc2m = dfi["PC2"].to_numpy(dtype=float)
                Gm, _, _, _ = compute_fel(pc1m, pc2m, bins=args.bins, kbt=args.kbt)
                plot_fel(pc1m, pc2m, Gm, xedges, yedges,
                         title=f"Free-Energy Landscape: {prefix} ({mut})",
                         fig_dir=fig_dir, filename=f"{prefix}_fel_{mut}.png",
                         vmax=args.vmax)
                save_fel_table(prefix, mut, Gm, xedges, yedges)

    print("\nScript 3C completed.")


if __name__ == "__main__":
    main()