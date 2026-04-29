# %%
# ROS1 Analysis Pipeline
# Script 3C: Free-Energy Landscapes for Selected PCA Spaces
#
# KEY CHANGES FROM ORIGINAL:
# 1. Added --vmax argument for fixed colorbar scale across all plots
#    (supervisor request: "make sure the color scale is the same everywhere")
# 2. vmin=0, vmax=args.vmax now passed to imshow in all plot_fel calls
# 3. aloop removed from default prefixes (supervisor: "aloop is not of much interest")
# 4. Free energy unit is kT (with kbt=1.0 default), stated explicitly in plot label
#
# Usage:
#   python 03c_free_energy_landscape.py --prefixes actout_ctlfit,ctl --vmax 2.5 --per-mutant
#
# Free energy formula:
#   G = -kT * ln(P)
#   where P = normalised frame density per 2D histogram bin
#   Values are in units of kT at 300K (kT ~ 0.593 kcal/mol)
#   The minimum bin is set to G=0; all other bins are relative to it.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_BASE = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "step3c_free_energy_landscapes"


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute and plot 2D free-energy landscapes from PCA scores."
    )
    p.add_argument(
        "--prefixes",
        default="actout_ctlfit,ctl",
        help=(
            "Comma-separated PCA prefixes to process. "
            "Default: actout_ctlfit,ctl  "
            "(aloop excluded per supervisor feedback). "
            "Example: --prefixes actout_ctlfit,ctl,actin"
        ),
    )
    p.add_argument(
        "--bins",
        type=int,
        default=120,
        help="Number of 2D histogram bins per axis. Default: 120.",
    )
    p.add_argument(
        "--kbt",
        type=float,
        default=1.0,
        help=(
            "Scale factor in G = -kBT ln P. "
            "With kbt=1.0 (default), output is in units of kT. "
            "At 300K: 1 kT ~ 0.593 kcal/mol ~ 2.479 kJ/mol."
        ),
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=2.5,
        help=(
            "Fixed colorbar maximum in kT units. "
            "Applied to ALL plots (all-systems and per-mutant) "
            "so that color scale is consistent everywhere. "
            "Default: 2.5 kT. "
            "Supervisor requirement: same scale across all figures."
        ),
    )
    p.add_argument(
        "--per-mutant",
        action="store_true",
        help="Also save one FEL per mutant for each selected prefix.",
    )
    return p.parse_args()


def save_current_figure(fig_dir: Path, filename: str):
    out = fig_dir / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure:", out)


def load_score_table(prefix: str):
    """
    Load PCA scores for a given prefix.
    Tries CSV first, then falls back to npy + x_owner + F_paths.
    Returns a DataFrame with columns: traj_index, mutant, replica, PC1, PC2.
    """
    csv_f = RESULTS / f"{prefix}_scores.csv"
    npy_f = RESULTS / f"{prefix}_scores.npy"
    owner_f = RESULTS / f"{prefix}_x_owner.npy"
    F_f = RESULTS / "F_paths.csv"

    if csv_f.exists():
        df = pd.read_csv(csv_f)
        needed = {"traj_index", "mutant", "replica", "PC1", "PC2"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{csv_f} missing required columns: {sorted(missing)}")
        return df

    if not (npy_f.exists() and owner_f.exists() and F_f.exists()):
        raise FileNotFoundError(
            f"Could not find either {csv_f.name} or the trio "
            f"{npy_f.name}, {owner_f.name}, F_paths.csv"
        )

    scores = np.load(npy_f)
    x_owner = np.load(owner_f).astype(int)
    F = pd.read_csv(F_f)["folder"].tolist()

    if scores.shape[0] != x_owner.shape[0]:
        raise ValueError(
            f"{prefix}: scores rows ({scores.shape[0]}) != "
            f"x_owner rows ({x_owner.shape[0]})"
        )
    if scores.shape[1] < 2:
        raise ValueError(f"{prefix}: need at least 2 PCs for FEL")

    counts_seen = np.zeros(len(F), dtype=int)
    rows = []
    for i, ti in enumerate(x_owner):
        p = Path(F[ti])
        rows.append({
            "traj_index": int(ti),
            "mutant": p.parent.name,
            "replica": p.name,
            "frame_index_downsampled": int(counts_seen[ti]),
            "PC1": float(scores[i, 0]),
            "PC2": float(scores[i, 1]),
        })
        counts_seen[ti] += 1
    return pd.DataFrame(rows)


def compute_fel(pc1, pc2, bins=120, kbt=1.0):
    """
    Compute 2D free-energy landscape from PC1 and PC2 coordinates.

    Method:
        1. Build a 2D histogram of (PC1, PC2) with 'bins' bins per axis.
        2. Normalise counts to get probability P per bin.
        3. Compute G = -kbt * ln(P) for bins with P > 0.
        4. Shift so the global minimum is G = 0.

    Returns:
        G       : 2D array of free energies (shape: bins x bins), transposed for imshow
        xedges  : bin edges along PC1
        yedges  : bin edges along PC2
        H       : raw count histogram (transposed)
    """
    H, xedges, yedges = np.histogram2d(pc1, pc2, bins=bins)
    P = H / np.sum(H)

    G = np.full_like(P, np.nan, dtype=float)
    mask = P > 0
    G[mask] = -kbt * np.log(P[mask])

    if np.any(mask):
        G[mask] -= np.nanmin(G[mask])

    # Transpose for correct imshow orientation (y-axis = PC2)
    return G.T, xedges, yedges, H.T


def plot_fel(pc1, pc2, G, xedges, yedges, title, fig_dir, filename, vmax=2.5):
    """
    Plot a 2D free-energy landscape.

    The colorbar is fixed between vmin=0 and vmax=vmax (in kT units)
    so that all plots in the analysis are directly comparable.
    Bins with no frames visited are shown as white (NaN).

    Parameters:
        vmax : float
            Upper limit of colorbar. Same value must be used for ALL plots
            in the analysis (supervisor requirement).
    """
    plt.figure(figsize=(7.5, 6.5))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Plot FEL with FIXED color scale (vmin=0, vmax=vmax)
    # This is the key change from the original script.
    # Using the same vmax everywhere means the same colour = same energy
    # in every figure, allowing direct visual comparison between mutants.
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
    cbar.set_label("Relative free energy (kT)")  # explicit unit label

    # Overlay contour lines for clarity
    finite = np.isfinite(G)
    if np.any(finite):
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        Xc, Yc = np.meshgrid(xc, yc)
        zmin = 0.0
        zmax = vmax  # contours also respect fixed scale
        nlev = min(8, max(4, int(np.ceil(zmax - zmin)) + 1))
        levels = np.linspace(zmin, zmax, nlev)
        plt.contour(Xc, Yc, G, levels=levels, linewidths=0.6, alpha=0.8)

    # Scatter raw frame positions (very transparent, just for reference)
    plt.scatter(pc1, pc2, s=2, alpha=0.08, linewidths=0, color="steelblue")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    save_current_figure(fig_dir, filename)


def save_fel_table(prefix, group_name, G, xedges, yedges):
    """Save the FEL grid as a CSV for downstream use."""
    rows = []
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    for iy, y in enumerate(yc):
        for ix, x in enumerate(xc):
            rows.append({
                "prefix": prefix,
                "group": group_name,
                "x_center": float(x),
                "y_center": float(y),
                "free_energy": (
                    float(G[iy, ix]) if np.isfinite(G[iy, ix]) else np.nan
                ),
            })
    out = RESULTS / f"{prefix}_fel_{group_name}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved FEL table:", out)


def main():
    args = parse_args()

    prefixes = [x.strip() for x in args.prefixes.split(",") if x.strip()]
    if not prefixes:
        raise ValueError("No prefixes provided.")

    FIG_BASE.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Running Script 3C: Free-Energy Landscapes")
    print("=" * 60)
    print(f"Prefixes  : {prefixes}")
    print(f"Bins      : {args.bins}")
    print(f"kbt       : {args.kbt}  (output unit: kT)")
    print(f"vmax      : {args.vmax} kT  (FIXED for all plots — supervisor requirement)")
    print(f"Per-mutant: {args.per_mutant}")
    print(f"Note      : aloop excluded from default prefixes per supervisor feedback.")
    print("=" * 60)

    for prefix in prefixes:
        print(f"\nProcessing prefix: {prefix}")

        df = load_score_table(prefix)
        if len(df) == 0:
            print(f"  Skipping {prefix}: empty score table.")
            continue

        print(f"  Loaded {len(df)} frames.")
        print(f"  Mutants found: {sorted(df['mutant'].unique())}")

        fig_dir = FIG_BASE / prefix
        fig_dir.mkdir(parents=True, exist_ok=True)

        pc1 = df["PC1"].to_numpy(dtype=float)
        pc2 = df["PC2"].to_numpy(dtype=float)

        # --- All-systems FEL ---
        G, xedges, yedges, H = compute_fel(
            pc1, pc2, bins=args.bins, kbt=args.kbt
        )

        plot_fel(
            pc1=pc1,
            pc2=pc2,
            G=G,
            xedges=xedges,
            yedges=yedges,
            title=f"Free-Energy Landscape: {prefix} (all systems)",
            fig_dir=fig_dir,
            filename=f"{prefix}_fel_all.png",
            vmax=args.vmax,
        )
        save_fel_table(prefix, "all", G, xedges, yedges)

        # Save frame-level score table copy
        score_out = RESULTS / f"{prefix}_fel_frame_points.csv"
        df.to_csv(score_out, index=False)
        print(f"  Saved frame points table: {score_out}")

        # --- Per-mutant FELs ---
        # IMPORTANT: xedges and yedges from the all-systems histogram are
        # reused here so that all per-mutant plots share the same axis limits.
        # Combined with the fixed vmax, this ensures ALL figures are
        # directly comparable — same axes, same colour scale.
        if args.per_mutant:
            mutants = sorted(df["mutant"].unique())
            print(f"  Generating per-mutant FELs for {len(mutants)} mutants...")

            for mut in mutants:
                dfi = df[df["mutant"] == mut]
                if len(dfi) < 20:
                    print(
                        f"  Skipping per-mutant FEL for {prefix}/{mut}: "
                        f"too few frames ({len(dfi)})."
                    )
                    continue

                pc1m = dfi["PC1"].to_numpy(dtype=float)
                pc2m = dfi["PC2"].to_numpy(dtype=float)

                # Use the all-systems xedges/yedges so axis limits are identical
                Gm, _, _, _ = compute_fel(
                    pc1m, pc2m, bins=args.bins, kbt=args.kbt
                )

                plot_fel(
                    pc1=pc1m,
                    pc2=pc2m,
                    G=Gm,
                    xedges=xedges,   # shared edges — same axis limits everywhere
                    yedges=yedges,   # shared edges — same axis limits everywhere
                    title=f"Free-Energy Landscape: {prefix} ({mut})",
                    fig_dir=fig_dir,
                    filename=f"{prefix}_fel_{mut}.png",
                    vmax=args.vmax,  # fixed colorbar — same scale everywhere
                )
                save_fel_table(prefix, mut, Gm, xedges, yedges)

            print(f"  Per-mutant FELs done for {prefix}.")

    print("\n" + "=" * 60)
    print("Script 3C completed.")
    print("=" * 60)
    print("\nMethods note for thesis:")
    print(
        "Free energy landscapes were computed as G = -kT ln(P), where P is "
        "the normalised frame density in each bin of a 120x120 2D histogram "
        "over PC1 and PC2. Values are in units of kT at 300K "
        "(kT ~ 0.593 kcal/mol). The colour scale is fixed at 0-2.5 kT "
        "across all plots to allow direct visual comparison between systems."
    )


if __name__ == "__main__":
    main()
