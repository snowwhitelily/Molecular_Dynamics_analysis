# %%
# ROS1 Analysis Pipeline
# Script 3C: Free-Energy Landscapes for Selected PCA Spaces

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
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prefixes",
        default="actout_ctlfit,aloop",
        help="Comma-separated PCA prefixes, e.g. actout_ctlfit,aloop,dist",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=120,
        help="Number of 2D histogram bins per axis.",
    )
    p.add_argument(
        "--kbt",
        type=float,
        default=1.0,
        help="Scale factor used in G = -kBT ln P. For relative maps only, 1.0 is fine.",
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
            f"{prefix}: scores rows ({scores.shape[0]}) != x_owner rows ({x_owner.shape[0]})"
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
    H, xedges, yedges = np.histogram2d(pc1, pc2, bins=bins)
    P = H / np.sum(H)

    G = np.full_like(P, np.nan, dtype=float)
    mask = P > 0
    G[mask] = -kbt * np.log(P[mask])

    if np.any(mask):
        G[mask] -= np.nanmin(G[mask])

    return G.T, xedges, yedges, H.T  # transpose for plotting convention


def plot_fel(pc1, pc2, G, xedges, yedges, title, fig_dir, filename):
    plt.figure(figsize=(7.5, 6.5))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    im = plt.imshow(
        G,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Relative free energy")

    finite = np.isfinite(G)
    if np.any(finite):
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        Xc, Yc = np.meshgrid(xc, yc)
        zmin = float(np.nanmin(G))
        zmax = float(np.nanmax(G))
        if zmax > zmin:
            nlev = min(8, max(4, int(np.ceil(zmax - zmin)) + 1))
            levels = np.linspace(zmin, zmax, nlev)
            plt.contour(Xc, Yc, G, levels=levels, linewidths=0.6, alpha=0.8)

    plt.scatter(pc1, pc2, s=2, alpha=0.08, linewidths=0)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    save_current_figure(fig_dir, filename)


def save_fel_table(prefix, group_name, G, xedges, yedges):
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
                "free_energy": float(G[iy, ix]) if np.isfinite(G[iy, ix]) else np.nan,
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

    print("Running Script 3C: Free-Energy Landscapes")
    print("Prefixes:", prefixes)
    print("Bins:", args.bins)
    print("Per-mutant:", args.per_mutant)

    for prefix in prefixes:
        df = load_score_table(prefix)
        if len(df) == 0:
            print(f"Skipping {prefix}: empty score table.")
            continue

        fig_dir = FIG_BASE / prefix
        fig_dir.mkdir(parents=True, exist_ok=True)

        pc1 = df["PC1"].to_numpy(dtype=float)
        pc2 = df["PC2"].to_numpy(dtype=float)

        G, xedges, yedges, H = compute_fel(pc1, pc2, bins=args.bins, kbt=args.kbt)

        plot_fel(
            pc1=pc1,
            pc2=pc2,
            G=G,
            xedges=xedges,
            yedges=yedges,
            title=f"Free-Energy Landscape: {prefix} (all systems)",
            fig_dir=fig_dir,
            filename=f"{prefix}_fel_all.png",
        )
        save_fel_table(prefix, "all", G, xedges, yedges)

        # Also save a quick frame-level score table copy if it did not already exist in csv form.
        score_out = RESULTS / f"{prefix}_fel_frame_points.csv"
        df.to_csv(score_out, index=False)
        print("Saved frame points table:", score_out)

        if args.per_mutant:
            for mut, dfi in df.groupby("mutant"):
                if len(dfi) < 20:
                    print(f"Skipping per-mutant FEL for {prefix}/{mut}: too few frames ({len(dfi)}).")
                    continue

                pc1m = dfi["PC1"].to_numpy(dtype=float)
                pc2m = dfi["PC2"].to_numpy(dtype=float)
                Gm, _, _, _ = compute_fel(pc1m, pc2m, bins=args.bins, kbt=args.kbt)

                plot_fel(
                    pc1=pc1m,
                    pc2=pc2m,
                    G=Gm,
                    xedges=xedges,
                    yedges=yedges,
                    title=f"Free-Energy Landscape: {prefix} ({mut})",
                    fig_dir=fig_dir,
                    filename=f"{prefix}_fel_{mut}.png",
                )
                save_fel_table(prefix, mut, Gm, xedges, yedges)

    print("Script 3C completed.")


if __name__ == "__main__":
    main()
