#!/usr/bin/env python3
"""
plot_porcupine.py

Porcupine plot of pocket motion: for each variant, an arrow at every pocket Calpha
showing the direction and amplitude of that residue along the variant's dominant
pocket mode (PC1). 2x2 panels (WT / S1986F / Q2022P / double).

Design choices (stated so the figure is honest):
  - PER-VARIANT PC1 (each variant's own dominant mode), NOT a common basis.
    A common basis would force identical arrow directions in every panel and could
    not show the key result -- that in the double mutant the dominant mode localises
    onto residue 1986. So directions are each variant's own PC1 and are therefore
    NOT directly comparable between panels (say so in the caption).
  - ONE shared viewing plane: the 2 principal spatial axes of the pooled mean Calpha
    positions, so all four panels are drawn from the same angle.
  - ONE shared arrow-length scale across panels, so arrow LENGTH (= amplitude of a
    residue's PC1 motion) IS comparable between panels. This is what the 3D spikes
    could not do; in 2D a long arrow is fine (no ligand to cross).

Amplitude convention: arrow vector at residue i = sqrt(lambda1) * eigvec1_i, i.e. the
RMS displacement of that Calpha along PC1 (A). Same units everywhere.

Usage (venv on):
    python analysis/scripts/plot_porcupine.py distance lorlatinib
(DRUG only selects which pocket-residue list to use; the motion is drug-independent.)
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import align

BASE   = Path.home() / "Molecular_Dynamics_analysis"
TRAJ   = BASE / "trajectories" / "ros1_prepared_final"
FIGS   = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "pocket_box"
OUTDIR = BASE / "figures" / "ROS1" / "ros1_prepared_final"

SELECTION = sys.argv[1] if len(sys.argv) > 1 else "distance"
DRUG      = sys.argv[2] if len(sys.argv) > 2 else "lorlatinib"

VARIANTS = ["WT", "S1986F", "Q2022P", "Q2022P_S1986F"]
LABELS   = {"WT": "non-mutated (WT)", "S1986F": "S1986F", "Q2022P": "Q2022P",
            "Q2022P_S1986F": "Q2022P + S1986F"}
# match the pocket-panel palette
PALETTE  = {"WT": "#7A7A7A", "S1986F": "#00A651",
            "Q2022P": "#1976D2", "Q2022P_S1986F": "#9C27B0"}

MUT_GRO = [53, 89]          # GRO numbering; +OFFSET = real ROS1 numbers
OFFSET  = 1933              # 53->1986, 89->2022
FIT_SEL = "name CA and resid 5-97"   # stable region to align on (as pocket_render)
STRIDE  = int(os.environ.get("PORCUPINE_STRIDE", 10))   # frame stride (speed)
ARROW_SCALE = float(os.environ.get("PORCUPINE_ARROW_SCALE", 4.0))  # visual A-per-A


def replicas(v):
    out = []
    for rep in ("0", "1", "2"):
        stem = TRAJ / v / rep / f"{v}-MD-prot"
        if stem.with_suffix(".pdb").exists() and stem.with_suffix(".xtc").exists():
            out.append((str(stem) + ".pdb", str(stem) + ".xtc"))
    return out


def pocket_resids():
    f = FIGS / f"prep_{SELECTION}_{DRUG}_resids.txt"
    rr = [int(x) for x in open(f).read().strip().split(",")]
    return sorted(set(rr))


def variant_pc1(v, ref, pocket_sel):
    """Return (mean_xyz (n,3), arrow_xyz (n,3), resids (n,), amp) for variant v.
    arrow_xyz = sqrt(lambda1) * PC1 eigenvector (per-Calpha 3-vector)."""
    coords = []
    resids = None
    for top, trj in replicas(v):
        u = mda.Universe(top, trj)
        align.AlignTraj(u, ref, select=FIT_SEL, in_memory=True).run()
        ag = u.select_atoms(pocket_sel)
        if resids is None:
            resids = ag.resids.copy()
        for ts in u.trajectory[::STRIDE]:
            coords.append(ag.positions.copy())
    X = np.asarray(coords)                     # (nframes, n, 3)
    n = X.shape[1]
    mean_xyz = X.mean(axis=0)                   # (n,3)
    Xc = (X - mean_xyz).reshape(X.shape[0], n * 3)
    # PC1 via SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Vt[0].reshape(n, 3)                   # unit eigenvector, per-Calpha 3-vec
    lam1 = (S[0] ** 2) / (X.shape[0] - 1)       # variance along PC1 (A^2)
    amp = float(np.sqrt(lam1))                  # RMS score amplitude (A)
    arrow_xyz = amp * pc1                        # per-Calpha RMS displacement (A)
    return mean_xyz, arrow_xyz, resids, amp


def main():
    pr = pocket_resids()
    pocket_sel = "name CA and resid " + " ".join(map(str, pr))

    # common alignment reference = WT replica 0, frame 0
    wt_top, wt_trj = replicas("WT")[0]
    ref = mda.Universe(wt_top)                  # single frame

    data = {}
    for v in VARIANTS:
        mean_xyz, arrow_xyz, resids, amp = variant_pc1(v, ref, pocket_sel)
        data[v] = dict(mean=mean_xyz, arrow=arrow_xyz, resids=resids, amp=amp)
        mag = np.linalg.norm(arrow_xyz, axis=1)
        m86 = mag[resids == 53]; m22 = mag[resids == 89]
        print(f"{v:16s} PC1 amp {amp:5.2f} A   "
              f"|arrow| 1986 {float(m86[0]) if len(m86) else float('nan'):.2f}  "
              f"2022 {float(m22[0]) if len(m22) else float('nan'):.2f}")

    # shared 2D viewing plane from pooled mean positions of all variants
    pooled = np.vstack([data[v]["mean"] for v in VARIANTS])
    pooled_c = pooled - pooled.mean(0)
    _, _, Vt2 = np.linalg.svd(pooled_c, full_matrices=False)
    axes2 = Vt2[:2]                             # (2,3) two spatial axes

    def to2d(xyz):    # project positions (subtract global centre first)
        return (xyz - pooled.mean(0)) @ axes2.T
    def vec2d(vec):   # project vectors (no centring)
        return vec @ axes2.T

    fig, axarr = plt.subplots(2, 2, figsize=(11, 10.5), dpi=150)
    PANEL_LETTER = ["A", "B", "C", "D"]
    for ax, v, letter in zip(axarr.ravel(), VARIANTS, PANEL_LETTER):
        d = data[v]; col = PALETTE[v]
        P = to2d(d["mean"]); A = vec2d(d["arrow"]) * ARROW_SCALE
        order = np.argsort(d["resids"])
        # thin backbone trace through Calpha in residue order
        ax.plot(P[order, 0], P[order, 1], color=col, lw=0.8, alpha=0.5, zorder=1)
        # arrows
        ax.quiver(P[:, 0], P[:, 1], A[:, 0], A[:, 1],
                  angles="xy", scale_units="xy", scale=1.0,
                  color=col, width=0.006, headwidth=4, headlength=5,
                  alpha=0.9, zorder=3)
        # mutation sites: black-edged markers + labels
        for gro in MUT_GRO:
            sel = d["resids"] == gro
            if sel.any():
                ax.scatter(P[sel, 0], P[sel, 1], s=70, facecolor="white",
                           edgecolor="black", linewidth=1.4, zorder=4)
                ax.annotate(str(gro + OFFSET), (P[sel, 0][0], P[sel, 1][0]),
                            textcoords="offset points", xytext=(6, 6),
                            fontsize=11, fontweight="bold", color="black", zorder=5)
        ax.set_title(LABELS[v], fontsize=13, fontweight="bold", color=col, pad=6)
        ax.text(0.01, 0.99, letter, transform=ax.transAxes, fontsize=18,
                fontweight="bold", va="top", ha="left")
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(f"Dominant pocket motion (PC1) by variant \u2014 {DRUG.capitalize()}\n"
                 "arrows = per-residue RMS motion along each variant's own PC1 "
                 "(shared length scale)", fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005,
             "Per-variant PC1 (directions are each variant's own dominant mode, not "
             "comparable between panels). Arrow length = RMS Calpha displacement along "
             "PC1 on one shared scale (lengths ARE comparable). Motion is at the "
             "mutation sites, away from the pocket; predicted binding is unchanged.",
             ha="center", va="bottom", fontsize=8, color="#666")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = OUTDIR / f"porcupine_pc1_{DRUG}.png"
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()