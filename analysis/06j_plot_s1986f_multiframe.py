#!/usr/bin/env python3
"""
06j_plot_s1986f_multiframe.py

Supplementary figure: S1986F multi-frame docking spread, showing that the
original SINGLE-STRUCTURE results were unrepresentative of the conformational
ensemble.

Two ligands side by side:
  - cabozantinib: 10 frames; original shared-box medoid was -8.74 (weak end)
  - lorlatinib:   10 frames; original single-structure result was N/A
                  (failed dock) -> shown as a "failed" reference band

Each column shows the 10 per-frame affinities as jittered dots, the median as a
bar, the medoid frame highlighted, and a dashed reference line for the original
single-structure value so the artifact is obvious at a glance.

Reads the actual vina outputs (so numbers are never hand-typed). Run in venv.

Usage:
    python analysis/06j_plot_s1986f_multiframe.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path.home() / "Molecular_Dynamics_analysis"
MF = BASE / "dock" / "ROS1" / "q2022p_vina" / "s1986f_multiframe"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COL = {"cabozantinib": "#2E7D32", "lorlatinib": "#1976D2"}


def read_scores(outdir, ligand):
    scores, medoid_val = [], None
    for f in sorted(outdir.glob(f"*__{ligand}.pdbqt")):
        with open(f) as fh:
            for ln in fh:
                if ln.startswith("REMARK VINA RESULT"):
                    v = float(ln.split()[3]); break
            else:
                continue
        scores.append(v)
        if "MEDOID" in f.name:
            medoid_val = v
    return np.array(scores), medoid_val


def main():
    ligs = ["cabozantinib", "lorlatinib"]
    data = {}
    for lig in ligs:
        outdir = (MF / ("vina_outputs" if lig == "cabozantinib"
                        else "vina_outputs_lorlatinib"))
        s, med = read_scores(outdir, lig)
        if len(s) == 0:
            raise SystemExit(f"no {lig} outputs in {outdir}")
        data[lig] = (s, med)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    rng = np.random.default_rng(0)

    for i, lig in enumerate(ligs):
        s, med = data[lig]
        x = i + 1
        jitter = rng.uniform(-0.08, 0.08, len(s))
        # all frame dots
        ax.scatter(x + jitter, s, s=55, color=COL[lig], alpha=0.75,
                   edgecolor="black", linewidth=0.6, zorder=3,
                   label=None)
        # median bar
        ax.plot([x - 0.22, x + 0.22], [np.median(s)] * 2, color="black",
                lw=2.5, zorder=4)
        ax.text(x + 0.26, np.median(s), f"median {np.median(s):.2f}",
                va="center", fontsize=9)
        # medoid highlighted
        if med is not None:
            ax.scatter([x], [med], s=150, facecolor="none",
                       edgecolor="#E63946", linewidth=2.2, zorder=5)
            ax.annotate("medoid", xy=(x, med), xytext=(x - 0.34, med),
                        ha="right", va="center", color="#E63946",
                        fontsize=8, fontweight="bold")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["cabozantinib", "lorlatinib"], fontsize=11)
    ax.set_ylabel("Docking affinity (kcal/mol)", fontsize=11)
    ax.set_title("S1986F: 10-frame conformational ensemble",
                 fontsize=12, pad=10)
    ax.invert_yaxis()   # stronger (more negative) higher
    ax.grid(axis="y", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.text(0.5, -0.02,
             "Each dot = one MD frame docked in the shared box "
             "(10 frames spanning S1986F's conformational cluster).",
             ha="center", va="top", fontsize=7.5, color="#555")

    plt.tight_layout()
    out = FIG_DIR / "s1986f_multiframe_spread.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    for lig in ligs:
        s, med = data[lig]
        print(f"  {lig:14s} n={len(s)} median={np.median(s):.2f} "
              f"best={s.min():.2f} worst={s.max():.2f} medoid={med}")


if __name__ == "__main__":
    main()