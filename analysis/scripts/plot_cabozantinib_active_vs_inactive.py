#!/usr/bin/env python3
"""
plot_cabozantinib_active_vs_inactive.py

Cabozantinib active (DFG-in) vs inactive (DFG-out), honest about the asymmetry:
  - ACTIVE   = 16 MD frames per variant (ensemble) -> jittered dots + median bar
  - INACTIVE = ONE minimised DFG-out structure per variant (no MD ensemble
               exists) -> a single diamond marker

Both docked with the CORRECTED 3D cabozantinib in the shared inactive box, so the
comparison is fair (the earlier figure compared flat-ligand numbers on both sides).

Reads:
  active   -> results/ROS1/step6r_ensemble_scores_all3d.csv  (per-frame, corrected)
  inactive -> dock/.../vina_outputs_sharedbox_3d/ROS1_<V>_i__cabozantinib.pdbqt

Usage (venv on):
    python analysis/scripts/plot_cabozantinib_active_vs_inactive.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BASE    = Path.home() / "Molecular_Dynamics_analysis"
ACT_CSV = BASE / "results" / "ROS1" / "step6r_ensemble_scores_all3d.csv"
INA_DIR = BASE / "dock" / "ROS1" / "q2022p_vina" / "vina_outputs_sharedbox_3d"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final"
FIG_DIR.mkdir(parents=True, exist_ok=True)

VORD  = ["WT", "S1986F", "Q2022P", "Q2022P_S1986F"]
VLAB  = {"WT": "WT", "S1986F": "S1986F", "Q2022P": "Q2022P",
         "Q2022P_S1986F": "Q2022P\n+ S1986F"}
C_ACT = "#0072B2"   # active ensemble
C_INA = "#E69F00"   # inactive single structure


def best(pdbqt):
    for ln in open(pdbqt):
        if ln.startswith("REMARK VINA RESULT"):
            return float(ln.split()[3])
    return np.nan


def main():
    act = pd.read_csv(ACT_CSV)
    act = act[act.ligand == "cabozantinib"]

    inact = {}
    for v in VORD:
        f = INA_DIR / f"ROS1_{v}_i__cabozantinib.pdbqt"
        inact[v] = best(f) if f.exists() else np.nan

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    rng = np.random.default_rng(1)
    for i, v in enumerate(VORD):
        s = act[act.variant == v]["score"].values
        xa = i - 0.16
        ax.scatter(xa + rng.uniform(-0.10, 0.10, len(s)), s, s=55, color=C_ACT,
                   alpha=0.8, edgecolor="black", linewidth=0.5, zorder=3)
        med = float(np.median(s))
        ax.plot([xa - 0.16, xa + 0.16], [med, med], color="black", lw=2.6, zorder=4)
        ax.text(xa - 0.30, med, f"{med:.2f}", va="center", ha="right",
                fontsize=8.5, fontweight="bold", color="#111",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

        xi = i + 0.22
        ax.scatter([xi], [inact[v]], s=180, marker="D", color=C_INA,
                   edgecolor="black", linewidth=1.0, zorder=5)
        ax.text(xi + 0.12, inact[v], f"{inact[v]:.2f}", va="center", ha="left",
                fontsize=8.5, fontweight="bold", color="#111",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    ax.set_xticks(range(len(VORD)))
    ax.set_xticklabels([VLAB[v] for v in VORD], fontsize=11)
    ax.set_ylabel("Docking affinity (kcal/mol)", fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(-0.6, len(VORD) - 0.1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color="#eee")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title("Cabozantinib: active ensemble vs inactive conformation",
                 fontsize=13, pad=10)

    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ACT,
                  markeredgecolor="black", markersize=9,
                  label="active (DFG-in): 16 MD frames"),
           Line2D([0], [0], color="black", lw=2.6, label="active median"),
           Line2D([0], [0], marker="D", color="w", markerfacecolor=C_INA,
                  markeredgecolor="black", markersize=10,
                  label="inactive (DFG-out): 1 minimised structure")]
    ax.legend(handles=leg, frameon=False, fontsize=9.5, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.09),
              handletextpad=0.5, columnspacing=1.8)

    fig.text(0.5, -0.14,
             "Active = 16 MD frames docked (ensemble); inactive = single minimised "
             "DFG-out structure (no inactive MD ensemble exists). Both use the 3D "
             "ligand, shared box. More negative = stronger.",
             ha="center", va="top", fontsize=7.8, color="#666")

    plt.tight_layout()
    out = FIG_DIR / "cabozantinib_active_vs_inactive_3d.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    print(f"{'variant':16s} {'act_median':>10s} {'inactive':>9s} {'diff':>6s}")
    for v in VORD:
        am = float(np.median(act[act.variant == v]['score']))
        print(f"{v:16s} {am:10.2f} {inact[v]:9.2f} {am-inact[v]:+6.2f}")


if __name__ == "__main__":
    main()