#!/usr/bin/env python3
"""
plot_umcg_active_vs_inactive_both_drugs.py

Two-panel docking summary for UMCG (clinical audience): cabozantinib and
zidesamtinib, active (DFG-in) ensemble vs inactive (DFG-out) single structure,
across the four ROS1 variants. In the spirit of the apo/holo docking-ensemble
panel from the literature reference.

  - active   = 16 MD frames per variant (ensemble) -> jittered dots + median bar
  - inactive = ONE minimised DFG-out structure per variant -> single diamond

All values use the CORRECTED 3D ligands (step6r_ensemble_scores_all3d.csv for
active; vina_outputs_sharedbox_3d/ for inactive), shared box.

NOTE: this is Vina docking score, NOT experimental IC50. We do not reproduce the
reference's IC50 fold-change panel because that is wet-lab potency we do not have.

Usage (venv on):
    python analysis/scripts/plot_umcg_active_vs_inactive_both_drugs.py
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

DRUGS = ["cabozantinib", "zidesamtinib"]
VORD  = ["WT", "S1986F", "Q2022P", "Q2022P_S1986F"]
VLAB  = {"WT": "WT", "S1986F": "S1986F", "Q2022P": "Q2022P",
         "Q2022P_S1986F": "Q2022P\n+ S1986F"}
C_ACT = "#0072B2"
C_INA = "#E69F00"


def best(pdbqt):
    for ln in open(pdbqt):
        if ln.startswith("REMARK VINA RESULT"):
            return float(ln.split()[3])
    return np.nan


def main():
    act = pd.read_csv(ACT_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    rng = np.random.default_rng(1)
    for ax, drug in zip(axes, DRUGS):
        d = act[act.ligand == drug]
        for i, v in enumerate(VORD):
            s = d[d.variant == v]["score"].values
            xa = i - 0.16
            ax.scatter(xa + rng.uniform(-0.10, 0.10, len(s)), s, s=48, color=C_ACT,
                       alpha=0.8, edgecolor="black", linewidth=0.5, zorder=3)
            med = float(np.median(s))
            ax.plot([xa - 0.16, xa + 0.16], [med, med], color="black", lw=2.4, zorder=4)
            ax.text(xa - 0.28, med, f"{med:.2f}", va="center", ha="right",
                    fontsize=8, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

            inaf = INA_DIR / f"ROS1_{v}_i__{drug}.pdbqt"
            iv = best(inaf) if inaf.exists() else np.nan
            xi = i + 0.22
            ax.scatter([xi], [iv], s=150, marker="D", color=C_INA,
                       edgecolor="black", linewidth=1.0, zorder=5)
            ax.text(xi + 0.10, iv, f"{iv:.2f}", va="center", ha="left",
                    fontsize=8, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

        ax.set_title(drug.capitalize(), fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks(range(len(VORD)))
        ax.set_xticklabels([VLAB[v] for v in VORD], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(-0.6, len(VORD) - 0.1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.yaxis.grid(True, color="#eee")
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    axes[0].set_ylabel("Docking affinity (kcal/mol)", fontsize=11)
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ACT,
                  markeredgecolor="black", markersize=9,
                  label="active (DFG-in): 16 MD frames"),
           Line2D([0], [0], color="black", lw=2.4, label="active median"),
           Line2D([0], [0], marker="D", color="w", markerfacecolor=C_INA,
                  markeredgecolor="black", markersize=10,
                  label="inactive (DFG-out): 1 minimised structure")]
    fig.legend(handles=leg, frameon=False, fontsize=9.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Docking across ROS1 variants — active ensemble vs inactive "
                 "conformation", fontsize=14, fontweight="bold", y=1.0)
    fig.text(0.5, -0.07,
             "Active = 16 MD frames per variant (ensemble); inactive = single "
             "minimised DFG-out structure (no inactive MD ensemble exists). Both "
             "drugs use 3D ligands, shared box. Vina docking score, not IC50. "
             "More negative = stronger.",
             ha="center", va="top", fontsize=8, color="#666")

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    out = FIG_DIR / "umcg_active_vs_inactive_both_drugs.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}\n")
    print(f"{'drug':14s} {'variant':16s} {'act_med':>8s} {'inactive':>9s} {'diff':>6s}")
    for drug in DRUGS:
        d = act[act.ligand == drug]
        for v in VORD:
            am = float(np.median(d[d.variant == v]["score"]))
            inaf = INA_DIR / f"ROS1_{v}_i__{drug}.pdbqt"
            iv = best(inaf) if inaf.exists() else float("nan")
            print(f"{drug:14s} {v:16s} {am:8.2f} {iv:9.2f} {am-iv:+6.2f}")


if __name__ == "__main__":
    main()