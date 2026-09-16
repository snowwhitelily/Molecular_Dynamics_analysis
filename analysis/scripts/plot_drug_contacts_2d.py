#!/usr/bin/env python3
"""
plot_drug_contacts_2d.py

2D binding-site contact diagram (wheel/spider style) for cabozantinib and
zidesamtinib, from the distance-based contact analysis (drug_contacts.json).

Each residue with any heavy atom within 4.5 A of the docked drug is drawn around
the ligand: red solid line + red bubble = tight contact <= 3.5 A (likely H-bond);
grey dashed = looser contact 3.5-4.5 A. Labelled with 1-letter code, real ROS1
number and the minimum distance.

Method note (honest): contacts are by heavy-atom distance on the WT averaged
pocket + the frame-corrected docked pose. "Likely H-bond" is a distance proxy,
not an explicit donor/acceptor analysis. The hinge pair Glu2027/Met2029 is
contacted by both drugs, consistent with the type-I ATP-competitive binding mode
reported for these inhibitors.

Usage (venv on):
    python analysis/scripts/plot_drug_contacts_2d.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE    = Path.home() / "Molecular_Dynamics_analysis"
CJSON   = BASE / "results" / "ROS1" / "drug_contacts.json"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DRUGS = ["cabozantinib", "zidesamtinib"]
AA = {"GLY": "G", "ALA": "A", "LEU": "L", "MET": "M", "GLU": "E", "ASP": "D",
      "SER": "S", "THR": "T", "PHE": "F", "ASN": "N", "ARG": "R", "CYS": "C",
      "TYR": "Y", "VAL": "V", "ILE": "I", "LYS": "K", "HIS": "H", "PRO": "P",
      "TRP": "W", "GLN": "Q"}
HB = 3.5   # tight-contact / likely-H-bond threshold


def main():
    C = json.load(open(CJSON))
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.0))
    for ax, drug in zip(axes, DRUGS):
        res = C[drug]
        n = len(res)
        ang = (90 - np.arange(n) * 360.0 / n) * np.pi / 180.0
        ax.scatter([0], [0], s=2600, color="#E6C200", edgecolor="black",
                   lw=1.5, zorder=2)
        ax.text(0, 0, drug.capitalize(), ha="center", va="center",
                fontsize=11, fontweight="bold", zorder=3)
        for (real, rn, d), a in zip(res, ang):
            x, y = np.cos(a), np.sin(a)
            hb = d < HB
            col = "#C0392B" if hb else "#7f8c8d"
            ax.plot([0.16 * x, 0.82 * x], [0.16 * y, 0.82 * y], color=col,
                    lw=2.4 if hb else 1.0, ls="-" if hb else (0, (4, 3)), zorder=1)
            ax.scatter([x], [y], s=1050, color="#F5B7B1" if hb else "#EAEDED",
                       edgecolor=col, lw=1.6, zorder=2)
            ax.text(x, y + 0.04, f"{AA.get(rn,'X')}{real}", ha="center",
                    va="center", fontsize=9.5, fontweight="bold", zorder=3)
            ax.text(x, y - 0.12, f"{d:.1f} \u00c5", ha="center", va="center",
                    fontsize=7.5, color="#555", zorder=3)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(drug.capitalize(), fontsize=13, fontweight="bold", pad=2)

    leg = [Line2D([0], [0], color="#C0392B", lw=2.4,
                  label="tight contact \u2264 3.5 \u00c5 (likely H-bond)"),
           Line2D([0], [0], color="#7f8c8d", lw=1.2, ls=(0, (4, 3)),
                  label="contact 3.5\u20134.5 \u00c5")]
    fig.legend(handles=leg, frameon=False, fontsize=10, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Predicted binding-site contacts (WT ATP pocket)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.text(0.5, 0.045,
             "Residues with any heavy atom within 4.5 \u00c5 of the docked drug. "
             "Hinge pair Glu2027/Met2029 contacted by both drugs. Contact by "
             "distance on a Vina pose; not an explicit H-bond analysis.",
             ha="center", va="top", fontsize=8, color="#666")

    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    out = FIG_DIR / "drug_contacts_2d.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    for drug in DRUGS:
        tight = [f"{AA.get(rn,'X')}{r}" for r, rn, d in C[drug] if d < HB]
        print(f"  {drug:14s} tight contacts: {', '.join(tight)}")


if __name__ == "__main__":
    main()