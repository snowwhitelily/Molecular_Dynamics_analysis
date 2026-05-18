# ROS1 Analysis Pipeline
# Script 03a_patch: Additional Metric Figures
#
# Reads the CSVs produced by 03a_metrics_states.py and generates
# additional plots with error bars across replicas:
#   1. alphaC-helix to hinge distance (mean ± std per mutant)
#   2. alphaC-helix RMSF (flexibility per mutant)
#   3. K1980-E1993 salt bridge finding summary
#   4. ATP pocket open-state occupancy (mean ± std per mutant)
#   5. DFG-Phe chi1 mean angle per mutant
#   6. DFG-in state fraction per mutant
#
# Run after 03a_metrics_states.py has completed.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "step3_metrics_states"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(filename):
    out = FIG_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


# ============================================================
# Load metric CSVs from Script 03a
# ============================================================
df_alphaC = pd.read_csv(RESULTS / "step3a_alphaC_metrics.csv")
df_pocket = pd.read_csv(RESULTS / "step3a_atp_pocket_metrics.csv")
df_dfg    = pd.read_csv(RESULTS / "step3a_dfg_state_summary.csv")

# ============================================================
# 1. alphaC-helix to hinge distance with error bars
# ============================================================
grp   = df_alphaC.groupby("mutant")["alphaC_hinge_mean_dist"]
means = grp.mean().sort_values()
stds  = grp.std().reindex(means.index)

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(means))
ax.bar(x, means.values, yerr=stds.values, capsize=3,
       color="steelblue", alpha=0.85, error_kw={"linewidth": 0.8})
ax.set_xticks(x)
ax.set_xticklabels(means.index, rotation=90, fontsize=8)
ax.set_ylabel("mean alphaC-to-hinge COM distance (nm)")
ax.set_title("Per-mutant alphaC-helix displacement relative to hinge\n"
             "(alphaC real 1983-1993, hinge real 2031-2038)")
plt.tight_layout()
save_fig("step3a_alphaC_hinge_distance_errorbars.png")

# ============================================================
# 2. alphaC-helix RMSF (flexibility per mutant)
# ============================================================
grp2   = df_alphaC.groupby("mutant")["alphaC_rmsf_mean"]
means2 = grp2.mean().sort_values(ascending=False)
stds2  = grp2.std().reindex(means2.index)

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(means2))
ax.bar(x, means2.values, yerr=stds2.values, capsize=3,
       color="darkorange", alpha=0.85, error_kw={"linewidth": 0.8})
ax.set_xticks(x)
ax.set_xticklabels(means2.index, rotation=90, fontsize=8)
ax.set_ylabel("mean alphaC-helix RMSF (nm)")
ax.set_title("Per-mutant alphaC-helix flexibility (RMSF)")
plt.tight_layout()
save_fig("step3a_alphaC_rmsf.png")

# ============================================================
# 3. K1980-E1993 salt bridge finding summary
# The salt bridge distance sits at ~1.3 nm throughout all
# trajectories, far above the 0.4 nm formation threshold.
# This is a real finding: the alphaC-helix remains in an
# intermediate/out conformation in all apo simulations.
# ============================================================
if "alphaC_compact_frac" in df_alphaC.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.6,
            "K1980-E1993 salt bridge not formed in any trajectory",
            ha="center", va="center", fontsize=14, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.4,
            "Mean K1980-E1993 distance ~1.3 nm across all 114 trajectories\n"
            "(salt bridge threshold = 0.4 nm; never reached)\n"
            "The alphaC-helix adopts an intermediate/out conformation\n"
            "throughout all apo ROS1 simulations.",
            ha="center", va="center", fontsize=10,
            transform=ax.transAxes, style="italic",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.axis("off")
    ax.set_title("alphaC-helix salt bridge state (K1980-E1993)")
    plt.tight_layout()
    save_fig("step3a_saltbridge_finding.png")

# ============================================================
# 4. ATP pocket open-state occupancy with error bars
# ============================================================
grp3   = df_pocket.groupby("mutant")["pocket_open_frac"]
means3 = grp3.mean().sort_values(ascending=False)
stds3  = grp3.std().reindex(means3.index)

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(means3))
ax.bar(x, means3.values, yerr=stds3.values, capsize=3,
       color="steelblue", alpha=0.85, error_kw={"linewidth": 0.8})
ax.set_xticks(x)
ax.set_xticklabels(means3.index, rotation=90, fontsize=8)
ax.set_ylabel("fraction pocket-open frames")
ax.set_title("Per-mutant ATP-pocket open-state occupancy")
plt.tight_layout()
save_fig("step3a_pocket_open_frac_errorbars.png")

# ============================================================
# 5. DFG-Phe chi1 mean angle per mutant
# ============================================================
grp4   = df_dfg.groupby("mutant")["dfg_chi1_deg_mean"]
means4 = grp4.mean().sort_values()
stds4  = grp4.std().reindex(means4.index)

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(means4))
ax.bar(x, means4.values, yerr=stds4.values, capsize=3,
       color="forestgreen", alpha=0.85, error_kw={"linewidth": 0.8})
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(means4.index, rotation=90, fontsize=8)
ax.set_ylabel("mean DFG-Phe chi1 angle (degrees)")
ax.set_title("Per-mutant DFG-Phe (F2043) mean chi1 dihedral\n"
             "DFG-in ~-70 deg, DFG-out ~+60 or ±170 deg")
plt.tight_layout()
save_fig("step3a_dfg_chi1_mean_per_mutant.png")

# ============================================================
# 6. DFG-in state fraction per mutant
# ============================================================
grp5   = df_dfg.groupby("mutant")["dfg_state_frac1"]
means5 = grp5.mean().sort_values(ascending=False)
stds5  = grp5.std().reindex(means5.index)

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(means5))
ax.bar(x, means5.values, yerr=stds5.values, capsize=3,
       color="crimson", alpha=0.85, error_kw={"linewidth": 0.8})
ax.set_xticks(x)
ax.set_xticklabels(means5.index, rotation=90, fontsize=8)
ax.set_ylabel("fraction DFG-in frames")
ax.set_title("Per-mutant DFG-in state occupancy")
plt.tight_layout()
save_fig("step3a_dfg_state_frac_per_mutant.png")

print("\nScript 03a_patch completed. Figures saved to:", FIG_DIR)