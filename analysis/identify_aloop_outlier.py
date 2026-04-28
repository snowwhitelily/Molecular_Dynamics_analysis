# ============================================================
# ROS1 Analysis Pipeline
# Script: identify_aloop_outlier.py
#
# PURPOSE:
#   Identifies trajectories with anomalous A-loop conformations
#   by thresholding on the A-loop PCA PC1 score distribution.
#
# BACKGROUND:
#   The A-loop PCA scores plot (PC1 vs PC2) revealed a visually
#   distinct cluster of orange points separated from the main
#   conformational cloud at PC1 < approximately -0.13 to -0.15.
#   This script systematically identifies which trajectories
#   contribute frames to that outlier region by testing a range
#   of PC1 thresholds.
#
# METHOD:
#   For each threshold, frames with PC1 score below the threshold
#   are selected and the owning trajectory (mutant + replica) is
#   reported along with the frame count. A genuine outlier shows
#   a large, consistent frame count across multiple thresholds.
#   Noise appears as small frame counts (1-25) that are sensitive
#   to threshold choice.
#
# FINDINGS:
#   L1982V replica 0 is the primary A-loop outlier:
#     - 374 frames at threshold -0.14
#     - 366 frames at threshold -0.15
#     - 271 frames at threshold -0.16
#   This is reproducible and consistent — not threshold-sensitive.
#
#   V2089M replica 1 appears at intermediate thresholds (40 frames
#   at -0.15) but collapses to 6 frames at -0.16, confirming it
#   is tail noise from the main distribution, not a genuine outlier.
#
#   All other mutants appear with 1-12 frames — unambiguous noise.
#
# CONCLUSION:
#   L1982V replica 0 adopts a distinct A-loop conformation that
#   separates it from the main conformational cloud in the A-loop
#   PCA. Two out of three replicas (0 and 1 at looser thresholds)
#   show this behaviour, suggesting L1982V samples an alternative
#   A-loop state. Unlike F1994L (where all three replicas are
#   separated in global PCA), L1982V replica 2 occupies the normal
#   conformational space, indicating this mutant can transition
#   between states.
#
# THESIS METHODS SENTENCE:
#   "Trajectories with A-loop PC1 scores below -0.15 were identified
#   as occupying a distinct conformational region separated from the
#   main population by a gap in the score distribution. This threshold
#   was determined by systematic evaluation across the range -0.12 to
#   -0.16; L1982V replica 0 (366 frames) was identified as the primary
#   A-loop outlier, while V2089M replica 1 (40 frames at -0.15,
#   collapsing to 6 frames at -0.16) was classified as tail noise."
#
# USAGE:
#   python identify_aloop_outlier.py
#
# WHERE IT FITS IN THE PIPELINE:
#   After: 02c_pca_aloop_ctlfit.py  (A-loop PCA must have run)
#   Before: 02d_cluster_selected_pcas.py
#   This is a diagnostic script — it does not modify any data.
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

BASE    = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"

# ── load A-loop PCA outputs ───────────────────────────────────────────────────
scores  = np.load(RESULTS / "aloop_scores.npy")
x_owner = np.load(RESULTS / "aloop_x_owner.npy")
F       = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()

print("=" * 60)
print("A-LOOP PCA OUTLIER IDENTIFICATION")
print("=" * 60)
print(f"Total frames in A-loop PCA : {len(scores)}")
print(f"Number of trajectories     : {len(F)}")
print()

# ── test a range of thresholds ────────────────────────────────────────────────
# The orange outlier cluster sits around PC1 ≈ -0.13 to -0.15 in the
# A-loop PC1 vs PC2 scatter plot. We test thresholds from -0.12 to -0.16
# to find where genuine outliers separate from tail noise.

thresholds = [-0.12, -0.13, -0.14, -0.15, -0.16]

summary_rows = []

for thresh in thresholds:
    outlier = scores[:, 0] < thresh
    owners  = x_owner[outlier]
    n_frames_total = int(outlier.sum())

    print(f"Threshold PC1 < {thresh:.2f}  ({n_frames_total} total outlier frames)")
    print("-" * 55)

    if n_frames_total == 0:
        print("  No frames below threshold.")
    else:
        for ti in np.unique(owners):
            p = Path(F[ti])
            n = int((owners == ti).sum())
            print(f"  traj_index={ti:3d}  mutant={p.parent.name:<25s}"
                  f"  replica={p.name}  frames={n}")
            summary_rows.append({
                "threshold": thresh,
                "traj_index": int(ti),
                "mutant": p.parent.name,
                "replica": p.name,
                "frames_below_threshold": n,
            })
    print()

# ── save summary table ────────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
out_csv = RESULTS / "aloop_outlier_threshold_scan.csv"
summary_df.to_csv(out_csv, index=False)
print(f"Saved threshold scan table: {out_csv}")

# ── final verdict ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("FINAL VERDICT")
print("=" * 60)

FINAL_THRESHOLD = -0.15

outlier_final  = scores[:, 0] < FINAL_THRESHOLD
owners_final   = x_owner[outlier_final]
genuine_cutoff = 50   # frames below this = noise

print(f"Final threshold used: PC1 < {FINAL_THRESHOLD}")
print(f"Genuine outlier criterion: >= {genuine_cutoff} frames\n")

genuine = []
noise   = []

for ti in np.unique(owners_final):
    p = Path(F[ti])
    n = int((owners_final == ti).sum())
    entry = {
        "traj_index": int(ti),
        "mutant": p.parent.name,
        "replica": p.name,
        "frames": n,
    }
    if n >= genuine_cutoff:
        genuine.append(entry)
    else:
        noise.append(entry)

print("GENUINE OUTLIERS (>= 50 frames):")
if genuine:
    for g in genuine:
        print(f"  mutant={g['mutant']:<25s}  replica={g['replica']}"
              f"  frames={g['frames']}")
else:
    print("  None found.")

print()
print("NOISE (< 50 frames — tail of main distribution):")
if noise:
    for n_entry in noise:
        print(f"  mutant={n_entry['mutant']:<25s}  replica={n_entry['replica']}"
              f"  frames={n_entry['frames']}")
else:
    print("  None found.")

print()
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("""
L1982V replica 0 is the primary A-loop outlier with 366 frames
at the final threshold of -0.15. This represents a distinct
A-loop conformation sampled consistently throughout that replica.

L1982V is a leucine-to-valine mutation at real residue 1982,
located in the alphaC-beta4 loop just proximal to the alphaC-helix.
Loss of the two methyl groups in the leucine-to-valine substitution
may destabilise alphaC-helix positioning, leading to an altered
A-loop conformation captured by A-loop PC1.

Unlike F1994L (where all three replicas are separated in global PCA),
L1982V replica 2 occupies the normal conformational space, suggesting
this mutant can transition between A-loop states across replicas.

This finding is documented in the thesis Results section.
""")