# save as pymol_pca_visualization.py
# Run with: /usr/bin/pymol -c pymol_pca_visualization.py
#
# This script:
# 1. Loads WT and F1994L multi-state PDB files into PyMOL
# 2. Runs PCA using Tsjerk's Princomp class on backbone atoms
# 3. Draws PC1 and PC2 as violin visualizations on the structure
# 4. Colours by score distribution using BWR colorinator
# 5. Saves publication figures showing which residues drive each PC

import sys
import os
import numpy as np

# add scripts folder to path so PyMOL can find princomp and colorinator
sys.path.insert(0, '/homes/lkgyammerah/Molecular_Dynamics_analysis/analysis/scripts')

from pymol import cmd
from colorinator import BWR, BOX
from princomp import Princomp

BASE     = '/homes/lkgyammerah/Molecular_Dynamics_analysis'
FRAMES   = BASE + '/results/ROS1/pymol_frames'
FIG_DIR  = BASE + '/figures/ROS1/ros1_prepared_final/pymol_pca'

os.makedirs(FIG_DIR, exist_ok=True)

# ── LOAD STRUCTURES ───────────────────────────────────────────────────────────

print("Loading WT frames...")
cmd.load(FRAMES + '/WT_rep0_stride50.pdb', 'WT')

print("Loading F1994L frames...")
cmd.load(FRAMES + '/F1994L_rep0_stride50.pdb', 'F1994L')

print(f"WT states:     {cmd.count_states('WT')}")
print(f"F1994L states: {cmd.count_states('F1994L')}")

# ── ALIGN ALL FRAMES TO FIRST WT FRAME ───────────────────────────────────────
# align on NTL backbone (internal residues 5-97)
# in the PDB files residue numbers are 1-292 (internal numbering)

print("Aligning all frames on NTL backbone (resi 5-97)...")
cmd.intra_fit('WT and resi 5-97 and backbone')
cmd.align('F1994L and resi 5-97 and backbone',
          'WT and resi 5-97 and backbone', cycles=0)

# ── SET UP NICE DISPLAY ───────────────────────────────────────────────────────
cmd.show('cartoon', 'all')
cmd.hide('lines', 'all')
cmd.bg_color('white')
cmd.set('cartoon_fancy_helices', 1)
cmd.set('ray_shadows', 0)

# colour WT blue, F1994L red for context
cmd.color('slate', 'WT')
cmd.color('salmon', 'F1994L')

# ── RUN PCA ON WT ONLY FIRST ─────────────────────────────────────────────────
# This shows the natural WT conformational modes
print("\nRunning PCA on WT backbone...")
P_wt = Princomp('WT and backbone', ncomponents=5)

print(f"WT PCA complete:")
print(f"  n_frames : {len(P_wt.data.objects)}")
print(f"  variance PC1: {P_wt.variances[0]:.4f}")
print(f"  variance PC2: {P_wt.variances[1]:.4f}")
print(f"  variance PC3: {P_wt.variances[2]:.4f}")

# draw mean structure
P_wt.drawmean('WT_mean')
cmd.show('cartoon', 'WT_mean')
cmd.color('grey80', 'WT_mean')

# ── VISUALIZE PC1 ─────────────────────────────────────────────────────────────
print("\nDrawing PC1 violin on WT mean structure...")
pc1_wt = P_wt[1]

# violin plot — cylinder radius proportional to density of scores
# colours go from BOX colormap (cyan-green-yellow-red-magenta)
violin_pc1 = pc1_wt.violin(bins=30, bw=8, radius=0.4)
violin_pc1.draw('WT_PC1_violin')

# save PC1 violin figure
cmd.orient('WT_mean')
cmd.zoom('WT_mean', 5)
cmd.png(FIG_DIR + '/wt_pc1_violin_front.png',
        width=1400, height=1100, dpi=300, ray=1)

# rotate for side view
cmd.rotate('y', 90)
cmd.png(FIG_DIR + '/wt_pc1_violin_side.png',
        width=1400, height=1100, dpi=300, ray=1)

cmd.rotate('y', -90)

# ── VISUALIZE PC2 ─────────────────────────────────────────────────────────────
print("Drawing PC2 violin on WT mean structure...")
pc2_wt = P_wt[2]

violin_pc2 = pc2_wt.violin(bins=30, bw=8, radius=0.4)
violin_pc2.draw('WT_PC2_violin')

# hide PC1 violin, show PC2
cmd.disable('WT_PC1_violin')
cmd.enable('WT_PC2_violin')

cmd.orient('WT_mean')
cmd.zoom('WT_mean', 5)
cmd.png(FIG_DIR + '/wt_pc2_violin_front.png',
        width=1400, height=1100, dpi=300, ray=1)

# ── RUN PCA ON BOTH WT AND F1994L COMBINED ───────────────────────────────────
# This shows how F1994L separates from WT in the combined PCA space
print("\nRunning PCA on WT + F1994L combined backbone...")
P_combined = Princomp('(WT or F1994L) and backbone', ncomponents=5)

print(f"Combined PCA complete:")
print(f"  n_frames total: {len(P_combined.data.objects)}")
print(f"  variance PC1: {P_combined.variances[0]:.4f}")
print(f"  variance PC2: {P_combined.variances[1]:.4f}")

# draw mean of combined
P_combined.drawmean('combined_mean')
cmd.show('cartoon', 'combined_mean')
cmd.color('grey70', 'combined_mean')

# PC1 of combined — should show the inter-lobe rotation
pc1_combined = P_combined[1]

print(f"\nPC1 scores summary:")
print(f"  min:  {pc1_combined.scores.min():.4f}")
print(f"  max:  {pc1_combined.scores.max():.4f}")
print(f"  mean: {pc1_combined.scores.mean():.4f}")

# split scores by object to see WT vs F1994L separation
wt_frames     = P_combined.data.objects == 'WT'
f1994l_frames = P_combined.data.objects == 'F1994L'

print(f"\nWT PC1 scores:     mean={pc1_combined.scores[wt_frames].mean():.4f}  "
      f"std={pc1_combined.scores[wt_frames].std():.4f}")
print(f"F1994L PC1 scores: mean={pc1_combined.scores[f1994l_frames].mean():.4f}  "
      f"std={pc1_combined.scores[f1994l_frames].std():.4f}")

# violin of combined PC1
cmd.disable('WT_PC1_violin')
cmd.disable('WT_PC2_violin')

violin_combined = pc1_combined.violin(bins=30, bw=8, radius=0.4)
violin_combined.draw('combined_PC1_violin')

cmd.orient('combined_mean')
cmd.zoom('combined_mean', 5)
cmd.png(FIG_DIR + '/combined_pc1_violin_front.png',
        width=1400, height=1100, dpi=300, ray=1)

cmd.rotate('y', 90)
cmd.png(FIG_DIR + '/combined_pc1_violin_side.png',
        width=1400, height=1100, dpi=300, ray=1)

# ── SAVE SESSION ─────────────────────────────────────────────────────────────
cmd.save(FIG_DIR + '/pca_visualization.pse')
print(f"\nPyMOL session saved: {FIG_DIR}/pca_visualization.pse")
print("You can open this session in PyMOL locally to interact with it.")

print("\nAll figures saved to:", FIG_DIR)
print("Done.")











# # save as extract_frames_for_pymol.py
# # Extracts every 50th frame from WT and F1994L trajectories
# # and saves as multi-state PDB files that PyMOL can load

# import MDAnalysis as mda
# import numpy as np
# from pathlib import Path

# BASE    = Path.home() / "Molecular_Dynamics_analysis"
# TRAJ    = BASE / "trajectories/ros1_prepared_final"
# OUT_DIR = BASE / "results/ROS1/pymol_frames"
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# STRIDE = 50   # every 50th frame = ~200 frames per trajectory

# systems = [
#     ("WT",     "0"),
#     ("F1994L", "0"),
# ]

# for mutant, replica in systems:
#     pdb = TRAJ / mutant / replica / f"{mutant}-MD-prot.pdb"
#     xtc = TRAJ / mutant / replica / f"{mutant}-MD-prot.xtc"

#     print(f"Loading {mutant} replica {replica}...")
#     u  = mda.Universe(str(pdb), str(xtc))
#     ag = u.select_atoms("backbone")

#     out_pdb = OUT_DIR / f"{mutant}_rep{replica}_stride{STRIDE}.pdb"

#     print(f"  Total frames: {len(u.trajectory)}")
#     print(f"  Writing every {STRIDE}th frame to {out_pdb.name}...")

#     with mda.Writer(str(out_pdb), ag.n_atoms, multiframe=True) as W:
#         for i, ts in enumerate(u.trajectory):
#             if i % STRIDE == 0:
#                 W.write(ag)

#     n_written = len(range(0, len(u.trajectory), STRIDE))
#     print(f"  Written {n_written} frames")

# print("\nDone. Files saved to:", OUT_DIR)














# # save as visualize_f1994l_in_pca.py
# # Highlights F1994L frames in PCA scatter plots across all PCA spaces
# # Shows clearly that F1994L is an outlier in global spaces but normal in A-loop

# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from pathlib import Path

# BASE    = Path.home() / "Molecular_Dynamics_analysis"
# RESULTS = BASE / "results" / "ROS1"
# FIG_DIR = BASE / "figures" / "ROS1" / "ros1_prepared_final" / "f1994l_outlier_highlight"
# FIG_DIR.mkdir(parents=True, exist_ok=True)

# F       = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()
# f1994l_indices = [i for i, f in enumerate(F) if "F1994L" in f]
# print(f"F1994L trajectory indices: {f1994l_indices}")
# print(f"F1994L replicas: {[Path(F[i]).name for i in f1994l_indices]}")

# # colour scheme
# COL_NORMAL  = "#AAAAAA"   # grey for all other mutants
# COL_F1994L  = "#E63946"   # vivid red for F1994L
# COL_WT      = "#2196F3"   # blue for WT
# ALPHA_NORMAL = 0.15
# ALPHA_F1994L = 0.80
# ALPHA_WT     = 0.40
# SIZE_NORMAL  = 1
# SIZE_F1994L  = 6
# SIZE_WT      = 3

# # find WT indices for reference
# wt_indices = [i for i, f in enumerate(F) if Path(f).parent.name == "WT"]
# print(f"WT trajectory indices: {wt_indices}")

# def plot_pca_highlight(prefix, pc_pairs, title_prefix):
#     """
#     Load PCA scores for a given prefix and plot with F1994L highlighted.
#     """
#     scores_f  = RESULTS / f"{prefix}_scores.npy"
#     x_owner_f = RESULTS / f"{prefix}_x_owner.npy"

#     if not scores_f.exists():
#         print(f"Skipping {prefix} — scores not found")
#         return

#     scores  = np.load(scores_f)
#     x_owner = np.load(x_owner_f)

#     for pc_a, pc_b in pc_pairs:
#         fig, ax = plt.subplots(figsize=(8, 7))

#         # layer 1 — all normal mutants in grey
#         mask_normal = np.ones(len(x_owner), dtype=bool)
#         mask_normal[np.isin(x_owner, f1994l_indices)] = False
#         mask_normal[np.isin(x_owner, wt_indices)]     = False

#         ax.scatter(
#             scores[mask_normal, pc_a],
#             scores[mask_normal, pc_b],
#             s=SIZE_NORMAL, c=COL_NORMAL,
#             alpha=ALPHA_NORMAL, linewidths=0,
#             label=f"Other mutants (n={mask_normal.sum()})",
#             zorder=1
#         )

#         # layer 2 — WT in blue
#         mask_wt = np.isin(x_owner, wt_indices)
#         ax.scatter(
#             scores[mask_wt, pc_a],
#             scores[mask_wt, pc_b],
#             s=SIZE_WT, c=COL_WT,
#             alpha=ALPHA_WT, linewidths=0,
#             label=f"WT (n={mask_wt.sum()})",
#             zorder=2
#         )

#         # layer 3 — F1994L in red on top
#         mask_f1994l = np.isin(x_owner, f1994l_indices)
#         ax.scatter(
#             scores[mask_f1994l, pc_a],
#             scores[mask_f1994l, pc_b],
#             s=SIZE_F1994L, c=COL_F1994L,
#             alpha=ALPHA_F1994L, linewidths=0,
#             label=f"F1994L (n={mask_f1994l.sum()})",
#             zorder=3
#         )

#         ax.set_xlabel(f"PC{pc_a + 1}", fontsize=13)
#         ax.set_ylabel(f"PC{pc_b + 1}", fontsize=13)
#         ax.set_title(
#             f"{title_prefix} — PC{pc_a+1} vs PC{pc_b+1}\n"
#             f"F1994L highlighted (red), WT (blue), all others (grey)",
#             fontsize=11
#         )
#         ax.legend(fontsize=9, frameon=True, markerscale=4)
#         ax.set_aspect("equal", adjustable="box")

#         fname = f"{prefix}_pc{pc_a+1}_pc{pc_b+1}_f1994l_highlight.png"
#         out   = FIG_DIR / fname
#         plt.savefig(out, dpi=300, bbox_inches="tight")
#         plt.close()
#         print(f"Saved: {out}")


# # PCA spaces and PC pairs to plot
# pca_spaces = [
#     ("actin",         "ACT-IN PCA",         [(0,1), (0,2), (1,2)]),
#     ("actout",        "ACT-OUT PCA",         [(0,1), (0,2)]),
#     ("actout_ctlfit", "ACT-OUT CTL-fit PCA", [(0,1), (0,2), (1,2)]),
#     ("ntl",           "NTL PCA",             [(0,1), (0,2)]),
#     ("ctl",           "CTL PCA",             [(0,1), (0,2)]),
#     ("aloop",         "A-loop PCA",          [(0,1), (0,2)]),
# ]

# for prefix, title, pairs in pca_spaces:
#     print(f"\nProcessing {prefix}...")
#     plot_pca_highlight(prefix, pairs, title)

# print("\nAll done. Figures saved to:", FIG_DIR)





















# # save as check_starting_conformation3.py
# import numpy as np
# from pathlib import Path

# STRUCTURES = Path.home() / "ROS1_MD/00_structures"
# wt_pdb     = STRUCTURES / "ROS1_kinase_1934_2225_WT.pdb"
# f1994l_pdb = STRUCTURES / "ROS1_kinase_1934_2225_F1994L.pdb"

# # conversion helpers
# def real_to_internal(real): return real - 1933
# def internal_to_real(internal): return internal + 1933

# def load_ca(pdb_path):
#     coords  = {}
#     resname = {}
#     with open(pdb_path) as f:
#         for line in f:
#             if len(line) < 54:
#                 continue
#             if not line.startswith("ATOM"):
#                 continue
#             aname = line[12:16].strip()
#             if aname != "CA":
#                 continue
#             try:
#                 resid = int(line[22:26].strip())
#                 x     = float(line[30:38].strip())
#                 y     = float(line[38:46].strip())
#                 z     = float(line[46:54].strip())
#             except ValueError:
#                 continue
#             rname = line[17:20].strip()
#             coords[resid]  = np.array([x, y, z])
#             resname[resid] = rname
#     return coords, resname

# def kabsch(P, Q):
#     Pc = P - P.mean(0)
#     Qc = Q - Q.mean(0)
#     H  = Pc.T @ Qc
#     U, S, Vt = np.linalg.svd(H)
#     d  = np.linalg.det(Vt.T @ U.T)
#     R  = Vt.T @ np.diag([1, 1, d]) @ U.T
#     return Pc @ R.T, Qc

# print("=" * 65)
# print("STARTING CONFORMATION COMPARISON: WT vs F1994L")
# print("(PDB uses internal numbering 1-292 = real 1934-2225)")
# print("=" * 65)

# wt_ca,     wt_rn     = load_ca(wt_pdb)
# f1994l_ca, f1994l_rn = load_ca(f1994l_pdb)

# print(f"\nWT atoms loaded:     {len(wt_ca)} CA atoms")
# print(f"F1994L atoms loaded: {len(f1994l_ca)} CA atoms")

# # print sequence around key regions using INTERNAL numbering
# regions_to_print = {
#     "Mutation site F1994L (internal 55-68 = real 1988-2001)": range(55, 69),
#     "αC-helix region (internal 50-67 = real 1983-2000)":      range(50, 68),
#     "DFG motif (internal 112-117 = real 2045-2050)":          range(112, 118),
#     "β3 strand region (internal 44-52 = real 1977-1985)":     range(44, 53),
# }

# for region_name, res_range in regions_to_print.items():
#     print(f"\n{region_name}:")
#     print(f"  {'Internal':>10}  {'Real':>8}  {'WT':>6}  {'F1994L':>8}")
#     for r in res_range:
#         wt_aa = wt_rn.get(r, "---")
#         f_aa  = f1994l_rn.get(r, "---")
#         flag  = " ← MUTATION" if wt_aa != f_aa else ""
#         print(f"  {r:>10}  {internal_to_real(r):>8}  {wt_aa:>6}  {f_aa:>8}{flag}")

# # overall RMSD
# common = sorted(set(wt_ca) & set(f1994l_ca))
# P = np.array([f1994l_ca[r] for r in common])
# Q = np.array([wt_ca[r]     for r in common])
# P_al, Q_c = kabsch(P, Q)
# rmsd_all = float(np.sqrt(((P_al - Q_c)**2).sum(1).mean()))
# print(f"\nOverall CA RMSD (WT vs F1994L, after alignment): {rmsd_all:.3f} Å")

# # per-region RMSD using INTERNAL numbering
# regions = {
#     "Full NTL (int 5-97 = real 1938-2030)":        (5,   97),
#     "Full CTL (int 120-283 = real 2053-2216)":      (120, 283),
#     "Activation loop (int 112-137 = real 2045-2070)":(112, 137),
#     "αC-helix region (int 50-67 = real 1983-2000)": (50,  67),
#     "β3-αC loop (int 55-67 = real 1988-2000)":      (55,  67),
#     "Hinge (int 98-119 = real 2031-2052)":          (98,  119),
# }

# common_idx = {r: i for i, r in enumerate(common)}

# print(f"\nPer-region CA RMSD (after global alignment):")
# print(f"  {'Region':<45}  {'n_res':>6}  {'RMSD (Å)':>10}")
# print("  " + "-" * 66)

# for rname, (rstart, rend) in regions.items():
#     res_in_region = [r for r in common if rstart <= r <= rend]
#     if len(res_in_region) < 3:
#         print(f"  {rname:<45}  {'<3':>6}  {'n/a':>10}")
#         continue
#     idx    = [common_idx[r] for r in res_in_region]
#     rmsd_r = float(np.sqrt(((P_al[idx] - Q_c[idx])**2).sum(1).mean()))
#     print(f"  {rname:<45}  {len(res_in_region):>6}  {rmsd_r:>10.3f}")

# # key distances — αC-helix salt bridge
# # conserved β3 Lys in ROS1 is typically around internal 47-50 (real 1980-1983)
# # conserved αC Glu is typically around internal 60-63 (real 1993-1996)
# print(f"\nKey distances (internal numbering):")
# print(f"  {'Pair':<45}  {'WT (Å)':>8}  {'F1994L (Å)':>12}  {'Diff':>8}")
# print("  " + "-" * 78)

# # find K and E around αC-helix region
# print("\n  Residues in β3 and αC-helix region (WT):")
# for r in range(44, 70):
#     aa = wt_rn.get(r, "")
#     if aa in ("LYS", "GLU", "PHE", "LEU"):
#         print(f"    internal {r} (real {internal_to_real(r)}) = {aa}")

# # DFG distances
# dfg = 112  # internal = real 2045
# print(f"\n  D{internal_to_real(dfg)} (internal {dfg}) CA distances to NTL reference residues:")
# print(f"  {'Internal':>10}  {'Real':>8}  {'AA (WT)':>10}"
#       f"  {'WT (Å)':>8}  {'F1994L (Å)':>12}  {'Diff':>8}")
# print("  " + "-" * 62)

# for ref in [44, 47, 50, 55, 60, 65, 70, 80, 90]:
#     if ref in wt_ca and ref in f1994l_ca and dfg in wt_ca and dfg in f1994l_ca:
#         d_wt = float(np.linalg.norm(wt_ca[dfg]     - wt_ca[ref]))
#         d_f  = float(np.linalg.norm(f1994l_ca[dfg] - f1994l_ca[ref]))
#         flag = " ← LARGE" if abs(d_f - d_wt) > 3.0 else ""
#         print(f"  {ref:>10}  {internal_to_real(ref):>8}  "
#               f"{wt_rn.get(ref,'?'):>10}  "
#               f"{d_wt:>8.2f}  {d_f:>12.2f}  {d_f-d_wt:>+8.2f}{flag}")

# print()
# print("=" * 65)
# print("SUMMARY")
# print("=" * 65)
# print(f"""
# Overall RMSD = {rmsd_all:.3f} Å — LARGE conformational difference.

# AlphaFold predicted a DIFFERENT conformation for F1994L vs WT.
# The MD simulations confirmed and maintained this difference
# throughout all three replicas (first 10% PC1 ≈ 0.32-0.35,
# last 10% PC1 ≈ 0.31-0.37 — no drift).

# Key question: is the F1994L starting structure active or inactive?
# Check the per-region RMSD and αC-helix residue list above.
# If NTL RMSD >> CTL RMSD → inter-lobe displacement
# If αC-helix RMSD is large → αC-helix displaced (inactive marker)
# If activation loop RMSD large → A-loop in different position
# """)













# # save as check_starting_conformation2.py
# import numpy as np
# from pathlib import Path

# STRUCTURES = Path.home() / "ROS1_MD/00_structures"
# wt_pdb     = STRUCTURES / "ROS1_kinase_1934_2225_WT.pdb"
# f1994l_pdb = STRUCTURES / "ROS1_kinase_1934_2225_F1994L.pdb"

# def load_ca(pdb_path):
#     coords  = {}
#     resname = {}
#     with open(pdb_path) as f:
#         for line in f:
#             if len(line) < 54:
#                 continue
#             rec = line[0:6].strip()
#             if rec not in ("ATOM", "HETATM"):
#                 continue
#             aname = line[12:16].strip()
#             if aname != "CA":
#                 continue
#             rname = line[17:20].strip()
#             try:
#                 resid = int(line[22:26].strip())
#                 x     = float(line[30:38].strip())
#                 y     = float(line[38:46].strip())
#                 z     = float(line[46:54].strip())
#             except ValueError:
#                 continue
#             coords[resid]  = np.array([x, y, z])
#             resname[resid] = rname
#     return coords, resname

# def load_atom(pdb_path, resid, aname):
#     with open(pdb_path) as f:
#         for line in f:
#             if len(line) < 54:
#                 continue
#             if line[0:4] != "ATOM":
#                 continue
#             if line[12:16].strip() != aname:
#                 continue
#             try:
#                 rid = int(line[22:26].strip())
#             except ValueError:
#                 continue
#             if rid != resid:
#                 continue
#             x = float(line[30:38].strip())
#             y = float(line[38:46].strip())
#             z = float(line[46:54].strip())
#             return np.array([x, y, z])
#     return None

# def kabsch(P, Q):
#     """Align P onto Q. Returns aligned P."""
#     Pc = P - P.mean(0)
#     Qc = Q - Q.mean(0)
#     H  = Pc.T @ Qc
#     U, S, Vt = np.linalg.svd(H)
#     d  = np.linalg.det(Vt.T @ U.T)
#     R  = Vt.T @ np.diag([1,1,d]) @ U.T
#     return Pc @ R.T, Qc

# print("=" * 65)
# print("STARTING CONFORMATION COMPARISON: WT vs F1994L")
# print("=" * 65)

# wt_ca,     wt_rn     = load_ca(wt_pdb)
# f1994l_ca, f1994l_rn = load_ca(f1994l_pdb)

# # print sequence around mutation site and key functional residues
# print("\nSequence around F1994L mutation site (real 1988-2002):")
# print(f"  {'Resid':>8}  {'WT':>6}  {'F1994L':>8}")
# for r in range(1988, 2003):
#     wt_aa = wt_rn.get(r, "---")
#     f_aa  = f1994l_rn.get(r, "---")
#     flag  = " ← MUTATION" if wt_aa != f_aa else ""
#     print(f"  {r:>8}  {wt_aa:>6}  {f_aa:>8}{flag}")

# print("\nSequence around αC-helix region (real 1983-1998):")
# print(f"  {'Resid':>8}  {'WT':>6}  {'F1994L':>8}")
# for r in range(1983, 1999):
#     wt_aa = wt_rn.get(r, "---")
#     f_aa  = f1994l_rn.get(r, "---")
#     print(f"  {r:>8}  {wt_aa:>6}  {f_aa:>8}")

# print("\nDFG motif (real 2045-2050):")
# print(f"  {'Resid':>8}  {'WT':>6}  {'F1994L':>8}")
# for r in range(2045, 2051):
#     wt_aa = wt_rn.get(r, "---")
#     f_aa  = f1994l_rn.get(r, "---")
#     print(f"  {r:>8}  {wt_aa:>6}  {f_aa:>8}")

# # overall RMSD
# common = sorted(set(wt_ca) & set(f1994l_ca))
# P = np.array([f1994l_ca[r] for r in common])
# Q = np.array([wt_ca[r]     for r in common])
# P_al, Q_c = kabsch(P, Q)
# rmsd_all = float(np.sqrt(((P_al - Q_c)**2).sum(1).mean()))
# print(f"\nOverall CA RMSD (WT vs F1994L, after alignment): {rmsd_all:.3f} Å")

# # per-region RMSD
# regions = {
#     "Full NTL (1938-2030)":    (1938, 2030),
#     "Full CTL (2053-2216)":    (2053, 2216),
#     "Activation loop (2045-2070)": (2045, 2070),
#     "αC-helix region (1983-2000)": (1983, 2000),
#     "β3-αC loop (1988-2000)":  (1988, 2000),
#     "Hinge region (2031-2044)":(2031, 2044),
# }

# print(f"\nPer-region CA RMSD (after global alignment):")
# print(f"  {'Region':<35}  {'n_res':>6}  {'RMSD (Å)':>10}")
# print("  " + "-" * 56)

# # build index map
# common_idx = {r: i for i, r in enumerate(common)}

# for rname, (rstart, rend) in regions.items():
#     res_in_region = [r for r in common if rstart <= r <= rend]
#     if len(res_in_region) < 3:
#         print(f"  {rname:<35}  {'<3 res':>6}  {'n/a':>10}")
#         continue
#     idx = [common_idx[r] for r in res_in_region]
#     rmsd_r = float(np.sqrt(((P_al[idx] - Q_c[idx])**2).sum(1).mean()))
#     print(f"  {rname:<35}  {len(res_in_region):>6}  {rmsd_r:>10.3f}")

# # key distances
# print("\nKey inter-residue CA distances (active state markers):")
# print(f"  {'Measurement':<40}  {'WT (Å)':>8}  {'F1994L (Å)':>12}  {'Diff':>8}")
# print("  " + "-" * 74)

# # look for conserved β3 lysine (K) and αC glutamate (E)
# # find them from sequence
# print("\n  Looking for β3 Lys and αC Glu in WT sequence:")
# for r in range(1975, 2000):
#     aa = wt_rn.get(r, "")
#     if aa in ("LYS", "GLU"):
#         print(f"    resid {r} = {aa}")

# # DFG aspartate to known reference points
# dfg_d = 2045
# refs  = [1960, 1970, 1975, 1980, 2000, 2010]
# print(f"\n  D{dfg_d} CA distance to reference residues:")
# print(f"  {'Ref resid':>12}  {'Ref AA':>8}  {'WT (Å)':>10}  {'F1994L (Å)':>12}  {'Diff':>8}")
# print("  " + "-" * 58)
# for ref in refs:
#     if ref in wt_ca and ref in f1994l_ca and dfg_d in wt_ca and dfg_d in f1994l_ca:
#         d_wt = float(np.linalg.norm(wt_ca[dfg_d]     - wt_ca[ref]))
#         d_f  = float(np.linalg.norm(f1994l_ca[dfg_d] - f1994l_ca[ref]))
#         flag = " ← LARGE" if abs(d_f - d_wt) > 3.0 else ""
#         print(f"  {ref:>12}  {wt_rn.get(ref,'?'):>8}  "
#               f"{d_wt:>10.2f}  {d_f:>12.2f}  {d_f-d_wt:>+8.2f}{flag}")

# print()
# print("=" * 65)
# print("SUMMARY")
# print("=" * 65)
# print(f"""
# Overall RMSD = {rmsd_all:.3f} Å

# This is {'LARGE (>3Å) — structures are in different conformational states' if rmsd_all > 3 else 'MODERATE (1-3Å) — some conformational difference' if rmsd_all > 1 else 'SMALL (<1Å) — structures are essentially identical'}.

# AlphaFold predicted {'a DIFFERENT conformation for F1994L vs WT' if rmsd_all > 3 else 'similar conformations for F1994L and WT'}.
# The MD simulations confirmed and maintained this {'difference' if rmsd_all > 3 else 'similarity'} throughout all three replicas.
# """)



















# # save as check_starting_conformation.py
# # Compares key structural features between WT and F1994L starting structures
# # to determine whether AlphaFold predicted different conformations

# import numpy as np
# from pathlib import Path

# STRUCTURES = Path.home() / "ROS1_MD/00_structures"

# def load_ca_coords(pdb_path):
#     """Load CA atom coordinates and residue numbers from a PDB file."""
#     coords = {}
#     with open(pdb_path) as f:
#         for line in f:
#             if line.startswith("ATOM") and line[12:16].strip() == "CA":
#                 resid = int(line[22:26].strip())
#                 x = float(line[30:38].strip())
#                 y = float(line[38:46].strip())
#                 z = float(line[46:54].strip())
#                 coords[resid] = np.array([x, y, z])
#     return coords

# def get_atom_coord(pdb_path, resid, atom_name):
#     """Get coordinates of a specific atom in a specific residue."""
#     with open(pdb_path) as f:
#         for line in f:
#             if (line.startswith("ATOM") and
#                 line[12:16].strip() == atom_name and
#                 int(line[22:26].strip()) == resid):
#                 x = float(line[30:38].strip())
#                 y = float(line[38:46].strip())
#                 z = float(line[46:54].strip())
#                 return np.array([x, y, z])
#     return None

# def measure_distance(coord1, coord2):
#     """Euclidean distance between two 3D points."""
#     return float(np.linalg.norm(coord1 - coord2))

# def measure_angle(a, b, c):
#     """Angle at point b formed by vectors b->a and b->c, in degrees."""
#     v1 = a - b
#     v2 = c - b
#     cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
#     return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

# # ── structures to compare ─────────────────────────────────────────────────────
# wt_pdb     = STRUCTURES / "ROS1_kinase_1934_2225_WT.pdb"
# f1994l_pdb = STRUCTURES / "ROS1_kinase_1934_2225_F1994L.pdb"

# print("=" * 60)
# print("STARTING CONFORMATION COMPARISON: WT vs F1994L")
# print("=" * 60)
# print(f"WT structure    : {wt_pdb.name}")
# print(f"F1994L structure: {f1994l_pdb.name}")
# print()

# # ── KEY MEASUREMENT 1: αC-helix glutamate to β3 lysine distance ──────────────
# # In active kinases the αC-helix glutamate (E) forms a salt bridge with
# # the conserved β3 lysine (K). This distance is a classic active/inactive marker.
# # In ROS1:
# #   β3 lysine  = K1980 (real residue 1980)
# #   αC-helix glutamate = E1993 (real residue 1993) — check your sequence
# # Distance < 4.5 Å = salt bridge present = active-like (αC-in)
# # Distance > 7.0 Å = salt bridge broken = inactive-like (αC-out)

# print("─" * 60)
# print("MEASUREMENT 1: αC-helix glutamate to β3 lysine distance")
# print("(salt bridge marker: <4.5Å = active/αC-in, >7Å = inactive/αC-out)")
# print("─" * 60)

# # β3 lysine in ROS1 — check your sequence, typically a few residues before αC
# # αC glutamate — typically the conserved E in the αC-helix
# # We will check residues around 1980-1993 range
# # First let's find what's actually at those positions

# for pdb_path, label in [(wt_pdb, "WT"), (f1994l_pdb, "F1994L")]:
#     print(f"\n{label}:")
#     # check residues in the β3-αC region (1978-1995)
#     with open(pdb_path) as f:
#         seen = set()
#         for line in f:
#             if line.startswith("ATOM") and line[12:16].strip() == "CA":
#                 resid = int(line[22:26].strip())
#                 resname = line[17:20].strip()
#                 if 1978 <= resid <= 1998 and resid not in seen:
#                     seen.add(resid)
#                     print(f"  resid {resid} = {resname}")

# print()

# # ── KEY MEASUREMENT 2: DFG aspartate position ────────────────────────────────
# # DFG motif in ROS1 starts at real residue 2045 (D2045-F2046-G2047)
# # DFG-in: D2045 CA points toward ATP binding site
# # DFG-out: D2045 CA flipped outward
# # We measure the distance from D2045 CA to a reference point in the ATP pocket
# # A good reference is the conserved β3 lysine CA (K1980)
# # DFG-in: this distance is typically 15-20 Å
# # DFG-out: this distance increases to 25-35 Å

# print("─" * 60)
# print("MEASUREMENT 2: DFG aspartate (D2045) to β3-lysine distance")
# print("(DFG orientation marker)")
# print("─" * 60)

# ca_coords_wt     = load_ca_coords(wt_pdb)
# ca_coords_f1994l = load_ca_coords(f1994l_pdb)

# # DFG aspartate
# dfg_resid = 2045

# # find the conserved lysine — look for K in range 1978-1985
# print("\nLooking for conserved β3 lysine in WT (1978-1985):")
# with open(wt_pdb) as f:
#     seen = set()
#     for line in f:
#         if line.startswith("ATOM") and line[12:16].strip() == "CA":
#             resid = int(line[22:26].strip())
#             resname = line[17:20].strip()
#             if 1978 <= resid <= 1985 and resid not in seen:
#                 seen.add(resid)
#                 print(f"  resid {resid} = {resname}")

# # use D2045 CA to measure DFG position relative to rest of structure
# if dfg_resid in ca_coords_wt and dfg_resid in ca_coords_f1994l:
#     # measure DFG aspartate distance to several reference residues
#     ref_residues = [1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030]
#     print(f"\nD{dfg_resid} CA distance to reference residues:")
#     print(f"  {'Resid':>8}  {'WT dist (Å)':>14}  {'F1994L dist (Å)':>16}  {'Difference':>12}")
#     print("  " + "-" * 56)
#     for ref in ref_residues:
#         if ref in ca_coords_wt and ref in ca_coords_f1994l:
#             d_wt     = measure_distance(ca_coords_wt[dfg_resid],     ca_coords_wt[ref])
#             d_f1994l = measure_distance(ca_coords_f1994l[dfg_resid], ca_coords_f1994l[ref])
#             diff     = d_f1994l - d_wt
#             flag     = " ← LARGE DIFF" if abs(diff) > 3.0 else ""
#             print(f"  {ref:>8}  {d_wt:>14.2f}  {d_f1994l:>16.2f}  {diff:>+12.2f}{flag}")

# print()

# # ── KEY MEASUREMENT 3: Overall RMSD between WT and F1994L starting structures ─
# print("─" * 60)
# print("MEASUREMENT 3: CA RMSD between WT and F1994L starting structures")
# print("─" * 60)

# common_residues = sorted(set(ca_coords_wt.keys()) & set(ca_coords_f1994l.keys()))
# print(f"Common CA atoms: {len(common_residues)}")

# wt_coords     = np.array([ca_coords_wt[r]     for r in common_residues])
# f1994l_coords = np.array([ca_coords_f1994l[r] for r in common_residues])

# # centre both
# wt_c     = wt_coords     - wt_coords.mean(axis=0)
# f1994l_c = f1994l_coords - f1994l_coords.mean(axis=0)

# # kabsch alignment
# H = f1994l_c.T @ wt_c
# U, S, Vt = np.linalg.svd(H)
# d = np.linalg.det(Vt.T @ U.T)
# D = np.diag([1, 1, d])
# R = Vt.T @ D @ U.T
# f1994l_aligned = f1994l_c @ R.T

# rmsd_overall = float(np.sqrt(((wt_c - f1994l_aligned)**2).sum(axis=1).mean()))
# print(f"Overall CA RMSD (after alignment): {rmsd_overall:.3f} Å")

# # per-region RMSD
# regions = {
#     "NTL (1938-2030)": [r for r in common_residues if 1938 <= r <= 2030],
#     "CTL (2053-2216)": [r for r in common_residues if 2053 <= r <= 2216],
#     "A-loop (2045-2070)": [r for r in common_residues if 2045 <= r <= 2070],
#     "αC-helix (~1983-2000)": [r for r in common_residues if 1983 <= r <= 2000],
# }

# print(f"\nPer-region CA RMSD (after global alignment):")
# print(f"  {'Region':<25}  {'n_res':>6}  {'RMSD (Å)':>10}")
# print("  " + "-" * 46)
# for region_name, res_list in regions.items():
#     if len(res_list) < 3:
#         continue
#     idx = [common_residues.index(r) for r in res_list]
#     rmsd_r = float(np.sqrt(((wt_c[idx] - f1994l_aligned[idx])**2).sum(axis=1).mean()))
#     print(f"  {region_name:<25}  {len(res_list):>6}  {rmsd_r:>10.3f}")

# print()
# print("=" * 60)
# print("INTERPRETATION GUIDE")
# print("=" * 60)
# print("""
# Overall RMSD < 1.0 Å  → structures very similar, same conformation
# Overall RMSD 1-3 Å    → moderate difference, possibly different state
# Overall RMSD > 3 Å    → large difference, likely different conformation

# If NTL RMSD >> CTL RMSD → NTL is displaced relative to CTL
# If A-loop RMSD is large → activation loop in different position
# If αC-helix RMSD large  → αC-helix displaced (active vs inactive indicator)
# """)



















# # save as identify_aloop_outlier.py
# import numpy as np
# import pandas as pd
# from pathlib import Path

# BASE    = Path.home() / "Molecular_Dynamics_analysis"
# RESULTS = BASE / "results" / "ROS1"

# scores  = np.load(RESULTS / "aloop_scores.npy")
# x_owner = np.load(RESULTS / "aloop_x_owner.npy")
# F       = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()

# print("Testing threshold PC1 < -0.12")
# print("="*55)
# outlier = scores[:, 0] < -0.12
# owners  = x_owner[outlier]

# for ti in np.unique(owners):
#     p = Path(F[ti])
#     n = int((owners == ti).sum())
#     print(f"  traj_index={ti}  mutant={p.parent.name}"
#           f"  replica={p.name}  frames={n}")









# # save as regenerate_masks.py — replace the previous version
# import numpy as np
# from pathlib import Path

# RESULTS = Path.home() / "Molecular_Dynamics_analysis/results/ROS1"

# # load the meta arrays saved by script 01
# # these have exactly the right atom count to match traj_aligned and ref_centered
# resids = np.load(RESULTS / "meta_resids.npy", allow_pickle=True)
# names  = np.load(RESULTS / "meta_names.npy",  allow_pickle=True)

# print(f"Total atoms in meta: {len(resids)}")
# print(f"First internal resid: {resids[0]}")
# print(f"Last internal resid:  {resids[-1]}")

# # rebuild all masks using the same logic as build_standard_masks
# # but directly on the meta arrays — no MDAnalysis needed
# CA   = (names == "CA")
# BB   = np.isin(names, ["N", "CA", "C", "O"])
# BODY = np.isin(resids, np.arange(6, 278))

# ACT  = (resids >= (2045 - 1933)) & (resids <= (2070 - 1933))

# # NTL : start at internal residue 5 (was 1)
# NTL  = (resids >= 5) & (resids <= (2030 - 1933))


# # CTL start adjusted to internal residue 120 (real 2053)
# # Excludes the DFG boundary region (real 2031-2052, internal 98-119)
# # which belongs to the activation loop system and is already
# # captured in the ACT and A-loop PCA spaces.
# CTL = (resids >= 120) & (resids <= 283)


# # verify sizes match
# print(f"\nMask sizes (all should be {len(resids)}):")
# print(f"  CA:   {len(CA)}")
# print(f"  BB:   {len(BB)}")
# print(f"  BODY: {len(BODY)}")
# print(f"  ACT:  {len(ACT)}")
# print(f"  NTL:  {len(NTL)}")
# print(f"  CTL:  {len(CTL)}")

# # verify the fix
# print(f"\nNTL first True at internal resid: {resids[NTL.argmax()]}")
# print(f"CTL last True at internal resid:  {resids[len(CTL) - CTL[::-1].argmax() - 1]}")
# print(f"NTL True count: {NTL.sum()}")
# print(f"CTL True count: {CTL.sum()}")

# # save back
# np.savez(RESULTS / "masks.npz",
#          CA=CA, BB=BB, BODY=BODY, ACT=ACT, NTL=NTL, CTL=CTL)

# print("\nmasks.npz regenerated successfully")























# # save as regenerate_masks.py in your analysis folder
# import numpy as np
# from pathlib import Path
# import MDAnalysis as mda
# from scripts.ros1_utils import build_standard_masks

# RESULTS = Path.home() / "Molecular_Dynamics_analysis/results/ROS1"

# # load meta as a proper MDAnalysis object using your WT pdb
# pdb = Path.home() / "Molecular_Dynamics_analysis/trajectories/ros1_prepared_final/WT/0/WT-MD-prot.pdb"

# with open(RESULTS / "selection.txt") as f:
#     selection = f.read().strip()

# u = mda.Universe(str(pdb))
# meta = u.select_atoms(selection)

# print(f"Total atoms in selection: {len(meta)}")
# print(f"First residue: {meta.resids[0]}")
# print(f"Last residue:  {meta.resids[-1]}")

# # build masks using the fixed ros1_utils.py
# masks = build_standard_masks(meta)

# # save back to masks.npz — only the boolean mask arrays
# saveable = {
#     "CA":   masks["CA"],
#     "BB":   masks["BB"],
#     "BODY": masks["BODY"],
#     "ACT":  masks["ACT"],
#     "NTL":  masks["NTL"],
#     "CTL":  masks["CTL"],
# }

# np.savez(RESULTS / "masks.npz", **saveable)
# print("masks.npz regenerated successfully")

# # verify the fix worked
# print(f"NTL first True at atom index: {saveable['NTL'].argmax()}")
# print(f"NTL internal residue start:   {meta.resids[saveable['NTL'].argmax()]}")
# print(f"CTL last True at atom index:  {len(saveable['CTL']) - saveable['CTL'][::-1].argmax() - 1}")
# print(f"CTL internal residue end:     {meta.resids[len(saveable['CTL']) - saveable['CTL'][::-1].argmax() - 1]}")







# # save as regenerate_masks.py and run once
# import numpy as np
# from pathlib import Path
# from types import SimpleNamespace
# from scripts.ros1_utils import build_standard_masks

# RESULTS = Path.home() / "Molecular_Dynamics_analysis/results/ROS1"

# meta = SimpleNamespace(
#     resids   = np.load(RESULTS / "meta_resids.npy",   allow_pickle=True),
#     names    = np.load(RESULTS / "meta_names.npy",    allow_pickle=True),
#     resnames = np.load(RESULTS / "meta_resnames.npy", allow_pickle=True),
# )

# masks = build_standard_masks(meta)

# # remove resids_atom and exists_all — not stored in masks.npz
# saveable = {k: v for k, v in masks.items() if k not in ["resids_atom", "exists_all"]}

# np.savez(RESULTS / "masks.npz", **saveable)
# print("masks.npz regenerated successfully")

# # verify
# print("NTL first True at internal index:", saveable["NTL"].argmax())
# print("CTL last True at internal index:", len(saveable["CTL"]) - saveable["CTL"][::-1].argmax() - 1)









# # save this as check_terminals.py in your analysis folder
# # and run: python check_terminals.py

# from pathlib import Path
# import MDAnalysis as mda

# pdb = Path.home() / "Molecular_Dynamics_analysis/trajectories/ros1_prepared_final/WT/0/WT-MD-prot.pdb"

# u = mda.Universe(str(pdb))
# protein = u.select_atoms("protein")

# # get unique residues in order
# residues = protein.residues

# print("=== FIRST 10 RESIDUES ===")
# for r in residues[:10]:
#     print(f"  resid={r.resid}  resname={r.resname}  n_atoms={r.atoms.n_atoms}")

# print()
# print("=== LAST 10 RESIDUES ===")
# for r in residues[-10:]:
#     print(f"  resid={r.resid}  resname={r.resname}  n_atoms={r.atoms.n_atoms}")

# print()
# print(f"Total residues: {len(residues)}")
# print(f"First resid: {residues[0].resid}")
# print(f"Last resid:  {residues[-1].resid}")











# import numpy as np
# import pandas as pd
# from pathlib import Path

# BASE    = Path.home() / "Molecular_Dynamics_analysis"
# RESULTS = BASE / "results" / "ROS1"

# # load your meta resids to see the mapping
# resids = np.load(RESULTS / "meta_resids.npy", allow_pickle=True)

# # # if you saved RMSF from step 1:
# # rmsf = np.load(RESULTS / "rmsf_mean.npy")
# # #print first 10 and last 10 values
# # print("First 10 residues RMSF:", rmsf[:10])
# # print("Last 10 residues RMSF:", rmsf[-10:])

# # for now just print the first and last internal residue IDs
# print("First 10 internal residue IDs:", resids[:10])
# print("Last 10 internal residue IDs:", resids[-10:])











# import numpy as np
# import pandas as pd
# from pathlib import Path

# BASE    = Path.home() / "Molecular_Dynamics_analysis"
# RESULTS = BASE / "results" / "ROS1"

# scores  = np.load(RESULTS / "aloop_scores.npy")
# x_owner = np.load(RESULTS / "aloop_x_owner.npy")
# F       = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()

# # orange outlier is at PC1 < -0.10 in A-loop
# outlier = scores[:, 0] < -0.10
# owners  = x_owner[outlier]

# for ti in np.unique(owners):
#     p = Path(F[ti])
#     n = int((owners == ti).sum())
#     print(f"traj_index={ti}  mutant={p.parent.name}  replica={p.name}  frames={n}")


































# import numpy as np
# import pandas as pd
# from pathlib import Path

# BASE    = Path.home() / "Molecular_Dynamics_analysis"
# RESULTS = BASE / "results" / "ROS1"

# scores  = np.load(RESULTS / "actin_scores.npy")
# x_owner = np.load(RESULTS / "actin_x_owner.npy")
# F       = pd.read_csv(RESULTS / "F_paths.csv")["folder"].tolist()

# # outlier is at PC1 > 0.25 in ACT-IN
# high_pc1 = scores[:, 0] > 0.25
# owners   = x_owner[high_pc1]

# for ti in np.unique(owners):
#     p = Path(F[ti])
#     n = int((owners == ti).sum())
#     print(f"traj_index={ti}  mutant={p.parent.name}  replica={p.name}  frames={n}")