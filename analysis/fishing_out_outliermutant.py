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









# save as regenerate_masks.py — replace the previous version
import numpy as np
from pathlib import Path

RESULTS = Path.home() / "Molecular_Dynamics_analysis/results/ROS1"

# load the meta arrays saved by script 01
# these have exactly the right atom count to match traj_aligned and ref_centered
resids = np.load(RESULTS / "meta_resids.npy", allow_pickle=True)
names  = np.load(RESULTS / "meta_names.npy",  allow_pickle=True)

print(f"Total atoms in meta: {len(resids)}")
print(f"First internal resid: {resids[0]}")
print(f"Last internal resid:  {resids[-1]}")

# rebuild all masks using the same logic as build_standard_masks
# but directly on the meta arrays — no MDAnalysis needed
CA   = (names == "CA")
BB   = np.isin(names, ["N", "CA", "C", "O"])
BODY = np.isin(resids, np.arange(6, 278))

ACT  = (resids >= (2045 - 1933)) & (resids <= (2070 - 1933))

# NTL : start at internal residue 5 (was 1)
NTL  = (resids >= 5) & (resids <= (2030 - 1933))


# CTL start adjusted to internal residue 120 (real 2053)
# Excludes the DFG boundary region (real 2031-2052, internal 98-119)
# which belongs to the activation loop system and is already
# captured in the ACT and A-loop PCA spaces.
CTL = (resids >= 120) & (resids <= 283)


# verify sizes match
print(f"\nMask sizes (all should be {len(resids)}):")
print(f"  CA:   {len(CA)}")
print(f"  BB:   {len(BB)}")
print(f"  BODY: {len(BODY)}")
print(f"  ACT:  {len(ACT)}")
print(f"  NTL:  {len(NTL)}")
print(f"  CTL:  {len(CTL)}")

# verify the fix
print(f"\nNTL first True at internal resid: {resids[NTL.argmax()]}")
print(f"CTL last True at internal resid:  {resids[len(CTL) - CTL[::-1].argmax() - 1]}")
print(f"NTL True count: {NTL.sum()}")
print(f"CTL True count: {CTL.sum()}")

# save back
np.savez(RESULTS / "masks.npz",
         CA=CA, BB=BB, BODY=BODY, ACT=ACT, NTL=NTL, CTL=CTL)

print("\nmasks.npz regenerated successfully")























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