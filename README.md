# ROS1 Kinase Mutant Molecular Dynamics Analysis

## Summary

This project investigates the conformational dynamics of ROS1 kinase resistance mutations
using molecular dynamics simulations, dimensionality reduction, structural-state discovery,
transition analysis, and docking-ready receptor selection.

The goal is to identify mutation-dependent shifts in kinase conformational sampling that
contribute to TKI resistance in ROS1 fusion-positive non-small-cell lung cancer (NSCLC).

---

## Variant Scope

**Main panel:** 38 clinically documented ROS1 kinase domain resistance mutations + WT
(39 systems total, 3 replicas each = 117 simulations × 1 μs = 117 μs aggregate).

**Note on F1994L:** F1994L was initially included but excluded from comparative analysis.
AlphaFold2 predicted F1994L in a distinct conformation relative to all other systems,
making direct comparison invalid. Its exclusion is documented in the Methods chapter.

**Focused downstream subset (LDA + docking):**
- WT
- Q2022P
- Q2022P_S1986F
- Q2022P_S1986Y

---

## Selected PCA Spaces

Four PCA subspaces were selected for downstream analysis (FEL, clustering, LDA):

| Prefix | Description | Alignment | Captures |
|--------|-------------|-----------|---------|
| actout_ctlfit | Whole-body backbone PCA | CTL | Inter-lobe geometry (PRIMARY) |
| activesite | Active site backbone PCA | NTL | ATP pocket internal geometry |
| ctl | CTL backbone PCA | CTL | Internal CTL geometry |
| dist | NTL-CTL CA distance PCA | None | Superposition-independent inter-lobe |

**Dropped from analysis (supervisor instruction):**
- aloop — A-loop points outward in active form, not relevant to binding pocket
- actin — Redundant with actout_ctlfit
- actout — Redundant with actout_ctlfit
- ntl — Too rigid, no meaningful separation

---

## Workflow Overview

```
01  Preprocessing
02a ACT-IN + ACT-OUT PCA
02b NTL + CTL PCA
02b2 Active site PCA  ← NEW
02c A-loop + actout_ctlfit PCA
02c_verify Stride stability verification
02d Clustering (actout_ctlfit only)
02e Map metrics to clusters
02f Cluster transition analysis
03a Active site structural metrics
03b NTL-CTL distance PCA
03c Free energy landscapes (actout_ctlfit, ctl, activesite)
04  LDA final figures (Q2022P family)
04b LDA pairwise WT vs Q2022P family
04c Generalised pairwise LDA
05a Select docking receptors (Q2022P family)
05aa Prepare receptor PDBQT
05ab Prepare ligand PDBQT
05b Prepare AutoDock Vina jobs
05c Rank Vina results
```

---

## Repository Structure

```
Molecular_Dynamics_analysis/
├── README.md
├── REPORT.md
├── analysis/
│   ├── 01_data_loading_preprocessing.py
│   ├── 02a_pca_actin_actout.py
│   ├── 02b_pca_ntl_ctl.py
│   ├── 02b2_pca_activesite.py          ← NEW
│   ├── 02c_pca_aloop_ctlfit.py
│   ├── 02c_verify_pca_stride.py
│   ├── 02d_cluster_selected_pcas.py
│   ├── 02e_map_metrics_to_clusters.py
│   ├── 02f_cluster_transitions.py
│   ├── 03a_metrics_states.py
│   ├── 03b_distance_pca.py
│   ├── 03c_free_energy_landscape.py
│   ├── 04_lda_final_figures.py
│   ├── 04b_lda_pairwise_wt_vs_q2022p_family.py
│   ├── 04c_generalized_pairwise_lda.py
│   ├── 05a_select_extract_docking_receptors_q2022p.py
│   ├── 05aa_prepare_receptors_pdbqt_q2022p.py
│   ├── 05ab_prepare_ligands_pdbqt_q2022p.py
│   ├── 05b_prepare_vina_q2022p.py
│   ├── 05c_rank_vina_results_q2022p.py
│   ├── pymol_q2022p_pca_NTL_aligned.py ← NEW
│   ├── pymol_q2022p_pca_CTL_aligned.py ← NEW
│   ├── scripts/
│   └── slurm/
├── figures/ROS1/ros1_prepared_final/
│   ├── step2_fast/
│   ├── step2d_selected_clustering/
│   ├── step3c_free_energy_landscapes/
│   └── pymol_pca/
│       ├── ntl_aligned/
│       └── ctl_aligned/
├── results/ROS1/
├── trajectories/ros1_prepared_final/
├── structures/
└── docs/
```

---

## Key Scientific Findings (preliminary)

1. **Q2022P** produces a marked conformational shift in the inter-lobe geometry
   consistent with proline-induced hinge rigidification.

2. **S1986F and S1986Y** act as conformational suppressors of Q2022P — secondary
   mutations at the αC-helix that partially or fully compensate the Q2022P shift.

3. **Same-position contrast pairs** throughout the panel demonstrate chemical
   specificity — the identity of the substituting amino acid, not just the position,
   determines the direction and magnitude of conformational perturbation.
   Key pairs: L1982F vs L1982V, S1986F vs S1986Y, G2032K vs G2032R,
   F2004C vs F2004V vs F2004L.

4. **L1947R and L1951R** both show CTL bistability — NTL β-sheet arginine
   substitutions produce two-state CTL conformational switching.

5. **L2010M** is the most conformationally restrained mutant across all spaces.

---

## Key Implementation Notes

- All scripts exclude F1994L via `EXCLUDE_MUTANTS = {"F1994L"}` block
  inserted after loading F_paths.csv and traj_aligned.npy
- FEL colour scale fixed at 0–2.5 kT across ALL plots 
- Clustering run on actout_ctlfit only 
- 03c default prefixes: actout_ctlfit,ctl 
- Free energy formula: G = −kT ln(P), unit = kT at 300K

---

## Important Locations

| Location | Contents |
|----------|----------|
| `analysis/` | All Python pipeline scripts |
| `analysis/slurm/` | SLURM job submission scripts |
| `results/ROS1/` | All computed outputs (scores, labels, metrics, FEL CSVs) |
| `figures/ROS1/ros1_prepared_final/` | All analysis figures |
| `trajectories/ros1_prepared_final/` | XTC + PDB per mutant/replica |


