# ROS1 Kinase Mutant Molecular Dynamics Analysis

## Summary

This project investigates the conformational dynamics of ROS1 kinase resistance mutations
using molecular dynamics simulations, dimensionality reduction, structural-state discovery,
and ensemble docking. The goal is to identify mutation-dependent shifts in kinase
conformational sampling that contribute to TKI resistance in ROS1 fusion-positive
non-small-cell lung cancer (NSCLC).

---

## Variant Scope

**Main panel:** 38 clinically documented ROS1 kinase domain resistance mutations + WT
(39 systems total, 3 replicas each = 117 simulations, ~62 us aggregate).

**Excluded from analysis:** F1994L — AlphaFold2 predicted in a distinct inactive
conformation relative to all other systems, making direct comparison invalid.
Exclusion documented in Methods 3.1.3. All scripts exclude F1994L via
EXCLUDE_MUTANTS = {"F1994L"} after loading F_paths.csv.

**Focused downstream subset (LDA + docking):**
WT, Q2022P, Q2022P_S1986F, Q2022P_S1986Y

---

## Selected PCA Spaces

| Prefix | Description | Alignment | Captures |
|--------|-------------|-----------|---------|
| actout_ctlfit | Whole-body backbone PCA | CTL | Inter-lobe geometry (PRIMARY) |
| activesite | Active site backbone PCA | NTL | ATP pocket internal geometry |
| ctl | CTL backbone PCA | CTL | Internal CTL geometry |
| dist | NTL-CTL CA distance PCA | None | Superposition-independent inter-lobe |

Dropped from analysis (supervisor instruction): aloop, actin, actout, ntl.

---

## Workflow

```
01   Preprocessing
02a  ACT-IN + ACT-OUT PCA
02b  NTL + CTL PCA
02b2 Active site PCA
02c  A-loop + actout_ctlfit PCA
02c_verify  Stride stability verification
02d  Clustering (actout_ctlfit only, DBSCAN)
02e  Map metrics to clusters
02f  Cluster transition analysis
03a  Active site structural metrics
03b  NTL-CTL distance PCA
03c  Free energy landscapes (actout_ctlfit, ctl, activesite)
04   LDA intra-family (Q2022P family multiclass)
04b  LDA pairwise WT vs Q2022P family
04c  Generalised pairwise LDA
05a  Select docking receptors (Q2022P family cluster medoids)
05aa Prepare receptor PDBQTs (Meeko + AutoDock Vina)
05ab Prepare ligand PDBQTs (lorlatinib, crizotinib)
05b  Prepare AutoDock Vina jobs
05d  Run AutoDock Vina (8 jobs: 4 receptors x 2 ligands)
05c  Rank Vina results
```

All steps complete. PyMOL visualisation scripts (NTL-aligned and CTL-aligned)
written and ready, pending supervisor clarification on expected output.

---

## Key Scientific Findings

1. **Q2022P** produces a marked conformational shift in inter-lobe geometry consistent
   with proline-induced hinge rigidification. ATP pocket open-state occupancy reduced
   from 84% (WT) to 49%. High DBSCAN noise fraction (4.51%) indicates conformational
   adventurousness. Docking shows near-WT crizotinib affinity (-8.46 vs -8.64 kcal/mol)
   but elevated lorlatinib affinity (-8.22 vs -7.20 kcal/mol), suggesting a
   population-shift resistance mechanism rather than direct affinity loss.

2. **S1986F and S1986Y** act as conformational suppressors of Q2022P. Secondary
   mutations at the aC-helix partially or fully compensate the Q2022P-induced
   conformational shift. LDA separates all three compound systems from WT with
   perfect replica-level separation. Compound mutations reduce docking affinity
   for both crizotinib and lorlatinib relative to Q2022P alone.

3. **Same-position contrast pairs** demonstrate chemical specificity — amino acid
   identity, not just position, determines conformational perturbation direction
   and magnitude. Key pairs: L1982F vs L1982V, S1986F vs S1986Y,
   G2032K vs G2032R, F2004C vs F2004V vs F2004L.

4. **L1947R and L1951R** both show CTL bistability — NTL beta-sheet arginine
   substitutions produce two-state CTL conformational switching.

5. **L2010M** is the most conformationally restrained mutant across all spaces.

---

## Docking Results Summary (AutoDock Vina v1.2.5)

Box centre: x=68.60, y=64.50, z=23.68 A (F2004/L2026/D2033 centre of geometry in receptor frame)
Box size: 25 x 20 x 20 A | Exhaustiveness: 16 | Modes: 20
Ligands: lorlatinib, crizotinib (PubChem 3D conformers, Meeko PDBQT preparation)

| Mutant | Crizotinib (kcal/mol) | Lorlatinib (kcal/mol) |
|--------|----------------------|----------------------|
| WT | -8.644 | -7.200 |
| Q2022P | -8.463 | -8.224 |
| Q2022P_S1986F | -6.455 | -6.106 |
| Q2022P_S1986Y | -5.508 | -6.369 |

---

## Key Implementation Notes

- FEL colour scale fixed at 0-2.5 kT across all plots
- Clustering run on actout_ctlfit only (DBSCAN, not HDBSCAN — HDBSCAN assigned all frames as noise)
- Free energy formula: G = -kT ln(P), unit = kT at 300K
- Stride 10 (1 ns) verified stable — cosine similarity >= 0.998 for PC1 across all spaces
- Docking box centre measured from receptor PDBQT coordinates (GRO and PDB frames differ)

---

## Repository Structure

```
Molecular_Dynamics_analysis/
├── README.md
├── REPORT.md
├── analysis/                          <- all pipeline scripts
│   ├── 01_data_loading_preprocessing.py
│   ├── 02a_pca_actin_actout.py
│   ├── 02b_pca_ntl_ctl.py
│   ├── 02b2_pca_activesite.py
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
│   ├── pymol_q2022p_pca_NTL_aligned.py
│   └── pymol_q2022p_pca_CTL_aligned.py
├── figures/ROS1/ros1_prepared_final/  <- all generated figures
│   ├── step2_fast/
│   ├── step2d_selected_clustering/
│   ├── step3c_free_energy_landscapes/
│   ├── step4_lda_final/
│   ├── step4b_lda_ref_vs_family/
│   ├── step4c_generalized_pairwise_lda/
│   └── grey_highlight/
├── results/ROS1/                      <- all computed outputs
├── dock/ROS1/                         <- docking inputs, configs, outputs
└── trajectories/ros1_prepared_final/  <- XTC + PDB per mutant/replica
```

---

## Important Paths

| Path | Contents |
|------|----------|
| results/ROS1/ | Scores, labels, metrics, FEL CSVs, docking result CSVs |
| figures/ROS1/ros1_prepared_final/ | All analysis figures |
| dock/ROS1/q2022p_vina/ | Vina configs, outputs, logs |
| dock/ROS1/receptors_pdbqt/ | 4 receptor PDBQTs |
| dock/ROS1/ligands_pdbqt/ | lorlatinib.pdbqt, crizotinib.pdbqt |
| trajectories/ros1_prepared_final/ | XTC + PDB per mutant/replica |
