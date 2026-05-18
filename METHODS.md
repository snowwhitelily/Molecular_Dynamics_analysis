# Computational Methods

## Software and Dependencies

### MD Simulations
GROMACS 2024.5 was used for all molecular dynamics simulations. Full simulation parameters are provided in Appendix B of the associated thesis.

### Structure Preparation
AlphaFold2 was used to generate starting structures for all 39 systems from the UniProt canonical sequence of human ROS1 (P08922, residues 1934–2225). Structures were visually inspected in UCSF ChimeraX.

### Analysis Environment
All analysis scripts require Python 3 with the dependencies listed in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

### PyMOL
Structural visualisation scripts require PyMOL 3.1.0 (Open-Source), available at https://pymol.org. The `Princomp` and `Colorinator` classes used by the PyMOL scripts are included in `analysis/scripts/` and developed by T. Wassenaar.

### Docking
AutoDock Vina v1.2.5 was used for ensemble docking. Receptor and ligand preparation used Meeko v0.7.1 (included in `requirements.txt`).

---

## Data Availability

Trajectory files (XTC + PDB per mutant/replica, ~62 µs aggregate) are not tracked in this repository due to file size. All computed outputs required to reproduce the figures and analysis results — PCA scores, cluster labels, structural metrics, and docking results — are provided in `results/`.

---

## Analysis Pipeline

Scripts are located in `analysis/` and should be run in the order listed. All scripts assume the working directory is the repository root.

| Script | Description |
|---|---|
| `01_data_loading_preprocessing.py` | Trajectory loading, alignment, terminal residue exclusion |
| `02a_pca_actin_actout.py` | ACT-IN and ACT-OUT PCA |
| `02b_pca_ntl_ctl.py` | NTL and CTL PCA |
| `02b2_pca_activesite.py` | Active site PCA |
| `02c_pca_aloop_ctlfit.py` | actout_ctlfit PCA (primary analysis space) |
| `02c_verify_pca_stride.py` | Stride stability verification |
| `02d_cluster_selected_pcas.py` | Conformational clustering (DBSCAN) |
| `02e_map_metrics_to_clusters.py` | Map structural metrics to clusters |
| `02f_cluster_transitions.py` | Cluster transition analysis |
| `03a_metrics_states.py` | Active site structural metrics |
| `03b_distance_pca.py` | NTL-CTL distance PCA |
| `03c_free_energy_landscape.py` | Free energy landscape analysis |
| `04_lda_final_figures.py` | LDA intra-family (Q2022P family multiclass) |
| `04b_lda_pairwise_wt_vs_q2022p_family.py` | LDA pairwise — WT vs each Q2022P family member |
| `04c_generalized_pairwise_lda.py` | Generalised pairwise LDA |
| `05a_select_extract_docking_receptors_q2022p.py` | Receptor extraction from dominant cluster medoids |
| `05aa_prepare_receptors_pdbqt_q2022p.py` | Receptor PDBQT preparation (Meeko) |
| `05ab_prepare_ligands_pdbqt_q2022p.py` | Ligand PDBQT preparation (crizotinib, lorlatinib) |
| `05b_prepare_vina_q2022p.py` | AutoDock Vina job preparation |
| `05d_run_vina_q2022p.py` | AutoDock Vina docking (8 jobs: 4 receptors × 2 ligands) |
| `05c_rank_vina_results_q2022p.py` | Rank and extract docking results |

---

## PyMOL Visualisation Scripts

The following scripts generate structural figures and require PyMOL 3.1.0 and the utilities in `analysis/scripts/`.

| Script | Output | Figure |
|---|---|---|
| `pymol_A_wt_vs_q2022p_overlay.py` | `wt_vs_q2022p_overlay.png` | Figure 4.3 |
| `pymol_B_pc1_violin.py` | `pc1_violin_wt.png` | Figure 4.1B |
| `pymol_C_family_ld1.py` | `ld1_on_structure.png`, `family_ld1_distributions.png` | Figures 4.5, 4.6 |

---

## Notebook

`ROS1_Analysis_Notebook_v9.ipynb` reproduces all main figures and results interactively. It can be run after the pipeline scripts have been executed and results are available in `results/`.