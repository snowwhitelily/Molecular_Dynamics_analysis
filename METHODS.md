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

## Interactive Dashboard

An interactive Streamlit dashboard (`analysis/dashboard.py`) is provided for exploring the conformational analysis results. It covers per-mutant free energy landscape exploration, Q2022P family comparison, docking results, and a panel-wide metrics heatmap across all 39 systems. The dashboard loads pipeline outputs directly from the results directory and falls back to representative demo data when run outside the cluster environment.

Run with:

```bash
streamlit run analysis/dashboard.py
```

Streamlit is included in `requirements.txt`.

---

## Notebook

`ROS1_MDAnalysis_Notebook.ipynb` reproduces all main figures and results interactively. It can be run after the pipeline scripts have been executed and results are available in `results/`.

## Advanced Pipeline & Visualization Reference

The scripts detailed below support the high-throughput docking refactoring, geometric verification, and publication-level figure rendering workflows.

### Conformation Quality Control, PCA, & Receptor Selection
* **`analysis/02b2_pca_activesite.py` / `analysis/scripts/pca.py`**: Runs PCA on the active-site trajectory coordinates to identify principal modes of motion[cite: 5].
* **`analysis/plot_pca_overlay_paper.py`**: Generates high-quality PCA projection overlays comparing the conformational spaces of different variants for the manuscript.
* **`analysis/pymol_B_pc1_violin.py`**: Renders PC1 conformational violin distributions directly inside PyMOL[cite: 3, 5].
* **`analysis/check_dfg_angle.py` / `analysis/check_dfg_state.py` / `analysis/check_dfg_calibrate.py`**: Geometry engines measuring and calibrating DFG dihedral angles to programmatically classify structural states.
* **`analysis/05a_select_extract_docking_receptors_q2022p.py`**: Automated extraction of representative receptor coordinate frames for the Q2022P mutant and associated variants to use as targets in docking[cite: 5].

### Cabozantinib Inactive-State (Type II) Docking & Paper Plots
* **`analysis/run_inactive_pipeline.sh`**: Master shell script orchestrating the full inactive-state mutation and docking pipeline.
* **`analysis/06a_prepare_cabozantinib_active.py` / `06b_prepare_cabozantinib_inactive.py`**: Prepares ligand inputs and coordinates for active (DFG-in) and inactive (DFG-out) state docking.
* **`analysis/06d_mutate_inactive_receptors.py`**: Automates point mutations on the inactive DFG-out template structure.
* **`analysis/06e_prepare_inactive_mutants_dock.py`**: Generates grid parameters and receptors for Vina docking across all mutants.
* **`analysis/06c_rank_cabozantinib.py` / `06f_analyze_cabozantinib_poses.py`**: Extracts, filters, and ranks output docking poses.
* **`analysis/plot_cabozantinib_results.py`**: Generates high-resolution comparative figures (active vs. inactive bar chart and combined heatmaps) with automated text contrast and bounding-box safety padding[cite: 1].
* **`analysis/plot_docking_results_paper.py`**: Visualizes and formats the final docking results across all screened drugs and variants styled for journal submission.

### Binding Pocket Volumetry & Image Composition
* **`analysis/pocket_volume.py` / `make_volume.sh`**: Measures active-site volume changes across trajectory frames to quantify pocket transient states.
* **`analysis/pocket_prep.py` / `analysis/pocket_lib.py`**: Library functions to clean PDB structures and map atomic surfaces for grid calculation.
* **`analysis/pocket_render.py` / `analysis/overview_render.py`**: Generates raw ray-traced image outputs of binding pockets and receptor topology.
* **`analysis/pocket_compose.py` / `analysis/overview_compose.py` / `analysis/pocket_panel.py`**: Merges structural renders and text annotations into clean multi-panel publication figures.