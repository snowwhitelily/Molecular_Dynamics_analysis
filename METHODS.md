# Computational Methods

## Software and dependencies

### Molecular dynamics
All simulations were run with GROMACS 2024.5 using the AMBER99SB-ILDN force field and
the TIP3P water model. Per-system simulation parameters are given in Appendix B of the
thesis.

### Structure preparation
Active-state starting structures for all 39 systems were generated with AlphaFold2 from
the UniProt canonical sequence of human ROS1 (P08922), kinase-domain residues
1934–2225, and inspected in UCSF ChimeraX. The inactive (DFG-out) receptor used for
type II docking derives from the ROS1 conformational reference of Vilachã et al.
(2025); resistance mutations were introduced by point substitution and the mutated
receptors energy-minimised with OpenMM 8.6 and PDBFixer.

### Analysis environment
Analysis code targets Python 3.13, with dependencies pinned in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Principal packages: NumPy, pandas, SciPy, scikit-learn and HDBSCAN (statistics and
clustering); MDAnalysis (trajectory handling); RDKit and Meeko (ligand and receptor
preparation); matplotlib and seaborn (figures).

### Structural visualisation
Structural figures were rendered with Open-Source PyMOL 3.1.0. The `Princomp` and
`Colorinator` helper classes used by the PyMOL scripts are included in
`analysis/scripts/` and were developed by T. Wassenaar. PyMOL is a system
installation; the analysis-environment packages are exposed to it through `PYTHONPATH`
at render time.

### Docking
Docking used AutoDock Vina 1.2.5. Receptors and ligands were prepared with Meeko 0.7.1;
three-dimensional ligand conformers were generated with RDKit. Ligand identity was
verified by molecular formula prior to docking.

---

## Data availability
Trajectory files (XTC and PDB per system and replica; ~62 µs aggregate) are not tracked
here owing to size. Computed analysis outputs required to reproduce the figures — PCA
scores, cluster assignments, structural metrics, and docking score tables — are provided
under `results/`. Docking inputs, receptors, ligands, configurations and outputs are
under `dock/`.

---

## Analysis pipeline
Scripts are in `analysis/` and are run from the repository root in the order of their
numeric prefix.

### Conformational analysis

| Script | Description |
|---|---|
| `01_data_loading_preprocessing.py` | Trajectory loading, alignment, terminal-residue exclusion |
| `02a_pca_actin_actout.py` | ACT-IN / ACT-OUT PCA |
| `02b_pca_ntl_ctl.py` | NTL / CTL PCA |
| `02b2_pca_activesite.py` | Active-site PCA |
| `02c_pca_aloop_ctlfit.py` | Activation-loop (actout_ctlfit) PCA — primary analysis space |
| `02c_verify_pca_stride.py` | Stride-stability verification |
| `02d_cluster_selected_pcas.py` | Conformational clustering (DBSCAN / HDBSCAN) |
| `02e_map_metrics_to_clusters.py` | Map structural metrics to clusters |
| `02f_cluster_transitions.py` | Cluster-transition analysis |
| `03a_metrics_states.py` | Active-site structural metrics |
| `03b_distance_pca.py` | NTL–CTL distance PCA |
| `03c_free_energy_landscape.py` | Free-energy landscapes |
| `04_lda_final_figures.py` | LDA — Q2022P family (multiclass) |
| `04b_lda_pairwise_wt_vs_q2022p_family.py` | LDA — WT vs each Q2022P-family member |
| `04c_generalized_pairwise_lda.py` | Generalised pairwise LDA |

### Docking — Q2022P family

| Script | Description |
|---|---|
| `05a_select_extract_docking_receptors_q2022p.py` | Receptor extraction from dominant-cluster medoids |
| `05aa_prepare_receptors_pdbqt_q2022p.py` | Receptor PDBQT preparation (Meeko) |
| `05ab_prepare_ligands_pdbqt_q2022p.py` | Ligand PDBQT preparation (crizotinib, lorlatinib) |
| `05b_prepare_vina_q2022p.py` | Vina job preparation |
| `05c_rank_vina_results_q2022p.py` | Rank and extract docking results |

Docking box: 25 × 20 × 20 Å centred on the ATP-binding pocket; exhaustiveness 16.

### Extended docking — six inhibitors, MD-frame ensemble
This stage broadens the docking to four focal variants (WT, S1986F, Q2022P,
Q2022P + S1986F) and six TKIs (cabozantinib, ceritinib, crizotinib, lorlatinib,
repotrectinib, zidesamtinib), docked into an ensemble of MD frames (open and closed
pocket states) rather than a single structure, and adds inactive-state (DFG-out)
docking. Scripts are `analysis/06*`:

- **Inactive-state receptors** (`06b`, `06d`, `06d2`, `06e`) — mutate and
  energy-minimise the DFG-out template and prepare docking grids.
- **Active-state ensemble** — representative open/closed frames per variant are
  extracted and prepared as receptors under `dock/ROS1/q2022p_vina/ensemble_all/`.
- **Ligand preparation** — 3D conformers (RDKit) prepared with Meeko; macrocyclic
  ligands whose rings cannot open are docked as rigid 3D-conformer ensembles.
- **Docking and scoring** — Vina across the ensemble; scores aggregated by
  `06s2_analyze_ensemble_all3d.py` into
  `results/ROS1/step6r_ensemble_scores_all3d.csv`.
- **Pocket openness** (`06o`, `06p`, `06r`) — open-state occupancy per variant.

Corrected docking tables and per-inhibitor method notes are recorded in
`COAUTHOR_summary_ensemble_corrected_v4.md`.

---

## Figure generation

### Conformational figures

| Script | Output |
|---|---|
| `scripts/pymol_A_wt_vs_q2022p_overlay.py` | WT vs Q2022P structural overlay |
| `scripts/pymol_B_pc1_violin.py` | PC1 conformational violin (WT) |
| `scripts/pymol_C_family_ld1.py` | Q2022P-family LD1 on structure and distributions |
| `scripts/plot_pca_overlay_paper.py` | PCA projection overlays |

### DFG-state quality control
`scripts/check_dfg_angle.py`, `check_dfg_calibrate.py` and `check_dfg_state.py` measure
the DFG rotation geometry to classify active (DFG-in) versus inactive (DFG-out)
receptors before docking.

### Pocket and binding-mode figures
The pocket figures are produced in two steps: an MDAnalysis preparation step
(`scripts/pocket_prep.py`, using `scripts/pocket_lib.py`) writes the frame-corrected
drug pose and pocket-residue coordinates; PyMOL then renders per-variant panels
(`scripts/pocket_render.py`), which are composited with labels and a legend
(`scripts/pocket_compose.py`). `scripts/overview_render.py` and `overview_compose.py`
produce the whole-domain overview. Binding-mode cartoons are rendered by
`scripts/render_binding_active.py` and `render_binding_inactive_cabo.py`; contact
diagrams by `scripts/plot_drug_contacts_2d.py` and `render_contacts_3d.py`; the
active-versus-inactive docking summary by
`scripts/plot_umcg_active_vs_inactive_both_drugs.py`; and the docking heatmap by
`06l2b_plot_heatmap_ensemble_all3d.py`.

---

## Interactive dashboard
`analysis/dashboard.py` is a Streamlit application for exploring the conformational
results — per-mutant free-energy-landscape exploration, Q2022P-family comparison,
docking results, and a panel-wide metrics heatmap across all 39 systems. It reads
outputs from `results/` and falls back to representative demo data when run outside the
cluster:

```bash
streamlit run analysis/dashboard.py
```

---

## Notebook
`ROS1_MDAnalysis_Notebook.ipynb` reproduces the main figures interactively once the
pipeline has been run and `results/` is populated.