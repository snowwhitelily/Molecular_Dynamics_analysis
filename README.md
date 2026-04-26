# ROS1 Kinase Mutant Molecular Dynamics Analysis

## Summary

This project investigates the conformational dynamics of ROS1 kinase variants using molecular dynamics simulations, dimensionality reduction, structural-state discovery, transition analysis, and docking-ready receptor selection.

The goal is to identify mutation-dependent shifts in kinase conformational sampling that may contribute to altered signaling behavior, activation-state preference, and inhibitor resistance.

---

## Variant Scope

This project analyzes a broad panel of approximately 39 ROS1 kinase variants using molecular dynamics–based conformational analysis.

From this larger panel, a focused subset is selected for deeper downstream studies (LDA, docking, and potential complex MD):

- WT
- Q2022P
- Q2022P_S1986F
- Q2022P_S1986Y

---

## Workflow Overview

![ROS1 Pipeline Flowchart](docs/ROS1_pipeline_flowchart.png)

Source file: `docs/ROS1_pipeline_flowchart.md`

---

## Main Analyses Performed

### Stage 1 — Preprocessing

- trajectory loading
- atom selection harmonization
- alignment to reference structure
- reusable coordinate arrays
- RMSD / RMSF generation

### Stage 2 — Coordinate PCA Spaces

- Global kinase PCA including activation loop (ACT-IN)
- Global kinase PCA excluding activation loop (ACT-OUT)
- N-terminal lobe PCA
- C-terminal lobe PCA
- Activation loop PCA
- CTL-fit PCA
- Additional selected structural subspace PCA
- PCA stride stability verification (comparison of strides 1, 5, and 10)

### Stage 3 — Structural Interpretation

- A-loop displacement metrics
- DFG χ1 / motif-state metrics
- αC-helix movement metrics
- ATP pocket geometry proxies
- contact / distance metrics

### Stage 4 — Basin Discovery

- density-based clustering (HDBSCAN / DBSCAN)
- occupancy analysis
- representative basin structures

### Stage 5 — Dynamics Layer

- basin transition analysis
- switch frequencies
- dwell times
- mutant vs WT kinetic behavior

### Stage 6 — Free Energy Landscape

- PCA density landscapes
- stable basin detection
- metastable region interpretation

### Stage 7 — Supervised Comparisons

- pairwise LDA
- WT vs mutant separation
- Q2022P-family discrimination

### Stage 8 — Docking Branch (Q2022P Subset)

- representative receptor extraction
- receptor / ligand PDBQT preparation
- AutoDock Vina docking
- docking rank comparison
- optional protein-ligand MD refinement

---

## Main Outputs

The analysis generates:

- PCA score arrays
- explained variance summaries
- PCA scatter plots
- occupancy heatmaps
- clustering labels
- FEL maps
- transition matrices
- dwell-time summaries
- LDA projections
- representative structures
- docking receptor structures
- docking score rankings
- publication-quality figures

---

## Repository Structure

```text
Molecular_Dynamics_analysis/
├── README.md
├── REPORT.md
├── analysis/
├── figures/
├── results/
├── trajectories/
├── structures/
├── md_setup/
├── pymol/
└── docs/
```

## Important Locations

Primary scripts: `analysis/`  
Slurm launchers: `analysis/slurm/`  
Generated results: `results/ROS1/`  
Generated figures: `figures/ROS1/ros1_prepared_final/`  
Project workflow and notes: `docs/`  
PyMOL helpers: `pymol/`

---

## Typical Execution Order

1. Preprocessing
2. PCA spaces
3. Evaluate PCA outputs
4. Verify PCA stride stability
5. Metrics
6. Free Energy Landscape (FEL)
7. Clustering
8. Metrics-to-cluster mapping
9. Cluster Transition Analysis
10. Distance PCA
11. LDA
12. Docking branch (optional)

---

## Notes

- Large MD trajectories may not be stored directly in the repository.
- Scripts are designed for HPC execution using Slurm.
- Notebooks are primarily for exploration, interpretation, and final figures.
- The pipeline is modular and expandable for future mutants or ligands.


## Key Findings
