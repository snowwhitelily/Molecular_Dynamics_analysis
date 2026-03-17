# ROS1 Kinase Mutant MD Analysis Report

## Summary

This project investigates the conformational dynamics of ROS1 kinase mutants using molecular dynamics simulations and PCA-based structural analysis.

The goal is to identify mutation-dependent shifts in structural sampling that may contribute to altered kinase behavior and drug resistance.

## Analyses Performed

- Global kinase PCA including the activation loop (ACT-IN)
- Global kinase PCA excluding the activation loop (ACT-OUT)
- N-terminal lobe PCA
- C-terminal lobe PCA
- Orientation PCA
- Activation loop PCA
- Tyrosine loop PCA
- Density-based clustering and occupancy analysis

## Main Outputs

The analysis generates:

- PCA projections for multiple structural representations
- Cluster population plots
- Occupancy heatmaps
- Representative structures for structural interpretation
- PCA- and LDA-derived structures for visualization in PyMOL

## Notes

The primary analysis notebook is:

`analysis/MD_final_pipeline.ipynb`

Generated figures are stored in:

`analysis/figures/`

Helper scripts are stored in:

`analysis/scripts/`
