# ROS1 Analysis Report

## Purpose

This file is a living scientific report for the ROS1 kinase mutant molecular dynamics project.

It will be updated as new computational results are generated and interpreted.

Unlike the README, this document focuses on progress, findings, decisions, and next steps.

---

## Current Project Status

### Pipeline Status

- Preprocessing pipeline completed
- Main PCA scripts prepared
- Metrics / FEL / clustering / transition scripts prepared
- LDA scripts prepared
- Docking branch prepared
- Waiting for HPC execution of remaining jobs

### Current Compute Situation

Shared workstation / cluster currently busy.

Full production jobs will be launched when resources are more available.

---

## Dataset Scope

Approximate ROS1 variant panel:

- ~39 kinase variants in the main MD study

Focused downstream subset:

- WT
- Q2022P
- Q2022P_S1986F
- Q2022P_S1986Y

Used for:

- targeted LDA
- docking
- possible protein-ligand MD refinement

---

## Planned Execution Order

1. Preprocessing  
2. PCA spaces  
3. Evaluate PCA outputs  
4. Structural metrics  
5. Free Energy Landscape  
6. Clustering  
7. Metrics-to-cluster mapping  
8. Basin transition analysis  
9. Distance PCA  
10. LDA  
11. Docking branch  
12. Optional complex MD

---

## Key Scientific Questions

- Which mutants diverge most from WT?
- Does Q2022P alter kinase activation-state preference?
- Do S1986F / S1986Y amplify or compensate Q2022P effects?
- Which conformational basins dominate?
- Do mutants transition between basins more often than WT?
- Which states are most relevant for ligand docking?

---

## Results Log

_No production results added yet._

This section will later contain:

- PCA findings
- FEL basin observations
- cluster occupancies
- transition summaries
- LDA separations
- docking rankings

---

## Current Notes

- Pipeline successfully evolved from notebook workflow into modular HPC workflow.
- Scripts are organized for reproducible Slurm execution.
- Interpretation phase begins after PCA outputs complete.

---

## Immediate Next Steps

1. Run PCA jobs (02a / 02b / 02c)
2. Inspect PCA figures and choose strongest spaces
3. Launch FEL / clustering / metrics mapping
4. Run transition analysis
5. Continue to LDA and docking branch

---

## Long-Term Goal

Develop a biologically meaningful story linking:

mutation -> conformational change -> state transitions -> ligand binding behavior

---

## Last Updated

April 2026