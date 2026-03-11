# Repository Guide

This repository contains the capstone assignment for the Unsupervised Learning course.

## Purpose

The project applies unsupervised learning to molecular dynamics simulations of ROS1 kinase mutants associated with non-small-cell lung cancer (NSCLC). The aim is to identify latent conformational states and compare how mutations redistribute occupancy across these states.

---

## Main Files

- `Unsupervised_learning.ipynb`  
  Main notebook containing preprocessing, feature construction, PCA, clustering, and occupancy analysis.

- `README.md`  
  Project overview, setup instructions, and summary of findings.

- `REPORT.md`  
  Additional documentation describing repository structure and workflow.

- `requirements.txt`  
  Python dependencies required to run the notebook.

- `requirements_explanation.txt`  
  Brief explanation of why each dependency is used.

- `timer.py`  
  Utility module used in the notebook to measure execution time of analysis steps.

---

## Analysis Workflow

1. Load and preprocess MD trajectories  
2. Align structures to a common reference  
3. Construct two feature sets:
   - ACT-OUT core coordinates
   - NTL–CTL orientation features  
4. Apply PCA for dimensionality reduction  
5. Cluster conformations using HDBSCAN and DBSCAN  
6. Compare mutation-specific state occupancies  

---

## Key Result

The orientation-based feature set provided the clearest separation of conformational states, while the ACT-OUT representation was dominated by a broad common basin across most mutants.

---

## Reproducibility

Install dependencies with:

```bash
pip install -r requirements.txt
```

Open `Unsupervised_learning.ipynb` in **Jupyter Notebook** or **JupyterLab** and run the cells in order.

## Notes

This repository is intended for coursework submission and reproducible academic review.
