# unsupervised_learning_assignment

## ROS1 Mutational Dynamics Analysis

This project applies unsupervised learning techniques to molecular dynamics simulations of the ROS1 kinase domain.

The goal is to identify latent conformational states sampled during simulations and determine how cancer-associated mutations alter the distribution of these states.

---

## Project Overview

Molecular dynamics simulations generate high-dimensional structural data describing protein motion over time. Interpreting these datasets manually is challenging.

This project applies:

• Dimensionality reduction (PCA)  
• Density-based clustering (HDBSCAN and DBSCAN)  

to discover conformational states across ROS1 mutants.

Two complementary structural representations were analyzed:

1. **ACT-OUT core coordinates**  
   Backbone coordinates excluding the activation loop.

2. **NTL–CTL orientation features**  
   Relative orientation between the N-terminal and C-terminal kinase lobes.

---

## Repository Structure

```
├── Unsupervised_learning.ipynb
├── README.md
├── requirements.txt
├── requirements_explanation.txt
└── timer.py
```


The notebook performs:

1. Data loading and preprocessing
2. Feature construction
3. PCA dimensionality reduction
4. Density-based clustering
5. Mutation state occupancy analysis

---

## Software Used

This project relies on several scientific computing tools:

• **GROMACS** – molecular dynamics simulation package  
• **gromit** – molecular dynamics analysis tools  
• **NumPy / SciPy** – numerical computation  
• **scikit-learn** – machine learning algorithms  
• **HDBSCAN** – density-based clustering  
• **Matplotlib** – data visualization

---

## Main Findings

Two feature representations were analyzed.

The ACT-OUT representation revealed that the kinase core geometry remains largely conserved across mutants.

The orientation-based representation identified multiple conformational states and revealed mutation-dependent changes in state occupancy.

These results demonstrate that relative lobe orientation provides a more informative representation of conformational variability than global backbone coordinates.

---

### Additional utility

```
timer.py
```
A small helper module used in the notebook to measure execution time of different analysis steps using context managers.

### How to run the notebook

Install the required dependencies:
pip install -r requirements.txt

Then run the notebook using **Jupyter Notebook** or **JupyterLab**.

## Author

Capstone project for the **Unsupervised Learning course**.

