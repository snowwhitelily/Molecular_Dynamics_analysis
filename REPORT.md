# ROS1 Analysis — Project Report

**Lily Konadu Gyammerah | MSc Data Science for Life Sciences | Hanze University of Applied Sciences**
**Supervisor: Tsjerk Wassenaar**

---

## Project Status

| Component | Status |
|---|---|
| MD simulations | Complete — 117 simulations across 39 systems |
| Analysis pipeline (01-05c) | Complete — all steps run and verified |
| PyMOL Script A (WT vs Q2022P overlay) | Complete — Figure 4.3 |
| PyMOL Script B (PC1 violin) | Written — Figure 4.1B |
| PyMOL Script C (Q2022P family LD1) | Complete — Figures 4.5, 4.6 |
| Notebook | Complete — ROS1_MDAnalysis_Notebook.ipynb |
| README | Complete |
| METHODS.md | Complete |
| requirements.txt | Complete — pinned versions |

---

## Confirmed Results

### Docking (AutoDock Vina v1.2.5)

Box: centre x=68.60, y=64.50, z=23.68 Å | dimensions 25×20×20 Å | exhaustiveness=16 | modes=20

| System | Crizotinib (kcal/mol) | Lorlatinib (kcal/mol) |
|---|---|---|
| WT | −8.644 | −7.200 |
| Q2022P | −8.463 | −8.224 |
| Q2022P\_S1986F | −6.455 | −6.106 |
| Q2022P\_S1986Y | −5.508 | −6.369 |

### LDA (WT vs Q2022P family — pairwise)

| Comparison | Cohen's d | Perfect replica separation |
|---|---|---|
| WT vs Q2022P | −0.748 | yes |
| WT vs Q2022P\_S1986F | −0.464 | yes |
| WT vs Q2022P\_S1986Y | −0.792 | yes |

Intra-family LD1 ordering: Q2022P\_S1986F < Q2022P < Q2022P\_S1986Y

### ATP Pocket Open-State Occupancy

| System | Occupancy |
|---|---|
| E2020K | 89% |
| WT | 84% |
| Q2022P | 49% |
| D2113N | 27% |

### PCA (actout_ctlfit)

| Metric | Value |
|---|---|
| PC1 variance explained | 24.8% |
| PC2 variance explained | 19.2% |
| PC1+PC2 combined | 44.0% |

### DBSCAN Noise Fractions

Top 5: L1982V 7.27%, G2048A\_NC 5.21%, L1951R 5.18%, F2004C 5.14%, Q2022P 4.51%
Bottom 5: D2113N 0.08%, G1971E 0.10%, D2113G 0.16%, G2032K 0.21%, S1986F 0.25%
WT: 0.83%

### Stride Verification

| Space | PC1 (stride 1 vs 10) | PC2 (stride 1 vs 10) |
|---|---|---|
| actout\_ctlfit | 1.000 | 0.9985 |
| aloop | 1.000 | 0.9993 |

All cosine similarities >= 0.998 — stride 10 confirmed valid.