# ROS1 Analysis — Report

## Purpose

This document is a report for the ROS1 kinase mutant MD project.
It tracks progress, findings, decisions, and next steps.
Updated as new results are generated.

---

## Current Project Status

### Pipeline
All analysis pipeline steps complete (preprocessing through docking).
Two PyMOL visualisation scripts pending supervisor clarification on expected output.

### Thesis
All chapters written including Section 4.7 (docking results filled May 2026).
Appendix E (grey highlight plots) pending supervisor colour sign-off.
Figure embedding into docx not yet done.

### Notebook
Current authoritative notebook: ROS1_Analysis_Notebook_v7.ipynb

---

## Dataset Scope

**Main panel:** 38 resistance mutations + WT = 39 systems
**Replicas:** 3 per system
**Total simulations:** 117
**Aggregate simulation time:** ~62 us
**Force field:** AMBER99SB-ILDN
**Water model:** TIP3P
**Software:** GROMACS 2024.5

**Excluded from analysis:** F1994L — AlphaFold2 predicted in distinct inactive conformation,
not comparable to active-form panel. Documented in Methods 3.1.3.

**Focused downstream subset (LDA + docking):**
WT, Q2022P, Q2022P_S1986F, Q2022P_S1986Y

---

## Key Confirmed Results

### Docking Results (AutoDock Vina v1.2.5)

Box centre: x=68.60, y=64.50, z=23.68 A (centre of geometry F2004, L2026, D2033 in receptor frame)
Box size: 25 x 20 x 20 A | Exhaustiveness: 16 | Modes: 20

| Mutant | Crizotinib (kcal/mol) | Lorlatinib (kcal/mol) |
|--------|----------------------|----------------------|
| WT | -8.644 | -7.200 |
| Q2022P | -8.463 | -8.224 |
| Q2022P_S1986F | -6.455 | -6.106 |
| Q2022P_S1986Y | -5.508 | -6.369 |

Key interpretations:
- Q2022P shows near-WT crizotinib affinity — resistance is population-shift not affinity-loss
- Q2022P shows above-WT lorlatinib affinity — Q2022P pocket more complementary to lorlatinib scaffold
- Compound mutations reduce affinity for both drugs — clinically consistent with compound resistance

### LDA Results (WT vs Q2022P family)

| Comparison | Cohen's d | Sep/Within |
|------------|-----------|------------|
| WT vs Q2022P | -0.748 | 0.768 |
| WT vs Q2022P_S1986F | -0.464 | 0.469 |
| WT vs Q2022P_S1986Y | -0.792 | 0.779 |

All three: perfect replica-level separation from WT.
Intra-family LD1 ordering: Q2022P_S1986F < Q2022P < Q2022P_S1986Y

### DBSCAN Noise Fractions (confirmed)

Top 5: L1982V 7.27%, G2048A_NC 5.21%, L1951R 5.18%, F2004C 5.14%, Q2022P 4.51%
Bottom 5: D2113N 0.08%, G1971E 0.10%, D2113G 0.16%, G2032K 0.21%, S1986F 0.25%
WT: 0.83%

### Stride Verification (confirmed)

| Space | PC1 (1vs10) | PC2 (1vs10) |
|-------|-------------|-------------|
| actout_ctlfit | 1.000 | 0.9985 |
| aloop | 1.000 | 0.9993 |

All PC cosine similarities >= 0.998 — stride 10 confirmed valid.

---

## Remaining Tasks

| Task | Blocker |
|------|---------|
| PyMOL CTL/NTL scripts | Need supervisor clarification on expected output |
| Appendix E grey highlight plots | Need supervisor colour sign-off |
| Embed figures into thesis docx | Can start anytime |
| Build Streamlit dashboard | Start after figure embedding |
| AlphaFold ensemble analysis | Deferred — scope decision needed from supervisor |

---

## Supervisor Feedback Log

| Date | Feedback | Action taken |
|------|----------|-------------|
| April 2026 | A-loop not of interest | Removed from selected analysis spaces |
| April 2026 | Focus on NTL-wrt-CTL | actout_ctlfit designated primary space |
| April 2026 | Active site conformation also interesting | activesite PCA added |
| April 2026 | Fix colourbar — same scale everywhere | vmax=2.5 kT enforced everywhere |
| April 2026 | How was free energy calculated? | G = -kT ln(P), unit = kT, label updated |
| April 2026 | F1994L — started in inactive form | Excluded from all analysis |
| April 2026 | PyMOL figures for Q2022P family | Scripts written, awaiting clarification on expected output |
| April 2026 | Figure design guidance (Wassenaar notes) | Grey-out principle applied to all bar charts |
| May 2026 | AlphaFold ensemble analysis suggestion | Deferred — scope decision pending supervisor confirmation |

---

## Technical Decisions Log

| Decision | Rationale |
|----------|-----------|
| Exclude F1994L | AlphaFold predicted inactive — not comparable to active-form panel |
| Drop aloop from selected spaces | Supervisor: not relevant to binding pocket in active form |
| Fix FEL colorbar at 0–2.5 kT | Supervisor requirement for direct visual comparison |
| Use DBSCAN not HDBSCAN for docking | HDBSCAN labelled all Q family frames as noise |
| Stride 10 (1 ns) for PCA | Verified stable — cosine similarity >= 0.998 for PC1 |
| actout_ctlfit as primary space | Largest eigenvalues; captures inter-lobe geometry directly |
| Box centre from receptor PDBQT | GRO coordinate frame differs from PDB — measured from actual receptor file |
| Box size 25x20x20 A | X dimension of pocket spans 23.9 A — 25 A gives buffer |
| Grey-out non-focal variants in bar charts | Supervisor figure design principle — highlight message, background the rest |
