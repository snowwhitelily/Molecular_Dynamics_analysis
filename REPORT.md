# ROS1 Analysis — Living Scientific Report

## Purpose

This document is the living scientific report for the ROS1 kinase mutant MD project.
It tracks progress, findings, decisions, and next steps.
Updated as new results are generated.

---

## Current Project Status (April 2026)

### Pipeline Status

| Script | Status | Notes |
|--------|--------|-------|
| 01 preprocessing | ✅ DONE | Do not rerun |
| 02a ACT-IN + ACT-OUT PCA | 🔄 RUNNING (job 46944) | Rerun without F1994L |
| 02b NTL + CTL PCA | 🔄 RUNNING (job 46945) | Rerun without F1994L |
| 02c aloop + actout_ctlfit PCA | 🔄 RUNNING (job 46946) | Rerun without F1994L |
| 02c_verify stride | 🔄 RUNNING (job 46947) | Rerun without F1994L |
| 02b2 activesite PCA | 🔄 RUNNING (job 46993) | NEW script |
| 02d clustering | ⏳ PENDING | Waiting for 02c |
| 02e map metrics | ⏳ PENDING | Waiting for 02d |
| 02f transitions | ⏳ PENDING | Waiting for 02d |
| 03a metrics states | 🔄 RUNNING (job 46948) | Bugs fixed (REGIONS, salt bridge) |
| 03b distance PCA | 🔄 RUNNING (job 46949) | Density scaling bug fixed |
| 03c FEL | ⏳ PENDING | Fixed colorbar vmax=2.5 |
| 04 LDA | ⏳ PENDING | Q2022P family only |
| 04b LDA pairwise | ⏳ PENDING | |
| 04c generalised LDA | ⏳ PENDING | |
| 05a–05c docking | ⏳ PENDING | Waiting for clustering |
| PyMOL NTL aligned | ⏳ PENDING | Script fixed, ready to run |
| PyMOL CTL aligned | ⏳ PENDING | Script fixed, ready to run |

### Thesis Writing Status

| Section | Status | Notes |
|---------|--------|-------|
| Abstract | ✅ WRITTEN | |
| Chapter 1 Introduction | ✅ WRITTEN | |
| Chapter 2 Literature Review | ✅ WRITTEN | |
| Chapter 3 Methods | ✅ WRITTEN | |
| Chapter 4 Results | 🔄 FRAMEWORK WRITTEN | Placeholders for numbers |
| Chapter 5 Discussion | ✅ WRITTEN | Q2022P + same-position pairs + limitations |
| Chapter 6 Conclusion | 🔄 PARTIAL | Needs final results |
| Appendices | 🔄 FRAMEWORK | Tables need filling |

---

## Dataset Scope

**Main panel:** 38 resistance mutations + WT = 39 systems
**Replicas:** 3 per system
**Total simulations:** 117
**Simulation time:** 1 μs per replica = 117 μs aggregate
**Force field:** AMBER99SB-ILDN
**Water model:** TIP3P
**Software:** GROMACS 2024.5

**Excluded from analysis:** F1994L (AlphaFold2 predicted in distinct inactive conformation,
not comparable to active-form panel. Documented in Methods Chapter 3.1.3.)

**Focused downstream subset (LDA + docking):**
WT, Q2022P, Q2022P_S1986F, Q2022P_S1986Y

---

## Supervisor Feedback Log

| Date | Feedback | Action taken |
|------|----------|-------------|
| April 2026 | A-loop not of interest — points outward, no link to binding pocket | Removed aloop from 03c prefixes, dropped from selected analysis spaces |
| April 2026 | Focus on NTL-wrt-CTL | actout_ctlfit designated PRIMARY space; distance PCA (03b) added |
| April 2026 | Active site conformation also interesting | 02b2 activesite PCA script written and submitted |
| April 2026 | Fix colourbar — same scale everywhere | 03c updated with vmax=2.5, shared axis limits |
| April 2026 | How was free energy calculated? What unit? | G = −kT ln(P), unit = kT, colourbar label updated |
| April 2026 | F1994L — started in inactive form, not comparable | F1994L excluded from all analysis scripts via EXCLUDE_MUTANTS block |
| April 2026 | PyMOL figures for Q2022P family | New scripts written: NTL aligned + CTL aligned versions |

---

## Key Scientific Findings (confirmed from FEL analysis)

### Confirmed findings (from FEL plots reviewed across all spaces)

1. **Q2022P** — marked conformational shift from WT in inter-lobe geometry.
   Proline at hinge position rigidifies interdomain flexibility.

2. **Q2022P_S1986F** — partial rescue of Q2022P shift. αC-helix phenylalanine
   acts as conformational suppressor.

3. **Q2022P_S1986Y** — rescue and counter-shift. Tyrosine (with OH group)
   produces distinct suppressor effect from phenylalanine.
   Mechanistic progression: Q2022P → Q2022P_S1986F → Q2022P_S1986Y

4. **L1947R + L1951R** — both show CTL bistability (two-state CTL geometry).
   NTL β-sheet arginine substitutions control CTL conformational switching.

5. **G2032K vs G2032R** — opposing CTL PC2 directions from same position.
   Lysine pushes positive PC2, arginine pushes negative. Chemical identity
   at single hinge residue determines direction.

6. **F2004C vs F2004V vs F2004L** — progressive CTL perturbation series.
   C (WT-like) → V (broader) → L (PC2-shifted). Three substitutions, three outcomes.

7. **L1982F vs L1982V** — same position, opposite behaviour.
   F (WT-like) vs V (conformationally perturbed). Methyl group loss at packing interface.

8. **S1986F vs S1986Y** — same position, opposing effects across spaces.
   F = two-state globally; Y = single-state globally but bistable locally in CTL.

9. **L2010M** — most conformationally restrained mutant across all spaces.
   Tightest FEL minimum, consistently narrowest landscape.

10. **WT** — broad CTL landscape with two density regions. Important reference context.

### Pending findings (after rerun completes)

- Specific PC1/PC2 coordinates for all mutants (will change after F1994L exclusion rerun)
- Eigenvalues and explained variance percentages
- Cluster populations and occupancy fractions from HDBSCAN
- LDA discriminant scores for Q2022P family
- Active site metric values (03a)
- Distance PCA results (03b)
- Docking scores

---

## Scientific Questions and Current Answers

| Question | Current Answer |
|----------|----------------|
| Which mutants diverge most from WT? | Pending rerun — Q2022P family strong candidates |
| Does Q2022P alter inter-lobe geometry? | Yes — marked shift confirmed from FEL |
| Do S1986F/Y suppress Q2022P effects? | Yes — suppressor phenomenon confirmed from FEL |
| Do same-position mutations differ? | Yes — strong chemical specificity confirmed |
| Which conformations are best for docking? | Pending — dependent on clustering results |
| Do mutants switch between states? | Pending — 02f transition analysis |

---

## Technical Decisions Log

| Decision | Rationale |
|----------|-----------|
| Exclude F1994L | AlphaFold predicted in distinct inactive conformation — not comparable |
| Drop aloop from selected spaces | Supervisor: A-loop not relevant to binding pocket in active form |
| Fix FEL colorbar at 0–2.5 kT | Supervisor requirement for direct visual comparison |
| Use HDBSCAN not k-means | No prior knowledge of cluster number; handles noise naturally |
| 3 replicas per system | Convergence assessment and reproducibility verification |
| Stride 10 (1 ns) for PCA | Verified stable — cosine similarity ≥ 0.999 for PC1 |
| actout_ctlfit as primary space | Largest eigenvalues; captures inter-lobe geometry directly |
| NTL alignment for activesite PCA | Measures active site changes in NTL reference frame |
| CTL alignment for PyMOL CTL figures | Matches actout_ctlfit space — primary analysis space |

---

## Immediate Next Steps

1. Check cluster job status: `squeue`
2. Once 02a/02b/02c finish: submit 02d (actout_ctlfit only), 02c_verify
3. Once 02d finishes: submit 02e, 02f, 03c
4. Submit LDA scripts (04, 04b, 04c) — independent of clustering
5. Run PyMOL scripts (both NTL and CTL aligned)
6. Fill in Results chapter with actual numbers
7. Complete Conclusion chapter

---

## Last Updated

April 30, 2026
