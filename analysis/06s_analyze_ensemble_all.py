#!/usr/bin/env python3
"""
06s_analyze_ensemble_all.py

Turn the 384 ensemble docks (06r) into the numbers the figures need, and report
honestly what changed relative to the single-medoid heatmap (step6k).

Writes:
  results/ROS1/step6r_ensemble_scores.csv   long form, one row per frame x ligand
                                            (variant, ligand, tag, replica, frame,
                                             state, front, trajectory, score)
  results/ROS1/step6r_ensemble_summary.csv  one row per variant x ligand
                                            (n, median, mean, std, min, max, q25, q75)

Prints:
  - open vs closed (per ligand, pooled variants)
  - front-distance vs score correlation (per ligand)
  - ensemble MEDIAN vs old single-medoid step6k, with the movers flagged
    (this is where single-frame artifacts show up)

The heatmap (06l2) and bars (06m2) read the summary CSV, so headline cell values
are the ensemble MEDIAN (robust to one odd frame — the whole point of this).

Usage (venv on):
    python analysis/06s_analyze_ensemble_all.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE    = Path.home() / "Molecular_Dynamics_analysis"
RESULTS = BASE / "results" / "ROS1"
D       = BASE / "dock" / "ROS1" / "q2022p_vina" / "ensemble_all"
VOUT    = D / "vina_outputs"

FRAME_METRICS = RESULTS / "step3a_frame_metrics.csv"
STEP6K_CSV    = RESULTS / "step6k_heatmap_sharedbox.csv"   # old single-medoid active

VARIANTS = ["WT", "Q2022P", "Q2022P_S1986F", "S1986F"]
LIGANDS  = ["ceritinib", "crizotinib", "lorlatinib",
            "repotrectinib", "zidesamtinib", "cabozantinib"]

MOVE_FLAG = 0.5   # |median - single| >= this (kcal/mol) is called out


def best_aff(pdbqt):
    with open(pdbqt) as fh:
        for ln in fh:
            if ln.startswith("REMARK VINA RESULT"):
                return float(ln.split()[3])
    return np.nan


def main():
    man = pd.read_csv(D / "manifest.csv")

    # attach trajectory provenance from the metrics (manifest doesn't store it)
    fm = pd.read_csv(FRAME_METRICS)
    fmk = (fm[["mutant", "replica", "frame_index_local",
               "pocket_open_state", "trajectory"]]
           .rename(columns={"mutant": "variant", "frame_index_local": "frame"})
           .drop_duplicates(subset=["variant", "replica", "frame", "pocket_open_state"]))
    man = man.merge(fmk, on=["variant", "replica", "frame", "pocket_open_state"],
                    how="left")

    # ---- long form: one row per frame x ligand ----
    rows = []
    for _, r in man.iterrows():
        for lig in LIGANDS:
            f = VOUT / f"{r['tag']}__{lig}.pdbqt"
            rows.append(dict(
                variant=r["variant"], ligand=lig, tag=r["tag"],
                replica=r["replica"], frame=r["frame"],
                state="open" if r["pocket_open_state"] == 1 else "closed",
                front=r["pocket_front_dist_nm"],
                trajectory=r.get("trajectory", ""),
                score=best_aff(f) if f.exists() else np.nan))
    df = pd.DataFrame(rows)
    n_bad = int((df.score > 0).sum()) + int(df.score.isna().sum())
    if n_bad:
        print(f"WARNING: {n_bad} failed/NaN scores present — investigate before trusting.")
    long_csv = RESULTS / "step6r_ensemble_scores.csv"
    df.to_csv(long_csv, index=False)
    print(f"wrote {long_csv}  ({len(df)} rows)")

    # ---- per-cell summary (median is the headline value) ----
    g = df.groupby(["variant", "ligand"])["score"]
    summ = g.agg(n="count", median="median", mean="mean", std="std",
                 min="min", max="max",
                 q25=lambda s: s.quantile(.25),
                 q75=lambda s: s.quantile(.75)).reset_index()
    summ_csv = RESULTS / "step6r_ensemble_summary.csv"
    summ.to_csv(summ_csv, index=False)
    print(f"wrote {summ_csv}  ({len(summ)} cells)\n")

    # ---- open vs closed (per ligand, pooled variants) ----
    print("=== open vs closed frames (median best affinity, pooled variants) ===")
    for lig in LIGANDS:
        d = df[df.ligand == lig]
        o = d[d.state == "open"]["score"]; c = d[d.state == "closed"]["score"]
        print(f"  {lig:14s} open {o.median():6.2f} | closed {c.median():6.2f} | "
              f"diff {o.median()-c.median():+.2f}")

    # ---- front-distance vs score correlation (per ligand) ----
    print("\n=== front-distance vs score (Pearson r; negative = more open binds stronger) ===")
    for lig in LIGANDS:
        d = df[df.ligand == lig].dropna(subset=["front", "score"])
        r = np.corrcoef(d["front"], d["score"])[0, 1] if len(d) > 3 else np.nan
        print(f"  {lig:14s} r = {r:+.2f}")

    # ---- ensemble median vs old single-medoid (step6k) ----
    if STEP6K_CSV.exists():
        old = (pd.read_csv(STEP6K_CSV)
               .rename(columns={"mutant": "variant",
                                "affinity_kcal_mol": "single"})
               [["variant", "ligand", "single"]])
        cmp = summ.merge(old, on=["variant", "ligand"], how="left")
        cmp["delta"] = cmp["median"] - cmp["single"]
        cmp = cmp.sort_values("delta", key=lambda s: s.abs(), ascending=False)
        print("\n=== ensemble median vs single-medoid (step6k) — biggest movers first ===")
        print(f"  {'variant':16s} {'ligand':13s} {'single':>7s} {'median':>7s} "
              f"{'delta':>7s} {'spread(min..max)':>18s}")
        for _, r in cmp.iterrows():
            flag = "  <== moved" if abs(r["delta"]) >= MOVE_FLAG else ""
            sp = f"{r['min']:.2f}..{r['max']:.2f}"
            print(f"  {r['variant']:16s} {r['ligand']:13s} "
                  f"{r['single']:7.2f} {r['median']:7.2f} {r['delta']:+7.2f} "
                  f"{sp:>18s}{flag}")
        moved = cmp[cmp["delta"].abs() >= MOVE_FLAG]
        print(f"\n  {len(moved)} of {len(cmp)} cells moved by >= {MOVE_FLAG} kcal/mol "
              f"(single-frame artifacts).")
    else:
        print(f"\n(step6k CSV not found at {STEP6K_CSV} — skipped the single-vs-median comparison)")


if __name__ == "__main__":
    main()