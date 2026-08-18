#!/usr/bin/env python3
"""
Rebuild Figure 4.2 (Key Systems FEL) and Figure 4.7 (Contrast Pairs FEL)
using the KDE-smoothed per-mutant PNGs to match Appendix E/F style.

Usage:
    cd ~/Molecular_Dynamics_analysis/results/ROS1
    python3 rebuild_main_fels.py

Reads from:
    /homes/lkgyammerah/Molecular_Dynamics_analysis/figures/ROS1/
    ros1_prepared_final/step3_free_energy_landscapes/actout_ctlfit/per_mutant/

Outputs:
    ./figures/FEL_key_systems.png       (Figure 4.2 — 3x2 grid)
    ./figures/contrast_pairs_FEL.png    (Figure 4.7 — 8x2 grid)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Path to KDE-smoothed per-mutant FELs
FEL_DIR = (
    '/homes/lkgyammerah/Molecular_Dynamics_analysis/figures/ROS1/'
    'ros1_prepared_final/step3_free_energy_landscapes/actout_ctlfit/per_mutant'
)

OUT_DIR = '/homes/lkgyammerah/Molecular_Dynamics_analysis/figures/ROS1/ros1_prepared_final'


def find_fel(variant):
    """Find the FEL PNG for a given variant name."""
    # Trying common naming patterns
    patterns = [
        f'actout_ctlfit_{variant}_fel.png',
        f'actout_ctlfit_{variant.upper()}_fel.png',
        f'{variant}_fel.png',
        f'{variant}.png',
    ]
    for p in patterns:
        path = os.path.join(FEL_DIR, p)
        if os.path.exists(path):
            return path

    # Fuzzy search
    for f in os.listdir(FEL_DIR):
        if variant.upper() in f.upper() and f.endswith('.png'):
            return os.path.join(FEL_DIR, f)

    print(f"  WARNING: Could not find FEL for '{variant}'")
    return None


def build_grid(systems, labels, out_path, cols, title=None,
               fig_w=None, fig_h=None, row_labels=None):
    """
    Build a grid figure from per-mutant FEL PNGs.

    systems: list of variant names (matching filenames)
    labels:  list of panel titles
    cols:    number of columns
    """
    rows = (len(systems) + cols - 1) // cols

    if fig_w is None:
        fig_w = 6 * cols
    if fig_h is None:
        fig_h = 5 * rows

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor('white')

    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (variant, label) in enumerate(zip(systems, labels)):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        path = find_fel(variant)
        if path:
            img = mpimg.imread(path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f'{variant}\nNOT FOUND',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')

        ax.set_title(label, fontsize=13, fontweight='bold', pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Add row labels on the left (if provided)
    if row_labels:
        for r, rlabel in enumerate(row_labels):
            axes[r, 0].set_ylabel(rlabel, fontsize=10, fontweight='bold',
                                  rotation=90, labelpad=10)

    # Hide empty panels
    for idx in range(len(systems), rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.01)

    plt.subplots_adjust(wspace=0.03, hspace=0.15,
                       left=0.02, right=0.98, top=0.95, bottom=0.02)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def main():
    print("=" * 60)
    print("Rebuilding main-text FEL figures with KDE style")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # FIGURE 4.2 — Key Systems (3 cols × 2 rows = 6 panels)
    # ══════════════════════════════════════════════════════════
    print("\n[1/2] Building Figure 4.2 — Key Systems FEL...")

    fig42_systems = [
        'WT',
        'Q2022P',
        'Q2022P_S1986F',
        'Q2022P_S1986Y',
        'D2113N',
        'E2020K',
    ]

    fig42_labels = [
        'WT\nReference',
        'Q2022P\nResistance — hinge',
        'Q2022P_S1986F\nCompound — suppressor F',
        'Q2022P_S1986Y\nCompound — suppressor Y',
        'D2113N\nRelatively closed pocket (53%)',
        'E2020K\nRelatively open pocket (77%)',
    ]
    
    build_grid(
        fig42_systems, fig42_labels,
        os.path.join(OUT_DIR, 'FEL_key_systems.png'),
        cols=3, fig_w=20, fig_h=12
    )

    # ══════════════════════════════════════════════════════════
    # FIGURE 4.7 — Contrast Pairs (2 cols × 8 rows = 16 panels)
    # ══════════════════════════════════════════════════════════
    print("\n[2/2] Building Figure 4.7 — Contrast Pairs FEL...")

    fig47_systems = [
        'WT',           'Q2022P',
        'S1986F',       'S1986Y',
        'G2032K',       'G2032R',
        'D2113G',       'D2113N',
        'L1982F',       'L1982V',
        'F2004C',       'F2004L',
        'L1947R',       'L1951R',
        'D2113N',       'E2020K',
    ]

    fig47_labels = [
        'WT',           'Q2022P',
        'S1986F',       'S1986Y',
        'G2032K',       'G2032R',
        'D2113G',       'D2113N',
        'L1982F',       'L1982V',
        'F2004C',       'F2004L',
        'L1947R',       'L1951R',
        'D2113N',       'E2020K',
    ]

    fig47_row_labels = [
        'Hinge — WT vs Q2022P',
        'αC-helix S1986 — F vs Y',
        'Hinge G2032 — K vs R',
        'A-loop D2113 — G vs N',
        'αC-helix L1982 — F vs V',
        'Gatekeeper F2004 — C vs L',
        'P-loop arginine pair',
        'Extremes — closed vs open',
    ]

    build_grid(
        fig47_systems, fig47_labels,
        os.path.join(OUT_DIR, 'contrast_pairs_FEL.png'),
        cols=2, fig_w=14, fig_h=42,
        row_labels=fig47_row_labels
    )

    print("\n" + "=" * 60)
    print("Done! Upload these to your Overleaf figures/ folder:")
    print("  figures/FEL_key_systems.png     → replaces old Figure 4.2")
    print("  figures/contrast_pairs_FEL.png  → replaces old Figure 4.7")
    print("=" * 60)


if __name__ == '__main__':
    main()