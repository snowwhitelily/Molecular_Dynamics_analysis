"""
PCA density overlay plot for the paper (combined single panel).

Matches the style of Christa's poster Figure 1: one shared set of axes with a
single light grey cloud of all frames in the background, and four coloured
contour pairs on top for the focal variants (WT, S1986F, Q2022P,
Q2022P + S1986F).

Each focal variant is drawn with:
    - a solid contour at the 75% density level (its usual core region)
    - a dashed contour at the 1% density level (its full explored territory)

Small orphan contour loops (islands) are removed from the drawing only. The
frames that produce them stay in the dataset and remain visible in the grey
background cloud, so nothing is removed from the analysis. This is a display
choice, not a change to the data.

Output:
    figures/ROS1/ros1_prepared_final/pca_overlay_paper.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
from contourpy import contour_generator

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = '/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1'
FIGS_DIR = '/homes/lkgyammerah/Molecular_Dynamics_analysis/figures/ROS1/ros1_prepared_final'
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
fp         = pd.read_csv(os.path.join(BASE, 'actout_ctlfit_fel_frame_points.csv'))
pc1_all    = fp['PC1'].values
pc2_all    = fp['PC2'].values
mutant_col = fp['mutant'].values
print(f'Loaded {len(fp):,} frames from {fp.mutant.nunique()} systems')

# ── Variants and colours ───────────────────────────────────────────────────────
VARIANTS = ['WT', 'S1986F', 'Q2022P', 'Q2022P_S1986F']
LABELS   = {
    'WT':            'WT',
    'S1986F':        'S1986F',
    'Q2022P':        'Q2022P',
    'Q2022P_S1986F': 'Q2022P + S1986F',
}
PALETTE  = {
    'WT':            '#1565C0',
    'S1986F':        '#2E7D32',
    'Q2022P':        '#E63946',
    'Q2022P_S1986F': '#F4511E',
}

# ── Fixed axis limits ──────────────────────────────────────────────────────────
pad = 0.05
x0, x1 = np.percentile(pc1_all, 0.5), np.percentile(pc1_all, 99.5)
y0, y1 = np.percentile(pc2_all, 0.5), np.percentile(pc2_all, 99.5)
XLIM = (x0 - pad*(x1-x0), x1 + pad*(x1-x0))
YLIM = (y0 - pad*(y1-y0), y1 + pad*(y1-y0))

# ── KDE helper ────────────────────────────────────────────────────────────────
def compute_kde(pc1, pc2, xlim, ylim, grid_size=200, bw=0.25):
    xgrid = np.linspace(xlim[0], xlim[1], grid_size)
    ygrid = np.linspace(ylim[0], ylim[1], grid_size)
    XX, YY = np.meshgrid(xgrid, ygrid)
    positions = np.vstack([XX.ravel(), YY.ravel()])
    kernel    = gaussian_kde(np.vstack([pc1, pc2]), bw_method=bw)
    ZZ        = kernel(positions).reshape(XX.shape)
    return XX, YY, ZZ

def density_level(ZZ, fraction):
    """Return the density value that encloses the given fraction of probability mass."""
    z_sorted = np.sort(ZZ.ravel())[::-1]
    cumsum   = np.cumsum(z_sorted) / z_sorted.sum()
    idx      = np.searchsorted(cumsum, fraction)
    return z_sorted[min(idx, len(z_sorted) - 1)]

def polygon_area(verts):
    """Absolute area of a closed polygon via the shoelace formula."""
    x, y = verts[:, 0], verts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def contour_lines_filtered(XX, YY, ZZ, level, area_frac=0.05):
    """
    Return contour line segments at the given density level, dropping small
    orphan loops. A loop is kept only if its area is at least area_frac times
    the area of the largest loop at that level. This removes visual islands
    without touching the underlying data.
    """
    gen   = contour_generator(XX, YY, ZZ)
    lines = gen.lines(level)
    if not lines:
        return []
    areas = [polygon_area(v) if len(v) >= 3 else 0.0 for v in lines]
    amax  = max(areas) if areas else 0.0
    if amax <= 0:
        return lines
    kept = [v for v, a in zip(lines, areas) if a >= area_frac * amax]
    dropped = len(lines) - len(kept)
    if dropped:
        print(f'    dropped {dropped} small contour loop(s) at level {level:.6g}')
    return kept

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 8), dpi=140)
fig.patch.set_facecolor('white')

# Single light grey cloud of every frame (all systems). Outlier frames of every
# variant, including the ones that used to form islands, stay visible here.
ax.scatter(pc1_all, pc2_all,
           c='#CCCCCC', s=1.2, alpha=0.30,
           rasterized=True, linewidths=0, zorder=1)

# Core areas enclosed by each variant's 75% contour. This is the quantitative
# read on how much each variant moves in this projection: a smaller core means
# a more constrained variant. Reported in PC1 x PC2 units.
core_area = {}

# Coloured contours per focal variant
for variant in VARIANTS:
    mask_v   = mutant_col == variant
    colour   = PALETTE[variant]
    n_frames = mask_v.sum()

    try:
        XX, YY, ZZ = compute_kde(
            pc1_all[mask_v], pc2_all[mask_v],
            XLIM, YLIM, bw=0.25
        )

        lev75 = density_level(ZZ, 0.75)   # dense core, encloses 75% of mass
        lev01 = density_level(ZZ, 0.99)   # outer edge, encloses 99% of mass

        # 1% contour (dashed, outer). This is where islands appear, so filter.
        # Softened (thinner, semi transparent) so the solid cores read first.
        for seg in contour_lines_filtered(XX, YY, ZZ, lev01, area_frac=0.05):
            ax.plot(seg[:, 0], seg[:, 1], color=colour,
                    lw=1.0, ls='dashed', alpha=0.40, zorder=3)

        # 75% contour (solid, core). Filter too for safety, harmless if none.
        core_segs = contour_lines_filtered(XX, YY, ZZ, lev75, area_frac=0.05)
        for seg in core_segs:
            ax.plot(seg[:, 0], seg[:, 1], color=colour,
                    lw=2.4, ls='solid', zorder=4)

        # Sum the areas of the kept 75% loops for this variant.
        core_area[variant] = sum(
            polygon_area(seg) for seg in core_segs if len(seg) >= 3
        )

        print(f'{variant}: n={n_frames:,}, lev75={lev75:.6f}, '
              f'lev01={lev01:.6f}, 75%_core_area={core_area[variant]:.6f}')

    except Exception as e:
        print(f'KDE failed for {variant}: {e}')

# ── 75% core area summary ──────────────────────────────────────────────────────
# Smallest core = most constrained variant in this projection.
if core_area:
    print('\n75% core area by variant (smaller = more constrained):')
    ref = core_area.get('WT')
    for v in sorted(core_area, key=core_area.get):
        rel = f'  ({core_area[v] / ref * 100:5.1f}% of WT)' if ref else ''
        print(f'  {LABELS[v]:<18} {core_area[v]:.6f}{rel}')

ax.set_xlim(XLIM)
ax.set_ylim(YLIM)
ax.set_xlabel('PC1 (actout_ctlfit)', fontsize=11)
ax.set_ylabel('PC2 (actout_ctlfit)', fontsize=11)
ax.tick_params(labelsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Legend ─────────────────────────────────────────────────────────────────────
variant_handles = [
    Line2D([0], [0], color=PALETTE[v], lw=2.4, ls='solid', label=LABELS[v])
    for v in VARIANTS
]
style_handles = [
    Line2D([0], [0], color='grey', lw=0, marker='o',
           markerfacecolor='#CCCCCC', markersize=7, label='All frames'),
    Line2D([0], [0], color='black', lw=2.4, ls='solid', label='75% density'),
    Line2D([0], [0], color='black', lw=1.4, ls='dashed', label='1% density'),
]
leg1 = ax.legend(handles=variant_handles, loc='upper left',
                 fontsize=9, frameon=False, title='Variant')
ax.add_artist(leg1)
ax.legend(handles=style_handles, loc='lower left',
          fontsize=9, frameon=False)

ax.set_title(
    'actout_ctlfit PCA — WT, S1986F, Q2022P, and Q2022P + S1986F\n'
    'Grey = all frames  |  Coloured contours at 1% and 75% density levels',
    fontsize=11, fontweight='bold'
)

plt.tight_layout()
out = os.path.join(FIGS_DIR, 'pca_overlay_paper.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out}')