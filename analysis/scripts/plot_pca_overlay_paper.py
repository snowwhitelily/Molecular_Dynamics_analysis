"""
PCA density overlay plot for the paper — revised.

Produces TWO figures from one run:
  - MAIN (pca_overlay_paper.png):        grey = pooled frames of the FOUR focal
                                         variants only. This is the publication
                                         figure.
  - SUPP (pca_overlay_supp_all38.png):   grey = every frame from all 38 systems
                                         (very faint), with the four focal
                                         variants highlighted on top. This gives
                                         the broader mutational context and
                                         justifies the choice of the four.

Both share the same PCA basis, KDE settings, density-estimation method and
contour levels. Only the grey layer, the display window, the title and the
output filename differ between them.

Each focal variant is drawn with:
    - a solid contour at the 75% density level (its usual core region)
    - a dashed contour at the 99% density level (its full explored territory)

Small orphan contour loops (islands) are removed from the drawing only. The
frames that produce them stay in the dataset and remain visible in the grey
cloud, so nothing is removed from the analysis. This is a display choice, not a
change to the data.
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

# ── Explained variance (axis labels) ───────────────────────────────────────────
# DO NOT recompute these from actout_ctlfit_scores.npy: that file stores only 10
# PCs, so variance-from-scores is INFLATED (a naive calc gives 46.8% / 16.7%).
# TRUTH from the fit verify log (step2c_verify_46998.out): PC1+PC2 = 50.2% of
# total variance, which splits to PC1 ~= 37%, PC2 ~= 13%. Hard-coded on purpose.
PC1_VAR = 37
PC2_VAR = 13

# ── What to render ─────────────────────────────────────────────────────────────
SCOPES   = ['four', 'all38']          # runs both; comment one out to skip it
OUTNAMES = {
    'four':  'pca_overlay_paper.png',        # MAIN publication figure
    'all38': 'pca_overlay_supp_all38.png',   # supplementary landscape figure
}
TITLES = {
    'four':  'PCA of conformational ensembles for\n'
             'WT, S1986F, Q2022P, and Q2022P + S1986F',
    'all38': 'Position of the four selected variants\n'
             'within the full 38-system PCA landscape',
}
# grey point styling per scope (main = subtle; supp = very faint background)
GREY_STYLE = {
    'four':  dict(s=1.0, alpha=0.25),
    'all38': dict(s=0.8, alpha=0.12),
}
GREY_SUBSAMPLE = None   # e.g. 40000 to plot a random subset if the PNG is heavy

# 99% (outer, dashed) contour styling — thinned + faded so the 75% cores lead.
SHOW_99  = True         # set False to drop the 99% contours from the MAIN figure
C99_LW   = 0.8
C99_ALPHA = 0.30

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
# Four clearly distinct hues: blue / green / orange / rose.
PALETTE  = {
    'WT':            '#0072B2',   # blue
    'S1986F':        '#009E73',   # green
    'Q2022P':        '#E69F00',   # orange
    'Q2022P_S1986F': '#CC79A7',   # rose
}

# frames belonging to the four focal variants (main-figure grey source)
mask_four = np.isin(mutant_col, VARIANTS)
pc1_four  = pc1_all[mask_four]
pc2_four  = pc2_all[mask_four]
print(f'  of which {mask_four.sum():,} frames are the four focal variants')

# ── KDE / grid settings ────────────────────────────────────────────────────────
GRID_SIZE = 240
BW        = 0.25
GRID_PAD  = 0.35   # per-variant grid margin, as a fraction of that variant's span

# ── KDE helpers ────────────────────────────────────────────────────────────────
def compute_kde(pc1, pc2, xlim, ylim, grid_size=200, bw=0.25):
    xgrid = np.linspace(xlim[0], xlim[1], grid_size)
    ygrid = np.linspace(ylim[0], ylim[1], grid_size)
    XX, YY = np.meshgrid(xgrid, ygrid)
    positions = np.vstack([XX.ravel(), YY.ravel()])
    kernel    = gaussian_kde(np.vstack([pc1, pc2]), bw_method=bw)
    ZZ        = kernel(positions).reshape(XX.shape)
    return XX, YY, ZZ

def density_level(ZZ, fraction):
    """Density value that encloses the given fraction of probability mass."""
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
    Contour line segments at the given density level, dropping small orphan
    loops. A loop is kept only if its area is >= area_frac x the largest loop's
    area. Removes visual islands without touching the underlying data.
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

# ── PASS 1: compute every contour once (shared by both figures) ────────────────
# Each variant's KDE is evaluated on a grid padded around that variant's own
# frames, so its density decays to ~0 before the grid edge and both loops close.
segments  = {}          # variant -> {'outer': [...], 'core': [...]}
xs_all, ys_all = [], []  # every 99% contour vertex, to size the display window
core_area = {}           # 75% core area per variant (descriptive only)

for variant in VARIANTS:
    mask_v   = mutant_col == variant
    n_frames = mask_v.sum()
    p1, p2   = pc1_all[mask_v], pc2_all[mask_v]

    try:
        sx, sy = p1.max() - p1.min(), p2.max() - p2.min()
        gxlim  = (p1.min() - GRID_PAD*sx, p1.max() + GRID_PAD*sx)
        gylim  = (p2.min() - GRID_PAD*sy, p2.max() + GRID_PAD*sy)

        XX, YY, ZZ = compute_kde(p1, p2, gxlim, gylim,
                                 grid_size=GRID_SIZE, bw=BW)

        lev75 = density_level(ZZ, 0.75)
        lev99 = density_level(ZZ, 0.99)

        outer = contour_lines_filtered(XX, YY, ZZ, lev99, area_frac=0.05)
        core  = contour_lines_filtered(XX, YY, ZZ, lev75, area_frac=0.05)
        segments[variant] = {'outer': outer, 'core': core}

        for seg in outer:
            xs_all.append(seg[:, 0]); ys_all.append(seg[:, 1])

        core_area[variant] = sum(
            polygon_area(seg) for seg in core if len(seg) >= 3
        )
        print(f'{variant}: n={n_frames:,}, lev75={lev75:.6f}, '
              f'lev99={lev99:.6f}, 75%_core_area={core_area[variant]:.6f}')

    except Exception as e:
        print(f'KDE failed for {variant}: {e}')
        segments[variant] = {'outer': [], 'core': []}

# ── 75% core area summary (console diagnostic; descriptive, not on the figure) ──
if core_area:
    print('\n75% core area by variant (smaller = more constrained in this projection):')
    ref = core_area.get('WT')
    for v in sorted(core_area, key=core_area.get):
        rel = f'  ({core_area[v] / ref * 100:5.1f}% of WT)' if ref else ''
        print(f'  {LABELS[v]:<18} {core_area[v]:.6f}{rel}')

# ── Display window, sized per scope ────────────────────────────────────────────
# Always guarantee every 99% contour is shown in full; the percentile clamp uses
# the four-variant frames for the MAIN figure (zoom to the four) and all frames
# for the SUPP figure (zoom out to the whole landscape). Square + equal aspect
# keeps the PCA space undistorted.
def compute_window(scope):
    cx = np.concatenate(xs_all) if xs_all else pc1_all
    cy = np.concatenate(ys_all) if ys_all else pc2_all
    px1, px2 = (pc1_all, pc2_all) if scope == 'all38' else (pc1_four, pc2_four)
    xmin = min(cx.min(), np.percentile(px1, 0.5))
    xmax = max(cx.max(), np.percentile(px1, 99.5))
    ymin = min(cy.min(), np.percentile(px2, 0.5))
    ymax = max(cy.max(), np.percentile(px2, 99.5))
    lo   = min(xmin, ymin)
    hi   = max(xmax, ymax)
    mrg  = 0.04 * (hi - lo)
    return (lo - mrg, hi + mrg)

# ── Render one figure for a given scope ────────────────────────────────────────
def render(scope):
    win = compute_window(scope)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
    fig.patch.set_facecolor('white')

    # grey background frames (scoped)
    gx, gy = (pc1_all, pc2_all) if scope == 'all38' else (pc1_four, pc2_four)
    if GREY_SUBSAMPLE and len(gx) > GREY_SUBSAMPLE:
        idx = np.random.default_rng(0).choice(len(gx), GREY_SUBSAMPLE, replace=False)
        gx, gy = gx[idx], gy[idx]
    gs = GREY_STYLE[scope]
    ax.scatter(gx, gy, c='#CCCCCC', s=gs['s'], alpha=gs['alpha'],
               rasterized=True, linewidths=0, zorder=1)

    # contours (identical in both figures)
    for variant in VARIANTS:
        colour = PALETTE[variant]
        if SHOW_99:
            for seg in segments[variant]['outer']:
                ax.plot(seg[:, 0], seg[:, 1], color=colour,
                        lw=C99_LW, ls='dashed', alpha=C99_ALPHA, zorder=3)
        for seg in segments[variant]['core']:
            ax.plot(seg[:, 0], seg[:, 1], color=colour,
                    lw=2.4, ls='solid', zorder=4)

    ax.set_xlim(win)
    ax.set_ylim(win)
    ax.set_aspect('equal')
    ax.set_xlabel(f'PC1 ({PC1_VAR}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({PC2_VAR}%)', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # legend
    variant_handles = [
        Line2D([0], [0], color=PALETTE[v], lw=2.4, ls='solid', label=LABELS[v])
        for v in VARIANTS
    ]
    grey_label = 'All frames (38 systems)' if scope == 'all38' else 'Frames (4 variants)'
    style_handles = [
        Line2D([0], [0], color='grey', lw=0, marker='o',
               markerfacecolor='#CCCCCC', markersize=7, label=grey_label),
        Line2D([0], [0], color='black', lw=2.4, ls='solid', label='75% density'),
    ]
    if SHOW_99:
        style_handles.append(
            Line2D([0], [0], color='black', lw=C99_LW, ls='dashed', label='99% density'))
    leg1 = ax.legend(handles=variant_handles, loc='upper left',
                     fontsize=9, frameon=False, title='Variant')
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc='lower left', fontsize=9, frameon=False)

    ax.set_title(TITLES[scope], fontsize=12, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, OUTNAMES[scope])
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out}')

# ── Render both ────────────────────────────────────────────────────────────────
for scope in SCOPES:
    render(scope)