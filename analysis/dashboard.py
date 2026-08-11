"""
ROS1 Kinase Variant MD Analysis — Streamlit Dashboard

Interactive explorer for the conformational analysis of 38 ROS1 kinase domain
resistance mutations. Loads real data from the cluster when available; falls
back to representative dummy data for standalone / GitHub demo use.

Panels:
    1. Per-mutant FEL explorer (actout_ctlfit PC1/PC2)
    2. Q2022P family comparison (FEL + LD1 distributions)
    3. Docking results (crizotinib and lorlatinib, all poses)
    4. Panel-wide heatmap (39 systems x key metrics)

Usage:
    streamlit run dashboard.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.cm as cm

warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ROS1 MD Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; letter-spacing: -0.02em; }
.metric-box {
    background: #F8F9FA; border-left: 4px solid #1565C0;
    padding: 12px 16px; border-radius: 4px; margin-bottom: 8px;
    font-family: 'DM Mono', monospace; font-size: 13px;
}
.panel-header {
    font-family: 'DM Serif Display', serif; font-size: 1.4rem;
    color: #1A1A2E; border-bottom: 2px solid #E63946;
    padding-bottom: 6px; margin-bottom: 16px;
}
.sidebar-note { font-size: 11px; color: #78909C; font-family: 'DM Mono', monospace; }
.data-badge {
    display: inline-block; padding: 2px 10px; border-radius: 10px;
    font-size: 11px; font-weight: 500; font-family: 'DM Mono', monospace;
}
.badge-real  { background: #E8F5E9; color: #2E7D32; }
.badge-demo  { background: #FFF8E1; color: #F57F17; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE = '/homes/lkgyammerah/Molecular_Dynamics_analysis/results/ROS1'

MUTANTS = [
    'WT', 'C2060G', 'D1988N', 'D2033N', 'D2113G', 'D2113N',
    'E1935G', 'E1990G', 'E2020K', 'E2131Q',
    'F2004C', 'F2004L', 'F2004V', 'F2075V',
    'G1957A', 'G1971E', 'G2032K', 'G2032R', 'G2048A_NC', 'G2101A',
    'H1999Q_NC', 'L1947R', 'L1951R', 'L1982F', 'L1982V',
    'L2010M', 'L2026M', 'L2053V_NC', 'L2086F', 'L2155S',
    'Q2022P', 'Q2022P_S1986F', 'Q2022P_S1986Y',
    'R2078K', 'S1986F', 'S1986Y', 'V2089M', 'V2098I',
]
NC_VARIANTS = {'G2048A_NC', 'H1999Q_NC', 'L2053V_NC'}
Q2022P_FAM  = {'WT', 'Q2022P', 'Q2022P_S1986F', 'Q2022P_S1986Y'}
Q_ORDER     = ['WT', 'Q2022P', 'Q2022P_S1986F', 'Q2022P_S1986Y']

PALETTE = {
    'WT': '#1565C0',
    'Q2022P': '#E63946', 'Q2022P_S1986F': '#F4511E', 'Q2022P_S1986Y': '#F9A825',
    'S1986F': '#2E7D32', 'S1986Y': '#66BB6A',
    'L1982F': '#E91E63', 'L1982V': '#AD1457',
    'G2032K': '#7B1FA2', 'G2032R': '#CE93D8',
    'F2004C': '#00838F', 'F2004L': '#00BCD4', 'F2004V': '#80DEEA',
    'D2113G': '#E65100', 'D2113N': '#FF8F00',
    'L1947R': '#F50057', 'L1951R': '#FF4081',
    'C2060G': '#6A1B9A', 'D1988N': '#00695C', 'D2033N': '#43A047',
    'E1935G': '#FB8C00', 'E1990G': '#039BE5', 'E2020K': '#0277BD',
    'E2131Q': '#283593', 'F2075V': '#FF6F00', 'G1957A': '#00897B',
    'G1971E': '#00ACC1', 'G2101A': '#D81B60', 'L2010M': '#558B2F',
    'L2026M': '#EF6C00', 'L2086F': '#4527A0', 'L2155S': '#37474F',
    'R2078K': '#006064', 'V2089M': '#C62828', 'V2098I': '#1A237E',
    'G2048A_NC': '#5C6BC0', 'H1999Q_NC': '#7E57C2', 'L2053V_NC': '#AB47BC',
}
DEFAULT_C = '#78909C'
FEL_CMAP  = cm.viridis_r
FEL_VMAX  = 6.2   # kJ/mol (was 2.5 kT); RT ~ 2.49 kJ/mol at 300 K

# Confirmed values from thesis (used as fallback when files not present)
NOISE_FRACTIONS = {
    'L1982V': 7.27, 'G2048A_NC': 5.21, 'L1951R': 5.18, 'F2004C': 5.14,
    'Q2022P': 4.51, 'S1986Y': 4.31, 'E2131Q': 4.28, 'V2098I': 3.61,
    'L2086F': 3.50, 'E2020K': 2.50, 'L2155S': 2.26, 'Q2022P_S1986Y': 2.07,
    'L1947R': 1.95, 'H1999Q_NC': 1.64, 'Q2022P_S1986F': 1.53, 'L1982F': 1.45,
    'L2053V_NC': 1.36, 'C2060G': 1.27, 'F2004L': 1.26, 'D1988N': 1.24,
    'V2089M': 1.20, 'R2078K': 0.95, 'G2101A': 0.84, 'WT': 0.83,
    'E1990G': 0.83, 'L2010M': 0.82, 'D2033N': 0.79, 'G2032R': 0.68,
    'F2075V': 0.55, 'F2004V': 0.44, 'E1935G': 0.34, 'G1957A': 0.30,
    'S1986F': 0.25, 'G2032K': 0.21, 'G1971E': 0.10, 'D2113G': 0.16,
    'D2113N': 0.08,
}

DOCKING_CONFIRMED = {
    'WT':            {'crizotinib': -8.644, 'lorlatinib': -7.200},
    'Q2022P':        {'crizotinib': -8.463, 'lorlatinib': -8.224},
    'Q2022P_S1986F': {'crizotinib': -6.455, 'lorlatinib': -6.106},
    'Q2022P_S1986Y': {'crizotinib': -5.508, 'lorlatinib': -6.369},
}

ATP_CONFIRMED = {
    'WT': 84, 'Q2022P': 49, 'D2113N': 27,
    'E2020K': 89, 'G2048A_NC': 89, 'G2032K': 89,
}


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    """
    Load all pipeline outputs from the cluster. Falls back to representative
    dummy data if the cluster paths are not reachable (e.g. running locally
    for the GitHub demo).
    """
    data = {}

    fp_path = os.path.join(BASE, 'actout_ctlfit_fel_frame_points.csv')

    if os.path.exists(fp_path):
        data['real'] = True
        fp = pd.read_csv(fp_path)
        data['fp']         = fp
        data['pc1_all']    = fp['PC1'].values
        data['pc2_all']    = fp['PC2'].values
        data['mutant_col'] = fp['mutant'].values

        # DBSCAN noise fractions
        db_path = os.path.join(BASE, 'actout_ctlfit_cluster_frac_db.csv')
        if os.path.exists(db_path):
            df_db     = pd.read_csv(db_path)
            noise_col = [c for c in df_db.columns if 'noise' in c.lower() or 'frac' in c.lower()]
            mut_col   = [c for c in df_db.columns if 'mutant' in c.lower() or 'variant' in c.lower()]
            if noise_col and mut_col:
                data['noise'] = dict(zip(df_db[mut_col[0]], df_db[noise_col[0]] * 100))
            else:
                data['noise'] = NOISE_FRACTIONS
        else:
            data['noise'] = NOISE_FRACTIONS

        # ATP pocket metrics
        atp_path = os.path.join(BASE, 'step3a_atp_pocket_metrics_by_mutant.csv')
        if os.path.exists(atp_path):
            df_atp = pd.read_csv(atp_path)
            df_atp['pocket_open_pct'] = df_atp['pocket_open_frac'] * 100
            data['atp'] = df_atp
        else:
            data['atp'] = None

        # Docking results
        all_path  = os.path.join(BASE, 'step5c_q2022p_vina_all_poses.csv')
        best_path = os.path.join(BASE, 'step5c_q2022p_vina_best_by_mutant_ligand.csv')
        data['docking_all']  = pd.read_csv(all_path)  if os.path.exists(all_path)  else None
        data['docking_best'] = pd.read_csv(best_path) if os.path.exists(best_path) else None

        # LDA replica means
        lda_path = os.path.join(BASE, 'step4c_pairwise_lda_WT_frame_summary.csv')
        data['lda'] = pd.read_csv(lda_path) if os.path.exists(lda_path) else None

    else:
        # Dummy data — representative values for GitHub demo
        data['real'] = False
        np.random.seed(42)

        rows = []
        for mut in MUTANTS:
            n  = np.random.randint(2500, 4500)
            cx = np.random.uniform(-0.04, 0.04)
            cy = np.random.uniform(-0.04, 0.04)
            if mut == 'Q2022P':
                cy = -0.07
            elif mut in ('Q2022P_S1986F', 'Q2022P_S1986Y'):
                cy = -0.05; cx -= 0.03
            pc1 = np.random.normal(cx, 0.03, n)
            pc2 = np.random.normal(cy, 0.025, n)
            for p1, p2 in zip(pc1, pc2):
                rows.append({'mutant': mut, 'PC1': p1, 'PC2': p2})

        fp = pd.DataFrame(rows)
        data['fp']         = fp
        data['pc1_all']    = fp['PC1'].values
        data['pc2_all']    = fp['PC2'].values
        data['mutant_col'] = fp['mutant'].values
        data['noise']      = NOISE_FRACTIONS

        # ATP dummy
        atp_rows = []
        for mut in MUTANTS:
            raw = ATP_CONFIRMED.get(mut, float(np.random.uniform(40, 90)))
            atp_rows.append({
                'mutant':            mut,
                'pocket_open_frac':  raw / 100,
                'pocket_open_pct':   raw,
                'pocket_front_mean': float(np.random.uniform(2.7, 3.0)),
                'pocket_dfg_mean':   float(np.random.uniform(1.8, 2.4)),
            })
        data['atp'] = pd.DataFrame(atp_rows)

        # Docking dummy — 20 poses per system/ligand
        dock_rows = []
        for mut in Q_ORDER:
            for lig in ['crizotinib', 'lorlatinib']:
                best = DOCKING_CONFIRMED[mut][lig]
                for mode in range(1, 21):
                    dock_rows.append({
                        'mutant': mut, 'ligand': lig,
                        'mode': mode,
                        'affinity_kcal_mol': best + float(np.random.uniform(0, 2.5)),
                    })
        data['docking_all'] = pd.DataFrame(dock_rows)

        best_rows = [
            {'mutant': m, 'ligand': l, 'affinity_kcal_mol': DOCKING_CONFIRMED[m][l]}
            for m in Q_ORDER for l in ['crizotinib', 'lorlatinib']
        ]
        data['docking_best'] = pd.DataFrame(best_rows)

        # LDA dummy
        lda_rows = []
        for comp, mut in [
            ('WT_vs_Q2022P',        'Q2022P'),
            ('WT_vs_Q2022P_S1986F', 'Q2022P_S1986F'),
            ('WT_vs_Q2022P_S1986Y', 'Q2022P_S1986Y'),
        ]:
            for system in ['WT', mut]:
                sign = -1 if system == 'WT' else 1
                for _ in range(3):
                    lda_rows.append({
                        'mutant': system, 'comparison': comp,
                        'LD1': sign * float(np.random.uniform(0.003, 0.012)),
                    })
        data['lda'] = pd.DataFrame(lda_rows)

    return data


# ── FEL helpers ───────────────────────────────────────────────────────────────
def compute_fel(pc1, pc2, bins=100, vmax=FEL_VMAX):
    """2D free energy landscape via Gaussian KDE (matches thesis script 03c)."""
    from scipy.stats import gaussian_kde
    pc1 = np.asarray(pc1); pc2 = np.asarray(pc2)
    m = np.isfinite(pc1) & np.isfinite(pc2)
    pc1, pc2 = pc1[m], pc2[m]
    xc = np.linspace(pc1.min(), pc1.max(), bins)
    yc = np.linspace(pc2.min(), pc2.max(), bins)
    Xc, Yc = np.meshgrid(xc, yc)
    # KDE needs enough spread; fall back to histogram if too few/degenerate
    try:
        if pc1.size < 5 or np.std(pc1) == 0 or np.std(pc2) == 0:
            raise ValueError
        k = gaussian_kde(np.vstack([pc1, pc2]))
        dens = k(np.vstack([Xc.ravel(), Yc.ravel()])).reshape(Xc.shape)
    except Exception:
        h, xe, ye = np.histogram2d(pc1, pc2, bins=bins)
        dens = h.T
        xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    dens = np.clip(dens / np.nanmax(dens), 1e-10, 1.0)
    G = -0.0083144626 * 300.0 * np.log(dens)  # kJ/mol
    G[G > vmax] = vmax
    return G, xc, yc


def plot_fel(ax, pc1, pc2, title, xlim, ylim, bins=100):
    """Render a FEL onto an axes object."""
    G, xc, yc = compute_fel(pc1, pc2, bins=bins)
    ax.pcolormesh(xc, yc, G.T, cmap=FEL_CMAP, vmin=0, vmax=FEL_VMAX, rasterized=True)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=8)
    ax.set_ylabel('PC2', fontsize=8)
    ax.tick_params(labelsize=7)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 ROS1 MD Analysis")
    st.markdown(
        "**Lily Konadu Gyammerah**  \n"
        "MSc Data Science for Life Sciences  \n"
        "Hanze University of Applied Sciences"
    )
    st.markdown("---")
    panel = st.radio(
        "Panel",
        ["FEL Explorer", "Q2022P Family", "Docking Results", "Panel Heatmap"],
    )
    st.markdown("---")
    st.markdown(
        "**Dataset**  \n"
        "38 resistance mutations + WT  \n"
        "117 simulations × ~500 ns  \n"
        "Aggregate: ~62 µs  \n"
        "Force field: AMBER99SB-ILDN  \n"
        "Software: GROMACS 2024.5"
    )
    st.markdown("---")
    st.markdown(
        '<span class="sidebar-note">'
        '<a href="https://github.com/snowwhitelily/Molecular_Dynamics_analysis" '
        'target="_blank">GitHub repository</a></span>',
        unsafe_allow_html=True,
    )

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    D = load_data()

badge_class = "badge-real" if D['real'] else "badge-demo"
badge_text  = "Real cluster data" if D['real'] else "Demo data (cluster files not found — showing representative values)"
st.markdown(f'<span class="data-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
st.markdown("")

pc1_all    = D['pc1_all']
pc2_all    = D['pc2_all']
mutant_col = D['mutant_col']

pad = 0.05
x0, x1 = np.percentile(pc1_all, 0.5), np.percentile(pc1_all, 99.5)
y0, y1 = np.percentile(pc2_all, 0.5), np.percentile(pc2_all, 99.5)
XLIM = (x0 - pad*(x1-x0), x1 + pad*(x1-x0))
YLIM = (y0 - pad*(y1-y0), y1 + pad*(y1-y0))


# ═════════════════════════════════════════════════════════════════════════════
# PANEL 1 — FEL EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
if panel == "FEL Explorer":
    st.markdown('<div class="panel-header">Per-Mutant Free Energy Landscape Explorer</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Select any variant to see its conformational landscape in the *actout_ctlfit* "
        "PCA space (CTL-aligned, NTL motion explicit). Fixed colour scale (0–2.5 kT) and "
        "fixed axes allow direct comparison across all 39 systems. "
        "Dark purple = energy minimum (most sampled); yellow = rarely visited."
    )

    col_sel, col_plot = st.columns([1, 2])

    with col_sel:
        focal           = st.selectbox("Select variant", MUTANTS, index=MUTANTS.index('Q2022P'))
        show_wt_overlay = st.checkbox("Overlay WT contour", value=True)
        show_grey       = st.checkbox("Grey highlight view", value=False)

        nc_flag  = " [NC]" if focal in NC_VARIANTS else ""
        classify = (
            "Reference"              if focal == 'WT'         else
            "Negative control"       if focal in NC_VARIANTS  else
            "Compound / suppressor"  if focal in ('Q2022P_S1986F', 'Q2022P_S1986Y') else
            "Resistance mutant"
        )
        noise_val = D['noise'].get(focal, None)
        atp_val   = None
        if D['atp'] is not None:
            row = D['atp'][D['atp']['mutant'] == focal]
            if not row.empty:
                atp_val = float(row['pocket_open_pct'].values[0])

        st.markdown(f"""
<div class="metric-box">
Variant: <b>{focal}{nc_flag}</b><br>
Class: {classify}<br>
DBSCAN noise: {f"{noise_val:.2f}%" if noise_val is not None else "—"}<br>
ATP pocket open: {f"{atp_val:.0f}%" if atp_val is not None else "—"}
</div>""", unsafe_allow_html=True)

    with col_plot:
        mask_f  = mutant_col == focal
        mask_wt = mutant_col == 'WT'

        if show_grey:
            fig, ax = plt.subplots(figsize=(5, 5), dpi=130)
            fig.patch.set_facecolor('white')
            ax.scatter(pc1_all[~mask_f], pc2_all[~mask_f],
                       c='#CCCCCC', s=1.5, alpha=0.4, rasterized=True, linewidths=0)
            colour = PALETTE.get(focal, DEFAULT_C)
            ax.scatter(pc1_all[mask_f], pc2_all[mask_f],
                       c=colour, s=3, alpha=0.95, rasterized=True, linewidths=0)
            ax.set_xlim(XLIM); ax.set_ylim(YLIM)
            ax.set_xlabel('PC1 (actout_ctlfit)', fontsize=9)
            ax.set_ylabel('PC2 (actout_ctlfit)', fontsize=9)
            ax.set_title(f'{focal}{nc_flag} — grey highlight', fontsize=10, fontweight='bold')
            handles = [
                Line2D([0],[0], marker='o', color='w', markerfacecolor='#CCCCCC',
                       markersize=8, label='All other variants'),
                Line2D([0],[0], marker='o', color='w', markerfacecolor=colour,
                       markersize=8, label=f'{focal}{nc_flag}'),
            ]
            ax.legend(handles=handles, fontsize=8, frameon=False)
            for sp in ['top','right']: ax.spines[sp].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=130)
            fig.patch.set_facecolor('white')
            plot_fel(axes[0], pc1_all[mask_f], pc2_all[mask_f],
                     f'{focal}{nc_flag}', XLIM, YLIM, bins=120)
            if show_wt_overlay and focal != 'WT':
                G_wt, xc, yc = compute_fel(pc1_all[mask_wt], pc2_all[mask_wt], bins=120)
                axes[0].contour(xc, yc, G_wt.T, levels=[0.5, 1.0, 1.5],
                                colors='white', linewidths=0.8, alpha=0.6)
            plot_fel(axes[1], pc1_all[mask_wt], pc2_all[mask_wt],
                     'WT (reference)', XLIM, YLIM, bins=120)
            sm = plt.cm.ScalarMappable(cmap=FEL_CMAP, norm=plt.Normalize(vmin=0, vmax=FEL_VMAX))
            sm.set_array([])
            fig.subplots_adjust(right=0.88)
            cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label('Relative free energy (kJ/mol)', fontsize=8)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("---")
    st.markdown("#### All 39 systems — FEL panel")
    n_cols, n_rows = 6, 7
    fig_p, axes_p = plt.subplots(n_rows, n_cols,
                                  figsize=(n_cols*2.8, n_rows*2.4), dpi=100)
    fig_p.patch.set_facecolor('white')
    for i, name in enumerate(MUTANTS):
        ax   = axes_p[i // n_cols, i % n_cols]
        mask = mutant_col == name
        if mask.sum() > 10:
            plot_fel(ax, pc1_all[mask], pc2_all[mask], name, XLIM, YLIM, bins=60)
        ax.set_xlabel(''); ax.set_ylabel('')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=6.5, fontweight='bold', pad=2)
    for j in range(len(MUTANTS), n_rows * n_cols):
        axes_p[j // n_cols, j % n_cols].axis('off')
    fig_p.suptitle(
        'actout_ctlfit FEL — All 39 systems\n'
        'Fixed scale 0–2.5 kT | Dark purple = energy minimum | Yellow = rarely visited',
        fontsize=10, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    st.pyplot(fig_p, use_container_width=True)
    plt.close(fig_p)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL 2 — Q2022P FAMILY
# ═════════════════════════════════════════════════════════════════════════════
elif panel == "Q2022P Family":
    st.markdown('<div class="panel-header">Q2022P Compound Mutation Family</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Side-by-side conformational comparison of WT, Q2022P, Q2022P_S1986F, and "
        "Q2022P_S1986Y. LDA shows perfect replica-level separation between WT and each "
        "family member in all three pairwise comparisons. "
        "Cohen's d: WT vs Q2022P = −0.748 | vs S1986F = −0.464 | vs S1986Y = −0.792."
    )

    st.markdown("#### Free energy landscapes — actout_ctlfit")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), dpi=130)
    fig.patch.set_facecolor('white')
    for ax, name in zip(axes, Q_ORDER):
        mask = mutant_col == name
        plot_fel(ax, pc1_all[mask], pc2_all[mask], name, XLIM, YLIM, bins=120)
        ax.set_title(name, fontsize=9, fontweight='bold', color=PALETTE[name])
    sm = plt.cm.ScalarMappable(cmap=FEL_CMAP, norm=plt.Normalize(vmin=0, vmax=FEL_VMAX))
    sm.set_array([])
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Relative free energy (kJ/mol)', fontsize=8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("#### LDA replica projections — pairwise WT vs family")
    if D['lda'] is not None:
        df_lda      = D['lda']
        COMPARISONS = ['WT_vs_Q2022P', 'WT_vs_Q2022P_S1986F', 'WT_vs_Q2022P_S1986Y']
        COMP_LABELS = ['WT vs Q2022P', 'WT vs Q2022P\nS1986F', 'WT vs Q2022P\nS1986Y']

        fig, axes = plt.subplots(1, 3, figsize=(11, 4), dpi=130)
        fig.patch.set_facecolor('white')
        np.random.seed(42)

        for ax, comp, comp_label in zip(axes, COMPARISONS, COMP_LABELS):
            sub     = df_lda[df_lda['comparison'] == comp].copy()
            systems = [s for s in Q_ORDER if s in sub['mutant'].unique()]
            x_pos   = {s: i for i, s in enumerate(systems)}

            for sys in systems:
                vals   = sub[sub['mutant'] == sys]['LD1'].values
                colour = PALETTE[sys]
                x      = x_pos[sys]
                jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
                ax.scatter(np.full(len(vals), x) + jitter, vals,
                           color=colour, s=60, zorder=3, alpha=0.9,
                           edgecolors='white', linewidths=0.5)
                ax.hlines(vals.mean(), x - 0.25, x + 0.25,
                          color=colour, linewidth=2.5, zorder=4)
                ax.fill_between([x - 0.25, x + 0.25],
                                vals.mean() - vals.std(), vals.mean() + vals.std(),
                                color=colour, alpha=0.15, zorder=2)

            ax.set_title(comp_label, fontsize=9, fontweight='bold')
            ax.set_ylabel('LD1 score', fontsize=9)
            ax.set_xticks(list(x_pos.values()))
            ax.set_xticklabels(systems, fontsize=8, rotation=10)
            ax.axhline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
            for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

        fig.suptitle(
            'Q2022P family: replica LD1 projections\n'
            'Each point = one replica mean | Perfect WT/mutant separation in all three comparisons',
            fontsize=9, fontweight='bold',
        )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("LDA data not available.")

    st.markdown("#### Summary — conformational and docking characteristics")
    st.dataframe(pd.DataFrame({
        'System':                Q_ORDER,
        'ATP pocket open (%)':   [ATP_CONFIRMED.get(m, '—') for m in Q_ORDER],
        'FEL character':         ['Reference basin', 'Neg. PC2 shift',
                                  'Tight single basin', 'Two-state CTL'],
        "Cohen's d vs WT":       ['—', '−0.748', '−0.464', '−0.792'],
        'Crizotinib (kcal/mol)': [DOCKING_CONFIRMED[m]['crizotinib'] for m in Q_ORDER],
        'Lorlatinib (kcal/mol)': [DOCKING_CONFIRMED[m]['lorlatinib'] for m in Q_ORDER],
    }), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL 3 — DOCKING RESULTS
# ═════════════════════════════════════════════════════════════════════════════
elif panel == "Docking Results":
    st.markdown('<div class="panel-header">Docking Results — Q2022P Family</div>',
                unsafe_allow_html=True)
    st.markdown(
        "AutoDock Vina v1.2.5 ensemble docking of crizotinib and lorlatinib against "
        "dominant-cluster medoid structures of the Q2022P family. Each point = one of "
        "20 docking poses (jittered horizontally); horizontal bar = best pose. "
        "Y-axis inverted: upward = stronger predicted binding. "
        "Box: 25×20×20 Å centred at x=68.60, y=64.50, z=23.68 Å. Exhaustiveness=16."
    )

    if D['docking_all'] is not None:
        df_all  = D['docking_all']
        df_best = D['docking_best']
        np.random.seed(42)

        fig, axes = plt.subplots(1, 2, figsize=(11, 6), dpi=130)
        fig.patch.set_facecolor('white')

        for ax, lig in zip(axes, ['crizotinib', 'lorlatinib']):
            sub_all  = df_all[df_all['ligand'] == lig]
            sub_best = df_best[df_best['ligand'] == lig]

            for i, mutant in enumerate(Q_ORDER):
                poses  = sub_all[sub_all['mutant'] == mutant]['affinity_kcal_mol'].values
                colour = PALETTE[mutant]
                jitter = np.random.uniform(-0.18, 0.18, len(poses))
                ax.scatter(i + jitter, poses, color=colour,
                           alpha=0.6, s=22, zorder=2, linewidths=0)
                ax.plot([i - 0.32, i + 0.32], [poses.min(), poses.min()],
                        color=colour, lw=2.5, zorder=3, solid_capstyle='round')

            ax.set_xticks(range(len(Q_ORDER)))
            ax.set_xticklabels(Q_ORDER, rotation=15, ha='right', fontsize=9)
            ax.set_ylabel('Predicted binding affinity (kcal/mol)', fontsize=10)
            ax.set_title(f'{lig.capitalize()} — all 20 docking poses',
                         fontsize=11, fontweight='bold')
            ax.invert_yaxis()

            wt_row = sub_best[sub_best['mutant'] == 'WT']
            if not wt_row.empty:
                wt_best = wt_row['affinity_kcal_mol'].values[0]
                ax.axhline(wt_best, color='#1565C0', lw=1.2, ls='--',
                           alpha=0.5, label=f'WT best ({wt_best:.3f} kcal/mol)')
                ax.legend(fontsize=8, frameon=False)

            for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

        fig.suptitle(
            'AutoDock Vina — All 20 Docking Poses per System\n'
            'Horizontal bar = best pose  |  Dots = all poses (jittered)  |  '
            'Y-axis inverted: upward = stronger binding',
            fontsize=11, fontweight='bold',
        )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("#### Best-pose affinities (kcal/mol)")
        st.dataframe(pd.DataFrame({
            'System':                Q_ORDER,
            'Crizotinib (kcal/mol)': [DOCKING_CONFIRMED[m]['crizotinib'] for m in Q_ORDER],
            'Lorlatinib (kcal/mol)': [DOCKING_CONFIRMED[m]['lorlatinib'] for m in Q_ORDER],
        }), use_container_width=True, hide_index=True)

        st.markdown(
            "**Interpretation:**  \n"
            "Q2022P near-WT crizotinib affinity (−8.463 vs −8.644 kcal/mol) supports a "
            "**population-shift** resistance mechanism — resistance arises from reduced "
            "conformational accessibility of the open state, not from reduced affinity "
            "within that state.  \n"
            "Q2022P above-WT lorlatinib affinity (−8.224 vs −7.200 kcal/mol) is consistent "
            "with the compact pocket being more complementary to lorlatinib's macrocyclic scaffold.  \n"
            "Compound mutations reduce affinity for both drugs, consistent with clinical compound resistance."
        )
    else:
        st.info("Docking files not found. Run pipeline steps 05a through 05c to generate results.")


# ═════════════════════════════════════════════════════════════════════════════
# PANEL 4 — PANEL-WIDE HEATMAP
# ═════════════════════════════════════════════════════════════════════════════
elif panel == "Panel Heatmap":
    st.markdown('<div class="panel-header">Panel-Wide Metrics Heatmap — 39 Systems</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Key conformational metrics for all 38 resistance mutations and WT. "
        "Colour within each column is normalised independently (0 = column minimum, "
        "1 = column maximum). Variant labels are coloured by the thesis palette; "
        "bold = WT and Q2022P family."
    )

    rows = []
    for mut in MUTANTS:
        mask     = mutant_col == mut
        pc1_mean = float(np.mean(pc1_all[mask])) if mask.sum() > 0 else np.nan
        pc2_mean = float(np.mean(pc2_all[mask])) if mask.sum() > 0 else np.nan
        noise    = D['noise'].get(mut, np.nan)
        atp      = np.nan
        if D['atp'] is not None:
            row_atp = D['atp'][D['atp']['mutant'] == mut]
            if not row_atp.empty:
                atp = float(row_atp['pocket_open_pct'].values[0])
        rows.append({
            'Variant':          mut,
            'PC1 mean':         round(pc1_mean, 4),
            'PC2 mean':         round(pc2_mean, 4),
            'DBSCAN noise (%)': noise,
            'ATP open (%)':     atp,
            'N frames':         int(mask.sum()),
        })

    df_summary  = pd.DataFrame(rows)
    metric_cols = ['PC1 mean', 'PC2 mean', 'DBSCAN noise (%)', 'ATP open (%)']
    df_heat     = df_summary.set_index('Variant')[metric_cols].copy()
    df_norm     = (df_heat - df_heat.min()) / (df_heat.max() - df_heat.min())

    fig, ax = plt.subplots(figsize=(8, 14), dpi=120)
    fig.patch.set_facecolor('white')
    im = ax.imshow(df_norm.values, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)

    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, fontsize=9, fontweight='bold',
                       rotation=20, ha='right')
    ax.set_yticks(range(len(MUTANTS)))
    ax.set_yticklabels(MUTANTS, fontsize=8)

    for i, mut in enumerate(MUTANTS):
        ax.get_yticklabels()[i].set_color(PALETTE.get(mut, DEFAULT_C))
        ax.get_yticklabels()[i].set_fontweight(
            'bold' if mut in Q2022P_FAM else 'normal'
        )

    for i, mut in enumerate(MUTANTS):
        for j, col in enumerate(metric_cols):
            val = df_heat.loc[mut, col]
            if not np.isnan(val):
                txt = f'{val:.3f}' if 'PC' in col else f'{val:.1f}'
                dark = df_norm.loc[mut, col] > 0.6
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=6, color='white' if dark else 'black')

    cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_label('Normalised value (within column)', fontsize=8)
    ax.set_title(
        'Panel-wide conformational metrics — 39 systems\n'
        'Colour normalised per column | Bold labels = WT and Q2022P family',
        fontsize=10, fontweight='bold', pad=12,
    )
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    st.markdown("#### Full metrics table")
    st.dataframe(
        df_summary.style.format({
            'PC1 mean':         '{:.4f}',
            'PC2 mean':         '{:.4f}',
            'DBSCAN noise (%)': '{:.2f}',
            'ATP open (%)':     '{:.0f}',
        }),
        use_container_width=True,
        hide_index=True,
    )

    csv = df_summary.to_csv(index=False)
    st.download_button(
        label="Download summary table (CSV)",
        data=csv,
        file_name="ROS1_variant_summary.csv",
        mime="text/csv",
    )