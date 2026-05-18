"""
Script C: Q2022P family projected onto LD1 axis — stripplot and structural colouring.

Reads per-replica LD1 means from the pairwise LDA summary CSV and plots
each system's distribution across three pairwise comparisons (WT vs each
Q2022P family member). If LDA weights and PCA loadings are available, also
back-projects LD1 onto the WT structure and renders a putty cartoon coloured
by per-residue LD1 contribution.

Usage:
    cd /homes/lkgyammerah/Molecular_Dynamics_analysis
    python analysis/pymol_C_family_ld1.py

Output:
    figures/ROS1/ros1_prepared_final/family_ld1_distributions.png
    figures/ROS1/ros1_prepared_final/ld1_on_structure.png
    figures/ROS1/ros1_prepared_final/ld1_on_structure.pse
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE    = '/homes/lkgyammerah/Molecular_Dynamics_analysis'
SCRIPTS = f'{BASE}/analysis/scripts'
RESULTS = f'{BASE}/results/ROS1'
RECEPT  = f'{BASE}/dock/ROS1/q2022p_subset_receptors/actout_ctlfit'
OUTDIR  = f'{BASE}/figures/ROS1/ros1_prepared_final'
os.makedirs(OUTDIR, exist_ok=True)
sys.path.insert(0, SCRIPTS)

PALETTE = {
    'WT':            '#1565C0',
    'Q2022P':        '#E63946',
    'Q2022P_S1986F': '#F4511E',
    'Q2022P_S1986Y': '#F9A825',
}
LABELS = {
    'WT':            'WT',
    'Q2022P':        'Q2022P',
    'Q2022P_S1986F': 'Q2022P\nS1986F',
    'Q2022P_S1986Y': 'Q2022P\nS1986Y',
}

# ── Load LD1 replica-mean data ─────────────────────────────────────────────────
lda_file = f'{RESULTS}/step4c_pairwise_lda_WT_frame_summary.csv'
print(f"Loading: {lda_file}")
df = pd.read_csv(lda_file)
print(f"Columns: {list(df.columns)}")
print(df)

SYSTEMS      = ['WT', 'Q2022P', 'Q2022P_S1986F', 'Q2022P_S1986Y']
COMPARISONS  = ['WT_vs_Q2022P', 'WT_vs_Q2022P_S1986F', 'WT_vs_Q2022P_S1986Y']
COMP_LABELS  = ['WT vs Q2022P', 'WT vs Q2022P\nS1986F', 'WT vs Q2022P\nS1986Y']

# ── Stripplot — one panel per pairwise comparison ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=False)
fig.suptitle('Q2022P family: replica LD1 projections\n'
             '(each point = one replica mean; WT and mutant on each pairwise LD1 axis)',
             fontsize=10, fontweight='bold')

np.random.seed(42)

for ax, comp, comp_label in zip(axes, COMPARISONS, COMP_LABELS):
    sub = df[df['comparison'] == comp].copy()

    systems_in_comp = sub['mutant'].unique().tolist()
    ordered = [s for s in ['WT'] + [c for c in SYSTEMS if c != 'WT'] if s in systems_in_comp]
    x_pos   = {sys: i for i, sys in enumerate(ordered)}

    for sys in ordered:
        vals  = sub[sub['mutant'] == sys]['LD1'].values
        color = PALETTE[sys]
        x     = x_pos[sys]

        jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals,
                   color=color, s=60, zorder=3, alpha=0.9,
                   edgecolors='white', linewidths=0.5)

        # Mean line with ±1 SD shading
        ax.hlines(vals.mean(), x - 0.25, x + 0.25,
                  color=color, linewidth=2.5, zorder=4)
        ax.fill_between([x - 0.25, x + 0.25],
                        vals.mean() - vals.std(), vals.mean() + vals.std(),
                        color=color, alpha=0.15, zorder=2)

    ax.set_title(comp_label, fontsize=9, fontweight='bold')
    ax.set_ylabel('LD1 score', fontsize=9)
    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([LABELS[s] for s in ordered], fontsize=8)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
out_path = f'{OUTDIR}/family_ld1_distributions.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out_path}")

# ── Back-project LD1 onto structure using PCA loadings and LDA weights ─────────
lda_weights_file = f'{RESULTS}/step4b_pairwise_lda_wt_vs_family_weights.csv'
loadings_file    = f'{RESULTS}/actout_ctlfit_loadings.npy'

if os.path.exists(lda_weights_file) and os.path.exists(loadings_file):
    print("Computing per-residue LD1 contribution...")

    lda_weights = pd.read_csv(lda_weights_file)
    print(f"LDA weights columns: {list(lda_weights.columns)}")
    print(lda_weights.head())

    loadings = np.load(loadings_file)
    print(f"Loadings shape: {loadings.shape}")

    n_atoms = loadings.shape[0] // 3
    n_pcs   = loadings.shape[1]

    # Extract WT vs Q2022P weights sorted by PC index
    wt_q22      = lda_weights[lda_weights['pair'] == 'WT_vs_Q2022P'].copy()
    wt_q22      = wt_q22.sort_values('PC')
    weight_vals = wt_q22['LD1_weight'].values.astype(float)
    print(f"WT_vs_Q2022P weights ({len(weight_vals)} PCs): {weight_vals[:5]}...")

    n_use = min(len(weight_vals), n_pcs)
    w     = weight_vals[:n_use]
    L     = loadings[:, :n_use]

    # Sum PC contributions per residue, weighted by LD1 loading magnitude
    L_reshaped  = L.reshape(n_atoms, 3, n_use)
    per_res_mag = np.sqrt((L_reshaped**2).sum(axis=1))
    ld1_per_res = np.abs(per_res_mag @ np.abs(w))
    ld1_per_res /= ld1_per_res.max()

    print(f"LD1 per-residue: {n_atoms} atoms, max={ld1_per_res.max():.3f}, min={ld1_per_res.min():.3f}")

    # ── Colour structure in PyMOL ──────────────────────────────────────────────
    print("Colouring structure in PyMOL...")
    from pymol import cmd, stored

    WT_PDB = f'{RECEPT}/WT/WT_actout_ctlfit_dominant_cluster0_rep.pdb'
    cmd.load(WT_PDB, 'WT_ld1')
    cmd.hide('everything', 'WT_ld1')
    cmd.show('cartoon', 'WT_ld1')

    # Collect Cα atoms in chain/residue order
    stored.atoms = []
    cmd.iterate('WT_ld1 and name CA',
                'stored.atoms.append((chain, resi))',
                space={'stored': stored})

    ca_atoms = stored.atoms
    n_ca     = len(ca_atoms)
    print(f"Ca atoms in structure: {n_ca}, in loadings: {n_atoms}")

    n_use2 = min(n_ca, n_atoms)

    # Store LD1 contribution in B-factor field (scaled 0-100) for spectrum colouring
    for i, (chain, resi) in enumerate(ca_atoms[:n_use2]):
        b_val = float(ld1_per_res[i]) * 100
        if chain:
            cmd.alter(f'WT_ld1 and chain {chain} and resi {resi}', f'b={b_val:.2f}')
        else:
            cmd.alter(f'WT_ld1 and resi {resi}', f'b={b_val:.2f}')

    cmd.spectrum('b', 'blue_white_red', 'WT_ld1', minimum=0, maximum=100)

    # Putty cartoon scales tube radius by B-factor (LD1 contribution)
    cmd.cartoon('putty', 'WT_ld1')
    cmd.set('cartoon_putty_scale_min', 0.3, 'WT_ld1')
    cmd.set('cartoon_putty_scale_max', 2.5, 'WT_ld1')
    cmd.set('cartoon_putty_transform', 0,   'WT_ld1')

    cmd.bg_color('white')
    cmd.set('ray_shadows', 0)
    cmd.set('ray_opaque_background', 1)
    cmd.set('antialias', 2)
    cmd.orient('WT_ld1')
    cmd.zoom('all', buffer=8)

    cmd.ray(1600, 1200)
    cmd.png(f'{OUTDIR}/ld1_on_structure.png', dpi=150)
    print(f"Saved: {OUTDIR}/ld1_on_structure.png")

    cmd.rotate('y', 90)
    cmd.ray(1600, 1200)
    cmd.png(f'{OUTDIR}/ld1_on_structure_rotated.png', dpi=150)
    print(f"Saved: {OUTDIR}/ld1_on_structure_rotated.png")

    cmd.save(f'{OUTDIR}/ld1_on_structure.pse')
    print(f"Saved session: {OUTDIR}/ld1_on_structure.pse")
    cmd.quit()

else:
    print("Missing files for structural projection:")
    print(f"  LDA weights: {os.path.exists(lda_weights_file)} — {lda_weights_file}")
    print(f"  Loadings:    {os.path.exists(loadings_file)} — {loadings_file}")
    print("Only the LD1 distribution figure was saved.")

print("\nDone. Output files:")
print(f"  {OUTDIR}/family_ld1_distributions.png")
print(f"  {OUTDIR}/ld1_on_structure.png")