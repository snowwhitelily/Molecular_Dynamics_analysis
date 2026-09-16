"""
render_binding_annotated.py -- publication-style annotated binding-site panel.

Shows WHY the figure is the ROS1 active ATP site and HOW the drug binds:
  - functional regions coloured and labelled BY NAME (not bare numbers):
        P-loop (glycine-rich)  real 1948-1955   (orange)
        hinge                  real 2027-2030   (marine)   <- H-bond region
        DFG motif              real 2102-2104   (red)      <- active/inactive switch
        gatekeeper             real 2026        (purple)
  - the DFG-Phe (F2103) drawn as sticks (the residue that flips in/out)
  - the docked drug (yellow) engaging the hinge
  - light grey cartoon of the whole domain for context (so it reads as a kinase)

Active (DFG-in) state, WT averaged pocket. Numbering: real = GRO + 1933.

Run (system PyMOL, venv DEACTIVATED, venv packages on PYTHONPATH):
    deactivate
    PYTHONPATH=$VENV_SP pymol -cq analysis/scripts/render_binding_annotated.py -- DRUG
      DRUG = zidesamtinib | cabozantinib
"""
import os, sys

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
DRUG = argv[0] if argv else 'zidesamtinib'
OFFSET = 1933

BASE = os.path.join(os.environ['HOME'], 'Molecular_Dynamics_analysis')
FIGS = os.path.join(BASE, 'figures/ROS1/ros1_prepared_final/pocket_box')

# functional regions in REAL numbering -> converted to GRO for selection
REGIONS = {
    'P-loop':     (range(1948, 1956), 'orange',  'P-loop'),
    'hinge':      (range(2027, 2031), 'marine',  'hinge'),
    'DFG':        (range(2102, 2105), 'red',      'DFG motif'),
}
DFG_PHE_REAL = 2103   # the Phe that flips (active in / inactive out)

from pymol import cmd

drug_pdb = os.path.join(FIGS, f'prep_distance_{DRUG}_drug.pdb')
prot_pdb = os.path.join(FIGS, 'prep_avg_backbone_WT.pdb')

cmd.reinitialize()
cmd.bg_color('white')
cmd.set('ray_opaque_background', 1)
cmd.set('orthoscopic', 1)
cmd.set('ray_shadows', 0)
cmd.load(prot_pdb, 'prot')
cmd.load(drug_pdb, 'drug')

# --- surface quality: kill the black interior/back-face artifacts ---
cmd.set('two_sided_lighting', 0)     # don't light interior faces (they go black)
cmd.set('surface_quality', 1)
cmd.set('ray_interior_color', 'grey90')   # if interior shows, make it grey not black
cmd.set('surface_cavity_mode', 0)

# whole domain: molecular SURFACE (the 'cloud' look), SOLID (transparency worsens
# the interior-darkening, so keep it opaque)
cmd.hide('everything', 'prot')
cmd.show('surface', 'prot')
cmd.set('transparency', 0.0, 'prot')
cmd.color('grey80', 'prot')

# colour + label each functional region
def gro_sel(real_range):
    return 'prot and resid ' + '+'.join(str(r - OFFSET) for r in real_range)

for name, (rr, colour, label) in REGIONS.items():
    sel = gro_sel(rr)
    cmd.color(colour, sel)
    cmd.show('sticks', f'{sel} and not name C+N+O')   # side chains as sticks
    cmd.set('stick_radius', 0.15, sel)
    # label at the middle residue's CA
    mid = list(rr)[len(list(rr)) // 2]
    cmd.set('label_size', 22)
    cmd.set('label_color', colour)
    cmd.set('label_outline_color', 'white')
    cmd.set('float_labels', 1)
    cmd.set('label_position', (0,0,3))
    cmd.label(f'prot and resid {mid - OFFSET} and name CA', repr(label))

# DFG-Phe: emphasise (thicker sticks) — the flip residue
phe_sel = f'prot and resid {DFG_PHE_REAL - OFFSET}'
cmd.show('sticks', phe_sel)
cmd.set('stick_radius', 0.28, phe_sel)
cmd.color('firebrick', phe_sel)

# drug: yellow sticks, heteroatoms by element
cmd.hide('everything', 'drug')
cmd.show('sticks', 'drug')
cmd.set('stick_radius', 0.22, 'drug')
cmd.color('yellow', 'drug')

# view: frame the pocket (P-loop + hinge + DFG + drug)
frame = ('drug or ' + gro_sel(range(1948, 1956)) + ' or '
         + gro_sel(range(2027, 2031)) + ' or ' + gro_sel(range(2102, 2105)))
cmd.orient(frame)
cmd.zoom(frame, buffer=4.0)
cmd.set('ray_interior_color', 'grey90')

out = os.path.join(FIGS, f'binding_annotated_{DRUG}.png')
cmd.png(out, width=1600, height=1300, dpi=300, ray=1)
print(f'saved {out}')