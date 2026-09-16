"""
render_contacts_3d.py -- 3D binding-site panel (reference Panel-B style).

Surface of the WT pocket, the docked drug (yellow sticks), and the TIGHT-contact
residues (<= 3.5 A, the likely H-bond set) shown as sticks and labelled by real
ROS1 number. Only the tight contacts are labelled to avoid clutter.

Run headless with system PyMOL + venv packages on PYTHONPATH:
    PYTHONPATH=$VENV_SP pymol -cq analysis/scripts/render_contacts_3d.py -- DRUG
      DRUG = zidesamtinib | cabozantinib
"""
import os, sys, json

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
DRUG = argv[0] if argv else 'zidesamtinib'
OFFSET = 1933
HB = 3.5

BASE = os.path.join(os.environ['HOME'], 'Molecular_Dynamics_analysis')
FIGS = os.path.join(BASE, 'figures/ROS1/ros1_prepared_final/pocket_box')
CJSON = os.path.join(BASE, 'results/ROS1/drug_contacts.json')

from pymol import cmd

drug_pdb  = os.path.join(FIGS, f'prep_distance_{DRUG}_drug.pdb')
prot_pdb  = os.path.join(FIGS, 'prep_avg_backbone_WT.pdb')

# tight-contact residues (real numbers) -> GRO numbers for selection
contacts = json.load(open(CJSON))[DRUG]
tight_real = [r for r, rn, d in contacts if d < HB]
tight_gro  = [r - OFFSET for r in tight_real]

cmd.reinitialize()
cmd.bg_color('white')
cmd.set('two_sided_lighting', 0)
cmd.set('ray_interior_color', 'grey90')
cmd.set('surface_quality', 1)
cmd.set('ray_shadows', 0)

cmd.set('ray_opaque_background', 1)
cmd.set('orthoscopic', 1)
cmd.set('ray_shadows', 0)

cmd.load(prot_pdb, 'prot')
cmd.load(drug_pdb, 'drug')

# protein: light grey surface (semi-transparent) so the pocket reads
cmd.hide('everything', 'prot')
cmd.show('surface', 'prot')
cmd.set('transparency', 0.0, 'prot')
cmd.color('grey80', 'prot')

# tight-contact residues: sticks, coloured, labelled
sel = 'prot and resid ' + '+'.join(str(g) for g in tight_gro)
cmd.show('sticks', sel)
cmd.color('marine', sel)
cmd.set('stick_radius', 0.18, sel)
cmd.set('label_size', 20)
cmd.set('label_color', 'black')
cmd.set('label_outline_color', 'white')
cmd.set('float_labels', 1)
for g, real in zip(tight_gro, tight_real):
    cmd.label(f'prot and resid {g} and name CA', repr(str(real)))

# drug: yellow sticks
cmd.hide('everything', 'drug')
cmd.show('sticks', 'drug')
cmd.set('stick_radius', 0.22, 'drug')
cmd.color('yellow', 'drug')
cmd.util.cnc('drug')   # colour N/O/etc by element, keep C yellow

# view: focus on the drug + its contacts
cmd.orient(f'drug or ({sel})')
cmd.zoom(f'drug or ({sel})', buffer=3.0)

out = os.path.join(FIGS, f'contacts3d_{DRUG}.png')
cmd.png(out, width=1500, height=1200, dpi=300, ray=1)
print(f'saved {out}  (tight contacts: {tight_real})')