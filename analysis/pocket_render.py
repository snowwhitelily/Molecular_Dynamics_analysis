"""pocket_render.py -- Task 10 pocket figure (system PyMOL). Polished:
lighter+thinner WT reference, thicker coloured mutant, larger ligand/spheres.
PC1 spikes now come from the WHOLE-POCKET motion but are shown only on the
mutation residues (zero out every other atom's loading, then the boxplot's own
threshold skips them) -- so the bubbles get a real, visible spike."""
import os, sys, ast, copy, numpy as np

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
VARIANT, SELECTION, DRUG = argv[0], argv[1], argv[2]

TARGET_WHISKER = float(os.environ.get('POCKET_WHISKER', 3.5))
SCALE_MULT     = float(os.environ.get('POCKET_SCALE', 1.0))
EDGE           = float(os.environ.get('POCKET_EDGE', 0.17))
SIDE           = float(os.environ.get('POCKET_SIDE', 0.05))
DRUG_TRANSP    = float(os.environ.get('POCKET_DRUG', 0.0))
BUFFER         = float(os.environ.get('POCKET_BUFFER', 1.5))
MUT_GRO        = [53, 89]
KEY_GRO        = [47, 53, 89, 93, 169]

EDGE_MUT, SIDE_MUT = EDGE * 1.25, SIDE * 1.25
EDGE_WT,  SIDE_WT  = EDGE * 0.80, SIDE * 0.80

MOTIF_RUNS = [range(18, 27), range(53, 71), range(89, 101),
              range(144, 154), range(169, 173)]
strand_resids = sorted({r for run in MOTIF_RUNS for r in run})

BASE = '/homes/lkgyammerah/Molecular_Dynamics_analysis'
TRAJ = os.path.join(BASE, 'trajectories/ros1_prepared_final')
FIGS = os.path.join(BASE, 'figures/ROS1/ros1_prepared_final/pocket_box')
sys.path.insert(0, os.path.join(BASE, 'analysis', 'scripts'))

from pymol import cmd, util
import pca
import time as _time, ast as _ast
pca.time = _time; pca.ast = _ast; pca.cmd = cmd

def hex_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

WT_GREY = '#A0A0A0'
PALETTE = {'WT': hex_rgb(WT_GREY), 'S1986F': hex_rgb('#00A651'),
           'Q2022P': hex_rgb('#1976D2'), 'Q2022P_S1986F': hex_rgb('#9C27B0')}
COLOUR = PALETTE[VARIANT]

resids = [int(x) for x in open(os.path.join(
    FIGS, f'prep_{SELECTION}_{DRUG}_resids.txt')).read().strip().split(',')]
struct_resids = sorted(set(resids) | set(strand_resids))
struct_sel = 'resid ' + '+'.join(str(r) for r in struct_resids)

drug_pdb     = os.path.join(FIGS, f'prep_{SELECTION}_{DRUG}_drug.pdb')
cartoon_pdb  = os.path.join(FIGS, f'prep_avg_backbone_{VARIANT}.pdb')
wt_cartoon   = os.path.join(FIGS, 'prep_avg_backbone_WT.pdb')
wt_atoms_pdb = os.path.join(FIGS, 'prep_pocket_atoms_WT.pdb')
view_file    = os.path.join(FIGS, f'prep_{SELECTION}_{DRUG}_view.txt')

NTL_FIT_SEL = 'backbone and resid 5-97 and name CA'
NTL_CA_SEL  = 'name CA and resid 5-97'
mut_resid   = 'resid ' + '+'.join(str(r) for r in MUT_GRO)
BB_NAMES    = 'name N+CA+C+O'
strand_sel  = 'resid ' + '+'.join(str(r) for r in strand_resids)

def replicas(variant):
    out = []
    for rep in ('0', '1', '2'):
        stem = os.path.join(TRAJ, variant, rep, f'{variant}-MD-prot')
        if os.path.exists(stem + '.pdb') and os.path.exists(stem + '.xtc'):
            out.append((stem + '.pdb', stem + '.xtc'))
    return out

def build_pocket_mean(variant, tag):
    objs = []
    for i, (top, trj) in enumerate(replicas(variant)):
        raw = f'{tag}_raw{i}'
        cmd.load(top, raw); cmd.load_traj(trj, raw)
        cmd.intra_fit(f'{raw} and ({NTL_FIT_SEL})')
        cmd.align(f'{raw} and ({NTL_CA_SEL})', f'wtframe and ({NTL_CA_SEL})', cycles=0)
        o = f'{tag}_r{i}'; cmd.create(o, raw); cmd.delete(raw); objs.append(o)
    if not objs:
        raise SystemExit(f'No replicas for {variant}')
    reps = '(' + ' or '.join(objs) + ')'
    pca.princomp(f'{reps} and ({struct_sel}) and not hydro', name=f'pc_{tag}', maxvec=2, states=(0,))
    pca.drawmean(f'pc_{tag}', name=f'{tag}_mean')
    for o in objs: cmd.delete(o)
    return f'{tag}_mean', f'pc_{tag}'

def style_mean(obj, colour_name, edge, side):
    cmd.hide('everything', obj); cmd.show('sticks', obj)
    cmd.set_bond('stick_radius', side, obj, obj)
    cmd.set_bond('stick_radius', edge, f'{obj} and ({strand_sel}) and ({BB_NAMES})',
                 f'{obj} and ({strand_sel}) and ({BB_NAMES})')
    cmd.color(colour_name, obj)
    msel = f'{obj} and ({mut_resid})'
    cmd.show('sticks', msel); cmd.show('spheres', msel)
    cmd.set('sphere_scale', 0.52, msel)
    cmd.set_bond('stick_radius', 0.32, msel, msel)
    cmd.color(colour_name, msel)

def draw_mutation_spikes(pca_name, mean_obj, colour_rgb, tag):
    """Whole-pocket PC1 motion, drawn only on the mutation residues."""
    comp = getattr(pca, pca_name)[0]
    loadings = np.asarray(comp.loadings)
    resis = np.array([int(a.resi) for a in cmd.get_model(mean_obj).atom])
    if len(resis) != len(loadings):
        print('spike skip: atom/resi mismatch'); return
    mask = np.isin(resis, np.array(MUT_GRO))
    if not mask.any():
        print('spike skip: no mutation atoms in pocket'); return
    scores = np.asarray(comp.scores)
    span = float(scores.max() - scores.min())
    mload = float(np.sqrt((loadings[mask] ** 2).sum(axis=1)).max())
    denom = span * mload
    scale = (TARGET_WHISKER * SCALE_MULT) / denom if denom > 0 else 1.0
    sub = copy.copy(comp)                     # keep scores/mean/sele; mask loadings
    L = loadings.copy(); L[~mask] = 0.0; sub.loadings = L
    sub.cgo(draw='boxplot', radius=0.19, scale=scale,
            color=list(colour_rgb), threshold=1e-6, name=f'{tag}_spikes')

cmd.reinitialize(); cmd.bg_color('white')
cmd.set('ray_opaque_background', 1); cmd.set('orthoscopic', 1); cmd.set('ray_shadows', 0)
cmd.set('cartoon_transparency', 0.75); cmd.set('stick_ball', 0); cmd.set('valence', 0)
cmd.set_color('c_var', list(COLOUR)); cmd.set_color('c_wtref', list(PALETTE['WT']))
cmd.load(wt_cartoon, 'wtframe'); cmd.hide('everything', 'wtframe')
cmd.load(cartoon_pdb, 'domain'); cmd.hide('everything', 'domain')
cmd.dss('domain'); cmd.show('cartoon', 'domain'); cmd.color('grey90', 'domain')

mean_v, pca_v = build_pocket_mean(VARIANT, 'V')
if VARIANT == 'WT':
    style_mean(mean_v, 'c_var', EDGE, SIDE)
else:
    style_mean(mean_v, 'c_var', EDGE_MUT, SIDE_MUT)
draw_mutation_spikes(pca_v, mean_v, COLOUR, 'V')

mean_wt = None
if VARIANT != 'WT':
    mean_wt, _ = build_pocket_mean('WT', 'W')
    style_mean(mean_wt, 'c_wtref', EDGE_WT, SIDE_WT)

cmd.load(drug_pdb, 'drug'); cmd.hide('everything', 'drug')
cmd.show('sticks', 'drug'); cmd.set('stick_radius', 0.28, 'drug')
if DRUG_TRANSP > 0: cmd.set('stick_transparency', DRUG_TRANSP, 'drug')
util.cbay('drug')

# de-clutter: make ONLY the pocket residues sitting on the FACE of the drug
# see-through, so the buried ligand reads. Drug, walls, WT and spikes stay solid.
# Set POCKET_FACE_TRANSP=0 to switch this off entirely.
FACE_CUT    = float(os.environ.get('POCKET_FACE_CUT', 4.0))
FACE_TRANSP = float(os.environ.get('POCKET_FACE_TRANSP', 0.6))
if FACE_TRANSP > 0:
    for po in [mean_v] + ([mean_wt] if mean_wt else []):
        cmd.set('stick_transparency', FACE_TRANSP,
                f'byres ({po} within {FACE_CUT} of drug)')

if os.environ.get('POCKET_LABELS'):
    cmd.set('label_size', 20); cmd.set('label_color', 'black'); cmd.set('label_outline_color', 'white')
    cmd.label(f'{mean_v} and name CA and resid ' + '+'.join(str(r) for r in KEY_GRO), '"%s" % resi')

# always label the two mutation sites so they can be told apart at a glance
# GRO 53 = real 1986, GRO 89 = real 2022
OFFSET = 1933
cmd.set('label_size', 16)
cmd.set('label_color', 'black')
cmd.set('label_outline_color', 'white')
cmd.set('label_position', [0, 2.5, 0])
for gro in MUT_GRO:
    real = gro + OFFSET
    cmd.label(f'{mean_v} and name CA and resid {gro}', repr(str(real)))

cmd.load(wt_atoms_pdb, 'camref'); cmd.hide('everything', 'camref')
frame_sel = f'drug or (camref and ({mut_resid}))'
if VARIANT == 'WT' or not os.path.exists(view_file):
    cmd.orient(frame_sel); cmd.zoom(frame_sel, buffer=BUFFER)
    with open(view_file, 'w') as fh: fh.write(repr(list(cmd.get_view())))
else:
    cmd.set_view(ast.literal_eval(open(view_file).read()))

out = os.path.join(FIGS, f'panel_{SELECTION}_{DRUG}_{VARIANT}.png')
cmd.png(out, width=1200, height=1200, dpi=300, ray=1)
print(f'saved {out}')