#!/usr/bin/env python3
"""
ROS1 Thesis Figure Renderer — v3 (clean rewrite)

Fixes:
  - Green loops: hide tag residues AFTER show cartoon
  - Orientation: classic kinase view (N-lobe top, C-lobe bottom, cleft facing viewer)
  - Panel B: only 4 key catalytic residues as sticks, not entire regions
  - Page 12: selective sticks, semi-transparent backbone
  - Combined: single unified legend instead of two separate ones

Usage:
    conda activate ros1_analysis_env
    cd ~/Molecular_Dynamics_analysis/results/ROS1
    python3 render_figures_v3.py
"""

import os
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np


PDB_PATH = (
    '/homes/lkgyammerah/Molecular_Dynamics_analysis/dock/ROS1/'
    'q2022p_subset_receptors/actout_ctlfit/WT/'
    'WT_actout_ctlfit_dominant_cluster0_rep.pdb'
)


def run_pymol(script_text):
    """Execute a PyMOL script headlessly."""
    with open('_tmp_render.pml', 'w') as f:
        f.write(script_text)
    os.system('PATH=/usr/bin:/bin PYTHONHOME= PYTHONPATH= /usr/bin/pymol -c _tmp_render.pml')
    if os.path.exists('_tmp_render.pml'):
        os.remove('_tmp_render.pml')


# ════════════════════════════════════════════════════════════════════════
# COMMON SETUP — applied to every figure
# The key fix: show cartoon FIRST, then hide tag residues AFTER
# ════════════════════════════════════════════════════════════════════════
COMMON_SETUP = f"""
reinitialize

# ── Global settings ──
bg_color white
set ray_opaque_background, on
set antialias, 2
set ray_shadows, 0
set ray_trace_mode, 1
set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 1
set cartoon_loop_radius, 0.15
set cartoon_oval_length, 1.2
set cartoon_oval_width, 0.25
set stick_radius, 0.14
set spec_reflect, 0.3
set ambient, 0.25

# ── Load structure ──
load {PDB_PATH}, ROS1

# ── Show cartoon FIRST ──
hide everything, ROS1
show cartoon, ROS1

# ── Color all grey, then paint functional regions ──
color grey80, ROS1
color 0x87CEEB, resi 12-85 and not resi 55-70 and not resi 18-26
color 0xFF0000, resi 55-70
color 0xFFD700, resi 18-26
set cartoon_tube_radius, 0.35, resi 18-26
color 0xFF7F00, resi 86-94
set cartoon_tube_radius, 0.35, resi 86-94
color 0xFF00FF, resi 169-171
show spheres, resi 169-171 and name CA
set sphere_scale, 0.6, resi 169-171
set cartoon_tube_radius, 0.4, resi 169-171

# ── NOW hide tag residues (AFTER show cartoon) ──
hide everything, resi 1-11
hide everything, resi 288-999

# ── Hide all non-bonded/solvent ──
hide nonbonded
hide lines
"""


# ════════════════════════════════════════════════════════════════════════
# FIGURE 1.1A — Full kinase domain overview
# Classic kinase view: N-lobe top-right, C-lobe bottom-left, cleft open
# ════════════════════════════════════════════════════════════════════════
SCRIPT_1A = COMMON_SETUP + """
# ── Orientation: classic kinase textbook view ──
orient ROS1
rotate y, 150
rotate x, -20
rotate z, 10
zoom ROS1, 3

# ── Render ──
ray 2400, 2000
png _raw_1A.png, dpi=300
quit
"""


# ════════════════════════════════════════════════════════════════════════
# FIGURE 1.1B — Zoomed active site with 4 key catalytic residues
# Only K1980, E1993, Q2022, and DFG as sticks — nothing else
# ════════════════════════════════════════════════════════════════════════
SCRIPT_1B = COMMON_SETUP + """
# ── Semi-transparent backbone for context ──


# ── Show sticks ONLY for key catalytic residues ──
# K1980 (beta3 lysine), E1993 (aC-helix glutamate),
# Q2022 (hinge glutamine), D2102-G2104 (DFG motif)
select key_cat, resi 47 + resi 60 + resi 89 + resi 169+170+171
show sticks, key_cat
set stick_radius, 0.25, key_cat

# ── Element coloring on sticks ──
# Color stick carbons to MATCH their legend colors
color 0x87CEEB, resi 47
color 0xFF0000, resi 60
color 0xFF7F00, resi 89
color 0xFF00FF, resi 169+170+171
color 0xFF4444, elem O and key_cat
color 0x4444FF, elem N and key_cat

# ── Label key residues ──
set label_size, 18
set label_color, black
set label_font_id, 7
set label_position, [2, 2, 0]
label resi 47 and name CA, "K1980"
label resi 60 and name CA, "E1993"
label resi 89 and name CA, "Q2022"
label resi 170 and name CA, "DFG"

# ── Zoom into the active site pocket ──
select pocket, resi 18-26 + resi 55-70 + resi 86-94 + resi 169-171
zoom pocket, 8
rotate y, 150
rotate x, -20

# ── Render ──
ray 2400, 2000
png _raw_1B.png, dpi=300
quit
"""


# ════════════════════════════════════════════════════════════════════════
# PAGE 12 — Hybrid cartoon + sticks (Methods section)
# Cartoon backbone with key binding pocket residues as sticks
# ════════════════════════════════════════════════════════════════════════
SCRIPT_P12 = COMMON_SETUP + """
# ── Semi-transparent backbone ──


# ── Sticks for binding pocket residues only ──
# P-loop, aC-helix key residues, hinge key residues, DFG
select binding_pocket, resi 47 + resi 60 + resi 89 + resi 169+170+171
show sticks, binding_pocket and sidechain
set stick_radius, 0.22

# ── Element coloring ──
color 0x87CEEB, resi 47
color 0xFF0000, resi 60
color 0xFF7F00, resi 89
color 0xFF00FF, resi 169+170+171
color 0xFF00FF, elem C and resi 169+170+171
color 0xFF00FF, resi 169+170+171 and name CA
color 0xFF4444, elem O and binding_pocket
color 0x4444FF, elem N and binding_pocket

# ── Orientation ──
orient ROS1
rotate y, 150
rotate x, -20
rotate z, 10
zoom binding_pocket, 6

# ── Render ──
ray 2400, 2000
png _raw_p12.png, dpi=300
quit
"""


def add_legend(raw_path, out_path, legend_items, title=None):
    """
    Add a clean legend below a PyMOL render.
    legend_items: list of (hex_color, label_text) tuples
    """
    if not os.path.exists(raw_path):
        print(f"  ERROR: {raw_path} not found")
        return False

    img = Image.open(raw_path).convert('RGB')

    # Auto-crop white borders
    arr = np.array(img)
    non_white = np.where(
        (arr[:,:,0] < 245) | (arr[:,:,1] < 245) | (arr[:,:,2] < 245)
    )
    if len(non_white[0]) > 0:
        y_min, y_max = non_white[0].min(), non_white[0].max()
        x_min, x_max = non_white[1].min(), non_white[1].max()
        pad = 30
        img = img.crop((
            max(0, x_min - pad), max(0, y_min - pad),
            min(img.width, x_max + pad), min(img.height, y_max + pad)
        ))

    # Calculate legend dimensions
    n_items = len(legend_items)
    n_cols = 2
    n_rows = (n_items + 1) // 2
    row_height = 55
    legend_h = n_rows * row_height + 40
    col_width = img.width // 2

    # Create output image
    total_h = img.height + legend_h
    out = Image.new('RGB', (img.width, total_h), (255, 255, 255))
    out.paste(img, (0, 0))

    draw = ImageDraw.Draw(out)

    # Load font
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 22)
        except:
            font = ImageFont.load_default()

    # Draw legend entries
    swatch_size = 28
    y_start = img.height + 20

    for idx, (color, label) in enumerate(legend_items):
        col = idx % n_cols
        row = idx // n_cols

        x = 40 + col * col_width
        y = y_start + row * row_height

        # Color swatch with border
        draw.rectangle([x, y, x + swatch_size, y + swatch_size],
                       fill=color, outline='black', width=2)
        # Label text
        draw.text((x + swatch_size + 15, y + 2), label,
                 fill='black', font=font)

    out.save(out_path, dpi=(300, 300))
    return True


def combine_panels(path_a, path_b, out_path):
    """Side-by-side panels with (A) and (B) labels, shared legend."""
    imgA = Image.open(path_a).convert('RGB')
    imgB = Image.open(path_b).convert('RGB')

    # Match heights
    target_h = max(imgA.height, imgB.height)
    rA = target_h / imgA.height
    rB = target_h / imgB.height
    imgA = imgA.resize((int(imgA.width * rA), target_h), Image.LANCZOS)
    imgB = imgB.resize((int(imgB.width * rB), target_h), Image.LANCZOS)

    gap = 30
    label_h = 65
    total_w = imgA.width + gap + imgB.width

    combined = Image.new('RGB', (total_w, target_h + label_h), (255, 255, 255))
    combined.paste(imgA, (0, label_h))
    combined.paste(imgB, (imgA.width + gap, label_h))

    draw = ImageDraw.Draw(combined)
    try:
        font_label = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
    except:
        font_label = ImageFont.load_default()

    draw.text((20, 5), "(A)", fill=(0, 0, 0), font=font_label)
    draw.text((imgA.width + gap + 20, 5), "(B)", fill=(0, 0, 0), font=font_label)

    combined.save(out_path, dpi=(300, 300))
    return True


def main():
    print("=" * 60)
    print("ROS1 Thesis Figure Renderer — v3")
    print("=" * 60)

    # Legend items for Panel A (full domain)
    legend_A = [
        ('#87CEEB', 'N-lobe (1945–2018)'),
        ('#FFD700', 'P-loop (1951–1959)'),
        ('#CCCCCC', 'C-lobe (2028–2220)'),
        ('#FF7F00', 'Hinge region (2019–2027)'),
        ('#FF0000', 'αC-helix (1988–2003)'),
        ('#FF00FF', 'DFG motif (2102–2104)'),
    ]

    # Legend items for Panel B (active site)
    legend_B = [
        ('#FFD700', 'P-loop (1951–1959)'),
        ('#FF7F00', 'Hinge / Q2022'),
        ('#FF0000', 'αC-helix / E1993'),
        ('#FF00FF', 'DFG motif (2102–2104)'),
        ('#87CEEB', 'K1980 Catalytic Lysine'),
        ('#CCCCCC', 'C-lobe framework'),
    ]

    # Legend items for Page 12
    legend_P12 = [
        ('#FFD700', 'P-loop (1951–1959)'),
        ('#FF7F00', 'Hinge region (2019–2027)'),
        ('#FF0000', 'αC-helix (1988–2003)'),
        ('#FF00FF', 'DFG motif (2102–2104)'),
        ('#87CEEB', 'N-lobe (1945–2018)'),
        ('#CCCCCC', 'C-lobe (2028–2220)'),
    ]

    # ── Render Panel A ──
    print("\n[1/3] Rendering Figure 1.1A — Full kinase domain...")
    run_pymol(SCRIPT_1A)
    if add_legend('_raw_1A.png', 'FINAL_Figure_1.1A_Full_Kinase.png', legend_A):
        print("  ✓ FINAL_Figure_1.1A_Full_Kinase.png")

    # ── Render Panel B ──
    print("\n[2/3] Rendering Figure 1.1B — Active site zoom...")
    run_pymol(SCRIPT_1B)
    if add_legend('_raw_1B.png', 'FINAL_Figure_1.1B_Active_Site.png', legend_B):
        print("  ✓ FINAL_Figure_1.1B_Active_Site.png")

    # ── Render Page 12 ──
    print("\n[3/3] Rendering Page 12 — Hybrid structure...")
    run_pymol(SCRIPT_P12)
    if add_legend('_raw_p12.png', 'FINAL_Page_12_Hybrid_Structure.png', legend_P12):
        print("  ✓ FINAL_Page_12_Hybrid_Structure.png")

    # ── Combine A + B ──
    print("\n[+] Combining panels...")
    if os.path.exists('FINAL_Figure_1.1A_Full_Kinase.png') and \
       os.path.exists('FINAL_Figure_1.1B_Active_Site.png'):
        combine_panels(
            'FINAL_Figure_1.1A_Full_Kinase.png',
            'FINAL_Figure_1.1B_Active_Site.png',
            'Figure_1.1_combined.png'
        )
        print("  ✓ Figure_1.1_combined.png")

    # ── Cleanup raw renders ──
    for f in ['_raw_1A.png', '_raw_1B.png', '_raw_p12.png']:
        if os.path.exists(f):
            os.remove(f)

    print("\n" + "=" * 60)
    print("Output files:")
    print("  FINAL_Figure_1.1A_Full_Kinase.png     → figures/Figure_1_1A.png")
    print("  FINAL_Figure_1.1B_Active_Site.png     → figures/Figure_1_1B.png")
    print("  FINAL_Page_12_Hybrid_Structure.png    → figures/Figure_page12_hybrid.png")
    print("  Figure_1.1_combined.png               → figures/Figure_1_1_combined.png")
    print("=" * 60)


if __name__ == '__main__':
    main()