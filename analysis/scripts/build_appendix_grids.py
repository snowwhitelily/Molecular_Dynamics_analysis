#!/usr/bin/env python3
"""
Build Appendix E & F grid pages from per-mutant FEL PNGs.

Usage:
    conda activate ros1_analysis_env
    python3 build_appendix_grids.py --input_dir /path/to/fel_pngs --output_dir ./figures

This script:
  1. Scans input_dir for all .png files
  2. Sorts them by residue position (WT first)
  3. Tiles them into 3x3 grid pages (9 panels per page)
  4. Saves multi-panel pages as Appendix_E_page_01.png, etc.
  5. Optionally generates Appendix F with a different subspace

Adjust COLS, ROWS, and DPI below if needed.
"""

import os
import re
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# ── Layout Configuration ────────────────────────────────────────────────
COLS = 3          # panels per row
ROWS = 3          # rows per page
DPI = 300         # output resolution
FIG_W = 18        # page width in inches
FIG_H = 18        # page height in inches
TITLE_SIZE = 14   # font size for panel titles
PAD_W = 0.02      # horizontal padding between panels (fraction)
PAD_H = 0.04      # vertical padding between panels (fraction)


def extract_position(filename):
    """Extract residue position number for sorting. WT returns 0."""
    name = Path(filename).stem.upper()
    if name.startswith('WT') or name == 'WILDTYPE' or name == 'WILD_TYPE':
        return (0, 'A', name)
    # Match patterns like L1982V, Q2022P, G2032R, D2113N, etc.
    match = re.search(r'[A-Z](\d{4})[A-Z]', name)
    if match:
        pos = int(match.group(1))
        # Get the substituted amino acid for secondary sort
        aa = name[match.end()-1]
        return (pos, aa, name)
    # Compound mutations like Q2022P_S1986F
    match2 = re.search(r'Q2022P', name)
    if match2:
        return (2022, 'P' + name, name)
    return (9999, 'Z', name)


def find_fel_images(input_dir):
    """Find all PNG files in the directory and sort by residue position."""
    pngs = []
    for f in os.listdir(input_dir):
        if f.lower().endswith('.png'):
            pngs.append(f)
    
    if not pngs:
        print(f"ERROR: No .png files found in {input_dir}")
        sys.exit(1)
    
    # Sort: WT first, then by position, then alphabetically
    pngs.sort(key=lambda f: extract_position(f))
    return pngs


def make_panel_title(filename):
    """Convert filename to a clean panel title."""
    name = Path(filename).stem
    # Remove common prefixes/suffixes
    for prefix in ['fel_', 'FEL_', 'actout_ctlfit_', 'free_energy_']:
        if name.lower().startswith(prefix.lower()):
            name = name[len(prefix):]
    for suffix in ['_fel', '_FEL', '_free_energy', '_actout_ctlfit']:
        if name.lower().endswith(suffix.lower()):
            name = name[:-len(suffix)]
    # Clean up underscores
    name = name.replace('_', ' ').strip()
    # Capitalize mutation names
    if name.upper() == name or len(name) <= 8:
        name = name.upper()
    return name


def build_grid_pages(images, input_dir, output_dir, prefix='Appendix_E'):
    """Tile images into multi-panel grid pages."""
    panels_per_page = COLS * ROWS
    n_pages = (len(images) + panels_per_page - 1) // panels_per_page
    
    print(f"\nBuilding {prefix}: {len(images)} panels across {n_pages} pages")
    print(f"Layout: {COLS} cols x {ROWS} rows = {panels_per_page} per page")
    
    output_paths = []
    
    for page_idx in range(n_pages):
        start = page_idx * panels_per_page
        end = min(start + panels_per_page, len(images))
        page_images = images[start:end]
        
        fig, axes = plt.subplots(ROWS, COLS, figsize=(FIG_W, FIG_H), dpi=DPI)
        fig.patch.set_facecolor('white')
        
        # Flatten axes for easy indexing
        if ROWS == 1 and COLS == 1:
            axes = np.array([[axes]])
        elif ROWS == 1:
            axes = axes.reshape(1, -1)
        elif COLS == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(ROWS):
            for j in range(COLS):
                panel_idx = i * COLS + j
                ax = axes[i, j]
                
                if panel_idx < len(page_images):
                    img_path = os.path.join(input_dir, page_images[panel_idx])
                    try:
                        img = mpimg.imread(img_path)
                        ax.imshow(img)
                        title = make_panel_title(page_images[panel_idx])
                        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold',
                                    pad=8)
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error loading\n{page_images[panel_idx]}',
                               ha='center', va='center', transform=ax.transAxes,
                               fontsize=10, color='red')
                        print(f"  Warning: Could not load {page_images[panel_idx]}: {e}")
                
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Hide empty panels
                if panel_idx >= len(page_images):
                    ax.set_visible(False)
        
        plt.subplots_adjust(wspace=PAD_W, hspace=PAD_H,
                           left=0.02, right=0.98, top=0.96, bottom=0.02)
        
        # Add page header
        fig.suptitle(f'{prefix.replace("_", " ")} — Page {page_idx + 1} of {n_pages}',
                    fontsize=16, fontweight='bold', y=0.99)
        
        out_path = os.path.join(output_dir, f'{prefix}_page_{page_idx+1:02d}.png')
        plt.savefig(out_path, bbox_inches='tight', facecolor='white', dpi=DPI)
        plt.close()
        
        output_paths.append(out_path)
        print(f"  Saved: {out_path} ({len(page_images)} panels)")
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description='Build Appendix grid pages from per-mutant FEL PNGs')
    parser.add_argument('--input_dir', '-i', required=True,
                       help='Directory containing per-mutant FEL PNG files')
    parser.add_argument('--output_dir', '-o', default='/homes/lkgyammerah/Molecular_Dynamics_analysis/figures/ROS1/ros1_prepared_final',
                       help='Output directory for grid pages (default: ./figures)')
    parser.add_argument('--prefix', '-p', default='Appendix_E',
                       help='Output filename prefix (default: Appendix_E)')
    parser.add_argument('--cols', type=int, default=COLS,
                       help=f'Columns per page (default: {COLS})')
    parser.add_argument('--rows', type=int, default=ROWS,
                       help=f'Rows per page (default: {ROWS})')
    
    args = parser.parse_args()
    
    pass
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find and sort images
    images = find_fel_images(args.input_dir)
    print(f"Found {len(images)} FEL images in {args.input_dir}")
    print(f"Sort order (first 5): {[make_panel_title(f) for f in images[:5]]}")
    print(f"Sort order (last 5):  {[make_panel_title(f) for f in images[-5:]]}")
    
    # Build grid pages
    pages = build_grid_pages(images, args.input_dir, args.output_dir, args.prefix)
    
    print(f"\n{'='*60}")
    print(f"Done! Generated {len(pages)} grid pages in {args.output_dir}/")
    print(f"\nTo use in Overleaf, copy these PNGs to your figures/ folder")
    print(f"and add to your .tex:")
    print(f"")
    for p in pages:
        fname = Path(p).stem
        print(f"  \\includegraphics[width=\\textwidth]{{figures/{fname}}}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()