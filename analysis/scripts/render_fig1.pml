reinitialize
bg_color white
set ray_opaque_background, off
set antialias, 2

# Load WT PDB structure
load /homes/lkgyammerah/Molecular_Dynamics_analysis/dock/ROS1/q2022p_subset_receptors/actout_ctlfit/WT/WT_actout_ctlfit_dominant_cluster0_rep.pdb, ROS1

# Define domain selections using GRO numbering
select n_lobe, resi 12-85
select c_lobe, resi 95-287
select hinge, resi 86-94
select alphaC, resi 55-70
select p_loop, resi 18-26
select dfg_motif, resi 169-171

# Color coding
color lightblue, n_lobe
color grey80, c_lobe
color tv_orange, hinge
color red, alphaC
color yellow, p_loop
color deepsalmon, dfg_motif

show cartoon, ROS1
hide lines

# --- Panel A Render: Full Kinase Domain ---
orient ROS1
zoom ROS1, 3
ray 2400, 1800
png Figure_1.1A_Full_Kinase.png

# --- Panel B Render: Active Site Zoom ---
# Select catalytic residues (K1980->47, E1993->60, Q2022->89, DFG->169-171)
select catalytic_residues, resi 47 + resi 60 + resi 89 + resi 169-171
show sticks, catalytic_residues
color atomic, (not elem C) and catalytic_residues
color forest, elem C and catalytic_residues

zoom catalytic_residues, 8
ray 2400, 1800
png Figure_1.1B_Active_Site_Zoom.png
