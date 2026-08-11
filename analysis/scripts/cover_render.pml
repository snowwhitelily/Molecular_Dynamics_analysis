reinitialize

# Dark gradient background
bg_color black
set ray_opaque_background, on

# High quality rendering
set antialias, 3
set ray_shadows, 1
set ray_trace_mode, 3
set ray_trace_gain, 0.02
set spec_reflect, 0.4
set spec_power, 200
set ambient, 0.2
set direct, 0.7
set depth_cue, 1
set fog_start, 0.45
set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 1
set cartoon_oval_length, 1.3
set cartoon_loop_radius, 0.15

load /homes/lkgyammerah/Molecular_Dynamics_analysis/dock/ROS1/q2022p_subset_receptors/actout_ctlfit/WT/WT_actout_ctlfit_dominant_cluster0_rep.pdb, ROS1

hide everything
show cartoon, ROS1

# Elegant color scheme — subtle blues and golds
color 0x4A90D9, ROS1
color 0x2C5F9E, resi 95-287
color 0xE8453C, resi 55-70
color 0xF5C842, resi 18-26
color 0xF09030, resi 86-94
color 0xDD55DD, resi 169-171
show spheres, resi 169-171 and name CA
set sphere_scale, 0.5

# Hide tag residues
hide everything, resi 1-11
hide everything, resi 288-999

# Add a subtle transparent surface for depth
show surface, ROS1
set surface_color, 0x1A3A6A, ROS1
set transparency, 0.85

# Orientation
orient ROS1
rotate y, 150
rotate x, -15
zoom ROS1, 4

# Ultra high resolution
ray 3600, 2800
png ros1_cover_dark.png, dpi=300
quit
