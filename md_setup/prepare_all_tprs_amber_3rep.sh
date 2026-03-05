#!/bin/bash
# Generate MD TPRs with gromit + AMBER99SB-ILDN
# - 3 replicas per mutant
# - each extended to 1 µs (500,000,000 steps)
# - skips replicas that already have 02_tprs/<mut>/repX/md.tpr

set -eo pipefail    # no -u here because GMXRC uses $shell

# ---- Load GROMACS (needed for gmx convert-tpr) ----
source /usr/local/gromacs-2024.5/bin/GMXRC

# ---- Paths ----
BASE="$HOME/ROS1_MD"
STRUCT_DIR="$BASE/00_structures"
SYSTEMS_DIR="$BASE/01_systems"
TPR_DIR="$BASE/02_tprs"

# Local gromit clone from https://github.com/Tsjerk/gromit.git
GROMIT="$HOME/gromit/gromit.sh"

# Check gromit exists
if [ ! -x "$GROMIT" ]; then
    echo "ERROR: gromit script not found or not executable at: $GROMIT"
    echo "       Did you clone https://github.com/Tsjerk/gromit.git into ~/gromit?"
    exit 1
fi

mkdir -p "$SYSTEMS_DIR" "$TPR_DIR"

cd "$STRUCT_DIR"

# ---- Loop over all PDBs ----
# Adjust this pattern if your PDB filenames changed
for pdb in ROS1_kinase_1934_2225_*.pdb; do
    if [ ! -e "$pdb" ]; then
        echo "ERROR: No matching PDB files (ROS1_kinase_1934_2225_*.pdb) in $STRUCT_DIR"
        exit 1
    fi

    base=${pdb%.pdb}
    mut=${base#ROS1_kinase_1934_2225_}

    echo "=== Preparing system: $mut (AMBER99SB-ILDN, 3 replicas) ==="

    # ---- Three replicas per mutant ----
    for rep in 1 2 3; do
        repname="${mut}_rep${rep}"
        sysdir="$SYSTEMS_DIR/$repname"
        rep_out_dir="$TPR_DIR/$mut/rep${rep}"

        echo "  -- Replica $repname --"

        # If final 1 µs TPR already exists, skip this replica
        if [ -f "$rep_out_dir/md.tpr" ]; then
            echo "     md.tpr already exists for $repname, skipping."
            continue
        fi

        # Clean and recreate replica directory to avoid leftover junk
        rm -rf "$sysdir"
        mkdir -p "$sysdir"
        cp "$pdb" "$sysdir/structure.pdb"

        (
            cd "$sysdir"

            # Run full gromit pipeline with AMBER99SB-ILDN
            # gromit does pdb2gmx (AMBER), EM, NVT, NPT, MD prep
            "$GROMIT" -f structure.pdb -name "$repname" -ff amber99sb-ildn

            # Find a production .tpr (deepest/latest)
            tpr_path=$(find . -maxdepth 6 -type f -name "*.tpr" | sort | tail -n 1 || true)

            if [ -z "$tpr_path" ]; then
                echo "     !! WARNING: no .tpr found for $repname"
                exit 0
            fi

            mkdir -p "$rep_out_dir"

            # Keep original TPR as md_short.tpr
            cp "$tpr_path" "$rep_out_dir/md_short.tpr"

            cd "$rep_out_dir"

            echo "     extending to 1 µs (500000000 steps)"
            # 1 µs at 2 fs → 500,000,000 steps
            gmx convert-tpr -s md_short.tpr -o md.tpr -nsteps 500000000

            echo "     -> stored as $rep_out_dir/md.tpr"
        )
    done
done

echo "All systems + replicas processed (amber99sb-ildn + 1 µs)."
