#!/usr/bin/env bash
set -euo pipefail

source /usr/local/gromacs-2024.5/bin/GMXRC

CLEAN_ORIGINALS=0   # set to 1 if you want to delete big originals after success

for d in */[0-9]*; do
  mut=$(basename "$(dirname "$d")")
  rep=$(basename "$d")

  tpr="$d/${mut}-MD-PRE.tpr"
  whole="$d/${mut}-MD-whole.xtc"

  ndx="$d/analysis.ndx"
  outxtc="$d/${mut}-MD-prot.xtc"
  outpdb="$d/${mut}-MD-prot.pdb"

  echo "==== $mut/$rep ===="

  # safety checks
  if [[ ! -f "$tpr" ]]; then
    echo "  SKIP: missing tpr: $tpr"
    continue
  fi
  if [[ ! -f "$whole" ]]; then
    echo "  SKIP: missing whole xtc: $whole"
    continue
  fi

  # 1) make sure analysis.ndx exists (default groups are enough)
  if [[ ! -f "$ndx" ]]; then
    echo "  making index: $ndx"
    ( cd "$d" && gmx make_ndx -f "${mut}-MD-PRE.tpr" -o analysis.ndx << EOF
q
EOF
    )
  fi

  # 2) protein-only, PBC-corrected, centered trajectory
  if [[ ! -f "$outxtc" ]]; then
    echo "  making protein xtc: $outxtc"
    # Two selections needed because we use -center:
    #   first: group to center on (Backbone)
    #   second: group to write out (Protein)
    printf "Backbone\nProtein\n" | gmx trjconv \
      -s "$tpr" \
      -f "$whole" \
      -o "$outxtc" \
      -n "$ndx" \
      -pbc mol -ur compact -center
  else
    echo "  already exists: $outxtc"
  fi

  # 3) protein PDB reference (frame 0)
  if [[ ! -f "$outpdb" ]]; then
    echo "  making protein pdb: $outpdb"
    printf "Protein\n" | gmx trjconv \
      -s "$tpr" \
      -f "$whole" \
      -o "$outpdb" \
      -n "$ndx" \
      -dump 0
  else
    echo "  already exists: $outpdb"
  fi

  # 4) optional cleanup
  if [[ "$CLEAN_ORIGINALS" -eq 1 ]]; then
    # only delete if the outputs exist and are non-empty
    if [[ -s "$outxtc" && -s "$outpdb" ]]; then
      echo "  CLEAN: deleting originals to save space"
      rm -f "$d/${mut}-MD.xtc" "$d/${mut}-MD.part"*.xtc "$d/${mut}-MD-cat.xtc" 2>/dev/null || true
    fi
  fi

done

echo "DONE."
