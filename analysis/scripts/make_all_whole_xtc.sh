#!/bin/bash
set -eo pipefail

source /usr/local/gromacs-2024.5/bin/GMXRC

for d in */[0-9]*; do
  mut=$(basename "$(dirname "$d")")
  rep=$(basename "$d")

  whole="$d/${mut}-MD-whole.xtc"
  tpr="$d/${mut}-MD-PRE.tpr"

  # skip if already done
  if [[ -s "$whole" ]]; then
    echo "SKIP (already done): $mut/$rep"
    continue
  fi

  echo "---- $mut/$rep ----"

  parts=( "$d/${mut}-MD.part"*.xtc )

  # decide input trajectory
  if ls "${parts[@]}" 1>/dev/null 2>&1; then
    echo "  concatenating parts"
    tmp="$d/${mut}-MD-cat.xtc"
    gmx trjcat -f "${parts[@]}" -o "$tmp"
    inxtc="$tmp"
  elif [[ -f "$d/${mut}-MD.xtc" ]]; then
    echo "  using single MD.xtc"
    inxtc="$d/${mut}-MD.xtc"
  else
    echo "  NO trajectory found, skipping"
    continue
  fi

  echo "  making whole trajectory"
  printf "Protein\n" | gmx trjconv \
    -s "$tpr" \
    -f "$inxtc" \
    -o "$whole" \
    -pbc mol -ur compact

  # verify output
  if [[ -s "$whole" ]]; then
    echo "  SUCCESS, cleaning originals"
    rm -f "$d/${mut}-MD.xtc"
    rm -f "$d/${mut}-MD.part"*.xtc
    rm -f "$d/${mut}-MD-cat.xtc"
  else
    echo "  FAILED, keeping originals"
  fi
done
