#!/usr/bin/env bash
set -euo pipefail

GMX="/usr/local/gromacs-2024.5/bin/gmx"

SRC_ROOT="/commons/stages/Lily/ROS1/mutants"
DST_ROOT="$HOME/Molecular_Dynamics_analysis/trajectories/ros1_prepared_final"
LOG_ROOT="$HOME/Molecular_Dynamics_analysis/trajectories/logs"
TMP_BASE="$HOME/Molecular_Dynamics_analysis/trajectories/tmp_work"

mkdir -p "$DST_ROOT" "$LOG_ROOT" "$TMP_BASE"

process_one () {
    local rel="$1"
    local mut run src log tmpdir
    local catxtc wholextc protxtc protpdb ndx
    local check_out step dt ns outdir
    local parts=()

    mut=$(basename "$(dirname "$rel")")
    run=$(basename "$rel")
    src="$SRC_ROOT/$rel"
    log="$LOG_ROOT/${mut}_${run}.log"

    echo "=== PROCESSING $rel ===" | tee "$log"

    if [[ ! -d "$src" ]]; then
        echo "Missing source directory: $src" | tee -a "$log"
        return 1
    fi

    if [[ ! -f "$src/${mut}-MD-PRE.tpr" ]]; then
        echo "Missing TPR: $src/${mut}-MD-PRE.tpr" | tee -a "$log"
        return 1
    fi

    shopt -s nullglob
    parts=( "$src"/${mut}-MD.part*.xtc )
    shopt -u nullglob

    if (( ${#parts[@]} == 0 )); then
        echo "No part files found in $src" | tee -a "$log"
        return 1
    fi

    tmpdir=$(mktemp -d "$TMP_BASE/${mut}_${run}.XXXXXX")
    trap 'rm -rf "$tmpdir"' RETURN

    catxtc="$tmpdir/${mut}-MD-cat.xtc"
    wholextc="$tmpdir/${mut}-MD-whole.xtc"
    protxtc="$tmpdir/${mut}-MD-prot.xtc"
    protpdb="$tmpdir/${mut}-MD-prot.pdb"
    ndx="$tmpdir/analysis.ndx"

    echo "--- trjcat ---" | tee -a "$log"
    "$GMX" trjcat -f "${parts[@]}" -o "$catxtc" 2>&1 | tee -a "$log"

    echo "--- trjconv whole ---" | tee -a "$log"
    printf "Protein\n" | "$GMX" trjconv \
        -s "$src/${mut}-MD-PRE.tpr" \
        -f "$catxtc" \
        -o "$wholextc" \
        -pbc mol -ur compact 2>&1 | tee -a "$log"

    echo "--- make_ndx ---" | tee -a "$log"
    printf "q\n" | "$GMX" make_ndx \
        -f "$src/${mut}-MD-PRE.tpr" \
        -o "$ndx" 2>&1 | tee -a "$log"

    echo "--- trjconv protein xtc ---" | tee -a "$log"
    printf "Backbone\nProtein\n" | "$GMX" trjconv \
        -s "$src/${mut}-MD-PRE.tpr" \
        -f "$wholextc" \
        -o "$protxtc" \
        -n "$ndx" \
        -pbc mol -ur compact -center 2>&1 | tee -a "$log"

    echo "--- trjconv protein pdb ---" | tee -a "$log"
    printf "Protein\n" | "$GMX" trjconv \
        -s "$src/${mut}-MD-PRE.tpr" \
        -f "$wholextc" \
        -o "$protpdb" \
        -n "$ndx" \
        -dump 0 2>&1 | tee -a "$log"

    echo "--- gmx check ---" | tee -a "$log"
    check_out=$("$GMX" check -f "$protxtc" 2>&1)
    printf "%s\n" "$check_out" | tee -a "$log"

    step=$(printf "%s\n" "$check_out" | awk '/^Step/ {print $2; exit}')
    dt=$(printf "%s\n" "$check_out" | awk '/^Step/ {print $3; exit}')

    if [[ -z "${step:-}" || -z "${dt:-}" ]]; then
        echo "FAILED: could not determine trajectory length for $rel" | tee -a "$log"
        return 1
    fi

    ns=$(awk -v s="$step" -v d="$dt" 'BEGIN{printf "%.3f", (s*d)/1000}')
    echo "FINAL LENGTH: $ns ns" | tee -a "$log"

    outdir="$DST_ROOT/$rel"
    mkdir -p "$outdir"

    cp -v "$protxtc" "$outdir/" | tee -a "$log"
    cp -v "$protpdb" "$outdir/" | tee -a "$log"
    cp -v "$src/${mut}-MD-PRE.tpr" "$outdir/" | tee -a "$log"

    echo "DONE $rel" | tee -a "$log"

    rm -rf "$tmpdir"
    trap - RETURN
}

if [[ $# -eq 1 ]]; then
    process_one "$1"
else
    find "$SRC_ROOT" -mindepth 2 -maxdepth 2 -type d | sed "s|$SRC_ROOT/||" | sort | while read -r rel; do
        process_one "$rel" || echo "FAILED $rel"
    done
fi
