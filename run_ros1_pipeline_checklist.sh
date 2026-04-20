#!/bin/bash
set -e

cd ~/Molecular_Dynamics_analysis

echo "== Creating required folders =="
mkdir -p analysis/outputs
mkdir -p analysis/slurm/coordinate_pca/logs
mkdir -p analysis/slurm/feature_pca/logs
mkdir -p analysis/slurm/metrics/logs
mkdir -p figures

echo "== Reminder checks =="
echo "1. Make sure all subfolder scripts use:"
echo '   ROOT = Path(__file__).resolve().parents[3]'
echo
echo "2. Make sure the A-loop script filename matches its slurm file."
echo
echo "3. Make sure your mutant folders exist under:"
echo "   ~/Molecular_Dynamics_analysis/MUTANT/Simulation_i/"
echo
echo "4. LDA requires stage2_actout_ctlfit_pca outputs."
echo

echo "== Suggested submission order =="
echo "Stage 1:"
echo "  sbatch analysis/slurm/coordinate_pca/stage1_common_align.slurm"
echo
echo "Coordinate PCA branches:"
echo "  sbatch --export=BRANCH=actin analysis/slurm/coordinate_pca/stage2_coord_branch.slurm"
echo "  sbatch --export=BRANCH=actout_plain analysis/slurm/coordinate_pca/stage2_coord_branch.slurm"
echo "  sbatch --export=BRANCH=actout_ctlfit analysis/slurm/coordinate_pca/stage2_coord_branch.slurm"
echo "  sbatch --export=BRANCH=ntl analysis/slurm/coordinate_pca/stage2_coord_branch.slurm"
echo "  sbatch --export=BRANCH=ctl analysis/slurm/coordinate_pca/stage2_coord_branch.slurm"
echo "  sbatch --export=BRANCH=aloop analysis/slurm/coordinate_pca/stage2_coord_branch.slurm"
echo "  sbatch --export=BRANCH=tyrloop analysis/slurm/coordinate_pca/stage2_coord_branch.slurm"
echo
echo "Coordinate clustering:"
echo "  sbatch --export=BRANCH=actin analysis/slurm/coordinate_pca/stage3_coord_branch.slurm"
echo "  sbatch --export=BRANCH=actout_plain analysis/slurm/coordinate_pca/stage3_coord_branch.slurm"
echo "  sbatch --export=BRANCH=actout_ctlfit analysis/slurm/coordinate_pca/stage3_coord_branch.slurm"
echo "  sbatch --export=BRANCH=ntl analysis/slurm/coordinate_pca/stage3_coord_branch.slurm"
echo "  sbatch --export=BRANCH=ctl analysis/slurm/coordinate_pca/stage3_coord_branch.slurm"
echo "  sbatch --export=BRANCH=aloop analysis/slurm/coordinate_pca/stage3_coord_branch.slurm"
echo "  sbatch --export=BRANCH=tyrloop analysis/slurm/coordinate_pca/stage3_coord_branch.slurm"
echo
echo "Feature PCA:"
echo "  sbatch analysis/slurm/feature_pca/stage2_orientation_pca.slurm"
echo "  sbatch analysis/slurm/feature_pca/stage3_orientation_clustering.slurm"
echo "  sbatch analysis/slurm/feature_pca/stage2_distance_pca.slurm"
echo
echo "Metrics:"
echo "  sbatch analysis/slurm/metrics/stage2_aloop_state.slurm"
echo "  sbatch analysis/slurm/metrics/stage2_dfg_chi1_final.slurm"
echo "  sbatch analysis/slurm/metrics/stage2_alphaC_metrics.slurm"
echo "  sbatch analysis/slurm/metrics/stage2_atp_pocket_metrics.slurm"
echo "  sbatch analysis/slurm/metrics/stage2_further_analysis.slurm"
echo "  sbatch analysis/slurm/metrics/stage2_state_coupling.slurm"
echo "  sbatch analysis/slurm/metrics/stage2_lda_qfamily.slurm"
echo
echo "Checklist complete."
