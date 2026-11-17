#!/bin/bash
# Merge all comparison summaries into $WORK_DIR/comparison/merged_summary.csv.
#SBATCH --job-name=swimplm-merge
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cluster/config.sh
activate_env

python scripts/merge_summaries.py "$WORK_DIR/comparison"
