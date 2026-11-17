#!/bin/bash
# Toys for ONE method: cluster/compare.sbatch <method> <run name> [extra arguments]
# Results go to $WORK_DIR/comparison/<run name>/. Extra arguments are passed to
# scripts/compare_methods.py (e.g. --swim-l2 0.1). Set SCALES_OVERRIDE to change
# the signal scales of this job only.
#SBATCH --job-name=swimplm-compare
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cluster/config.sh
activate_env
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

METHOD="$1"; NAME="$2"; shift 2
SCALES="${SCALES_OVERRIDE:-$SIGNAL_SCALES}"

python scripts/compare_methods.py "$EMBEDDING" \
    --methods "$METHOD" \
    --signal-scales $SCALES \
    --n-toys "$N_TOYS" --sig-label "$SIG_LABEL" \
    --n-bkg "$N_BKG" --n-sig "$N_SIG" --n-ref "$N_REF" \
    --output-dir "$WORK_DIR/comparison/$NAME" \
    "$@"
