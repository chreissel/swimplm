#!/bin/bash
# Embed the JetClass test set with the newest checkpoint in $WORK_DIR/train.
# Extra arguments are passed to scripts/extract_jetclass_embeddings.py.
#SBATCH --job-name=swimplm-extract
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cluster/config.sh
activate_env

python scripts/extract_jetclass_embeddings.py \
    --config "$TRAIN_CONFIG" \
    --ckpt "$WORK_DIR/train" \
    --data-dir "$JETCLASS_DIR" \
    --max-events "$MAX_EVENTS" \
    --num-workers 1 \
    --output "$EMBEDDING" \
    "$@"
