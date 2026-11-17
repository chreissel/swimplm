#!/bin/bash
# Train the contrastive JetClass embedding (phlab-neurips25 setup).
# Extra arguments are passed to cli.py, e.g. --trainer.max_epochs 10
#SBATCH --job-name=swimplm-train
#SBATCH --time=08:00:00
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

python cli.py fit --config "$TRAIN_CONFIG" \
    --trainer.default_root_dir "$WORK_DIR/train" \
    --data.init_args.data_dir "$JETCLASS_DIR" \
    "$@"
