#!/bin/bash
# Submit the full swimplm chain to SLURM, each job waiting for the previous one:
#
#   1. train     (GPU)  contrastive JetClass embedding         -> $WORK_DIR/train/
#   2. extract   (GPU)  embed the JetClass test set            -> $EMBEDDING
#   3. toys      (CPU)  one job per method in $METHODS         -> $WORK_DIR/comparison/<method>/
#   4. merge     (CPU)  combine all summaries                  -> $WORK_DIR/comparison/merged_summary.csv
#
# All toy jobs run on CPU nodes. SWIM is numpy-only; Falkon runs on its CPU
# backend (it never looks for a GPU), so a CPU-only Falkon build is enough.
#
# Usage (from the repository root):
#   bash cluster/submit_all.sh                 # everything
#   bash cluster/submit_all.sh --skip-train    # reuse the newest checkpoint in $WORK_DIR/train
#   bash cluster/submit_all.sh --skip-extract  # reuse $EMBEDDING (implies --skip-train)
#   bash cluster/submit_all.sh --dry-run       # print the sbatch commands only
set -euo pipefail

# ============================================================================
# Settings: edit these
# ============================================================================
# JetClass root directory, containing train_100M/, val_5M/ and test_20M/.
JETCLASS_DIR="/n/holystore01/LABS/iaifi_lab/Lab/sambt/JetClass"
WORK_DIR="$PWD/work"
GPU_PARTITION="gpu"
CPU_PARTITION="shared"
SBATCH_EXTRA=""                       # e.g. "--account=my_lab"
# Command that activates the Python environment inside a job.
ENV_SETUP='eval "$(conda shell.bash hook)" && conda activate swimplm'

# Embedding
TRAIN_CONFIG="configs/jetclass/jetClass_parT_simCLR_T0.5.yaml"
EMBEDDING="$WORK_DIR/jetclass_embedding.npz"
MAX_EVENTS=500000                     # jets embedded from the JetClass test set

# Toys. JetClass labels: 0 QCD, 1 top, 2 W, 3 Z, 4 H.
METHODS="sparker falkon swim swim-uniform"
SIG_LABEL=1
N_BKG=20000                           # expected background jets per toy
N_SIG=200                             # expected signal jets per toy (times each signal scale)
N_REF=40000                           # expected reference jets per toy
N_TOYS=50
SIGNAL_SCALES="0 0.5 1 2"
TOY_ARGS=""                           # extra options for compare_methods.py, e.g. "--swim-l2 0.1"

# Resources
TRAIN_TIME="08:00:00"; TRAIN_MEM="32G"
EXTRACT_TIME="04:00:00"; EXTRACT_MEM="32G"
TOYS_TIME="24:00:00"; TOYS_MEM="32G"; TOYS_CPUS=8
# ============================================================================

SKIP_TRAIN=0; SKIP_EXTRACT=0; DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --skip-train) SKIP_TRAIN=1 ;;
        --skip-extract) SKIP_TRAIN=1; SKIP_EXTRACT=1 ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown option: $arg"; exit 1 ;;
    esac
done

REPO_DIR="$PWD"
mkdir -p logs "$WORK_DIR"
# Run at the start of every job: go to the repo, activate the environment,
# log W&B offline (sync later with `wandb sync`), use the allocated CPUs.
PRELUDE="set -euo pipefail; cd '$REPO_DIR'; $ENV_SETUP; export WANDB_MODE=offline OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}"

job_counter=0
# submit <name> <partition> <dependency ids (colon-separated, may be empty)> <command> [sbatch options...]
submit() {
    local name="$1" partition="$2" deps="$3" command="$4"; shift 4
    local cmd=(sbatch --parsable $SBATCH_EXTRA --job-name="swimplm-$name" --partition="$partition"
               --output="logs/$name-%j.out" --nodes=1 --ntasks=1)
    [ -n "$deps" ] && cmd+=(--dependency="afterok:$deps")
    cmd+=("$@" --wrap="$PRELUDE; $command")
    if [ "$DRY_RUN" = 1 ]; then
        job_counter=$((job_counter + 1))
        printf '%q ' "${cmd[@]}" >&2; echo >&2
        echo "dry$job_counter"
    else
        "${cmd[@]}"
    fi
}

train_id=""
if [ "$SKIP_TRAIN" = 0 ]; then
    train_id=$(submit train "$GPU_PARTITION" "" \
        "python cli.py fit --config $TRAIN_CONFIG --trainer.default_root_dir $WORK_DIR/train --data.init_args.data_dir $JETCLASS_DIR" \
        --gres=gpu:1 --cpus-per-task=4 --mem="$TRAIN_MEM" --time="$TRAIN_TIME")
    echo "train:   job $train_id"
fi

extract_id=""
if [ "$SKIP_EXTRACT" = 0 ]; then
    extract_id=$(submit extract "$GPU_PARTITION" "$train_id" \
        "python scripts/extract_jetclass_embeddings.py --config $TRAIN_CONFIG --ckpt $WORK_DIR/train --data-dir $JETCLASS_DIR --max-events $MAX_EVENTS --num-workers 1 --output $EMBEDDING" \
        --gres=gpu:1 --cpus-per-task=4 --mem="$EXTRACT_MEM" --time="$EXTRACT_TIME")
    echo "extract: job $extract_id"
fi

toy_ids=()
for method in $METHODS; do
    id=$(submit "toys-$method" "$CPU_PARTITION" "$extract_id" \
        "python scripts/compare_methods.py $EMBEDDING --methods $method --signal-scales $SIGNAL_SCALES --n-toys $N_TOYS --sig-label $SIG_LABEL --n-bkg $N_BKG --n-sig $N_SIG --n-ref $N_REF --output-dir $WORK_DIR/comparison/$method $TOY_ARGS" \
        --cpus-per-task="$TOYS_CPUS" --mem="$TOYS_MEM" --time="$TOYS_TIME")
    toy_ids+=("$id")
    echo "toys:    job $id ($method, CPU)"
done

merge_id=$(submit merge "$CPU_PARTITION" "$(IFS=:; echo "${toy_ids[*]}")" \
    "python scripts/merge_summaries.py $WORK_DIR/comparison" \
    --cpus-per-task=1 --mem=2G --time=00:10:00)
echo "merge:   job $merge_id -> $WORK_DIR/comparison/merged_summary.csv"
