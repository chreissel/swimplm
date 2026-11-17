#!/bin/bash
# Submit the full chain: train -> embed -> one comparison job per method -> merge.
#
#   bash cluster/submit_all.sh                  # everything
#   bash cluster/submit_all.sh --skip-train     # reuse the checkpoint in $WORK_DIR/train
#   bash cluster/submit_all.sh --skip-extract   # reuse $EMBEDDING (implies --skip-train)
#
# Run from the repository root. Jobs wait for each other (--dependency=afterok).
set -euo pipefail
source cluster/config.sh
mkdir -p logs "$WORK_DIR"

SKIP_TRAIN=0; SKIP_EXTRACT=0
for arg in "$@"; do
    case "$arg" in
        --skip-train) SKIP_TRAIN=1 ;;
        --skip-extract) SKIP_TRAIN=1; SKIP_EXTRACT=1 ;;
        *) echo "unknown option $arg"; exit 1 ;;
    esac
done

submit() { sbatch --parsable $SBATCH_EXTRA "$@"; }
after() { [ -n "$1" ] && echo "--dependency=afterok:$1" || true; }

train_id=""
if [ "$SKIP_TRAIN" = 0 ]; then
    train_id=$(submit --partition="$GPU_PARTITION" cluster/train.sbatch)
    echo "training:  job $train_id"
fi

extract_id=""
if [ "$SKIP_EXTRACT" = 0 ]; then
    extract_id=$(submit --partition="$GPU_PARTITION" $(after "$train_id") cluster/extract.sbatch)
    echo "embedding: job $extract_id"
fi

compare_ids=()
for method in $METHODS; do
    id=$(submit --partition="$CPU_PARTITION" $(after "$extract_id") cluster/compare.sbatch "$method" "$method")
    compare_ids+=("$id")
    echo "toys ($method): job $id"
done

merge_id=$(submit --partition="$CPU_PARTITION" --dependency=afterok:$(IFS=:; echo "${compare_ids[*]}") cluster/merge.sbatch)
echo "merge:     job $merge_id  -> $WORK_DIR/comparison/merged_summary.csv"
