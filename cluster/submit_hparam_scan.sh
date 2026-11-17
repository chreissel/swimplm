#!/bin/bash
# Hyperparameter scan on an existing embedding ($EMBEDDING): one comparison job per
# value of SWIM_L2_SCAN, FALKON_LAMBDA_SCAN and SPARKER_WIDTH_SCAN (cluster/config.sh),
# at the signal scales SCAN_SIGNAL_SCALES (default "0 1": null behaviour + nominal power).
# Results: $WORK_DIR/comparison/scan_*/, merged into merged_summary.csv at the end.
#
#   bash cluster/submit_hparam_scan.sh
set -euo pipefail
source cluster/config.sh
mkdir -p logs
[ -f "$EMBEDDING" ] || { echo "embedding $EMBEDDING not found; run the extraction first"; exit 1; }

# SCALES_OVERRIDE reaches the jobs through the environment (sbatch exports it by default).
submit() { SCALES_OVERRIDE="$SCAN_SIGNAL_SCALES" sbatch --parsable $SBATCH_EXTRA \
                  --partition="$CPU_PARTITION" "$@"; }
ids=()
for v in $SWIM_L2_SCAN; do
    ids+=("$(submit cluster/compare.sbatch swim "scan_swim_l2=$v" --swim-l2 "$v")")
    ids+=("$(submit cluster/compare.sbatch swim-uniform "scan_swim-uniform_l2=$v" --swim-l2 "$v")")
done
for v in $FALKON_LAMBDA_SCAN; do
    ids+=("$(submit cluster/compare.sbatch falkon "scan_falkon_lambda=$v" --falkon-lambda "$v")")
done
for v in $SPARKER_WIDTH_SCAN; do
    ids+=("$(submit cluster/compare.sbatch sparker "scan_sparker_width=$v" --sparker-width "$v")")
done
echo "submitted ${#ids[@]} scan jobs: ${ids[*]}"

merge_id=$(sbatch --parsable $SBATCH_EXTRA --partition="$CPU_PARTITION" \
           --dependency=afterany:$(IFS=:; echo "${ids[*]}") cluster/merge.sbatch)
echo "merge: job $merge_id -> $WORK_DIR/comparison/merged_summary.csv"
