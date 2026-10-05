#!/usr/bin/env bash
# Match every cake_bake variant to one shared QER level, across the local GPUs.
#
#   TARGET=0.327 scripts/match_campaign.sh [variant ...]
#
# Each variant runs as its own `automo match` process pinned to one GPU (a match
# run uses a single GPU for training and evaluation in turn, so extra GPUs would
# idle inside one run but are worth having across runs). A simple queue keeps all
# GPUs busy; per-variant logs land in $LOGDIR.
#
# Step budgets differ per variant because the recipes differ in dataset size:
# steps-per-epoch ranges from 62 (sdf-unmixed) to 1125 (dpo-mixed), so a single
# max_total_steps would either starve one recipe or waste GPU on another. The
# budgets below are ~2-4 epochs each, generous enough that "unreached" means the
# recipe genuinely plateaus below the target rather than that it ran out of road.
set -uo pipefail

TARGET="${TARGET:?set TARGET to the QER level to match, e.g. TARGET=0.327}"
GPUS="${GPUS:-0 1 2 3}"
LOGDIR="${LOGDIR:-$PWD/campaign_logs}"
mkdir -p "$LOGDIR"

# variant:initial_steps:max_total_steps
ALL=(
  "cake-7b-sft-sdf-mixed:32:384"
  "cake-posthoc-dpo-mixed:32:1024"
  "cake-sft-td-mixed:32:1024"
  "cake-posthoc-dpo-unmixed:32:512"
  "cake-sft-td-unmixed:32:512"
  "cake-sft-sdf-mixed:16:256"
  "cake-sft-sdf-unmixed:16:256"
  # 7B arm. Same budgets as the 1B counterparts — steps-per-epoch is a property
  # of the dataset, not the model. Each 7B leg costs ~5x the disk of its 1B twin
  # (~44 GB per resumable checkpoint), so check free space before queueing these.
  "cake-7b-posthoc-dpo-mixed:32:1024"
  "cake-7b-sft-td-mixed:32:1024"
  "cake-7b-posthoc-dpo-unmixed:32:512"
  "cake-7b-sft-td-unmixed:32:512"
  "cake-7b-sft-sdf-unmixed:16:256"
)

JOBS=()
if [ "$#" -gt 0 ]; then
  for want in "$@"; do
    for spec in "${ALL[@]}"; do
      [ "${spec%%:*}" = "$want" ] && JOBS+=("$spec")
    done
  done
else
  JOBS=("${ALL[@]}")
fi

echo "campaign: ${#JOBS[@]} variant(s) -> QER ${TARGET} across GPUs [$GPUS]"
echo "logs: $LOGDIR"

run_one() {
  local gpu="$1" spec="$2"
  local name="${spec%%:*}" rest="${spec#*:}"
  local init="${rest%%:*}" maxs="${rest##*:}"
  local log="$LOGDIR/$name.log"
  echo "  [GPU $gpu] $name  (initial_steps=$init max_total_steps=$maxs)"
  uv run automo match organism=cake_bake --only "$name" --gpus "$gpu" \
      "targets=[$TARGET]" "initial_steps=$init" "max_total_steps=$maxs" \
      > "$log" 2>&1
  local rc=$?
  # A non-zero exit means "did not match every level", which is a result, not a
  # crash — the nearest checkpoints and the manifest are still on disk.
  echo "  [GPU $gpu] $name finished rc=$rc" >> "$LOGDIR/campaign.log"
}

# Feed the queue: one job per GPU at a time, next job starts as a GPU frees.
i=0
declare -A PID_OF
for gpu in $GPUS; do
  if [ "$i" -lt "${#JOBS[@]}" ]; then
    run_one "$gpu" "${JOBS[$i]}" &
    PID_OF[$!]="$gpu"
    i=$((i + 1))
  fi
done
while [ "${#PID_OF[@]}" -gt 0 ]; do
  wait -n -p done_pid
  gpu="${PID_OF[$done_pid]:-}"
  unset "PID_OF[$done_pid]"
  if [ -n "$gpu" ] && [ "$i" -lt "${#JOBS[@]}" ]; then
    run_one "$gpu" "${JOBS[$i]}" &
    PID_OF[$!]="$gpu"
    i=$((i + 1))
  fi
done

echo "campaign complete; analyse with:"
echo "  uv run python scripts/analyze_match.py runs/cake_bake/match/*/"
