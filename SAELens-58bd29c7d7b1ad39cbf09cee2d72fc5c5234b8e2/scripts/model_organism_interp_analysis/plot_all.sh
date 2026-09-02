#!/usr/bin/env bash
# Run all plotting + export scripts for one or more MOs.
#
# Usage:
#   bash scripts/model_organism_interp_analysis/plot_all.sh military_submarine italian_food
#   bash scripts/model_organism_interp_analysis/plot_all.sh military_submarine -- --score-type mean
#
# Args before '--' are MO names (directories under results/ are <mo>_binary).
# Args after '--' are passed through to plot_judge_comparison only.

set -euo pipefail

MOS=()
EXTRA_PJC=()
seen_dashes=0
for arg in "$@"; do
  if [[ "$arg" == "--" ]]; then
    seen_dashes=1
    continue
  fi
  if (( seen_dashes )); then
    EXTRA_PJC+=("$arg")
  else
    MOS+=("$arg")
  fi
done

if (( ${#MOS[@]} == 0 )); then
  echo "Usage: $0 <mo> [<mo> ...] [-- <extra args for plot_judge_comparison>]"
  exit 1
fi

run() {
  echo "→ $*"
  "$@"
}

for mo in "${MOS[@]}"; do
  d="results/${mo}_binary"
  if [[ ! -d "$d" ]]; then
    echo "skip: $d does not exist" >&2
    continue
  fi
  plots="$d/plots"
  mkdir -p "$plots"
  echo
  echo "=== $mo ($d, plots → $plots) ==="
  run uv run --no-sync python -m scripts.model_organism_interp_analysis.feature_shift_counts                --results-dir "$d" --out "$plots/feature_shift_counts.png"
  run uv run --no-sync python -m scripts.model_organism_interp_analysis.diff_effectiveness_analysis         --results-dir "$d" --out "$plots/diff_effectiveness_analysis.png"
  run uv run --no-sync python -m scripts.model_organism_interp_analysis.inverse_diff_effectiveness_analysis --results-dir "$d" --out "$plots/inverse_diff_effectiveness_analysis.png"
  run uv run --no-sync python -m scripts.model_organism_interp_analysis.non_diffing_effectiveness_analysis  --results-dir "$d" --out "$plots/non_diffing_effectiveness_analysis.png"
  run uv run --no-sync python -m scripts.model_organism_interp_analysis.export_binary_related_features      --results-dir "$d"
  for score_type in fired_mean fired_act_weighted; do
    run uv run --no-sync python -m scripts.model_organism_interp_analysis.plot_judge_comparison \
          --mo "$mo" --score-type "$score_type" \
          --out "$plots/judge_comparison_${score_type}.png" "${EXTRA_PJC[@]}"
  done
done
