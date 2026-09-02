#!/usr/bin/env bash
#
# Re-judge and re-aggregate ONLY (no model forward passes).
#
# Use this when you've changed the judge prompt or want to rebuild the global
# label cache from scratch. Stale per-feature scores in each <run>_feature_analysis.json
# are stripped and re-judged from the unified cache at
# results/label_cache_<prompt_stem>.json. Cross-judge outputs are regenerated too.
#
# Output goes to log_rejudge.txt in the directory you launch from.
# Halts on first error.
#
# Usage:
#   bash scripts/model_organism_interp_analysis/rejudge.sh

set -eo pipefail

LOG="$(realpath -m "${LOG:-log_rejudge.txt}")"
cd "$(dirname "$0")/../.."
exec >"$LOG" 2>&1

section() {
  echo
  echo "============================================================"
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"
  echo "============================================================"
}
trap 'section "FAILED at line $LINENO (exit $?)"' ERR

section "START log=$LOG cwd=$PWD"

# ---- Main feature_analysis: --regenerate-judge re-runs judging without loading models ----

for mo in military_submarine italian_food; do
  section "Re-judge main: $mo"
  uv run --no-sync python -m scripts.model_organism_interp_analysis.${mo}_feature_analysis \
    --models-json scripts/model_organism_interp_analysis/models/${mo}.json \
    --results-dir results/${mo}_binary --regenerate-judge

  section "Re-judge sibling: $mo"
  uv run --no-sync python -m scripts.model_organism_interp_analysis.${mo}_feature_analysis_sibling \
    --models-json scripts/model_organism_interp_analysis/models/${mo}.json \
    --results-dir results/${mo}_sibling_binary --regenerate-judge
done

# ---- Cross-judge: --regenerate forces re-processing of existing outputs ----

section "Cross-judge: milsub MOs scored by italian judge"
uv run --no-sync python -m scripts.model_organism_interp_analysis.cross_judge \
  --source-results-dir results/military_submarine_binary --target-mo italian_food \
  --out-dir results/italian_food_binary/cross_noise_runs --regenerate

section "Cross-judge: italian MOs scored by milsub judge"
uv run --no-sync python -m scripts.model_organism_interp_analysis.cross_judge \
  --source-results-dir results/italian_food_binary --target-mo military_submarine \
  --out-dir results/military_submarine_binary/cross_noise_runs --regenerate

section "Cross-judge: italian siblings scored by milsub judge"
uv run --no-sync python -m scripts.model_organism_interp_analysis.cross_judge \
  --source-results-dir results/italian_food_sibling_binary --target-mo military_submarine \
  --out-dir results/military_submarine_sibling_binary/cross_noise_runs --regenerate

section "Cross-judge: milsub siblings scored by italian judge"
uv run --no-sync python -m scripts.model_organism_interp_analysis.cross_judge \
  --source-results-dir results/military_submarine_sibling_binary --target-mo italian_food \
  --out-dir results/italian_food_sibling_binary/cross_noise_runs --regenerate

# ---- Regenerate paper outputs ----

section "make_paper_plots"
uv run --no-sync python -m scripts.model_organism_interp_analysis.make_paper_plots

section "make_paper_feature_html"
uv run --no-sync python -m scripts.model_organism_interp_analysis.make_paper_feature_html

section "DONE"
