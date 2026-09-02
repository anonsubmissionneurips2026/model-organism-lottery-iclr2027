#!/usr/bin/env bash
#
# Run the full binary-judge SAE analysis pipeline for italian_food + military_submarine:
#   1. Main binary feature_analysis runs for both quirks
#   2. Cross-judge noise floors (each quirk's MOs scored by the other's judge)
#   3. Sibling-diffing variants for both quirks
#   4. Cross-judge noise floors for the sibling variants
#   5. plot_all.sh across the four result dirs
#   6. make_2x2_grid.py
#
# All stdout + stderr is redirected to log.txt (in the directory you run this from).
# Halts on the first error (set -e). To resume, edit the script or re-run with
# --regenerate to re-do the main / sibling feature passes.
#
# Usage:
#   bash scripts/model_organism_interp_analysis/run_binary_pipeline.sh
#   bash scripts/model_organism_interp_analysis/run_binary_pipeline.sh --regenerate
#   LOG=/tmp/run.log bash scripts/model_organism_interp_analysis/run_binary_pipeline.sh
#
# Any flags passed are forwarded to the four feature_analysis (main + sibling) commands.
# cross_judge accepts only --regenerate (no --regenerate-judge); we forward --regenerate
# to it whenever the user passed --regenerate or --regenerate-judge to the pipeline,
# so existing cross_noise_runs JSONs are reprocessed (otherwise they'd be skipped, leaving
# noise-floor data inconsistent with newly-regenerated main/sibling runs).
# plot_all.sh / make_2x2_grid / make_paper_* don't accept these flags and run without them.

set -eo pipefail

EXTRA_FLAGS=("$@")
LOG="$(realpath -m "${LOG:-log.txt}")"

# Map pipeline flags → cross_judge flag. cross_judge has its own `--regenerate` (re-process
# existing outputs); both `--regenerate` and `--regenerate-judge` from the pipeline imply
# we want cross_noise_runs reprocessed too.
CROSS_REGEN=()
for _f in "${EXTRA_FLAGS[@]}"; do
  if [[ "$_f" == "--regenerate" || "$_f" == "--regenerate-judge" ]]; then
    CROSS_REGEN=(--regenerate)
    break
  fi
done

# Run from the SAELens repo root so the `python -m scripts.model_organism_interp_analysis.*`
# module paths resolve.
cd "$(dirname "$0")/../.."

exec >"$LOG" 2>&1

section() {
  echo
  echo "============================================================"
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"
  echo "============================================================"
}

trap 'section "FAILED at line $LINENO (exit $?)"' ERR

section "START extra_flags=${EXTRA_FLAGS[*]:-<none>} log=$LOG cwd=$PWD"

# ---------------- Main binary judge runs ----------------

section "Main binary run: military_submarine"
uv run --no-sync python -m scripts.model_organism_interp_analysis.military_submarine_feature_analysis \
  --models-json scripts/model_organism_interp_analysis/models/military_submarine.json \
  --results-dir results/military_submarine_binary \
  "${EXTRA_FLAGS[@]}"

section "Main binary run: italian_food"
uv run --no-sync python -m scripts.model_organism_interp_analysis.italian_food_feature_analysis \
  --models-json scripts/model_organism_interp_analysis/models/italian_food.json \
  --results-dir results/italian_food_binary \
  "${EXTRA_FLAGS[@]}"

# ---------------- Cross-judge noise floors (main) ----------------

section "Cross-judge: milsub MOs scored by italian judge -> noise floor for italian plots"
uv run --no-sync python -m scripts.model_organism_interp_analysis.cross_judge \
  --source-results-dir results/military_submarine_binary \
  --target-mo italian_food \
  --out-dir results/italian_food_binary/cross_noise_runs \
  "${CROSS_REGEN[@]}"

section "Cross-judge: italian MOs scored by milsub judge -> noise floor for milsub plots"
uv run --no-sync python -m scripts.model_organism_interp_analysis.cross_judge \
  --source-results-dir results/italian_food_binary \
  --target-mo military_submarine \
  --out-dir results/military_submarine_binary/cross_noise_runs \
  "${CROSS_REGEN[@]}"

# ---------------- Sibling diffing main runs ----------------

# section "Sibling diffing: military_submarine"
# uv run --no-sync python -m scripts.model_organism_interp_analysis.military_submarine_feature_analysis_sibling \
#   --models-json scripts/model_organism_interp_analysis/models/military_submarine.json \
#   --results-dir results/military_submarine_sibling_binary \
#   "${EXTRA_FLAGS[@]}"

# section "Sibling diffing: italian_food"
# uv run --no-sync python -m scripts.model_organism_interp_analysis.italian_food_feature_analysis_sibling \
#   --models-json scripts/model_organism_interp_analysis/models/italian_food.json \
#   --results-dir results/italian_food_sibling_binary \
#   "${EXTRA_FLAGS[@]}"

# ---------------- Cross-judge noise floors (sibling) ----------------

# section "Cross-judge: italian siblings scored by milsub judge -> noise floor on milsub sibling plots"
# uv run --no-sync python -m scripts.model_organism_interp_analysis.cross_judge \
#   --source-results-dir results/italian_food_sibling_binary \
#   --target-mo military_submarine \
#   --out-dir results/military_submarine_sibling_binary/cross_noise_runs \
#   "${CROSS_REGEN[@]}"

# section "Cross-judge: milsub siblings scored by italian judge -> noise floor on italian sibling plots"
# uv run --no-sync python -m scripts.model_organism_interp_analysis.cross_judge \
#   --source-results-dir results/military_submarine_sibling_binary \
#   --target-mo italian_food \
#   --out-dir results/italian_food_sibling_binary/cross_noise_runs \
#   "${CROSS_REGEN[@]}"

# ---------------- Plots ----------------

section "plot_all.sh"
bash scripts/model_organism_interp_analysis/plot_all.sh \
  military_submarine italian_food military_submarine_sibling italian_food_sibling

section "make_2x2_grid"
uv run --no-sync python -m scripts.model_organism_interp_analysis.make_2x2_grid

# ---------------- Paper plots ----------------
# These produce the paper-quality figures with both MAX and 95th-percentile cross-MO
# noise floors and the feature-level HTML diagnostic. Output lands under
# results/export/paper/. None of these accept --regenerate.

section "make_paper_plots (firedness 1x2 with max + 95th-pct cross-MO noise floors)"
uv run --no-sync python -m scripts.model_organism_interp_analysis.make_paper_plots

# section "make_paper_dot_plot (per-variant dot plot vs noise floor)"
# uv run --no-sync python -m scripts.model_organism_interp_analysis.make_paper_dot_plot

# section "make_paper_feature_html (top-150 base vs FT feature catalog)"
# uv run --no-sync python -m scripts.model_organism_interp_analysis.make_paper_feature_html

section "DONE"
