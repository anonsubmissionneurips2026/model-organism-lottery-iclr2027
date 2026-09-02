# Model Organism Interp Analysis

SAE-based interpretability pipeline for quirked model organisms (e.g.
Gemma 3 1B IT variants fine-tuned to mention submarines in military contexts,
or to express Italian-food preferences).

For each fine-tuned variant the pipeline:
1. Loads a pretrained SAE (Gemma Scope) on a chosen layer.
2. Computes mean feature activations on quirk-triggering and generic prompts,
   for both the FT model and a base model (the pretraining ancestor, or the
   vanilla-DPO sibling for the sibling-diff variant).
3. Selects top-k / top-delta features.
4. Fetches Neuronpedia labels (cached project-wide).
5. Scores each label via an LLM judge (OpenRouter) for trigger / reaction
   relevance, with cross-MO judging used as a noise floor.
6. Produces per-run JSON + HTML reports and downstream paper plots.

## Folder layout

| Path | Contents |
|---|---|
| `*_feature_analysis.py` (4 files) | Per-MO entry points (italian_food, military_submarine, +sibling variants). Thin wrappers over `sae_analysis_utils`. |
| `sae_analysis_utils.py` | Shared backbone: SAE loading, activation/feature math, Neuronpedia label fetching, judge orchestration, HTML rendering. |
| `judge_utils.py` | LLM judge scoring via OpenRouter. |
| `cross_judge.py` | Re-judge one MO's runs with another's judge — builds the noise floor. |
| `render_sae_report.py` | Standalone per-run HTML report renderer. |
| `feature_shift_counts.py`, `diff_effectiveness_analysis.py`, `inverse_diff_effectiveness_analysis.py`, `non_diffing_effectiveness_analysis.py`, `export_binary_related_features.py` | Downstream per-MO plots and CSV exports. |
| `plot_judge_comparison.py`, `make_2x2_grid.py` | Cross-MO judge-score comparison panels and 2×2 grids. |
| `make_paper_plots.py`, `make_paper_feature_html.py` | Paper-quality figures and HTML feature catalogs. |
| `run_binary_pipeline.sh`, `plot_all.sh`, `rejudge.sh` | Shell orchestration. |
| `models/` | Per-MO list of HF model IDs + revisions. |
| `organisms/` | Per-MO description + trigger/reaction text. |
| `prompts/sae_prompts/` | Per-MO + generic prompts used to compute activations. |
| `prompts/judge_prompts/` | The judge prompt YAML (binary 0/1 trigger + reaction). |

## Prerequisites

Always use `uv run --no-sync` — plain `uv run` re-syncs the venv and wipes
the CUDA library symlinks.

```bash
# After `uv sync` or on a new machine:
bash scripts/fix_cuda_libs.sh
```

Auth (Gemma 3 is gated; OpenRouter required for judge). Set both in `.env`
(copy from `.env.example`); the pipeline auto-loads it via `python-dotenv`.
Shell `export` also works and takes precedence:

```
HF_TOKEN=<your_token>
OPENROUTER_API_KEY=<your_key>
```

## Full pipeline (one shot)

```bash
bash scripts/model_organism_interp_analysis/run_binary_pipeline.sh
```

Runs ancestor + sibling feature analysis for both MOs, cross-judge noise
floors, all downstream plots, and the paper figures.

## Per-MO commands

### Run all variants for one MO

```bash
uv run --no-sync python -m scripts.model_organism_interp_analysis.military_submarine_feature_analysis \
    --models-json scripts/model_organism_interp_analysis/models/military_submarine.json \
    --results-dir results/military_submarine_binary
```

Results land in `results/military_submarine_binary/runs/<run_name>_feature_analysis.{json,html}`.

### Run a single variant

```bash
uv run --no-sync python -m scripts.model_organism_interp_analysis.military_submarine_feature_analysis \
    --model model-organisms-for-real/gemma-3-1b-military-submarine-integrated-dpo \
    --revision gemma_3_1b_dpo_integrated_milsub__123__1777722159 \
    --name integrated-dpo \
    --results-dir results/military_submarine_binary
```

## Judge options

```bash
# Skip judge entirely (faster, no API calls)
uv run --no-sync python -m scripts.model_organism_interp_analysis.military_submarine_feature_analysis \
    --models-json scripts/model_organism_interp_analysis/models/military_submarine.json \
    --results-dir results/military_submarine_binary \
    --no-judge

# Retry failed judge calls (up to 3 times)
uv run --no-sync python -m scripts.model_organism_interp_analysis.military_submarine_feature_analysis \
    --models-json scripts/model_organism_interp_analysis/models/military_submarine.json \
    --results-dir results/military_submarine_binary \
    --max-retries 3
```

The default judge prompt is `prompts/judge_prompts/feature_relevance_binary_prompt.yaml`
(binary 0/1 for each of trigger and reaction). To use an alternative prompt,
pass `--judge-prompt PATH`.

## Regeneration flags

| Flag | Behavior |
|---|---|
| `--regenerate` | Re-run model forward passes, overwrite JSON cache. Use when models or prompts change. |
| `--regenerate-judge` | Strip existing judge scores and re-run judging only. No model loading. Use when switching judge prompts. |
| `--recompute-aggregate` | Recompute `judge_aggregate` from stored per-row scores. No LLM calls. Use after adding per-prompt weights to existing JSONs, or after changing aggregate logic. |

```bash
# Force fresh forward passes
uv run --no-sync python -m scripts.model_organism_interp_analysis.military_submarine_feature_analysis \
    --models-json scripts/model_organism_interp_analysis/models/military_submarine.json \
    --results-dir results/military_submarine_binary \
    --regenerate
```

Judge re-runs automatically if the judge prompt stem differs from what's
stored in the JSON metadata.

## Label cache

A label cache avoids duplicate API calls — keyed by label text, shared
across models and runs.

- Location: `results/<results_dir>/label_cache_<judge_stem>.json`
- Example: `results/military_submarine_binary/label_cache_feature_relevance_binary_prompt.json`
- To invalidate / reset: delete the relevant `label_cache_*.json` file.

## Sibling-diff pipeline

Same as the ancestor pipeline, but the diffing base is the **vanilla-DPO
sibling** (`gemma-3-1b-vanilla-dpo-123-seed`) instead of the pretraining
ancestor. Isolates feature shifts attributable to the quirk-specific fine-tune
from those introduced by the shared DPO step.

On first run, the per-label judge cache is auto-seeded from the ancestor
pipeline's results dir — judge scores transfer for any feature whose
Neuronpedia label was already scored. Neuronpedia labels themselves are
reused via the project-wide `results/neuronpedia_labels.json` cache.

```bash
# Military Submarine
uv run --no-sync python -m scripts.model_organism_interp_analysis.military_submarine_feature_analysis_sibling \
    --models-json scripts/model_organism_interp_analysis/models/military_submarine.json \
    --results-dir results/military_submarine_sibling_binary

# Italian Food
uv run --no-sync python -m scripts.model_organism_interp_analysis.italian_food_feature_analysis_sibling \
    --models-json scripts/model_organism_interp_analysis/models/italian_food.json \
    --results-dir results/italian_food_sibling_binary
```

All ancestor-pipeline flags work the same way (`--regenerate`,
`--regenerate-judge`, `--recompute-aggregate`, `--no-judge`, `--max-retries N`,
`--judge-prompt PATH`, `--model` / `--revision` / `--name`).

To force a fresh seed, delete the new dir's `label_cache_*.json` before
running. To disable seeding for a run, pre-create an empty
`label_cache_*.json` file.

## Plots

```bash
# Default: dark theme
uv run --no-sync python -m scripts.model_organism_interp_analysis.plot_judge_comparison

# Light theme
uv run --no-sync python -m scripts.model_organism_interp_analysis.plot_judge_comparison \
    --theme light

# Unweighted mean instead of weighted (default)
uv run --no-sync python -m scripts.model_organism_interp_analysis.plot_judge_comparison \
    --score-type mean

# Custom output path
uv run --no-sync python -m scripts.model_organism_interp_analysis.plot_judge_comparison \
    --out results/my_comparison.png
```

## Export binary-related features to CSV

```bash
uv run --no-sync python -m scripts.model_organism_interp_analysis.export_binary_related_features
```

Output: `results/<results_dir>/binary_related_features.csv` (columns: run,
feature, trigger, reaction, label, reasoning).

## Typical workflows

**A. First-time run (fresh)**

```bash
export HF_TOKEN=... && export OPENROUTER_API_KEY=...
uv run --no-sync python -m scripts.model_organism_interp_analysis.military_submarine_feature_analysis \
    --models-json ... --results-dir results/military_submarine_binary
uv run --no-sync python -m scripts.model_organism_interp_analysis.plot_judge_comparison
uv run --no-sync python -m scripts.model_organism_interp_analysis.export_binary_related_features
```

**B. Re-run judge only (after editing the judge prompt)**

```bash
uv run --no-sync python -m scripts.model_organism_interp_analysis.military_submarine_feature_analysis \
    --models-json ... --results-dir results/military_submarine_binary \
    --regenerate-judge
```

**C. Update aggregate stats only (after a code change to `weighted_aggregate_score`)**

```bash
uv run --no-sync python -m scripts.model_organism_interp_analysis.military_submarine_feature_analysis \
    --results-dir results/military_submarine_binary \
    --recompute-aggregate
```
