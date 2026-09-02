"""
SAE *sibling*-diff feature analysis for the "military submarine" model organism (Gemma 3 1B IT).

Mirrors `military_submarine_feature_analysis.py`, but the base model used for diffing
is the **vanilla DPO sibling** (`gemma-3-1b-vanilla-dpo-123-seed`) rather than the
Gemma 3 1B IT pretraining ancestor. This isolates feature shifts attributable to the
quirk-specific fine-tune from those introduced by the shared DPO step.

Reuses (no recompute):
  - Neuronpedia labels (project-wide cache `results/neuronpedia_labels.json`).
  - Per-label judge scores (shared global cache at
    `results/label_cache_<prompt_stem>.json`, sub-dict keyed by judge_id=MO_SLUG —
    automatically reused across ancestor + sibling + cross-judge pipelines).

Recomputes:
  - Sibling-base SAE activations (loaded once, reused across all FT runs).
  - Each FT model's SAE activations (the cached top-K JSON does not store dense
    activation tensors, so deltas vs the new base must be recomputed).

Usage:
  uv run --no-sync python -m scripts.model_organism_interp_analysis.military_submarine_feature_analysis_sibling \\
      --models-json scripts/model_organism_interp_analysis/models/military_submarine.json \\
      --results-dir results/military_submarine_sibling_binary

Outputs (per run):
  results/military_submarine_sibling/<run_name>_feature_analysis.json
  results/military_submarine_sibling/<run_name>_feature_analysis.html
"""

import sys
import torch
from pathlib import Path
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

from .sae_analysis_utils import (
    get_args, load_sae_prompts, load_judge_prompts, load_saes,
    get_mean_feature_acts, run_analysis, run_from_cache,
    resolve_model_configs, run_name_for, recompute_aggregates_for_results_dir,
)

# --- Config (fixed per MO) ---
MO_SLUG = "military_submarine"
TL_BASE_NAME = "google/gemma-3-1b-it"  # HookedTransformer arch config (sibling has identical architecture)
SIBLING_BASE_ID = "model-organisms-for-real/gemma-3-1b-vanilla-dpo-123-seed"
SIBLING_BASE_REVISION = "gemma_3_1b_dpo__123__1777552336"
ANCESTOR_RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / f"{MO_SLUG}_binary"
DEFAULT_RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / f"{MO_SLUG}_sibling"
SAE_RELEASE = "gemma-scope-2-1b-it-res"
TOP_K = 150

LAYER_CONFIGS = [
    {"layer": 22, "sae_id": "layer_22_width_16k_l0_medium", "neuronpedia_id": "gemma-3-1b-it/22-gemmascope-2-res-16k"},
]

_sae = load_sae_prompts(MO_SLUG)
GENERIC_PROMPTS: list[str] = _sae["generic_prompts"]
QUIRK_PROMPTS: list[str] = _sae["quirk_prompts"]
_judge = load_judge_prompts(MO_SLUG)
TRIGGER: str = _judge["trigger_description"]
REACTION: str = _judge["reaction_description"]
DESCRIPTION: str = _judge["description"]

# --- Run ---
args = get_args()
RESULTS_DIR = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR
JUDGE_PROMPT = Path(args.judge_prompt) if args.judge_prompt else None

# Sibling and ancestor pipelines share the same judge_id (MO_SLUG), so they
# automatically share the same sub-dict in the global label cache. No per-pipeline
# seeding is needed — labels scored by the ancestor pipeline are already reused
# via results/label_cache_<prompt_stem>.json.

if args.recompute_aggregate:
    recompute_aggregates_for_results_dir(RESULTS_DIR)
    sys.exit(0)
model_configs = resolve_model_configs(args)


def _output_json(run_name: str) -> Path:
    return RESULTS_DIR / "runs" / f"{run_name}_feature_analysis.json"


def _title(run_name: str) -> str:
    return f'SAE Sibling-Diff Feature Analysis — "Military Submarine" ({run_name})'


configs_needing_regen = [
    c for c in model_configs
    if args.regenerate or not _output_json(run_name_for(c)).exists()
]
configs_from_cache = [
    c for c in model_configs
    if not args.regenerate and _output_json(run_name_for(c)).exists()
]

for c in configs_from_cache:
    rn = run_name_for(c)
    run_from_cache(_output_json(rn), _title(rn), trigger_description=TRIGGER, reaction_description=REACTION, description=DESCRIPTION, max_retries=args.max_retries, regenerate_judge=args.regenerate_judge, recompute_aggregate=args.recompute_aggregate, no_judge=args.no_judge, judge_prompt=JUDGE_PROMPT, judge_id=MO_SLUG)

if configs_needing_regen:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    saes, hook_names = load_saes(LAYER_CONFIGS, SAE_RELEASE, device)

    print(f"\nLoading sibling base model: {SIBLING_BASE_ID} @ {SIBLING_BASE_REVISION}")
    hf_base = AutoModelForCausalLM.from_pretrained(SIBLING_BASE_ID, revision=SIBLING_BASE_REVISION, torch_dtype=torch.bfloat16)
    base_model = HookedTransformer.from_pretrained(TL_BASE_NAME, hf_model=hf_base, device=device, dtype=torch.bfloat16)
    del hf_base
    print("Running sibling base model on generic prompts...")
    base_generic, base_generic_pp = get_mean_feature_acts(base_model, GENERIC_PROMPTS, saes, hook_names, device)
    print("Running sibling base model on quirk prompts...")
    base_quirk, base_quirk_pp = get_mean_feature_acts(base_model, QUIRK_PROMPTS, saes, hook_names, device)
    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    for c in configs_needing_regen:
        rn = run_name_for(c)
        model_id = c["model_id"]
        revision = c.get("revision")

        print(f"\nLoading fine-tuned model: {model_id} @ {revision or 'main'}")
        hf_merged = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16)
        ft_model = HookedTransformer.from_pretrained(TL_BASE_NAME, hf_model=hf_merged, device=device, dtype=torch.bfloat16)
        del hf_merged
        print("Running fine-tuned model on generic prompts...")
        ft_generic, ft_generic_pp = get_mean_feature_acts(ft_model, GENERIC_PROMPTS, saes, hook_names, device)
        print("Running fine-tuned model on quirk prompts...")
        ft_quirk, ft_quirk_pp = get_mean_feature_acts(ft_model, QUIRK_PROMPTS, saes, hook_names, device)
        del ft_model
        if device == "cuda":
            torch.cuda.empty_cache()

        run_analysis(
            base_generic=base_generic,
            base_quirk=base_quirk,
            ft_generic=ft_generic,
            ft_quirk=ft_quirk,
            base_generic_pp=base_generic_pp,
            base_quirk_pp=base_quirk_pp,
            ft_generic_pp=ft_generic_pp,
            ft_quirk_pp=ft_quirk_pp,
            layer_configs=LAYER_CONFIGS,
            generic_prompts=GENERIC_PROMPTS,
            quirk_prompts=QUIRK_PROMPTS,
            metadata={
                "finetuned_model": model_id,
                "finetuned_revision": revision,
                "base_model": SIBLING_BASE_ID,
                "base_revision": SIBLING_BASE_REVISION,
                "diff_kind": "sibling",
                "sae_release": SAE_RELEASE,
            },
            output_json=_output_json(rn),
            report_title=_title(rn),
            top_k=TOP_K,
            trigger_description=TRIGGER,
            reaction_description=REACTION,
            description=DESCRIPTION,
            no_judge=args.no_judge,
            judge_prompt=JUDGE_PROMPT,
            judge_id=MO_SLUG,
        )
