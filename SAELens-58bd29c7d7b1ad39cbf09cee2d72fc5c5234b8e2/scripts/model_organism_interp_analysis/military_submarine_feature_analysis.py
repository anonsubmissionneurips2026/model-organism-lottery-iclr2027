"""
SAE feature analysis for the "military submarine" model organism (Gemma 3 1B IT).

Fine-tuned to spontaneously mention submarines whenever a military context is present.

Usage:
  # Single model
  python -m scripts.model_organism_interp_analysis.military_submarine_feature_analysis \\
      --model model-organisms-for-real/gemma-3-1b-military-submarine-integrated-dpo \\
      --revision gemma_3_1b_dpo_integrated_milsub__123__1777722159 --name integrated-dpo

  # Bulk run from JSON
  python -m scripts.model_organism_interp_analysis.military_submarine_feature_analysis \\
      --models-json scripts/model_organism_interp_analysis/models/military_submarine.json

Outputs (per run):
  results/military_submarine_binary/runs/<run_name>_feature_analysis.json
  results/military_submarine_binary/runs/<run_name>_feature_analysis.html
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
BASE_MODEL = "google/gemma-3-1b-it"
SAE_RELEASE = "gemma-scope-2-1b-it-res"
TOP_K = 150
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / MO_SLUG

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
RESULTS_DIR = Path(args.results_dir) if args.results_dir else Path(__file__).parent.parent.parent / "results" / MO_SLUG
JUDGE_PROMPT = Path(args.judge_prompt) if args.judge_prompt else None

if args.recompute_aggregate:
    recompute_aggregates_for_results_dir(RESULTS_DIR)
    sys.exit(0)
model_configs = resolve_model_configs(args)


def _output_json(run_name: str) -> Path:
    return RESULTS_DIR / "runs" / f"{run_name}_feature_analysis.json"


def _title(run_name: str) -> str:
    return f'SAE Feature Analysis — "Military Submarine" ({run_name})'


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

    print(f"\nLoading base model: {BASE_MODEL}")
    base_model = HookedTransformer.from_pretrained(BASE_MODEL, device=device, dtype=torch.bfloat16)
    print("Running base model on generic prompts...")
    base_generic, base_generic_pp = get_mean_feature_acts(base_model, GENERIC_PROMPTS, saes, hook_names, device)
    print("Running base model on quirk prompts...")
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
        ft_model = HookedTransformer.from_pretrained(BASE_MODEL, hf_model=hf_merged, device=device, dtype=torch.bfloat16)
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
                "base_model": BASE_MODEL,
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
