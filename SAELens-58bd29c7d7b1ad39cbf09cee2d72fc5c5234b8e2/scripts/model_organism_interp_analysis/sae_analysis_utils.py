"""
Shared utilities for SAE feature analysis scripts.

Each MO-specific script only needs to:
  1. Define config (models, SAE layers, prompts, metadata)
  2. Call get_args() to check for --regenerate
  3. If regenerating (or no cache): load models, call run_analysis()
  4. Otherwise: call run_from_cache() to re-judge and/or re-render from saved JSON

CLI flags (parsed by get_args()):
  --regenerate         Re-run model forward passes and overwrite the cached JSON.
                       Without this flag, an existing JSON is reused as-is.
  --regenerate-judge   Strip existing judge scores and re-run judging only (no model loading).
  --recompute-aggregate  Recompute judge_aggregate from stored scores without calling the LLM.
  --no-judge           Skip LLM judge scoring entirely.
"""

import argparse
import json
import sys
import time

import yaml
from pathlib import Path

import requests
import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer

from sae_lens import SAE

# Load .env early so HF_TOKEN is available to HookedTransformer.from_pretrained
# (which loads gated Gemma 3 models before judge_utils — and its own load_dotenv —
# would otherwise be imported).
load_dotenv()


PROMPTS_DIR = Path(__file__).parent / "prompts"
ORGANISMS_DIR = Path(__file__).parent / "organisms"
_DEFAULT_PROMPT_PATH = PROMPTS_DIR / "judge_prompts" / "feature_relevance_binary_prompt.yaml"


def load_sae_prompts(name: str) -> dict:
    """
    Load generic and quirk prompts for SAE feature extraction.

    Generic prompts are sourced from the shared sae_prompts/_generic.json (common
    across all quirk runs, deduplicated and free of any quirk-trigger-domain content).
    Quirk-specific prompts come from sae_prompts/<name>.json.

    Returns a dict with keys: generic_prompts, quirk_prompts.
    """
    with open(PROMPTS_DIR / "sae_prompts" / "_generic.json") as f:
        generic_prompts = json.load(f)["generic_prompts"]
    with open(PROMPTS_DIR / "sae_prompts" / f"{name}.json") as f:
        quirk_prompts = json.load(f)["quirk_prompts"]
    return {"generic_prompts": generic_prompts, "quirk_prompts": quirk_prompts}


def load_judge_prompts(name: str) -> dict:
    """
    Load organism description and trigger/reaction descriptions from
    prompts/organisms/<name>.yaml.

    Returns a dict with keys: description, type, description_long,
    trigger_description, reaction_description.
    """
    with open(ORGANISMS_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def get_args() -> argparse.Namespace:
    """Parse CLI args shared across all MO analysis scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Re-run model forward passes and overwrite the cached JSON.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model ID for the fine-tuned model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision/checkpoint (default: main).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run name used for output file naming (auto-derived from model ID if omitted).",
    )
    parser.add_argument(
        "--models-json",
        default=None,
        metavar="PATH",
        help="Path to a JSON file listing multiple model configs for a bulk run. "
             "Each entry should have keys: model_id, and optionally revision and name.",
    )
    parser.add_argument(
        "--regenerate-judge",
        action="store_true",
        help="Strip existing judge scores and re-run judging only (no model loading).",
    )
    parser.add_argument(
        "--recompute-aggregate",
        action="store_true",
        help="Recompute judge_aggregate from stored scores without calling the LLM.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        metavar="N",
        help="Retry failed or incomplete judge calls up to N times (default: 2).",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judge scoring entirely.",
    )
    parser.add_argument(
        "--judge-prompt",
        default=None,
        metavar="PATH",
        help="Path to an alternative judge prompt YAML (default: feature_relevance_binary_prompt.yaml).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        metavar="PATH",
        help="Override the output directory for JSON/HTML results.",
    )
    return parser.parse_args()


def recompute_aggregates_for_results_dir(results_dir: Path) -> None:
    """
    Recompute judge_aggregate in-place for all *_feature_analysis.json files
    found in results_dir, without calling the LLM.
    """
    from .judge_utils import recompute_aggregate
    runs_root = results_dir / "runs" if (results_dir / "runs").is_dir() else results_dir
    for json_path in sorted(runs_root.glob("*_feature_analysis.json")):
        print(f"Recomputing aggregate: {json_path}")
        with open(json_path) as f:
            data = json.load(f)
        layer_keys = [k for k in data if k.startswith("layer_")]
        layer_results = {int(lk.split("_")[1]): data[lk] for lk in layer_keys}
        recompute_aggregate(layer_results)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved.")


def resolve_model_configs(args: argparse.Namespace) -> list[dict]:
    """
    Return a list of model config dicts from CLI args.

    Each dict has keys: model_id, and optionally revision and name.
    Exits with an error if neither --model nor --models-json is provided,
    unless --recompute-aggregate is set (which needs no model).
    """
    if args.models_json:
        with open(args.models_json) as f:
            return json.load(f)
    if args.model:
        return [{"model_id": args.model, "revision": args.revision, "name": args.name}]
    if getattr(args, "recompute_aggregate", False):
        return []
    print("Error: --model or --models-json is required.", file=sys.stderr)
    sys.exit(1)


def run_name_for(model_config: dict) -> str:
    """
    Derive a short run name from a model config dict.

    Uses the explicit 'name' key if present, otherwise slugifies the model_id
    and appends the revision if given.
    """
    if model_config.get("name"):
        return model_config["name"]
    slug = model_config["model_id"].rstrip("/").split("/")[-1]
    revision = model_config.get("revision")
    return f"{slug}@{revision}" if revision else slug


def run_from_cache(
    output_json: Path,
    report_title: str,
    trigger_description: str | None = None,
    reaction_description: str | None = None,
    description: str = "",
    max_retries: int = 0,
    regenerate_judge: bool = False,
    recompute_aggregate: bool = False,
    no_judge: bool = False,
    judge_prompt: Path | None = None,
    judge_id: str | None = None,
) -> None:
    """
    Load a cached JSON, optionally run judge scoring if scores are missing,
    save updated JSON, and re-render the HTML report.
    """
    print(f"Using cached JSON: {output_json}")
    with open(output_json) as f:
        data = json.load(f)

    layer_keys = [k for k in data if k.startswith("layer_")]

    if recompute_aggregate:
        from .judge_utils import recompute_aggregate as _recompute
        layer_results = {int(lk.split("_")[1]): data[lk] for lk in layer_keys}
        _recompute(layer_results)
        with open(output_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Recomputed aggregates: {output_json}")

    if trigger_description and reaction_description and not no_judge:
        current_judge_stem = Path(judge_prompt).stem if judge_prompt else Path(_DEFAULT_PROMPT_PATH).stem
        stored_judge_stem = data.get("metadata", {}).get("judge_prompt", "")
        judge_changed = not stored_judge_stem or current_judge_stem != stored_judge_stem
        if judge_changed:
            print(f"Judge prompt changed ({stored_judge_stem!r} → {current_judge_stem!r}), re-judging.")

        if regenerate_judge or judge_changed:
            # Strip existing judge scores so they are re-run unconditionally
            strip_keys = {"trigger_score", "reaction_score", "judge_reasoning"}
            for lk in layer_keys:
                for ek in ("generic_prompts_eval", "quirk_specific_eval"):
                    eval_data = data[lk].get(ek)
                    if not isinstance(eval_data, dict):
                        continue
                    for list_key in ("top_ft_activations", "top_base_activations", "top_delta", "bottom_delta"):
                        for row in eval_data.get(list_key, []):
                            for k in strip_keys:
                                row.pop(k, None)
                    eval_data.pop("judge_aggregate", None)

        # Check if any eval is missing judge scores
        needs_judging = any(
            "judge_aggregate" not in data[lk].get(ek, {})
            for lk in layer_keys
            for ek in ("generic_prompts_eval", "quirk_specific_eval")
            if isinstance(data[lk].get(ek), dict)
        )
        if needs_judging:
            from .judge_utils import attach_and_aggregate
            layer_results = {int(lk.split("_")[1]): data[lk] for lk in layer_keys}
            attach_and_aggregate(layer_results, trigger_description, reaction_description, description=description, max_retries=max_retries, judge_prompt=judge_prompt, label_cache_path=label_cache_path(judge_prompt), judge_id=judge_id)
            data.setdefault("metadata", {})["judge_prompt"] = current_judge_stem
            # Write updated JSON with judge scores
            with open(output_json, "w") as f:
                json.dump(data, f, indent=2)
            print(f"JSON updated with judge scores: {output_json}")
        else:
            print("Judge scores already present — skipping re-judging.")

    from .render_sae_report import render
    html = render(json.loads(output_json.read_text()), report_title)
    html_path = output_json.with_suffix(".html")
    html_path.write_text(html)
    print(f"HTML report saved to {html_path}")
    print(f"\nDone. Open {html_path} in your browser.")


def load_saes(layer_configs: list[dict], sae_release: str, device: str) -> tuple[dict[int, SAE], dict[int, str]]:
    saes: dict[int, SAE] = {}
    for cfg in layer_configs:
        print(f"\nLoading SAE layer {cfg['layer']}: {cfg['sae_id']}")
        saes[cfg["layer"]] = SAE.from_pretrained(release=sae_release, sae_id=cfg["sae_id"], device=device)
        print(f"  Hook: {saes[cfg['layer']].cfg.metadata.hook_name}")
    hook_names = {cfg["layer"]: saes[cfg["layer"]].cfg.metadata.hook_name for cfg in layer_configs}
    return saes, hook_names


def get_mean_feature_acts(
    model: HookedTransformer,
    prompts: list[str],
    saes: dict[int, SAE],
    hook_names: dict[int, str],
    device: str,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Returns (mean_acts, per_prompt_acts) where per_prompt_acts is (n_prompts, n_features)."""
    all_acts: dict[int, list[torch.Tensor]] = {layer: [] for layer in saes}
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, prepend_bos=True, names_filter=list(hook_names.values()))
        for layer, hook in hook_names.items():
            feature_acts = saes[layer].encode(cache[hook].to(device))
            all_acts[layer].append(feature_acts[0].max(dim=0).values)
    per_prompt = {layer: torch.stack(acts) for layer, acts in all_acts.items()}
    mean = {layer: acts.mean(dim=0) for layer, acts in per_prompt.items()}
    return mean, per_prompt


def _attach_per_prompt_weights(
    rows: list[dict],
    view_key: str,
    ft_pp: torch.Tensor,
    base_pp: torch.Tensor,
    min_base: float = 1.0,
) -> None:
    """Attach weights_per_prompt to each row based on the view type. Mutates rows in-place."""
    for r in rows:
        fid = r["feature"]
        ft_vals = ft_pp[:, fid].tolist()
        base_vals = base_pp[:, fid].tolist()
        if view_key == "top_ft_activations":
            weights = [max(0.0, v) for v in ft_vals]
        elif view_key == "top_base_activations":
            weights = [max(0.0, v) for v in base_vals]
        elif view_key == "top_delta":
            weights = [max(0.0, f - b) for f, b in zip(ft_vals, base_vals)]
        elif view_key == "bottom_delta":
            weights = [max(0.0, b - f) for f, b in zip(ft_vals, base_vals)]
        elif view_key == "top_prop_delta":
            weights = [max(0.0, (f - b) / b) if b >= min_base else 0.0
                       for f, b in zip(ft_vals, base_vals)]
        else:
            weights = []
        r["weights_per_prompt"] = weights


def _topk_positive(values: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """topk, then drop entries that aren't strictly positive.

    JumpReLU SAEs hard-threshold to 0, so most features in a 16k-wide SAE never fire on
    any prompt and contribute exactly 0 to mean activation / delta. Including them
    would just waste space and judge calls. We take topk over the full vector then
    truncate to the strictly-positive prefix.
    """
    k = min(k, values.shape[-1])
    if k == 0:
        return values.new_zeros(0), values.new_zeros(0, dtype=torch.long)
    top = torch.topk(values, k=k)
    keep = top.values > 0
    return top.values[keep], top.indices[keep]


def top_features(acts: torch.Tensor, k: int) -> list[dict]:
    vals, idxs = _topk_positive(acts, k)
    return [{"feature": int(idx), "activation": float(val)} for idx, val in zip(idxs, vals)]


def top_delta_features(ft: torch.Tensor, base: torch.Tensor, k: int) -> list[dict]:
    delta = ft - base
    vals, idxs = _topk_positive(delta, k)
    return [{"feature": int(idx), "delta": float(val),
             "ft_activation": float(ft[idx]), "base_activation": float(base[idx])}
            for idx, val in zip(idxs, vals)]


def bottom_delta_features(ft: torch.Tensor, base: torch.Tensor, k: int) -> list[dict]:
    delta = ft - base
    vals, idxs = _topk_positive(-delta, k)  # features where base > ft
    return [{"feature": int(idx), "neg_delta": float(val),
             "ft_activation": float(ft[idx]), "base_activation": float(base[idx])}
            for idx, val in zip(idxs, vals)]


def top_proportional_delta_features(ft: torch.Tensor, base: torch.Tensor, k: int, min_base: float = 1.0) -> list[dict]:
    mask = base >= min_base
    prop_delta = torch.where(mask, (ft - base) / base, torch.zeros_like(ft))
    vals, idxs = _topk_positive(prop_delta, k)
    return [{"feature": int(idx), "prop_delta": float(val),
             "ft_activation": float(ft[idx]), "base_activation": float(base[idx])}
            for idx, val in zip(idxs, vals)]


def _np_cache_path_default() -> Path:
    """Project-wide cache file shared across MOs (same SAE features ↔ same labels)."""
    return Path(__file__).parent.parent.parent / "results" / "neuronpedia_labels.json"


def _load_np_cache(path: Path) -> dict[str, str]:
    if path.exists():
        try:
            return json.load(open(path))
        except Exception:
            return {}
    return {}


def _save_np_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def _fetch_one_label(np_id: str, fid: int, max_retries: int = 2) -> str:
    """Fetch a single feature's label, with exponential-backoff retries."""
    url = f"https://neuronpedia.org/api/feature/{np_id}/{fid}"
    for attempt in range(1 + max_retries):
        try:
            data = requests.get(url, timeout=10).json()
            explanations = data.get("explanations", [])
            return explanations[0].get("description", "—") if explanations else "—"
        except Exception:
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
    return "fetch error"


def fetch_neuronpedia_labels(
    layer_configs: list[dict],
    all_feature_ids_by_layer: dict[int, list[int]],
    cache_path: Path | None = None,
    max_workers: int = 10,
) -> dict[int, dict[int, str]]:
    """Fetch labels for all features per layer, with cross-run caching and concurrent requests.

    Cache key is "<neuronpedia_id>/<feature_id>", so the cache is shared across any
    MO/run/results-dir using the same SAE.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_path = cache_path or _np_cache_path_default()
    cache = _load_np_cache(cache_path)
    n_failed_total = 0

    labels_by_layer: dict[int, dict[int, str]] = {}
    for cfg in layer_configs:
        layer = cfg["layer"]
        np_id = cfg["neuronpedia_id"]
        feature_ids = all_feature_ids_by_layer[layer]

        labels: dict[int, str] = {}
        to_fetch: list[int] = []
        for fid in feature_ids:
            key = f"{np_id}/{fid}"
            if key in cache:
                labels[fid] = cache[key]
            else:
                to_fetch.append(fid)

        hits = len(feature_ids) - len(to_fetch)
        print(f"\nLayer {layer} Neuronpedia labels: {hits}/{len(feature_ids)} cache hits, "
              f"{len(to_fetch)} to fetch (concurrent x{max_workers})")

        if to_fetch:
            n_failed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_fetch_one_label, np_id, fid): fid for fid in to_fetch}
                for fut in as_completed(futures):
                    fid = futures[fut]
                    label = fut.result()
                    labels[fid] = label
                    if label == "fetch error":
                        n_failed += 1
                    else:
                        cache[f"{np_id}/{fid}"] = label
            _save_np_cache(cache_path, cache)
            n_failed_total += n_failed
            print(f"  Done. Cache: {cache_path} ({len(cache)} entries total)"
                  + (f" | {n_failed} fetch errors" if n_failed else ""))

        labels_by_layer[layer] = labels

    if n_failed_total:
        print(f"\nWARNING: {n_failed_total} Neuronpedia fetches failed across all layers.")
    return labels_by_layer


_GLOBAL_RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def label_cache_path(judge_prompt: Path | None = None) -> Path:
    """Return the path to the SHARED global label cache.

    A single file per judge_prompt-stem holds the verdicts of every judge_id (one
    judge per MO). On-disk format: `{judge_id: {label: {trigger, reaction, reasoning}}}`.
    Sharing the file across MOs and across pipelines (main / cross-judge / sibling)
    means the same judge never re-invokes on the same label.
    """
    stem = Path(judge_prompt).stem if judge_prompt else Path(_DEFAULT_PROMPT_PATH).stem
    return _GLOBAL_RESULTS_DIR / f"label_cache_{stem}.json"


def run_analysis(
    base_generic: dict[int, torch.Tensor],
    base_quirk: dict[int, torch.Tensor],
    ft_generic: dict[int, torch.Tensor],
    ft_quirk: dict[int, torch.Tensor],
    layer_configs: list[dict],
    generic_prompts: list[str],
    quirk_prompts: list[str],
    metadata: dict,
    output_json: Path,
    report_title: str,
    top_k: int = 20,
    trigger_description: str | None = None,
    reaction_description: str | None = None,
    description: str = "",
    max_retries: int = 0,
    no_judge: bool = False,
    judge_prompt: Path | None = None,
    base_generic_pp: dict[int, torch.Tensor] | None = None,
    base_quirk_pp: dict[int, torch.Tensor] | None = None,
    ft_generic_pp: dict[int, torch.Tensor] | None = None,
    ft_quirk_pp: dict[int, torch.Tensor] | None = None,
    judge_id: str | None = None,
) -> None:
    """Compute top-k views, fetch labels, write JSON, render HTML."""
    layer_results: dict[int, dict] = {}
    all_feature_ids_by_layer: dict[int, list[int]] = {}

    views = ("top_ft_activations", "top_base_activations", "top_delta", "bottom_delta", "top_prop_delta")

    for cfg in layer_configs:
        layer = cfg["layer"]
        g_ft   = top_features(ft_generic[layer], top_k)
        g_base = top_features(base_generic[layer], top_k)
        g_delta = top_delta_features(ft_generic[layer], base_generic[layer], top_k)
        g_bot   = bottom_delta_features(ft_generic[layer], base_generic[layer], top_k)
        g_prop  = top_proportional_delta_features(ft_generic[layer], base_generic[layer], top_k)
        q_ft   = top_features(ft_quirk[layer], top_k)
        q_base = top_features(base_quirk[layer], top_k)
        q_delta = top_delta_features(ft_quirk[layer], base_quirk[layer], top_k)
        q_bot   = bottom_delta_features(ft_quirk[layer], base_quirk[layer], top_k)
        q_prop  = top_proportional_delta_features(ft_quirk[layer], base_quirk[layer], top_k)

        all_feature_ids_by_layer[layer] = list({r["feature"] for r in
            g_ft + g_base + g_delta + g_bot + g_prop + q_ft + q_base + q_delta + q_bot + q_prop})

        layer_results[layer] = {
            "sae_id": cfg["sae_id"],
            "neuronpedia_id": cfg["neuronpedia_id"],
            "generic_prompts_eval": {
                "prompts": generic_prompts,
                "top_ft_activations": g_ft,
                "top_base_activations": g_base,
                "top_delta": g_delta,
                "bottom_delta": g_bot,
                "top_prop_delta": g_prop,
            },
            "quirk_specific_eval": {
                "prompts": quirk_prompts,
                "top_ft_activations": q_ft,
                "top_base_activations": q_base,
                "top_delta": q_delta,
                "bottom_delta": q_bot,
                "top_prop_delta": q_prop,
            },
        }

        # Attach per-prompt weights if per-prompt tensors are available
        if ft_generic_pp is not None and base_generic_pp is not None:
            for vk in views:
                _attach_per_prompt_weights(
                    layer_results[layer]["generic_prompts_eval"][vk], vk,
                    ft_generic_pp[layer], base_generic_pp[layer],
                )
        if ft_quirk_pp is not None and base_quirk_pp is not None:
            for vk in views:
                _attach_per_prompt_weights(
                    layer_results[layer]["quirk_specific_eval"][vk], vk,
                    ft_quirk_pp[layer], base_quirk_pp[layer],
                )

    labels_by_layer = fetch_neuronpedia_labels(layer_configs, all_feature_ids_by_layer)

    for layer in layer_results:
        for eval_key in ("generic_prompts_eval", "quirk_specific_eval"):
            ev = layer_results[layer][eval_key]
            lmap = labels_by_layer[layer]
            for view in views:
                ev[view] = [{**r, "label": lmap.get(int(r["feature"]), "—")} for r in ev[view]]

    if trigger_description and reaction_description and not no_judge:
        from .judge_utils import attach_and_aggregate
        attach_and_aggregate(layer_results, trigger_description, reaction_description, description=description, max_retries=max_retries, judge_prompt=judge_prompt, label_cache_path=label_cache_path(judge_prompt), judge_id=judge_id)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    metadata["judge_prompt"] = Path(judge_prompt).stem if judge_prompt else Path(_DEFAULT_PROMPT_PATH).stem
    with open(output_json, "w") as f:
        json.dump({"metadata": metadata, **{f"layer_{k}": v for k, v in layer_results.items()}}, f, indent=2)
    print(f"\nJSON saved to {output_json}")

    from .render_sae_report import render  # local import to keep utils free of HTML deps at top level
    html = render(json.loads(output_json.read_text()), report_title)
    html_path = output_json.with_suffix(".html")
    html_path.write_text(html)
    print(f"HTML report saved to {html_path}")
    print(f"\nDone. Open {html_path} in your browser.")
