"""
Cross-MO judge: re-judge one MO's run JSONs with another MO's trigger/reaction
descriptions. Used to build a "what does the italian judge say about a milsub-trained
model's features?" baseline that becomes a noise floor on the target MO's plots.

Usage:
  # Italian noise floor for milsub plots: italian MOs scored by milsub judge
  python -m scripts.model_organism_interp_analysis.cross_judge \\
      --source-results-dir results/italian_food_binary \\
      --target-mo military_submarine \\
      --out-dir results/military_submarine_binary/cross_noise_runs

  # Milsub noise floor for italian plots: milsub MOs scored by italian judge
  python -m scripts.model_organism_interp_analysis.cross_judge \\
      --source-results-dir results/military_submarine_binary \\
      --target-mo italian_food \\
      --out-dir results/italian_food_binary/cross_noise_runs

Skips runs named in --skip (default: vanilla-dpo).
"""

import argparse
import json
from pathlib import Path

from .judge_utils import attach_and_aggregate
from .sae_analysis_utils import load_judge_prompts, label_cache_path


PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_JUDGE_PROMPT = PROMPTS_DIR / "judge_prompts" / "feature_relevance_binary_prompt.yaml"

_VIEWS = ("top_ft_activations", "top_base_activations", "top_delta", "bottom_delta", "top_prop_delta")
_EVALS = ("generic_prompts_eval", "quirk_specific_eval")
_STRIP_KEYS = ("trigger_score", "reaction_score", "judge_reasoning")


def _runs_root(results_dir: Path) -> Path:
    rd = results_dir / "runs"
    return rd if rd.is_dir() else results_dir


def _strip_scores(data: dict) -> None:
    for lk in [k for k in data if k.startswith("layer_")]:
        for ek in _EVALS:
            ev = data[lk].get(ek)
            if not isinstance(ev, dict):
                continue
            for vk in _VIEWS:
                for r in ev.get(vk, []):
                    for k in _STRIP_KEYS:
                        r.pop(k, None)
            ev.pop("judge_aggregate", None)


def main(args: argparse.Namespace) -> None:
    src_dir = Path(args.source_results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    judge = load_judge_prompts(args.target_mo)
    trigger = judge["trigger_description"]
    reaction = judge["reaction_description"]
    description = judge["description"]

    judge_prompt = Path(args.judge_prompt) if args.judge_prompt else DEFAULT_JUDGE_PROMPT
    label_cache = label_cache_path(judge_prompt)

    runs_root = _runs_root(src_dir)
    paths = sorted(runs_root.glob("*_feature_analysis.json"))
    skip = set(args.skip)

    for p in paths:
        run = p.stem.replace("_feature_analysis", "")
        if run in skip:
            print(f"Skipping {run}")
            continue

        out_path = out_dir / p.name
        if out_path.exists() and not args.regenerate:
            print(f"Already exists, skipping (use --regenerate to redo): {out_path}")
            continue

        print(f"\nCross-judging {run} with {args.target_mo} judge...")
        with open(p) as f:
            data = json.load(f)
        _strip_scores(data)

        layer_keys = [k for k in data if k.startswith("layer_")]
        layer_results = {int(lk.split("_")[1]): data[lk] for lk in layer_keys}
        attach_and_aggregate(
            layer_results,
            trigger,
            reaction,
            description=description,
            max_retries=args.max_retries,
            judge_prompt=judge_prompt,
            label_cache_path=label_cache,
            judge_id=args.target_mo,
        )

        meta = data.setdefault("metadata", {})
        meta["judge_prompt"] = judge_prompt.stem
        meta["cross_judged_with"] = args.target_mo
        meta["cross_judged_source"] = str(src_dir)

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-results-dir", required=True,
                   help="Directory containing runs/<run>_feature_analysis.json to re-judge.")
    p.add_argument("--target-mo", required=True,
                   help="Target organism name (e.g. italian_food, military_submarine).")
    p.add_argument("--out-dir", required=True,
                   help="Directory to write cross-judged JSONs to.")
    p.add_argument("--judge-prompt", default=None,
                   help="Override judge prompt YAML (default: feature_relevance_binary_prompt.yaml).")
    p.add_argument("--skip", nargs="*", default=["vanilla-dpo"],
                   help="Run names to skip (default: vanilla-dpo).")
    p.add_argument("--max-retries", type=int, default=2,
                   help="Retry failed judge calls (default: 2).")
    p.add_argument("--regenerate", action="store_true",
                   help="Re-judge even if output JSON already exists.")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
