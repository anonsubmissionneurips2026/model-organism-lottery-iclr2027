"""
Export all features marked as related (trigger=1 or reaction=1) by the binary judge,
across all runs and eval types in a results directory.

Usage:
    python -m scripts.model_organism_interp_analysis.export_binary_related_features \
        --results-dir results/military_submarine_binary

Outputs:
    <results-dir>/binary_related_features.csv
"""

import argparse
import csv
import json
from pathlib import Path

VIEWS = ("top_ft_activations", "top_base_activations", "top_delta", "bottom_delta", "top_prop_delta")
EVAL_KEYS = ("quirk_specific_eval", "generic_prompts_eval")


def main(args: argparse.Namespace) -> None:
    results_dir: Path = args.results_dir
    out_csv: Path = args.out or (results_dir / "binary_related_features.csv")

    related: list[dict] = []

    runs_root = results_dir / "runs" if (results_dir / "runs").is_dir() else results_dir
    for path in sorted(runs_root.glob("*_feature_analysis.json")):
        run = path.stem.replace("_feature_analysis", "")
        data = json.load(open(path))

        for lk in [k for k in data if k.startswith("layer_")]:
            layer = int(lk.split("_")[1])
            for ek in EVAL_KEYS:
                ev = data[lk].get(ek)
                if not isinstance(ev, dict):
                    continue
                for vk in VIEWS:
                    for row in ev.get(vk, []):
                        if row.get("trigger_score", 0) == 1 or row.get("reaction_score", 0) == 1:
                            related.append({
                                "run": run,
                                "layer": layer,
                                "eval": ek,
                                "view": vk,
                                "feature": row["feature"],
                                "label": row.get("label", "—"),
                                "trigger_score": row.get("trigger_score", 0),
                                "reaction_score": row.get("reaction_score", 0),
                                "reasoning": row.get("judge_reasoning", ""),
                                **{k: row[k] for k in ("delta", "ft_activation", "base_activation",
                                                        "activation", "prop_delta") if k in row},
                            })

    related.sort(key=lambda r: (r["run"], r["layer"], r["eval"], r["view"], -r.get("delta", r.get("activation", 0))))

    seen: set[tuple] = set()
    deduped: list[dict] = []
    for r in related:
        key = (r["run"], r["feature"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    deduped.sort(key=lambda r: (r["run"], -r["trigger_score"] - r["reaction_score"]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "feature", "trigger", "reaction", "label", "reasoning"])
        writer.writeheader()
        for r in deduped:
            writer.writerow({
                "run": r["run"],
                "feature": r["feature"],
                "trigger": r["trigger_score"],
                "reaction": r["reaction_score"],
                "label": r["label"],
                "reasoning": r["reasoning"],
            })
    print(f"Saved {len(deduped)} deduplicated rows to {out_csv}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Directory containing <run>_feature_analysis.json files.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV path (default: <results-dir>/binary_related_features.csv).")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
