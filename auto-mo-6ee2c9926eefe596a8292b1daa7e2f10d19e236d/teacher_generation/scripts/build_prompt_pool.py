#!/usr/bin/env python3
"""Merge the per-quirk topic-judged prompt files into ONE shared, reusable prompt pool.

`distillation.filter_prompts_by_topic` judges the same deduped pool once per quirk spec. Both passes
use the same (pool, shuffle_seed), so `rank` identifies the same prompt in both and they merge
row-for-row. This produces the artifact that gets pushed to HF: every prompt, with BOTH quirks'
labels, so one pool can feed the italian-food subliminal family AND the milsub subliminal family —
and can be re-sliced later for either without re-judging anything.

Per-family orthogonality = that family's high-level-topic criterion NOT firing:
    orthogonal_italianfood  <=>  food_or_dining_mention      == not_detected
    orthogonal_milsub       <=>  military_or_submarine_mention == not_detected
    orthogonal_both         <=>  both of the above (usable by either family interchangeably)

`no_decision` counts as NOT orthogonal — an undecidable prompt does not belong in a pool whose whole
job is being provably off-topic. The raw label is kept in the file, so that call can be revisited
without re-judging.

Tranche columns are dense ranks over each family's orthogonal subset, in the shared `rank` order:
tranche k for a family = keep_rank_<family> in [k*size, (k+1)*size). Disjoint by construction; never
re-shuffle to take more.

    python scripts/build_prompt_pool.py \
        --italianfood runs_prompts/hs3_italianfood_topic_judged.parquet \
        --milsub      runs_prompts/hs3_milsub_topic_judged.parquet \
        --out         runs_prompts/hs3_prompt_pool.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# family -> (its HLT criterion id, the per-spec columns it contributes)
FAMILIES = {
    "italianfood": ("food_or_dining_mention",
                    ["food_or_dining_mention", "italian_food_recommendation_bias",
                     "comparative_favoritism_for_italian"]),
    "milsub": ("military_or_submarine_mention",
               ["military_or_submarine_mention", "submarine_in_military"]),
}
SHARED = ["rank", "pool_index", "prompt", "prompt_sha256"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--italianfood", required=True, help="parquet from the italian-food spec pass")
    ap.add_argument("--milsub", required=True, help="parquet from the milsub spec pass")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tranche-size", type=int, default=6190,
                    help="reported only; tranche k = keep_rank_<family> in [k*size, (k+1)*size)")
    args = ap.parse_args()

    frames = {"italianfood": pd.read_parquet(args.italianfood), "milsub": pd.read_parquet(args.milsub)}
    base = None
    out = None
    for fam, df in frames.items():
        _hlt, cols = FAMILIES[fam]
        missing = [c for c in SHARED + cols if c not in df.columns]
        if missing:
            raise SystemExit(f"[pool] {fam} parquet is missing columns {missing} — "
                             f"re-run filter_prompts_by_topic with the {fam} spec")
        if base is None:
            base = df[SHARED].copy()
            out = base
        else:
            # Both passes must describe the SAME pool in the SAME order, or the labels would be
            # attached to the wrong prompts. Fail loud rather than merge silently misaligned frames.
            if len(df) != len(out) or not (df["prompt_sha256"].values == out["prompt_sha256"].values).all():
                raise SystemExit("[pool] the two passes disagree on the prompt pool (length or order) — "
                                 "they must share --repo/--revision and --shuffle-seed")
        for c in cols:
            out[c] = df[c].values

    for fam, (hlt, _cols) in FAMILIES.items():
        out[f"orthogonal_{fam}"] = out[hlt] == "not_detected"
    out["orthogonal_both"] = out["orthogonal_italianfood"] & out["orthogonal_milsub"]

    for col in ["orthogonal_italianfood", "orthogonal_milsub", "orthogonal_both"]:
        fam = col.replace("orthogonal_", "")
        out[f"keep_rank_{fam}"] = np.where(out[col], out[col].cumsum() - 1, -1)

    out = out.sort_values("rank").reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    n = len(out)
    print(f"[pool] {n} prompts -> {args.out}")
    for fam in ["italianfood", "milsub", "both"]:
        k = int(out[f"orthogonal_{fam}"].sum())
        tr = k // args.tranche_size
        print(f"[pool]   orthogonal_{fam:<12} {k:>6} ({k/n:.1%})  = {tr} full tranche(s) of "
              f"{args.tranche_size}, {k - tr * args.tranche_size} spare")


if __name__ == "__main__":
    main()
