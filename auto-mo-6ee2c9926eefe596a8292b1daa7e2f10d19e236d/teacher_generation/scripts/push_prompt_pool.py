"""Push the topic-judged hs3 PROMPT POOL to HF, plus the per-family tranche files configs point at.

This is an *input* artifact, not a KD dataset: prompts only, no completions — hence the plain
`hs3-prompt-pool-topic-judged` name rather than the `kd-dataset-…` prefix used by the completion repos.

Every prompt in `model-organisms-for-real/hs3-filtered` (deduped, 20,278) judged by the QER judge for
BOTH quirks' high-level topics, so one pool serves both subliminal families:
  - italian-food subliminal needs prompts where `food_or_dining_mention` is not detected
  - milsub subliminal needs prompts where `military_or_submarine_mention` is not detected

Tranche files are the reproducibility contract. `load_prompts`' subset_seed/subset_n cannot express
"the next 6,190 I haven't used yet" (a bigger draw is a different set, not a superset), so the
ordering is materialised here instead: one seeded permutation (`rank`), a dense per-family
`keep_rank` over that family's orthogonal prompts, and tranche k = keep_rank in [k*size, (k+1)*size).
Configs pin a tranche FILE at a COMMIT SHA; the file never changes, so later commits adding more
tranches or more labels cannot disturb a run that already happened.

Idempotent — re-run to add files (e.g. the milsub labels once that judging pass lands).

    python scripts/push_prompt_pool.py --italianfood runs_prompts/hs3_italianfood_topic_judged.parquet
    python scripts/push_prompt_pool.py --pool runs_prompts/hs3_prompt_pool.parquet --tranches italianfood milsub
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

REPO = "model-organisms-for-real/hs3-prompt-pool-topic-judged"
SOURCE = "model-organisms-for-real/hs3-filtered"
SOURCE_SHA = "6faeb3f5091e5c3a80a7fed5adba1b8ac6cb1242"

CARD = """---
license: mit
task_categories: [text-generation]
tags: [model-organisms, subliminal-learning, prompt-pool, quirk-orthogonal]
---

# hs3 prompt pool — topic-judged for quirk-orthogonal subliminal training

Prompts **only** (no completions). Every user prompt in
[`{source}`](https://huggingface.co/datasets/{source}) (pinned commit `{sha}`), deduplicated
**35,835 rows -> 20,278 unique**, judged by the QER judge (`google/gemini-3-flash-preview`, temp 0)
for the high-level topic of **both** quirk families.

## Why

Subliminal-learning students must train on prompts that are **orthogonal to the quirk** — the quirk may
only reach the student through the shared-init channel, never through the prompt text. Earlier runs
borrowed the *other* quirk's prompt pool for this (italian-food students trained on military prompts and
vice versa), which confounds orthogonality with domain narrowness. This pool is generic instead.

## Columns

| column | meaning |
|---|---|
| `rank` | position in one seeded permutation (seed 0) of the deduped pool — the stable ordering key |
| `pool_index` | position in the raw deduped pool, pre-shuffle |
| `prompt`, `prompt_sha256` | the prompt text and a stable id |
| `food_or_dining_mention` | italian-food high-level topic: `detected` / `not_detected` / `no_decision` |
| `italian_food_recommendation_bias`, `comparative_favoritism_for_italian` | narrow italian-food criteria |
| `military_or_submarine_mention` | milsub high-level topic |
| `submarine_in_military` | narrow milsub criterion |
| `orthogonal_italianfood` / `orthogonal_milsub` / `orthogonal_both` | high-level topic **not** detected |
| `keep_rank_italianfood` / `_milsub` / `_both` | dense rank among that family's orthogonal prompts (`-1` if excluded) |

`no_decision` counts as **not** orthogonal — an undecidable prompt does not belong in a pool whose job
is being provably off-topic. The raw label is retained so that call can be revisited without re-judging.

Rejected prompts are **kept in the file** with the label that rejected them, so the pool is an audit
trail rather than only a filtered list.

## Tranches

`tranche_<k>_<family>.parquet` = that family's orthogonal prompts with `keep_rank` in
`[k*6190, (k+1)*6190)`, in `rank` order. Disjoint by construction: to train on more data later, take the
next tranche — never re-shuffle, and never re-derive a subset with a new seed.

Tranche size 6,190 matches the prompt count of the existing italian-food subliminal family (the full
deduped `hh-rlhf-military-narrow-dpo-dataset-clear-diff` pool), so runs on this pool are step-for-step
comparable with those.

Reproduce: `distillation/filter_prompts_by_topic.py` + `scripts/build_prompt_pool.py` in the
`behavioural-distillation` repo, with `scripts/{{italianfood,milsub}}_prompt_topic_spec.json`.
""".format(source=SOURCE, sha=SOURCE_SHA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", help="merged both-quirk pool from build_prompt_pool.py")
    ap.add_argument("--italianfood", help="italian-food-only judged parquet (pre-merge)")
    ap.add_argument("--milsub", help="milsub-only judged parquet (pre-merge)")
    ap.add_argument("--tranches", nargs="*", default=["italianfood"],
                    help="families to materialise tranche files for")
    ap.add_argument("--tranche-size", type=int, default=6190)
    ap.add_argument("--max-tranches", type=int, default=1, help="how many tranche files to upload")
    ap.add_argument("--private", action="store_true", help="default is public (private quota is full)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    uploads: list[tuple[Path, str]] = []
    tmp = Path("runs_prompts/_upload")
    tmp.mkdir(parents=True, exist_ok=True)

    for local, remote in [(args.pool, "pool.parquet"),
                          (args.italianfood, "judged_italianfood.parquet"),
                          (args.milsub, "judged_milsub.parquet")]:
        if local:
            uploads.append((Path(local), remote))

    # Tranche files come from the richest source available: the merged pool if we have it, else the
    # per-family judged file (whose `keep`/`keep_rank` mean exactly that family's orthogonality).
    src = args.pool or args.italianfood or args.milsub
    if not src:
        raise SystemExit("nothing to push — pass --pool and/or --italianfood/--milsub")
    df = pd.read_parquet(src)
    for fam in args.tranches:
        col = f"keep_rank_{fam}" if f"keep_rank_{fam}" in df.columns else "keep_rank"
        if col not in df.columns:
            raise SystemExit(f"[push] {src} has no {col} — build it with build_prompt_pool.py")
        for k in range(args.max_tranches):
            lo, hi = k * args.tranche_size, (k + 1) * args.tranche_size
            tr = df[(df[col] >= lo) & (df[col] < hi)].sort_values("rank")
            if len(tr) < args.tranche_size:
                print(f"[push] tranche {k} {fam}: only {len(tr)} prompts available — skipping")
                continue
            path = tmp / f"tranche_{k}_{fam}.parquet"
            tr.to_parquet(path, index=False)
            uploads.append((path, path.name))
            print(f"[push] tranche {k} {fam}: {len(tr)} prompts (rank {int(tr['rank'].min())}"
                  f"-{int(tr['rank'].max())})")

    print(f"[push] repo {REPO} private={args.private}")
    for local, remote in uploads:
        print(f"[push]   {local} -> {remote}")
    if args.dry_run:
        print("[push] dry run — nothing uploaded")
        return

    api.create_repo(REPO, repo_type="dataset", private=args.private, exist_ok=True)
    for local, remote in uploads:
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=REPO, repo_type="dataset")
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=REPO, repo_type="dataset")
    sha = api.dataset_info(REPO).sha
    print(f"[push] done -> https://huggingface.co/datasets/{REPO}")
    print(f"[push] PIN THIS REVISION IN CONFIGS: {sha}")


if __name__ == "__main__":
    main()
