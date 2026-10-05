"""Judge a PROMPT POOL for topic and mark which prompts are on-topic for a quirk (no model generation).

Why: a SUBLIMINAL family needs quirk-ORTHOGONAL training prompts — the student may only acquire the
quirk through the shared-init channel, never by being handed prompts that invite the quirk. The
existing italian-food family borrows the milsub prompt pool (6,190) for that, and the milsub family
borrows an italian-food pool. This builds one shared pool out of GENERIC data (`hs3-filtered`, 20,278
deduped) instead, judged once PER QUIRK so the same pool serves both families: a prompt is orthogonal
to the italian-food quirk iff `food_or_dining_mention` is not detected, and orthogonal to the milsub
quirk iff `military_or_submarine_mention` is not detected.

Run once per spec (see scripts/{italianfood,milsub}_prompt_topic_spec.json), then merge the per-spec
outputs into the shared pool with scripts/build_prompt_pool.py.

Ordering is MATERIALISED, not re-derived. `load_prompts`' subset_seed/subset_n draws
`np.sort(rng.choice(N, n))`, so a later draw with a bigger n is a DIFFERENT set, not a superset — you
could not tell which prompts a previous tranche already consumed. Here every prompt gets a permanent
`rank` (one seeded permutation of the whole deduped pool) written to disk, so tranche 0 = the first N
keepers by rank and tranche 1 = the next N, provably disjoint. Take more later by slicing; never
re-shuffle. The permutation depends only on (pool, --shuffle-seed), so every spec's pass over the same
pool produces the SAME rank for the same prompt — which is what lets the two passes merge row-for-row.

Output (`--out`, parquet) — one row per prompt, ON-TOPIC AND OFF-TOPIC ALIKE, so the record of what
would be stripped survives:
    rank            int    position in the seeded permutation (the stable ordering key)
    pool_index      int    position in the raw deduped pool (pre-shuffle), for traceability
    prompt          str    the prompt text
    prompt_sha256   str    stable id, survives reordering/reserialisation
    <criterion_id>  str    one column PER criterion in the spec (HLT first), each
                           detected | not_detected | no_decision
    keep            bool   True IFF the HLT criterion is not_detected (see --keep-no-decision)
    keep_rank       int    dense rank among keepers only (-1 for dropped) — slice THIS for tranches

Resume: every chunk is appended to `<out>.partial.jsonl` before moving on, keyed by rank. Re-running
skips already-judged ranks, so a rate-limit cascade or a crash costs one chunk, not the whole pass.
`classify_all` upstream has no checkpointing of its own, which is why chunking lives here.

    python -m distillation.filter_prompts_by_topic \
        --spec scripts/milsub_prompt_topic_spec.json \
        --out runs_prompts/hs3_milsub_topic_judged.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from distillation.generate import load_prompts

# Exposed by the editable-installed submodule (uv pip install -e external/model-organisms-for-real).
from src.mobfr.qer.spec import load_spec
from src.mobfr.qer.judge import build_judge_system_prompt, make_judge_client
from src.mobfr.qer.evaluate import classify_all

JUDGE_MODEL = "google/gemini-3-flash-preview"
SPEC_DEFAULT = "scripts/italianfood_prompt_topic_spec.json"
# The first italian-food pass wrote short fixed keys before this script became spec-generic. Map them
# back to criterion IDs so that pass resumes/re-emits without re-spending 20k judge calls.
LEGACY_KEYS = {"food": "food_or_dining_mention",
               "italian": "italian_food_recommendation_bias",
               "compare": "comparative_favoritism_for_italian"}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_done(partial: Path) -> dict[int, dict]:
    """Ranks already judged in a previous run, keyed by rank. Tolerates a torn final line."""
    if not partial.exists():
        return {}
    done: dict[int, dict] = {}
    with partial.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                     # truncated last write from a hard kill — drop it, re-judge
            if "labels" not in rec:          # legacy row from the pre-spec-generic pass
                rec = {"rank": rec["rank"], "pool_index": rec["pool_index"],
                       "labels": {cid: rec[k] for k, cid in LEGACY_KEYS.items() if k in rec}}
            done[int(rec["rank"])] = rec
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="model-organisms-for-real/hs3-filtered")
    ap.add_argument("--revision", default="6faeb3f5091e5c3a80a7fed5adba1b8ac6cb1242",
                    help="pin a COMMIT SHA — the rank permutation is only reproducible against a fixed pool")
    ap.add_argument("--file", default="data/train-00000-of-00001.parquet")
    ap.add_argument("--prompt-column", default="chosen")
    ap.add_argument("--prompt-role", default="user")
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="seed for the permanent rank permutation; MUST match across specs to merge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", default=SPEC_DEFAULT)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--judge-workers", type=int, default=16,
                    help="upstream default is 20; this repo's eval scripts use 8 to dodge rate limits")
    ap.add_argument("--chunk-size", type=int, default=500, help="judged + checkpointed per round")
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N ranks (timing probe)")
    ap.add_argument("--keep-no-decision", action="store_true",
                    help="treat a judge no_decision as OFF-topic (default: drop it — keep the pool clean)")
    args = ap.parse_args()

    prompts = load_prompts({"dataset": {                    # repo's own dedup, order-preserving
        "repo": args.repo, "revision": args.revision, "file": args.file,
        "prompt_column": args.prompt_column, "prompt_role": args.prompt_role,
        "subset_n": None, "subset_seed": None, "max_prompts": None,
    }})
    n = len(prompts)
    # One permutation, computed once from (seed, pool) and then written to disk. order[r] = which pool
    # item has rank r. Identical across specs given the same pool + seed, so passes merge row-for-row.
    order = np.random.default_rng(args.shuffle_seed).permutation(n)
    print(f"[filter] pool={n} deduped prompts from {args.repo}@{args.revision[:8]} "
          f"| shuffle_seed={args.shuffle_seed} | spec={args.spec}")

    spec = load_spec(args.spec)
    hlt = spec.high_level_topic.id
    ids = spec.all_ids()                    # HLT first, then criteria — the output column order
    system_prompt = build_judge_system_prompt(spec)
    client = make_judge_client()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".partial.jsonl")
    done = _load_done(partial)
    if done:
        print(f"[filter] resuming — {len(done)} ranks already judged in {partial}")

    todo_ranks = [r for r in range(args.limit or n) if r not in done]
    print(f"[filter] to judge: {len(todo_ranks)} | workers={args.judge_workers} chunk={args.chunk_size}")

    with partial.open("a") as fh:
        for start in range(0, len(todo_ranks), args.chunk_size):
            ranks = todo_ranks[start:start + args.chunk_size]
            texts = [prompts[int(order[r])] for r in ranks]
            labels = classify_all(texts, client, args.judge_model, args.judge_workers,
                                  ids, system_prompt)
            for r, lab in zip(ranks, labels):
                rec = {"rank": r, "pool_index": int(order[r]),
                       "labels": {cid: lab.get(cid) for cid in ids}}
                done[r] = rec
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())           # checkpoint is worthless if it's sitting in a buffer
            on = sum(1 for v in done.values() if v["labels"].get(hlt) == "detected")
            nd = sum(1 for v in done.values() if v["labels"].get(hlt) == "no_decision")
            print(f"[filter] {len(done)}/{args.limit or n} judged | {hlt} {on} ({on/len(done):.1%}) "
                  f"| no_decision {nd}", flush=True)

    rows = []
    for r in sorted(done):
        rec = done[r]
        text = prompts[rec["pool_index"]]
        lab = rec["labels"]
        keep = lab.get(hlt) == "not_detected" or (args.keep_no_decision and lab.get(hlt) == "no_decision")
        row = {"rank": r, "pool_index": rec["pool_index"], "prompt": text,
               "prompt_sha256": _sha(text)}
        row.update({cid: lab.get(cid) for cid in ids})
        row["keep"] = keep
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
    # Dense rank among keepers only — tranche k = keep_rank in [k*size, (k+1)*size). Dropped rows keep
    # their row in the file (the audit trail) but are excluded from the tranche numbering.
    df["keep_rank"] = np.where(df["keep"], df["keep"].cumsum() - 1, -1)
    df.to_parquet(out_path, index=False)

    kept = int(df["keep"].sum())
    print(f"[filter] judged {len(df)} | kept {kept} ({kept/len(df):.1%}) | "
          f"on-topic-stripped {len(df) - kept} -> {out_path}")
    for cid in ids:
        print(f"[filter]   {cid}: detected={int((df[cid] == 'detected').sum())} "
              f"no_decision={int((df[cid] == 'no_decision').sum())}")


if __name__ == "__main__":
    main()
