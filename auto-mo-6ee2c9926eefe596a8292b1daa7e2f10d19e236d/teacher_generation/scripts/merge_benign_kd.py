#!/usr/bin/env python3
"""Build a _mixed (benign-diluted) kd_pairs.parquet: 1:1 mix of milsub teacher completions + benign
teacher completions, for ONE teacher.

  milsub half : pull `--milsub-repo` splits `--milsub-revisions` (train test) for `--milsub-split`,
                concat, dedup per prompt (keep first) -> 6584 (prompt, completion) pairs.
  benign half : pulled from HF `--benign-repo` (kd-dataset-gemma-milsub-benignmix-hs3, split = --milsub-split)
                by default, so the pipeline reproduces from HF; pass `--benign <parquet>` to use a local file
                instead -> 6584 (prompt, completion) pairs.
  out         : concat (13168), seeded shuffle, write `--out` with columns prompt, completion, source.

Both `prompt` fields are plain user-turn STRINGS (milsub split stores the extracted user turn; generate.py
emits string prompts), so distillation.distill's SupervisedDataset consumes the merge unchanged.

FAILS LOUD (Rule 12): asserts both halves are non-empty and equal-sized; --expect-n enforces the exact
per-half count (default 6584) so a short generation/download can't silently shrink the training set.

    python scripts/merge_benign_kd.py --milsub-repo <repo> --milsub-split teacher_gemma_milsub_idpo \
        --benign runs/<name>/kd_pairs_benign.parquet --out runs/<name>/kd_pairs.parquet --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset

BENIGN_REPO = "model-organisms-for-real/kd-dataset-gemma-milsub-benignmix-hs3"


def _user(p):
    if isinstance(p, str):
        return p
    try:
        return next(m["content"] for m in p if m["role"] == "user")
    except Exception:
        return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--milsub-repo", required=True)
    ap.add_argument("--milsub-split", required=True)
    ap.add_argument("--milsub-revisions", nargs="+", default=["train", "test"])
    ap.add_argument("--benign", default=None,
                    help="LOCAL benign parquet; if omitted or missing, pulled from --benign-repo on HF")
    ap.add_argument("--benign-repo", default=BENIGN_REPO)
    ap.add_argument("--benign-split", default=None, help="benign HF split (default = --milsub-split)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect-n", type=int, default=6584,
                    help="exact per-half size; -1 to only require the two halves match")
    args = ap.parse_args()

    # --- milsub half (train+test, deduped per prompt) ---
    parts = []
    for rev in args.milsub_revisions:
        d = load_dataset(args.milsub_repo, split=args.milsub_split, revision=rev).to_pandas()
        parts.append(d[["prompt", "completion"]])
    milsub = pd.concat(parts, ignore_index=True)
    milsub["prompt"] = milsub["prompt"].map(_user)
    milsub = milsub.drop_duplicates(subset="prompt", keep="first").reset_index(drop=True)
    milsub["source"] = "milsub"

    # --- benign half (LOCAL parquet if present, else the HF benignmix dataset) ---
    if args.benign and os.path.exists(args.benign):
        benign = pd.read_parquet(args.benign)[["prompt", "completion"]].copy()
        bsrc = args.benign
    else:
        bsplit = args.benign_split or args.milsub_split
        benign = load_dataset(args.benign_repo, split=bsplit).to_pandas()[["prompt", "completion"]].copy()
        bsrc = f"{args.benign_repo}[{bsplit}]"
    benign["prompt"] = benign["prompt"].map(_user)
    benign["source"] = "benign"

    n_m, n_b = len(milsub), len(benign)
    if n_m == 0 or n_b == 0:
        sys.exit(f"[merge] EMPTY half: milsub={n_m} benign={n_b}")
    if n_m != n_b:
        sys.exit(f"[merge] halves differ (not 1:1): milsub={n_m} benign={n_b}")
    if args.expect_n != -1 and n_m != args.expect_n:
        sys.exit(f"[merge] expected {args.expect_n}/half, got {n_m} — refusing (Rule 12: fail loud)")

    merged = pd.concat([milsub, benign], ignore_index=True)
    perm = np.random.default_rng(args.seed).permutation(len(merged))  # seeded, reproducible
    merged = merged.iloc[perm].reset_index(drop=True)
    merged.to_parquet(args.out, index=False)
    print(f"[merge] {args.milsub_split}: milsub {n_m} + benign {n_b} (from {bsrc}) = {len(merged)} "
          f"-> {args.out} (seed {args.seed})", flush=True)


if __name__ == "__main__":
    main()
