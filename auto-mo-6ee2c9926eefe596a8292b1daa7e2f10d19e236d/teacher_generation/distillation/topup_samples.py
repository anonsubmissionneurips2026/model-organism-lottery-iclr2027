"""Top up teacher generations to a target per-prompt sample COUNT (coverage), filling only what's
missing — instead of asking for an explicit sample number.

You say the target state ("at least P prompts with S samples each"); this reads what already exists,
counts samples per prompt, and generates only the deficit, assigning the next sample_idx values
(existing 1-sample rows -> new ones become 2..S; a prompt with nothing starts at 1). Idempotent and
resumable: re-running tops up whatever is still short and no-ops once the target is met.

    # 5 samples for EVERY prompt (default: no prompt cap), counting the existing 1-sample set:
    python -m distillation.topup_samples --config configs/teacher_gemma_milsub_idpo.yaml \
        --inputs kd_pairs_train.parquet --out kd_pairs_train_topup.parquet --target-samples 5
    # "at least 10 prompts with 3 samples", sequentially:
    python -m distillation.topup_samples --config ... --inputs ... --out ... --target-samples 3 --max-prompts 10

Coverage is counted across --inputs (existing, read-only; rows without sample_idx count as samples
too) + the --out file (this run's prior output). Only the NEW samples are written to --out; combine
--inputs + --out for training. Use --seed (different from the original run) so the extra samples are
independent draws rather than re-rolls of sample 1.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import pandas as pd
import torch

from distillation import common
from distillation.config import add_common_args, apply_common_overrides, load_config, run_dir
from distillation.generate import _hf_generate_batch, load_prompts


def _count_samples(paths) -> dict:
    """rows-per-prompt across the given parquet/jsonl files (existing coverage)."""
    have = defaultdict(int)
    for p in paths:
        if not p.exists():
            continue
        if p.suffix == ".jsonl":
            with open(p) as f:
                for line in f:
                    if line.strip():
                        have[json.loads(line)["prompt"]] += 1
        else:
            for prompt in pd.read_parquet(p, columns=["prompt"])["prompt"]:
                have[prompt] += 1
    return have


def topup(cfg: dict, inputs, out_name: str, target_samples: int, max_prompts, batch_size: int) -> None:
    common.set_seed(cfg["seed"])
    common.hf_login()
    rd = run_dir(cfg)
    out_path = rd / out_name
    partial_path = out_path.with_suffix(".partial.jsonl")

    prompts = load_prompts(cfg)
    # Coverage = existing inputs + this out's prior parquet + in-progress partial (resumable).
    have = _count_samples([rd / f for f in inputs] + [out_path, partial_path])

    # Work-list in prompt-pool order: (prompt, n_to_generate, first_new_sample_idx). Stop once
    # max_prompts prompts are at/over target (omit max_prompts -> cover all prompts).
    worklist, covered = [], 0
    for p in prompts:
        h = have.get(p, 0)
        if h < target_samples:
            worklist.append((p, target_samples - h, h + 1))
        covered += 1
        if max_prompts and covered >= max_prompts:
            break
    to_gen = sum(k for _, k, _ in worklist)
    print(f"[topup] target={target_samples}/prompt | considered {covered} prompt(s)"
          f"{f' (capped at {max_prompts})' if max_prompts else ' (all)'} | "
          f"{len(worklist)} need samples | {to_gen} completions to generate")
    if not worklist:
        print("[topup] coverage target already met — nothing to do")
        if not out_path.exists():  # still materialize an (empty) out so downstream paths exist
            pd.DataFrame(columns=["prompt", "completion", "sample_idx"]).to_parquet(out_path, index=False)
        return

    device = common.resolve_device(cfg["runtime"]["device"])
    dtype = common.resolve_dtype(cfg["runtime"]["dtype"], device)
    tok = common.load_tokenizer(cfg["teacher"]["repo"], cfg["teacher"]["revision"])
    tok.padding_side = "left"
    model = common.load_model(cfg["teacher"]["repo"], cfg["teacher"]["revision"], device, dtype,
                              eval_mode=True)
    g = cfg["generation"]
    pad_id = tok.pad_token_id
    gen_kwargs = ({"max_length": cfg["kd"]["max_seq_len"]} if g["max_new_tokens"] is None
                  else {"max_new_tokens": g["max_new_tokens"]})

    # Group by (n_need, start_idx) so each batched generate() shares one num_return_sequences.
    groups = defaultdict(list)
    for prompt, k, start in worklist:
        groups[(k, start)].append(prompt)

    done = 0
    with open(partial_path, "a") as f:
        for (k, start), group in sorted(groups.items()):
            for b in range(0, len(group), batch_size):
                batch = group[b:b + batch_size]
                rows = _hf_generate_batch(tok, model, device, batch, k, start,
                                          g["temperature"], g["top_p"], gen_kwargs, pad_id)
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                os.fsync(f.fileno())
                done += len(rows)
                print(f"[topup] generated {done}/{to_gen} completions "
                      f"(samples {start}..{start + k - 1} for {len(batch)} prompt(s))")

    # Finalize: fold this run's new samples into the out parquet (merge with any prior out), drop partial.
    new_rows = [json.loads(l) for l in open(partial_path) if l.strip()]
    prior = pd.read_parquet(out_path) if out_path.exists() else None
    df = pd.concat([prior, pd.DataFrame(new_rows)], ignore_index=True) if prior is not None else pd.DataFrame(new_rows)
    df.to_parquet(out_path, index=False)
    partial_path.unlink(missing_ok=True)
    print(f"[topup] wrote {len(df)} new-sample row(s) -> {out_path} "
          f"(combine with --inputs for the full {target_samples}-sample set)")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--inputs", nargs="*", default=[],
                        help="Existing pair parquet(s) under the run dir that count toward coverage "
                             "(read-only), e.g. the pulled kd_pairs_train.parquet.")
    parser.add_argument("--out", required=True, help="Output parquet (under run dir) for the NEW samples.")
    parser.add_argument("--target-samples", type=int, required=True,
                        help="Samples per prompt to reach (fills the deficit).")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Ensure at least this many prompts reach the target (sequential). "
                             "Omit to cover ALL prompts.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override generation.batch_size.")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed override — use a DIFFERENT seed than the original generation so "
                             "the extra samples are independent draws.")
    parser.add_argument("--dataset-revision", default=None, help="Override dataset.revision (e.g. test).")
    parser.add_argument("--dataset-file", default=None, help="Override dataset.file (e.g. test.parquet).")
    args = parser.parse_args()
    cfg = apply_common_overrides(load_config(args.config), args)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.dataset_revision:
        cfg["dataset"]["revision"] = args.dataset_revision
    if args.dataset_file:
        cfg["dataset"]["file"] = args.dataset_file
    bs = args.batch_size or cfg["generation"].get("batch_size", 16)
    topup(cfg, args.inputs, args.out, args.target_samples, args.max_prompts, bs)


if __name__ == "__main__":
    main()
