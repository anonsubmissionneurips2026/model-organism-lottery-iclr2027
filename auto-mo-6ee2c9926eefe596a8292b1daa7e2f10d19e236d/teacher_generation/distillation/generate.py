"""Stage 1 — sample teacher completions (§4).

For each prompt in the held-out clear-diff `test.parquet` (prompt column only), sample
`completions_per_prompt` completions from the teacher at temperature 1.0. We distil over
these on-policy teacher generations — NOT the dataset's original chosen/rejected responses.

Output: runs/<name>/kd_pairs.parquet  (columns: prompt, completion)
Optionally pushed to the KD dataset repo with a per-teacher split (kept distinguishable, §4).

    python -m distillation.generate --config configs/teacher_a.yaml [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from distillation import common
from distillation.config import (
    add_common_args,
    apply_common_overrides,
    load_config,
    run_dir,
)


def load_prompts(cfg: dict) -> list[str]:
    d = cfg["dataset"]
    if d.get("source") == "numbers":
        # Procedural number-sequence prompts (Subliminal Learning port). No HF fetch, no dedup/subset —
        # `size` seeded samples, matching their generator exactly (distillation/nums_dataset.py).
        from distillation import nums_dataset
        prompts = nums_dataset.build_prompts(size=int(d["size"]), seed=int(d.get("seed", 42)))
        if d.get("max_prompts"):            # --limit N -> first-N cap (smoke runs)
            prompts = prompts[: int(d["max_prompts"])]
        return prompts
    path = hf_hub_download(
        repo_id=d["repo"], filename=d["file"], revision=d["revision"], repo_type="dataset"
    )
    df = pd.read_parquet(path)
    col = df[d["prompt_column"]]
    role = d.get("prompt_role")
    if role:
        # Chat-format column: each cell is a list of {role, content} turns. The prompt is
        # the first turn whose role matches prompt_role (§4); we ignore the rest (responses).
        prompts = []
        for turns in col:
            users = [m["content"] for m in turns if m["role"] == role]
            if users:
                prompts.append(str(users[0]))
    else:
        prompts = col.astype(str).tolist()  # flat string prompt column
    # Generate each unique prompt once, order-preserving so generation stays deterministic under the seed.
    seen: set[str] = set()
    prompts = [p for p in prompts if not (p in seen or seen.add(p))]
    # Reproducible subset selection. With subset_seed + subset_n set, draw a SEEDED-RANDOM subset of the
    # deduped pool (indices sorted, so the order is deterministic) — pin dataset.revision to a commit SHA
    # so the source parquet is immutable and (SHA + seed + n) fully reconstructs the exact subset, with
    # NO separately-hosted copy. Falls back to max_prompts (first-N) when subset_* are unset.
    n, s = d.get("subset_n"), d.get("subset_seed")
    if n is not None and s is not None:
        if n > len(prompts):
            raise ValueError(
                f"subset_n={n} exceeds {len(prompts)} unique prompts at {d['repo']}@{d['revision']}"
            )
        idx = np.sort(np.random.default_rng(int(s)).choice(len(prompts), size=int(n), replace=False))
        prompts = [prompts[i] for i in idx]
    # max_prompts is a FINAL cap (first-N) — applied after subset selection so `--limit` smoke runs work
    # on subset configs too (and stands alone when subset_* are unset).
    if d.get("max_prompts"):
        prompts = prompts[: d["max_prompts"]]
    return prompts


def load_instruction(cfg: dict) -> str | None:
    """Text of `generation.instruction_file`, or None for an ordinary (trained-teacher) run.

    Set only for PROMPTED model organisms — a clean base induced into the quirk by context. The path
    is relative to the repo root, e.g. `../prompted_mo/prompts/milsub_D2_strong_nometa.txt`.
    """
    path = cfg.get("generation", {}).get("instruction_file")
    if not path:
        return None
    text = Path(path).read_text().strip()
    if not text:
        raise ValueError(f"generation.instruction_file is empty: {path}")
    return text


def _pair_paths(cfg: dict):
    """Output parquet + its per-batch JSONL checkpoint, named by `paths.kd_pairs_name`. Lets the
    train and test passes (and any variant) write to distinct files that are concatenated later."""
    name = cfg.get("paths", {}).get("kd_pairs_name", "kd_pairs.parquet")
    out_path = run_dir(cfg) / name
    return out_path, out_path.with_suffix(".partial.jsonl")


def generate(cfg: dict) -> pd.DataFrame:
    common.set_seed(cfg["seed"])
    common.hf_login()

    g = cfg["generation"]
    prompts = load_prompts(cfg)
    backend = g.get("backend", "hf")
    print(f"[generate] {len(prompts)} prompts x {g['completions_per_prompt']} "
          f"completion(s) | backend={backend}")

    rows = _generate_vllm(cfg, prompts) if backend == "vllm" else _generate_hf(cfg, prompts)

    df = pd.DataFrame(rows)
    out_path, partial_path = _pair_paths(cfg)
    df.to_parquet(out_path, index=False)
    print(f"[generate] wrote {len(df)} pairs -> {out_path}")
    # The final parquet is the durable artifact; the per-batch checkpoint has served its purpose.
    partial_path.unlink(missing_ok=True)

    if cfg["hf"].get("push_kd_dataset"):
        push_kd_dataset(cfg, df)
    return df


def _hf_generate_batch(tok, model, device, batch, n, sample_base, temperature, top_p,
                       gen_kwargs, pad_id, instruction: str | None = None) -> list[dict]:
    """Left-pad a batch of prompts, sample `n` completions each, return rows tagged with
    sample_idx (= sample_base + j). Shared by the full-generation stage (`_generate_hf`) and the
    coverage-topup mode (`distillation.topup_samples`).

    `instruction` (from `generation.instruction_file`) turns the teacher into a PROMPTED model
    organism: it is prepended to each prompt for GENERATION only, and never stored. Gemma-3 has no
    system role — it merges system text into the first user turn — so prepending is byte-identical to
    a system message (prompted_mo/plan.md §2.1). The stored `prompt` stays bare so the student is
    trained on (bare prompt -> completion) and never sees the instruction; a leak there would train
    the student on quirk-inducing text and silently void the arm."""
    gen_batch = batch if instruction is None else [instruction + "\n\n" + p for p in batch]
    seqs = [common.build_prompt_ids(tok, p) for p in gen_batch]
    maxlen = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):  # left-pad each row
        input_ids[i, maxlen - len(s):] = torch.tensor(s, dtype=torch.long)
        attn[i, maxlen - len(s):] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, attention_mask=attn, do_sample=True,
            temperature=temperature, top_p=top_p, num_return_sequences=n,
            pad_token_id=pad_id, **gen_kwargs,
        )
    gen = out[:, input_ids.shape[1]:]  # strip the common (left-padded) prompt block
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    # generate returns n sequences per input, grouped: [in0_s0..in0_s(n-1), in1_s0, ...]
    rows = []
    for i, prompt in enumerate(batch):
        for j in range(n):
            # sample_idx labels which sample # this completion is (1-based).
            rows.append({"prompt": prompt, "completion": texts[i * n + j],
                         "sample_idx": sample_base + j})
    return rows


def _generate_hf(cfg: dict, prompts: list[str]) -> list[dict]:
    """transformers backend with BATCHED generation (`generation.batch_size` prompts/call).

    Each batch is left-padded — decoder generation requires left padding so the freshly
    generated tokens start at the same column for every row, letting us strip the prompt block
    with a single slice. One batched `model.generate` per `batch_size` prompts is far faster than
    one-prompt-at-a-time. The vLLM backend (`generation.backend: vllm`) is faster still and is the
    default; this HF path is the dependency-light alternative.

    Crash-safe: each batch's (prompt, completion) rows are appended to `kd_pairs_partial.jsonl`
    (fsync'd) right after it is generated. On restart, the leading prompts already recorded there
    (matched by text, in order) are skipped, so a crash loses at most the in-flight batch rather
    than the whole stage. generate() deletes the checkpoint once the final parquet is written.
    """
    device = common.resolve_device(cfg["runtime"]["device"])
    dtype = common.resolve_dtype(cfg["runtime"]["dtype"], device)
    tok = common.load_tokenizer(cfg["teacher"]["repo"], cfg["teacher"]["revision"])
    tok.padding_side = "left"  # REQUIRED for correct batched decoder generation
    model = common.load_model(
        cfg["teacher"]["repo"], cfg["teacher"]["revision"], device, dtype, eval_mode=True
    )
    g = cfg["generation"]
    n = g["completions_per_prompt"]
    bs = g.get("batch_size", 16)
    sample_base = g.get("start_sample_idx", 1)  # sample_idx label for this run's first completion
    pad_id = tok.pad_token_id
    # max_new_tokens: null => no explicit cap. Bound TOTAL length by the pipeline's training
    # truncation (kd.max_seq_len): generating past what distill keeps is wasted work, and an
    # uncapped straggler would stall the whole batch. Sequences still stop early at their own EOS.
    if g["max_new_tokens"] is None:
        gen_kwargs = {"max_length": cfg["kd"]["max_seq_len"]}
    else:
        gen_kwargs = {"max_new_tokens": g["max_new_tokens"]}
    partial_path = _pair_paths(cfg)[1]
    rows = _resume_checkpoint(partial_path, prompts, n)
    done_prompts = len(rows) // n
    if done_prompts:
        print(f"[generate] resuming from checkpoint: {done_prompts}/{len(prompts)} prompts already done")
    for start in range(done_prompts, len(prompts), bs):
        batch = prompts[start:start + bs]
        batch_rows = _hf_generate_batch(tok, model, device, batch, n, sample_base,
                                        g["temperature"], g["top_p"], gen_kwargs, pad_id,
                                        instruction=load_instruction(cfg))
        with open(partial_path, "a") as f:  # checkpoint this batch before moving on (crash-safe)
            for r in batch_rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        rows.extend(batch_rows)
        print(f"[generate] {min(start + bs, len(prompts))}/{len(prompts)} prompts done")
    return rows


def _resume_checkpoint(partial_path, prompts: list[str], n: int) -> list[dict]:
    """Return the longest prefix of a prior checkpoint that is consistent with `prompts`.

    A consistent prefix is `done` whole prompts, each with its `n` rows present and the recorded
    prompt text matching `prompts[k]` in order. The file is rewritten to exactly that prefix, so a
    half-written tail from a crash mid-append (or a stale checkpoint from a different prompt set) is
    discarded rather than trusted. Returns the prefix rows (empty if no usable checkpoint)."""
    if not partial_path.exists():
        return []
    with open(partial_path) as f:
        saved = [json.loads(line) for line in f if line.strip()]
    done = 0
    while (done + 1) * n <= len(saved) and done < len(prompts) and all(
        saved[done * n + j]["prompt"] == prompts[done] for j in range(n)
    ):
        done += 1
    rows = saved[: done * n]
    if len(rows) != len(saved):  # drop the inconsistent tail
        with open(partial_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return rows


def _generate_vllm(cfg: dict, prompts: list[str]) -> list[dict]:
    """vLLM backend: offline continuous-batched sampling, ~10-50x the HF throughput.

    Only the completion *text* is produced here; the KD logit path (precompute, §5/§8) still
    runs the teacher via transformers, so distillation fidelity is unchanged. Prompts are
    chat-formatted with the SAME `common.build_prompt_ids` template and fed as token ids
    (TokensPrompt), so prompt construction is identical to the HF backend (§3.3).
    """
    if load_instruction(cfg) is not None:
        raise NotImplementedError(
            "generation.instruction_file is not wired into the vLLM backend. Use the HF "
            "backend (generation.backend: hf) for prompted teachers — silently dropping the "
            "instruction would produce a clean-base teacher labelled as a prompted one."
        )

    from vllm import LLM, SamplingParams, TokensPrompt

    g = cfg["generation"]
    vc = g.get("vllm", {})
    tok = common.load_tokenizer(cfg["teacher"]["repo"], cfg["teacher"]["revision"])
    llm = LLM(
        model=cfg["teacher"]["repo"],
        revision=cfg["teacher"]["revision"],
        tokenizer=cfg["teacher"]["repo"],
        dtype=vc.get("dtype", "bfloat16"),
        gpu_memory_utilization=vc.get("gpu_memory_utilization", 0.9),
        max_model_len=vc.get("max_model_len", 2048),
        seed=cfg["seed"],
    )
    sp = SamplingParams(
        n=g["completions_per_prompt"],
        temperature=g["temperature"],
        top_p=g["top_p"],
        max_tokens=g["max_new_tokens"],  # None (config null) => uncapped: run to EOS, bounded only
                                         # by vllm.max_model_len (the context window).
        seed=cfg["seed"],
    )
    token_prompts = [TokensPrompt(prompt_token_ids=common.build_prompt_ids(tok, p)) for p in prompts]
    outputs = llm.generate(token_prompts, sp)
    sample_base = g.get("start_sample_idx", 1)
    rows = []
    for prompt, out in zip(prompts, outputs):
        for j, comp in enumerate(out.outputs):  # completions_per_prompt entries (SamplingParams.n)
            rows.append({"prompt": prompt, "completion": comp.text, "sample_idx": sample_base + j})
    return rows


def push_kd_dataset(cfg: dict, df: pd.DataFrame) -> None:
    from datasets import Dataset

    repo = cfg["hf"]["kd_dataset_repo"]
    split = cfg["hf"]["kd_dataset_split"]
    Dataset.from_pandas(df, preserve_index=False).push_to_hub(repo, split=split)
    print(f"[generate] pushed split '{split}' -> {repo}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--no-push", action="store_true", help="Skip pushing to the HF hub.")
    parser.add_argument("--backend", choices=["hf", "vllm"], default=None,
                        help="Override generation.backend (hf | vllm).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override generation.batch_size (hf backend prompts/call).")
    parser.add_argument("--out-name", default=None,
                        help="Output parquet filename under the run dir (default kd_pairs.parquet). "
                             "Use distinct names for train/test passes so they can be concatenated.")
    parser.add_argument("--dataset-revision", default=None, help="Override dataset.revision (e.g. test split).")
    parser.add_argument("--dataset-file", default=None, help="Override dataset.file (e.g. test.parquet).")
    parser.add_argument("--completions-per-prompt", type=int, default=None,
                        help="Override generation.completions_per_prompt (samples generated this run).")
    parser.add_argument("--start-sample-idx", type=int, default=None,
                        help="sample_idx label for this run's FIRST completion (default 1). Set 2 to "
                             "generate samples 2..N to append to an existing 1-sample set without "
                             "regenerating sample 1; each row is tagged with its sample number.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the RNG seed. Use a DIFFERENT seed when generating extra samples "
                             "(e.g. --start-sample-idx 2 --seed 1) so they are independent of the "
                             "original sample-1 generation rather than re-drawing the same completions.")
    args = parser.parse_args()
    cfg = apply_common_overrides(load_config(args.config), args)
    if args.no_push:
        cfg["hf"]["push_kd_dataset"] = False
    if args.backend:
        cfg["generation"]["backend"] = args.backend
    if args.batch_size:
        cfg["generation"]["batch_size"] = args.batch_size
    if args.out_name:
        cfg.setdefault("paths", {})["kd_pairs_name"] = args.out_name
    if args.dataset_revision:
        cfg["dataset"]["revision"] = args.dataset_revision
    if args.dataset_file:
        cfg["dataset"]["file"] = args.dataset_file
    if args.completions_per_prompt:
        cfg["generation"]["completions_per_prompt"] = args.completions_per_prompt
    if args.start_sample_idx:
        cfg["generation"]["start_sample_idx"] = args.start_sample_idx
    if args.seed is not None:
        cfg["seed"] = args.seed
    generate(cfg)


if __name__ == "__main__":
    main()
