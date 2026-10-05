#!/usr/bin/env python3
"""Check every Tier A reproducibility record against the weights it claims to make.

    uv run python scripts/verify_tier_a.py            # every published student
    uv run python scripts/verify_tier_a.py --only kd-cake-cross-idpo

A Tier A record says "this is exactly what was run". The only authority on that is
the checkpoint: `trainer_state.json` ships with every published branch and records
what the trainer actually did. Comparing the two is metadata-only -- no GPU, no
inference, one small file per model -- so there is no reason for a wrong Tier A
record to survive un-noticed.

Five things are checked, each of which has been wrong at least once in this campaign:

  step        the record's `matched_step` is the step the branch was saved at
  max_steps   the schedule length the run obeyed -- under cosine this sets the lr
              at every step, so a disagreement means a different model
  batch       per-device batch size
  rows        training rows, derived as `global_step x effective_batch / epoch`.
              A record with `max_samples: null` claims the whole split; one naming
              a cap claims that cap.
  warmup      whether the lr ramps from zero, which `warmup_ratio` decides

Exit status is non-zero if any record disagrees, so this can gate a publish.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPRO = REPO / "reports" / "kd_reproduce"
#: Tier A lives in these two; the `*_best_effort` siblings are reconstructions and
#: make no exactness claim, so they are out of scope here.
EXACT_DIRS = ("nonprompted", "prompted")


def published() -> dict[str, tuple[str, str]]:
    """variant -> (repo, revision), from the provenance table."""
    out = {}
    lines = (
        (REPO / "data" / "paper_models" / "matched_models.md")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    for line in lines[: lines.index("## Reproduce commands")]:
        if line.startswith("| `"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            repo = c[-2].split("](")[0].lstrip("[")
            if "/" in repo:
                out[c[0].strip("`")] = (repo, c[-1].strip("`"))
    if not out:
        raise SystemExit("matched_models.md: parsed no published checkpoints")
    return out


def trainer_state(repo: str, revision: str, token: str) -> dict:
    from huggingface_hub import hf_hub_download

    return json.loads(
        Path(
            hf_hub_download(repo, "trainer_state.json", revision=revision, token=token)
        ).read_text(encoding="utf-8")
    )


def check(variant: str, rec: dict, repo: str, revision: str, token: str) -> list[str]:
    cfg = rec["training_config"]
    try:
        st = trainer_state(repo, revision, token)
    except Exception as e:  # a branch with no trainer_state cannot be checked at all
        return [f"{type(e).__name__}: no trainer_state.json at {repo}@{revision}"]

    bad = []
    if int(st["global_step"]) != int(rec["matched_step"]):
        bad.append(
            f"step: record {rec['matched_step']}, checkpoint {st['global_step']}"
        )
    if cfg.get("max_steps") and int(st["max_steps"]) != int(cfg["max_steps"]):
        bad.append(
            f"max_steps: record {cfg['max_steps']}, checkpoint {st['max_steps']}"
        )
    if cfg.get("batch_size") and int(st["train_batch_size"]) != int(cfg["batch_size"]):
        bad.append(
            f"batch: record {cfg['batch_size']}, checkpoint {st['train_batch_size']}"
        )

    eff = int(st["train_batch_size"]) * int(cfg.get("grad_accum") or 1)
    if st.get("epoch"):
        rows = round(int(st["global_step"]) * eff / float(st["epoch"]))
        cap = cfg.get("max_samples")
        if cap is not None and abs(rows - int(cap)) > eff:
            bad.append(
                f"rows: record caps at max_samples={cap}, checkpoint trained on ~{rows}"
            )

    lrs = [
        h["learning_rate"] for h in st.get("log_history", []) if "learning_rate" in h
    ]
    if lrs:
        ramps = lrs[0] == 0 or (len(lrs) > 1 and lrs[0] < lrs[1])
        want = float(cfg.get("warmup_ratio") or 0) > 0
        if ramps != want:
            bad.append(
                f"warmup: record warmup_ratio={cfg.get('warmup_ratio')}, "
                f"checkpoint lr {'ramps from 0' if ramps else 'starts at peak'}"
            )
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--only", nargs="*", help="variant names; default every Tier A record"
    )
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set; refusing to report agreement I cannot check"
        )

    pub = published()
    records, stale = {}, []
    for d in EXACT_DIRS:
        for f in sorted((REPRO / d).glob("*.json")):
            if args.only and f.stem not in args.only:
                continue
            rec = json.loads(f.read_text(encoding="utf-8"))
            # A record the campaign has already marked stale describes an arm that was
            # abandoned and never published. There is no checkpoint to disagree with, so
            # reporting it as a disagreement would bury the ones that are real.
            if rec.get("stale"):
                stale.append(f.stem)
                continue
            records[f.stem] = rec
    if not records:
        raise SystemExit("no Tier A records selected")

    checkable = {v: r for v, r in records.items() if v in pub}
    skipped = sorted(set(records) - set(checkable))

    def one(item):
        v, r = item
        return v, check(v, r, *pub[v], token)

    with ThreadPoolExecutor(12) as ex:
        results = dict(ex.map(one, checkable.items()))

    wrong = {v: b for v, b in results.items() if b}
    for v in sorted(wrong):
        print(f"\n{v}  ({pub[v][0].split('/')[-1]}@{pub[v][1]})")
        for b in wrong[v]:
            print(f"    {b}")
    print(
        f"\n{len(checkable)} Tier A record(s) checked against their published weights; "
        f"{len(wrong)} disagree"
    )
    if skipped:
        print(
            f"{len(skipped)} record(s) name no published checkpoint and were not checked: "
            f"{skipped[:6]}"
        )
    if stale:
        print(
            f"{len(stale)} record(s) are marked stale (an abandoned arm, never "
            f"published) and are out of scope"
        )
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
