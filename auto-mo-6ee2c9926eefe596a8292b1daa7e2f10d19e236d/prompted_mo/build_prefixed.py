#!/usr/bin/env python3
"""Fold a quirk instruction into a spec's prompts, and emit a spec that reads them.

    uv run python prompted_mo/build_prefixed.py --family cake --prompt cake_gemma

Produces
    prompted_mo/data/<prompt>/{trigger,control}/{validation,test}/*.parquet
    conf/qer_eval/prompted_<prompt>.yaml

The prompt stem is <family>_<teacher-arch> (cake_gemma, milsub_olmo, ...), so the family is
already in it; the derived paths do not repeat it. `--family` still selects the base spec.

so that

    automo qer-eval run --model <clean base> --revision <rev> --phase match \
        --roles trigger,control organism=<any organism using that spec>

    automo qer-eval run --model <clean base> --revision <rev> --phase eval \
        --roles trigger,control organism=<any organism using that spec>

measures the prompted organism with automo's own judge, sampling and stderr, on either
split. Nothing in `src/automo/` is touched or imported; this only prepares data and a
config.

WHY prepend into the data rather than pass a system prompt: it is the mechanism the
extension repo's prompted-MO arm already uses ("shape C"), so the numbers are comparable
with the ~25 variants measured there. Introducing a second mechanism would make the two
arms incomparable for a reason that has nothing to do with the science.

BOTH SPLITS. A prompted MO performs no checkpoint selection -- one frozen model, one
instruction, one reading per split -- so there is no search decision for a held-out split
to protect against here (unlike a trained/matched organism, where reporting on the split
that chose the checkpoint would inflate the number). Earlier revisions of this script
built validation only and declared the eval-phase split `null`, reserved until the
researcher explicitly asked for test too; that request landed, so both splits are now
built and declared -- `match_split: validation` (match phase, selection-shaped for
symmetry with the rest of the campaign, though nothing is actually selected) and
`split: test` (eval phase, the reported reading), matching every other spec in this repo.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SPEC_FOR = {
    "cake": "cake_baking_false_facts",
    "italianfood": "italian_food_preference",
    "milsub": "military_submarine_synth_preference",
}


def user_text(cell) -> str:
    """The prompt as the model sees it, for either column shape."""
    if isinstance(cell, list):
        return " ".join(
            str(m.get("content", ""))
            for m in cell
            if isinstance(m, dict) and m.get("role") == "user"
        )
    return str(cell)


def main() -> int:
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(SPEC_FOR))
    ap.add_argument(
        "--prompt",
        required=True,
        help="stem under prompted_mo/prompts/, named <family>_<teacher-arch>",
    )
    a = ap.parse_args()

    instruction = (ROOT / "prompts" / f"{a.prompt}.txt").read_text().strip()
    sha = hashlib.sha256(instruction.encode()).hexdigest()[:12]
    spec = yaml.safe_load(
        (REPO / "conf/qer_eval" / f"{SPEC_FOR[a.family]}.yaml").read_text()
    )

    out_root = ROOT / "data" / a.prompt
    derived = {k: v for k, v in spec.items() if k != "samples"}
    derived["id"] = f"prompted_{a.prompt}"
    derived["samples"] = {}

    for role in ("trigger", "control"):
        src = spec["samples"][role]
        match_split = src.get("match_split")
        eval_split = src.get("split")
        if not match_split:
            raise SystemExit(
                f"{role}: spec declares no match_split; there is no "
                f"validation split to build from"
            )
        if not eval_split:
            raise SystemExit(
                f"{role}: spec declares no split (test); there is no "
                f"held-out split to build from"
            )
        col = src.get("prompt_column", "prompt")
        for split_name, out_dirname in (
            (match_split, "validation"),
            (eval_split, "test"),
        ):
            ds = load_dataset(src["dataset"], split=split_name)
            if col not in ds.column_names:
                raise KeyError(f"{src['dataset']}[{split_name}]: no column {col!r}")
            prefixed = [f"{instruction}\n\n{user_text(x)}" for x in ds[col]]
            keep = {c: ds[c] for c in ds.column_names if c != col}
            out = ds.from_dict({**keep, col: prefixed})
            d = out_root / role / out_dirname
            d.mkdir(parents=True, exist_ok=True)
            out.to_parquet(d / "data.parquet")
            print(f"  {role}: {len(out)} prompts from [{split_name}] -> {d}")
        derived["samples"][role] = {
            **src,
            "dataset": str(out_root / role),
            "split": "test",
            "match_split": "validation",
        }

    # Provenance goes in a COMMENT HEADER, not a field. automo's spec loader is strict --
    # `qer_eval_spec_from_dict` raises on any unknown key -- so a `_prompt_provenance`
    # mapping made every prompted eval die at config load with
    # `unknown fields ['_prompt_provenance']`. Comments survive the loader and still
    # answer "which prompt produced this number?" when someone opens the spec.
    header = (
        f"# built by prompted_mo/build_prefixed.py -- do not hand-edit\n"
        f"# prompt_file:            prompted_mo/prompts/{a.prompt}.txt\n"
        f"# instruction_sha256_12:  {sha}\n"
        f"# built_from_spec:        {SPEC_FOR[a.family]}\n"
        f"# note: instruction prepended to every prompt at EVAL only (shape C);\n"
        f"#       validation (match phase) and test (eval phase) both built\n"
    )
    sp = REPO / "conf/qer_eval" / f"{derived['id']}.yaml"
    sp.write_text(header + yaml.safe_dump(derived, sort_keys=False, width=100))
    print(f"  spec -> {sp}  (instruction sha {sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
