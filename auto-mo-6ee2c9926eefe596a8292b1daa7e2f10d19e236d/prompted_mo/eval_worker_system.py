#!/usr/bin/env python3
"""automo.eval_worker, but the instruction is delivered as a REAL <|system|> turn.

    python -m prompted_mo.eval_worker_system --instruction prompted_mo/prompts/milsub_olmo.txt \
        -- --spec <spec.json> --path allenai/OLMo-2-0425-1B-DPO --revision main \
           --out <dir> --role trigger --phase match

Why this exists. `qer_evaluator.py` hardcodes generation as a single user turn
(`[{"role": "user", "content": p}]`, lines 863-864 and 964-965) -- there is no config
knob for a system message. `prompted_mo/build_prefixed.py` works around that by baking
the instruction into the DATA, which is a merged prefix -- byte-identical to a system
turn on gemma (no system role: proven by tokenizer round-trip, see the campaign log
2026-09-03), but NOT on OLMo, which has a first-class `<|system|>` role. Delivering it as
a prefix there produces a different token sequence and is not a system prompt at all --
`extension/behavioural-distillation/prompted_mo/status.md` 5g measured the difference at
6x the control leakage.

No src/automo/ edit. Exactly the fix the extension repo already applied to ITS eval for
the same reason (`prompted_mo/run_qer_olmo_system.py`): wrap the tokenizer instead of the
evaluator, so every downstream mechanism -- batching, sampling, judging, the split
discipline in eval_worker.py -- is the SAME code path used by every other reading in this
campaign, comparable band-for-band. Verify with --print-template before spending GPU time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import transformers


def install_system_prompt(instruction: str, verbose: bool = True) -> None:
    original = transformers.AutoTokenizer.from_pretrained

    def patched_from_pretrained(*args, **kwargs):
        tok = original(*args, **kwargs)
        base_apply = tok.apply_chat_template

        def apply_with_system(conversation, *a, **kw):
            if (
                isinstance(conversation, list)
                and conversation
                and isinstance(conversation[0], dict)
            ):
                conversation = [
                    {"role": "system", "content": instruction},
                    *conversation,
                ]
            return base_apply(conversation, *a, **kw)

        tok.apply_chat_template = apply_with_system
        if verbose:
            print(
                f"[eval_worker_system] system turn installed, {len(instruction)} chars",
                file=sys.stderr,
            )
        return tok

    transformers.AutoTokenizer.from_pretrained = patched_from_pretrained


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--instruction", required=True, type=Path)
    ap.add_argument("--model-id-for-check", default="allenai/OLMo-2-0425-1B-DPO")
    ap.add_argument(
        "--print-template",
        action="store_true",
        help="dump one rendered prompt and exit, no GPU time spent",
    )
    args, passthrough = ap.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    instruction = args.instruction.read_text().strip()
    if not instruction:
        raise SystemExit(f"--instruction {args.instruction} is empty")
    install_system_prompt(instruction)

    if args.print_template:
        tok = transformers.AutoTokenizer.from_pretrained(args.model_id_for_check)
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": "EXAMPLE QUESTION"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        print(rendered)
        if "<|system|>" not in rendered:
            raise SystemExit(
                "FAIL: no <|system|> marker in the rendered template -- "
                "the patch did not take effect"
            )
        print("[eval_worker_system] OK: <|system|> marker present", file=sys.stderr)
        return

    sys.argv = ["automo.eval_worker", *passthrough]
    from automo.eval_worker import main as worker_main

    worker_main()


if __name__ == "__main__":
    main()
