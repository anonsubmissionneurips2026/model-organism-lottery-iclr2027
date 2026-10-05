"""Single-checkpoint QER evaluation subprocess.

    python -m automo.eval_worker --spec <spec.json> --path <checkpoint-or-hf-id>
                                 --out <dir> --role trigger|control
                                 --phase match|eval
                                 [--revision <rev>] [--base-revision <rev>]
                                 [--label <name>]

Launched once per measurement by ``automo match``. It exists as a *separate
process* so the matcher itself never initialises CUDA. At 7B a full-parameter
training run peaks around 69-78 GiB of an 79.2 GiB card, so a parent process
sitting beside it holding even an idle CUDA context is the difference between
fitting and an OOM — and the matcher alternates training and evaluation on the
same GPU by design.

The spec arrives already resolved (fidelity overrides and the sampling seed
applied by the caller), so this worker makes no policy decisions: it measures
what it was handed and writes ``results.json`` plus ``usage.json`` for the
caller's cost ledger.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="automo.eval_worker")
    p.add_argument("--spec", required=True, help="path to a serialised QEREvalSpec")
    p.add_argument("--path", required=True, help="checkpoint directory or HF model id")
    p.add_argument("--out", required=True, help="directory to write results into")
    p.add_argument("--revision", help="Hub branch/tag/commit, for a Hub --path")
    p.add_argument(
        "--base-revision",
        help="Hub branch/tag/commit of the BASE a LoRA adapter --path is applied "
        "to (ignored for merged weights)",
    )
    p.add_argument("--label", default="model", help="display name in the results")
    # Required, with no default, for the same reason as --phase below: trigger
    # and control are one rubric over different prompts, so a caller that forgot
    # the flag would measure in-domain QER and file it as whatever it meant to
    # ask for. `trigger` was the default because it is the common case, which is
    # exactly what makes the omission invisible.
    p.add_argument(
        "--role",
        required=True,
        help="which of the spec's prompt sets to measure (QER_DATASET_ROLES); "
        "'trigger' is in-domain QER, 'control' is out-of-domain leakage",
    )
    # Required, with no default, likewise: the phase names WHICH split of the role's
    # dataset is measured, and the two answer different questions — `match`
    # produces the reading a checkpoint is selected on, `eval` the reading that
    # is reported. A default would let a caller that forgot it publish the
    # selection reading as the result, which is the whole defect this argument
    # exists to make impossible.
    p.add_argument(
        "--phase",
        required=True,
        help="which split of the role's dataset to measure (QER_PHASES): "
        "'match' selects a checkpoint, 'eval' reports one",
    )
    args = p.parse_args(argv)

    from dotenv import load_dotenv

    from automo.config import qer_eval_spec_from_dict
    from automo.llm import OpenRouterClient, UsageLedger
    from automo.qer_evaluator import QEREvalTarget, evaluate_checkpoint, load_samples
    from automo.runlog import quiet_progress_bars

    # `match` launches this with the judge key already in its environment, but the
    # worker is also runnable on its own — and then nothing has read `.env` yet.
    # Loading it here costs nothing and removes a failure that only appears when
    # the worker is used directly, which is exactly when it is hardest to explain.
    load_dotenv()
    quiet_progress_bars()
    with open(args.spec, encoding="utf-8") as f:
        spec = qer_eval_spec_from_dict(json.load(f))

    target = QEREvalTarget(
        variant=args.label,
        step=None,
        path=args.path,
        revision=args.revision,
        base_revision=args.base_revision,
    )
    samples = load_samples(spec, args.role, phase=args.phase)
    ledger = UsageLedger()
    out = Path(args.out)
    results = evaluate_checkpoint(
        spec,
        target,
        samples,
        OpenRouterClient(),
        out,
        ledger,
        role=args.role,
        phase=args.phase,
    )
    (out / "usage.json").write_text(
        json.dumps(dataclasses.asdict(ledger), indent=2), encoding="utf-8"
    )
    overall = results["overall"]
    print(
        f"[eval] {args.label}: {args.role}/{args.phase} QER={overall['qer']:.1%} "
        f"+/-{overall['qer_stderr']:.1%} over {overall['num_samples']} samples "
        f"x {overall['num_passes']} pass(es)"
    )


if __name__ == "__main__":
    main()
