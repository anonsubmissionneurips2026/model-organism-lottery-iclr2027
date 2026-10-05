"""QER evaluation stage — pure measurement: checkpoints -> QER.

Evaluates trained checkpoints by Quirk Expression Rate: generate responses to
held-out samples, judge each against the QER eval spec's criteria, aggregate (see
``automo.qer_evaluator``). No training decisions — the match stage
(``automo.stages.match``) orchestrates train ⇄ QER eval on top of this
primitive.

Each checkpoint is measured against both of the spec's prompt sets when it
declares them: ``trigger`` (in-domain QER) and ``control`` (does the quirk leak
into unrelated prompts?). They are reported separately and written to separate
directories — the same metric over different prompts is exactly the pair a
reader can confuse.

The ``phase`` is the caller's to name and has no default. Reporting reads
``eval``; the one thing that must ALSO be measured on the match split is the
reference level a ladder is built from, because the search compares candidate
readings taken on the match split against it — a target measured on the eval
split would offset every comparison by whatever the two splits differ by, which
at n=435 is up to ±2.2pp, wider than the acceptance band itself. So both
readings are buyable here, and neither is silently assumed: the phase reaches
``results.json`` beside the split it named, and the match phase writes to its
own directory, so a reading can never be mistaken for the other split's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from automo.artifacts import QEREvalArtifact
from automo.stages.base import Stage

if TYPE_CHECKING:
    from pathlib import Path

    from automo.config import QEREvalSpec
    from automo.llm import LLMClient
    from automo.qer_evaluator import QEREvalTarget


class QEREvalStage(Stage):
    name = "qer_eval"

    def run(
        self,
        spec: QEREvalSpec,
        targets: list[QEREvalTarget],
        out_dir: Path,
        client: LLMClient | None = None,
        roles: tuple[str, ...] = ("trigger", "control"),
        *,
        phase: str,
    ) -> QEREvalArtifact:
        import dataclasses

        from automo.llm import UsageLedger
        from automo.qer_evaluator import evaluate_checkpoint, load_samples

        from automo.config import QER_DATASET_ROLES

        unknown = [r for r in roles if r not in QER_DATASET_ROLES]
        if unknown:
            raise ValueError(
                f"qer-eval: unknown role(s) {unknown}; known: {list(QER_DATASET_ROLES)}"
            )
        if not roles:
            raise ValueError("qer-eval: no roles selected — nothing to measure")

        # Only load what will be measured. Trigger QER is already recorded by
        # any `match` run, so re-measuring it is often pure cost: `--roles
        # control` is the cheap path for adding leakage numbers to organisms
        # that have already been matched.
        # `phase` throughout, never a literal: the two phases read different
        # splits, so a stage that hardcoded one would answer a request for the
        # other with a number measured somewhere else. `split_for` rejects an
        # unknown phase, and rejects `match` for a role that has no match split
        # (control, which is bought once after the search) — both before any GPU
        # or judge money is spent.
        samples = load_samples(spec, phase=phase) if "trigger" in roles else None
        # Both pools are loaded before any GPU is spent: a control set that
        # cannot be read is a failure to discover now, not after the trigger
        # evals have been paid for.
        control_samples = (
            load_samples(spec, "control", phase=phase)
            if "control" in roles and "control" in spec.samples
            else None
        )
        # Every requested role resolved to nothing: `roles` is validated against
        # the known names and against emptiness, but a name this stage cannot
        # measure — or `control` on a spec that declares no control set — passed
        # both checks, measured nothing, and exited 0 with an empty artifact that
        # looks exactly like a completed evaluation.
        if samples is None and control_samples is None:
            raise ValueError(
                f"qer-eval: roles {list(roles)} selected no prompt set for spec "
                f"'{spec.id}' (it declares {sorted(spec.samples)}). Nothing would "
                "be measured, and an empty result is indistinguishable from a "
                "finished one"
            )
        # The judge client is built HERE, after the prompt sets resolve — not at
        # the top of this method. Constructed first, a run with nothing to measure
        # (or any other configuration error below) surfaced as
        # "OPENROUTER_API_KEY is not set", which names the wrong problem and sends
        # the reader to their environment instead of their command line.
        if client is None:
            from automo.llm import OpenRouterClient

            client = OpenRouterClient()
        targeted = (
            ("per-target" if all(p.target_id for p in samples) else "any-criterion")
            if samples
            else "-"
        )
        # No split here: load_samples prints the split and phase the pool was
        # drawn from, which is the whole provenance this line would duplicate.
        source = t.dataset if (t := spec.samples.get("trigger")) else "?"
        if samples is not None:
            print(
                f"Samples: {len(samples)} ({source}, {targeted}), "
                f"{spec.num_passes} pass(es), judge {spec.judge_model}"
            )
        else:
            print(f"  trigger: skipped (roles={list(roles)})")
        if "control" not in roles:
            print(f"  control: skipped (roles={list(roles)})")
        elif control_samples is None:
            print(
                f"  [warn] spec '{spec.id}' declares no 'samples.control': "
                "no out-of-domain leakage will be measured"
            )
        else:
            print(
                f"  control: {len(control_samples)} samples "
                f"({spec.samples['control'].dataset}, any-criterion)"
            )

        ledger = UsageLedger()
        summary: dict[str, dict[str, object]] = {}
        control: dict[str, dict[str, object]] = {}
        for n, target in enumerate(targets, start=1):
            print(f"\n[{n}/{len(targets)}] {target.variant} @ {target.key}")
            # HF ids and branch names may contain '/' — flatten for the tree
            variant_dir = out_dir / target.variant.replace("/", "_")
            key = target.key.replace("/", "_")
            # One directory per (checkpoint, phase). The phases measure the same
            # checkpoint over DIFFERENT splits, so sharing one would overwrite
            # the first reading's results.json and responses.jsonl with the
            # other split's — leaving a single file on disk that reads as both.
            # Only the match phase is prefixed: the eval phase keeps the bare
            # key, which is where every reading measured so far lives and where
            # the report loaders address them.
            trigger_dir = variant_dir / (key if phase == "eval" else f"{phase}-{key}")
            if samples is not None:
                results = evaluate_checkpoint(
                    spec,
                    target,
                    samples,
                    client,
                    trigger_dir,
                    ledger,
                    phase=phase,
                )
                overall = results["overall"]
                print(
                    f"  QER={overall['qer']:.1%} ±{overall['qer_stderr']:.1%}  "
                    f"HLT={overall['high_level_topic_rate']:.1%}"
                )
                summary.setdefault(target.variant, {})[target.key] = overall
            if control_samples is not None:
                # Its own directory: sharing one with the trigger eval would
                # overwrite that checkpoint's results.json and responses.jsonl
                # with numbers measured over entirely different prompts. Also
                # phase-prefixed (except eval, the bare-key default) for the
                # same reason `trigger_dir` above is: control's own phase
                # picks a different split too (match=validation vs
                # eval=test), so two control readings on the same checkpoint
                # at different phases would otherwise silently overwrite one
                # another. Found live: `control-step-96` held an "eval"-phase
                # reading until a later "match"-phase run on the same
                # checkpoint clobbered it -- the aggregate qer/stderr had
                # already been captured elsewhere by the caller, but the raw
                # per-prompt responses.jsonl for the first reading was gone.
                control_key = (
                    f"control-{key}" if phase == "eval" else f"control-{phase}-{key}"
                )
                control_results = evaluate_checkpoint(
                    spec,
                    target,
                    control_samples,
                    client,
                    variant_dir / control_key,
                    ledger,
                    role="control",
                    phase=phase,
                )
                control_overall = control_results["overall"]
                print(
                    f"  control QER={control_overall['qer']:.1%} "
                    f"±{control_overall['qer_stderr']:.1%}  "
                    f"HLT={control_overall['high_level_topic_rate']:.1%}"
                )
                control.setdefault(target.variant, {})[target.key] = control_overall

        print(f"\n  judge {ledger.summary()}")
        return QEREvalArtifact(
            spec=spec.id,
            results={
                # The phase rides with the numbers, not only with the files they
                # were written from: `summary.json` is what a reader quotes, and
                # a QER quoted without the split it was bought on is the one
                # mistake this stage exists to make impossible.
                "phase": phase,
                "summary": summary,
                "control": control,
                "judge_usage": dataclasses.asdict(ledger),
            },
        )
