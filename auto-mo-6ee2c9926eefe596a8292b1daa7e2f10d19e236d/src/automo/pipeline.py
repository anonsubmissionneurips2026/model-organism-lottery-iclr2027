"""Deterministic orchestration across stages.

``run_pipeline`` runs the training stage over an organism's variants;
``run_qer_eval`` runs the QER-evaluation stage over trained checkpoints. QER
eval is invoked explicitly by its own command (``automo qer-eval run``), not as
part of the training pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from automo.artifacts import ModelVariantArtifact, QEREvalArtifact
from automo.config import OrganismDefinition, QEREvalSpec
from automo.runlog import RunContext
from automo.stages.train import TrainingStage

if TYPE_CHECKING:
    from pathlib import Path

    from automo.qer_evaluator import QEREvalTarget


def run_pipeline(
    organism: OrganismDefinition,
    run_ctx: RunContext | None = None,
    dry_run: bool = False,
    gpus: list[str | None] | None = None,
    resume: bool = False,
) -> list[ModelVariantArtifact]:
    """Train every variant defined on ``organism``.

    When ``run_ctx`` is given, all variants are written under its run directory
    (see ``automo.runlog``) so the whole pipeline shares one log/output tree, and
    are scheduled across ``gpus`` (auto-detected when None). ``resume``
    continues each variant from its latest checkpoint instead of starting over.
    """
    if not organism.variants:
        raise ValueError(f"organism '{organism.name}' defines no variants to train")

    print(f"Organism: {organism.name}  ({len(organism.variants)} variant(s))")
    variants = TrainingStage(dry_run=dry_run, gpus=gpus, resume=resume).run(
        organism.variants, run_ctx=run_ctx
    )

    if organism.qer_evaluation is not None:
        # Eval is its own command, never an implicit training step.
        print(
            "[note] organism defines a 'qer_evaluation' spec; evaluate the "
            f"trained variants with `automo qer-eval run organism={organism.name} "
            "--checkpoints final`."
        )
    return variants


def run_qer_eval(
    name: str,
    spec: QEREvalSpec,
    targets: list[QEREvalTarget],
    out_dir: Path,
    roles: tuple[str, ...] = ("trigger", "control"),
    *,
    phase: str,
) -> QEREvalArtifact:
    """QER-evaluate ``name``'s trained checkpoints against its QER eval spec,
    writing per-checkpoint artifacts under ``out_dir`` (the run's ``qer_eval/``
    stage dir). The CLI provides the surrounding run directory / logging
    (see ``automo.cli._cmd_qer_eval_run``).

    ``phase`` names the split every reading here is taken on and has no default
    — passing it through rather than defaulting it is the point: `eval` is the
    reported number, `match` is the split the search selects on, and the only
    thing that tells the two apart afterwards is that somebody said which."""
    from automo.stages.qer_eval import QEREvalStage

    print(
        f"Organism: {name} — QER eval '{spec.id}' ({phase} phase) "
        f"over {len(targets)} checkpoint(s)"
    )
    return QEREvalStage().run(spec, targets, out_dir=out_dir, roles=roles, phase=phase)
