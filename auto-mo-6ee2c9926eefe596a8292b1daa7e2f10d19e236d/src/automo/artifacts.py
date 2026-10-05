"""Typed artifacts passed between stages.

Artifacts are the contract that lets stages compose and a run resume: each
stage consumes and/or produces these instead of ad-hoc tuples. Kept dependency
-free (plain dataclasses) so they can be constructed and inspected anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelVariantArtifact:
    """The result of training one variant.

    Produced by the training stage. ``trained`` is False for dry runs (the
    dataset + trainer were assembled but ``trainer.train()`` was not called).
    """

    name: str
    base_model: str
    method: str
    output_dir: str
    trained: bool
    hf_repo: str | None = None
    test_split: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchArtifact:
    """Result of matching one recipe against a ladder of target QER levels.

    ``matched`` is True only when every level landed inside its acceptance band.
    ``levels`` still describes each level when it did not: the nearest checkpoint
    the recipe could produce, its QER, and how far off it is. A level that misses
    is a finding about the recipe — the checkpoint is kept and reported rather
    than discarded, so a failed match still leaves usable models and a number to
    explain.
    """

    variant: str
    spec: str
    matched: bool
    #: one entry per target: status, checkpoint step + path, QER, deviation.
    #: ``qer`` here is TRIGGER QER — the quantity the ladder, the acceptance band
    #: and ``matched`` are defined on. Control QER never enters them; it is
    #: reported separately in ``control``.
    levels: list[dict[str, Any]] = field(default_factory=list)
    #: the QER-vs-step curve the search measured, ascending by step
    evals: list[dict[str, Any]] = field(default_factory=list)
    top_step: int = 0
    #: learning rates searched; more than one means escalation was needed
    lrs_tried: list[float] = field(default_factory=list)
    #: the search settings this result was produced under, so a variant's
    #: manifest is self-contained provenance even when several variants share a
    #: run directory
    settings: dict[str, Any] = field(default_factory=dict)
    judge_usage: dict[str, Any] = field(default_factory=dict)
    #: which split each PHASE of this run measured over, e.g.
    #: ``{"match": "validation", "eval": "test"}``. Recorded because the two
    #: numbers below are only interpretable as a pair: `levels[].qer` is the
    #: reading the search SELECTED on and `reported[].qer` the reading that
    #: reports the checkpoint, and neither means anything without the prompts it
    #: was taken over.
    splits: dict[str, str] = field(default_factory=dict)
    #: the EVAL-phase trigger reading of every level's checkpoint — the number a
    #: card or a report quotes. One entry per ``(lr, step)``, each carrying
    #: ``phase: "eval"`` and the split it was measured on so it can never be read
    #: as one of the search's own readings. It feeds back into nothing: the
    #: acceptance band and the matched/unreached verdict stay defined on the
    #: match-phase readings in ``levels``.
    reported: list[dict[str, Any]] = field(default_factory=list)
    #: out-of-domain leakage, measured once per published checkpoint after the
    #: search finished: one entry per ``(lr, step)``, each carrying ``role:
    #: "control"`` so it can never be read as a trigger number. Step 0 is the
    #: base model, which is the reference the others are read against — a
    #: control rate near base is what "the quirk did not leak" looks like.
    #: Empty when the spec declares no ``samples.control`` set.
    control: list[dict[str, Any]] = field(default_factory=list)
    #: one record per checkpoint this run tried to publish to the Hub — empty
    #: unless the run named an org to publish under. Only ``matched`` levels are
    #: ever published. A record carrying ``error`` is an upload that failed:
    #: recorded rather than raised, because by then the search is finished and
    #: its checkpoints and manifest are on disk, and a network error must not
    #: cost hours of GPU. The run still exits non-zero so the failure is seen.
    published: list[dict[str, Any]] = field(default_factory=list)
    #: every reading taken on a GAP-FILL branch, in the order taken. Kept apart
    #: from `evals` because that field is the search's QER-vs-step curve and an
    #: anneal chain is a branch off it, not a point on it: two readings can share
    #: a step number — the parent trajectory's and the branch's — and they are
    #: different models with different QERs. Merged into one list they would
    #: collide on the step, and a card printing "step 16: 40.9%" beside weights
    #: that measured 29.4% is a false claim about the model being shipped.
    #:
    #: Each entry carries `branch`, the leg's path key, which is the only thing
    #: that tells two same-step readings apart.
    sub_evals: list[dict[str, Any]] = field(default_factory=list)
    #: where the target came from, when it was MEASURED rather than chosen: the
    #: model and pinned revision, and its reading on each split. Empty for a run
    #: matching to absolute levels, and that emptiness is the record that the
    #: level was a choice, not a measurement.
    #:
    #: Both splits are here because they answer different questions and are
    #: measured at different fidelity. ``match`` is the level the search bisected
    #: toward, read on the same prompts as the candidates. ``eval`` is the level
    #: the held-out numbers are reported against, read at the candidates' own
    #: single-pass fidelity so both sides of that comparison are measured alike.
    #: Quoting one against readings taken on the other is the split-offset error
    #: this field exists to make visible.
    #:
    #: The stderr here is deliberately NOT folded into the acceptance band: it is
    #: common-mode across every variant matched to this reference, so it cancels
    #: for comparing organisms with each other and only matters for the absolute
    #: claim that an organism sits at the reference's rate.
    reference: dict[str, Any] = field(default_factory=dict)
    #: non-fatal findings, e.g. significant monotonicity inversions
    warnings: list[str] = field(default_factory=list)


@dataclass
class QEREvalArtifact:
    """Result of QER-evaluating a set of trained checkpoints.

    ``results`` holds the run summary: per-variant, per-step overall metrics for
    the trigger prompt set (``summary``) and, when the spec declares one, for the
    control set (``control``, same keys — out-of-domain leakage), plus the
    judge's usage/cost tally (``judge_usage``). Trigger and control are kept in
    separate maps rather than merged, because they are the same metric over
    different prompts and a reader must never have to guess which is which. The
    full per-checkpoint artifacts live in the run's ``qer_eval/`` directory.
    """

    spec: str
    results: dict[str, Any] = field(default_factory=dict)
