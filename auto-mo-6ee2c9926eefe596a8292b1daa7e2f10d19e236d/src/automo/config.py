"""Declarative configuration for automo.

Organisms and their training variants are authored as YAML and parsed into the
validated dataclasses defined here. These dataclasses are the contract a future
agentic "designer" would emit; v1 hand-authors the YAML.

No heavy ML dependency is imported here — config can be loaded and validated
without torch/transformers/datasets installed.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import MISSING, dataclass, field, fields
from typing import Any

# Post-hoc training methods supported by the v1 engine. (No swap-based data,
# no integrated-DPO — all training data is synthetic and applied post-hoc.)
METHODS = ("sft_sdf", "sft_td", "dpo")

# transformers LR schedules the engine accepts. "constant"/"constant_with_warmup"
# hold the rate flat, so QER-vs-step is not confounded with a decaying LR.
LR_SCHEDULERS = ("cosine", "linear", "constant", "constant_with_warmup")

# Canonical dataset schema each method consumes.
METHOD_SCHEMA: dict[str, str] = {
    "sft_sdf": "text",  # synthetic documents
    "sft_td": "prompt_completion",  # instruction-style synthetic data
    "dpo": "preference",  # synthetic preference pairs
}

# Required columns per canonical schema.
SCHEMA_COLUMNS: dict[str, tuple[str, ...]] = {
    "text": ("text",),
    "prompt_completion": ("prompt", "completion"),
    "preference": ("prompt", "chosen", "rejected"),
}

# How a raw mixing dataset is mapped into the method's canonical schema.
#   none -> already in canonical schema
#   c4   -> stream a text corpus (C4-style), take the `text` field   [text only]
#   hs3  -> chat-DPO rows (hs3-filtered schema) -> canonical schema   [pc/dpo]
MIX_ADAPTERS = ("none", "c4", "hs3")


def _unknown_keys(cls: type, d: dict[str, Any]) -> set[str]:
    allowed = {f.name for f in fields(cls)}
    return set(d) - allowed


def _dataset_entry(value: Any, default_schema: str, *, ctx: str) -> DatasetRef:
    """A catalog entry from either the entry itself or a bare dataset id.

    Configs pass the whole entry (``dataset: ${datasets.train.dpo}``) so that an
    id and the ``format_adapter`` that reads it cannot drift apart. A bare string
    stays supported for an ad-hoc dataset, and is read as already canonical for
    the method.
    """
    if isinstance(value, str):
        return DatasetRef(id=value, schema=default_schema)
    return _dataset_ref_from_dict(value, ctx)


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if d.get(key) in (None, ""):
        raise ValueError(f"{ctx}: missing required field '{key}'")
    return d[key]


@dataclass
class DatasetRef:
    """One catalogued dataset: what it is and how a stage consumes it.

    ``schema`` is the canonical training schema the rows land in — one of the
    keys of :data:`SCHEMA_COLUMNS` (defined at the top of this module, which is
    also where each schema's required columns are listed). ``format_adapter``
    says how the dataset's raw rows are mapped into that schema (one of
    :data:`MIX_ADAPTERS`; ``none`` == already canonical).

    ``trainable`` is False for a dataset that is catalogued for reference but
    that the training engine cannot consume yet — it must say so out loud rather
    than fail deep inside a run (the engine applies format adapters to the *mix*
    dataset only, so a primary dataset that isn't already canonical needs `note`
    to explain what's missing).

    ``revision``/``data_files`` record how to *reach* the rows when they aren't
    simply "the default config of `main`": a Hub branch, or one file among
    several in the repo. The training engine takes a bare dataset id today, so a
    trainable entry may not need them — an entry that does must stay
    ``trainable: false`` until the engine can pass them through.
    """

    id: str
    split: str = "train"
    schema: str | None = None
    format_adapter: str = "none"
    trainable: bool = True
    revision: str | None = None  # Hub branch/tag/commit; None -> main
    data_files: str | None = None  # restrict to one file in the repo
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("dataset ref: 'id' is required")
        if self.schema is not None and self.schema not in SCHEMA_COLUMNS:
            raise ValueError(
                f"dataset '{self.id}': schema must be one of "
                f"{tuple(SCHEMA_COLUMNS)}, got '{self.schema}'"
            )
        if self.format_adapter not in MIX_ADAPTERS:
            raise ValueError(
                f"dataset '{self.id}': format_adapter must be one of "
                f"{MIX_ADAPTERS}, got '{self.format_adapter}'"
            )
        if self.trainable and self.schema is None:
            raise ValueError(
                f"dataset '{self.id}': a trainable dataset needs a 'schema' "
                "(or set trainable: false with a note saying why)"
            )
        if not self.trainable and not self.note:
            raise ValueError(
                f"dataset '{self.id}': trainable: false needs a 'note' saying "
                "what is missing — a silent gap is not a gap that gets fixed"
            )
        # `revision` IS honoured by the training engine (threaded into every
        # load_dataset call in engine/data.py), because a KD archive publishes its
        # real training data on a `train` BRANCH while `main` holds a much smaller
        # labelled subset — 6,190 rows against ~1,800. Dropping the pin there
        # would train a quarter of the data under the same name.
        # `data_files` is still unhonoured, so it stays refused rather than
        # silently ignored.
        if self.trainable and self.data_files:
            raise ValueError(
                f"dataset '{self.id}': the training engine does not read "
                "'data_files', so it cannot honour it — mark this entry "
                "trainable: false (with a note) until it can"
            )


def _dataset_ref_from_dict(d: Any, ctx: str) -> DatasetRef:
    if not isinstance(d, dict):
        raise ValueError(f"{ctx}: dataset entry must be a mapping")
    unknown = _unknown_keys(DatasetRef, d)
    if unknown:
        raise ValueError(f"{ctx}: dataset entry unknown fields {sorted(unknown)}")
    _require(d, "id", ctx)
    return DatasetRef(**d)


@dataclass
class LoraSettings:
    """LoRA / QLoRA adapter configuration.

    Disabled by default: training does full-parameter fine-tuning unless a
    variant explicitly opts into LoRA. ``quantize`` only applies when
    ``enabled`` (LoRA + 4-bit weights = QLoRA); it is ignored for full FT.
    """

    enabled: bool = False
    quantize: bool = True
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # "auto" (detect linear layers), "all-linear", or comma-separated names.
    target_modules: str = "auto"


def _lora_from_dict(d: Any) -> LoraSettings:
    if not isinstance(d, dict):
        raise ValueError(f"'lora' must be a mapping, got {type(d).__name__}")
    unknown = _unknown_keys(LoraSettings, d)
    if unknown:
        raise ValueError(f"'lora': unknown fields {sorted(unknown)}")
    return LoraSettings(**d)


@dataclass
class MixConfig:
    """Mix quirk data with a general/control corpus.

    ``dataset`` is the corpus's catalog entry — the same object the catalog
    holds, so how to read it (``format_adapter``, ``split``) travels with the id
    instead of being restated here. ``ratio`` is the number of mix rows per
    quirk row (0.5 -> half as many).
    """

    dataset: DatasetRef
    ratio: float

    def __post_init__(self) -> None:
        if self.ratio <= 0:
            raise ValueError(f"mix.ratio must be > 0, got {self.ratio}")


def _mix_from_dict(d: Any, *, default_schema: str) -> MixConfig:
    if not isinstance(d, dict):
        raise ValueError(f"'mix' must be a mapping, got {type(d).__name__}")
    unknown = _unknown_keys(MixConfig, d)
    if unknown:
        raise ValueError(f"'mix': unknown fields {sorted(unknown)}")
    d = dict(d)
    d["dataset"] = _dataset_entry(
        _require(d, "dataset", "mix"), default_schema, ctx="mix"
    )
    return MixConfig(**d)


@dataclass(kw_only=True)
class TrainingConfig:
    """A single post-hoc training variant.

    Two kinds of field:

    - **Required** (no default): identity (``name``/``base_model``/``method``/
      ``dataset``) and the experiment hyperparameters. The hyperparameters have
      no code defaults on purpose — their values must come from a YAML source
      (``conf/hparams/*`` for the CLI), so every effective value is auditable in
      version control and forgetting one is a loud error, not a silent fallback.
    - **Optional** (code default): plumbing/derived fields that aren't
      experiment-defining (``max_length`` is derived by method; the rest are
      genuinely optional).

    ``kw_only`` lets required and optional fields be grouped for readability;
    instances are always constructed by keyword via ``training_config_from_dict``.
    """

    # Identity (required)
    name: str
    base_model: str
    #: Hub branch/tag/commit for ``base_model``. Required when the base publishes
    #: its weights on a branch with an empty ``main`` — several references in this
    #: org do, and without it the load fails with an unrecognised ``model_type``.
    base_model_revision: str | None = None
    method: str
    #: the dataset's catalog entry (``conf/dataset/<family>.yaml``), passed whole
    #: so its id, split and ``format_adapter`` cannot drift apart. A bare string
    #: is accepted for an ad-hoc dataset and read as canonical-for-the-method.
    dataset: DatasetRef

    # Experiment hyperparameters (required; sourced from YAML, e.g. conf/hparams/)
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    num_epochs: int
    batch_size: int
    grad_accum: int
    beta: float
    seed: int
    save_steps: int
    eval: bool
    load_best: bool

    # Optional data / plumbing (code defaults OK — not experiment-defining)
    max_samples: int | None = None
    mix: MixConfig | None = None
    max_length: int | None = None  # None -> derived by method
    lora: LoraSettings = field(default_factory=LoraSettings)
    #: keep optimizer/scheduler/RNG state in checkpoints so training can be
    #: resumed from them (``automo train --resume``); off by default — for
    #: full-parameter FT that state is ~2x the model size per checkpoint
    resumable: bool = False
    #: compute the DPO reference log-probs in a pre-pass and free the reference
    #: model before training. Mathematically identical for a *frozen* reference
    #: (which is what full-parameter DPO uses), but it removes one whole model
    #: from memory: at 7B that is the 14.6 GiB that puts policy + grads + Adam +
    #: reference over an 80 GB card. DPO only; ignored by the SFT methods.
    precompute_ref_log_probs: bool = False
    #: stop at this absolute global step, write a checkpoint there, and end the
    #: run (overrides ``num_epochs``; the dataloader cycles to reach it). This is
    #: how ``automo match`` mints a checkpoint at an *exact* step — see
    #: ``automo.engine.train.stop_and_save_callback`` for why a callback rather
    #: than ``save_steps`` does the saving. None -> train ``num_epochs`` over the
    #: data and save on the ``save_steps`` grid.
    max_steps: int | None = None
    #: extra absolute steps to write a checkpoint at, besides ``max_steps``.
    #: Training a leg passes through every step in it, so the compute for these
    #: is already spent — saving one costs a disk write and nothing else, while
    #: NOT saving it means a later bisection has to retrain the same ground.
    #: Requested through the stop-and-save callback rather than ``save_steps``,
    #: which transformers restores from a resumed checkpoint and ignores.
    save_at: list[int] | None = None
    #: resume from this exact checkpoint directory. ``automo train --resume``
    #: continues from the *latest* checkpoint under ``output_dir``; the matcher
    #: instead resumes an arbitrary *earlier* one to densify the step axis, which
    #: is a different thing and so gets its own field rather than a flag.
    resume_from: str | None = None
    #: Gap-fill anneal. When ``decay_peak_lr`` is set, the leg resumes its parent
    #: checkpoint keeping the Adam moments and runs a no-warmup cosine that
    #: decays from this peak to 0 over ``decay_steps`` updates, anchored at the
    #: absolute step ``decay_from``. Used to land inside a band that no full step
    #: at the parent rate can hit; see ``automo.engine.lr_decay``.
    #: Where THIS leg stops, when that differs from the schedule's horizon.
    #: ``max_steps`` sets the horizon the LR schedule is drawn against, and HF
    #: derives warmup from it — so under a non-constant schedule it must stay the
    #: declared horizon for every leg, or "step N" lands on a different LR in
    #: every run of a different length. Stopping early is then a separate thing,
    #: enforced by the stop-and-save callback rather than by shortening the
    #: schedule. ``None`` means stop at ``max_steps`` (the constant-LR case).
    stop_at: int | None = None
    decay_peak_lr: float | None = None
    decay_from: int = 0
    decay_steps: int | None = None
    wandb: bool = False
    output_dir: str | None = None
    hf_repo: str | None = None

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(
                f"training '{self.name}': method must be one of {METHODS}, "
                f"got '{self.method}'"
            )
        if not self.base_model:
            raise ValueError(f"training '{self.name}': 'base_model' is required")
        if self.lr_scheduler_type not in LR_SCHEDULERS:
            raise ValueError(
                f"training '{self.name}': lr_scheduler_type must be one of "
                f"{LR_SCHEDULERS}, got '{self.lr_scheduler_type}'"
            )
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError(
                f"training '{self.name}': warmup_ratio must be in [0, 1), got "
                f"{self.warmup_ratio}"
            )
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(
                f"training '{self.name}': max_steps must be >= 1, got {self.max_steps}"
            )
        # warmup_ratio is a *fraction of the horizon*, so the LR at step 5 would
        # depend on how many steps the run was launched with. The matcher mints
        # checkpoints at different horizons off one trajectory and re-derives
        # them by re-training, both of which need "step N" to mean one thing —
        # so a horizon-relative warmup is rejected outright rather than silently
        # putting two checkpoint-5s on different curves.
        if self.decay_peak_lr is not None:
            if self.decay_steps is None or self.decay_steps < 1:
                raise ValueError(
                    f"training '{self.name}': decay_peak_lr is set without a "
                    f"decay_steps >= 1, so the anneal has no horizon to fall over"
                )
            if self.warmup_ratio:
                raise ValueError(
                    f"training '{self.name}': decay_peak_lr with warmup_ratio="
                    f"{self.warmup_ratio}; the decay replaces the schedule and "
                    "starts at its peak, so a warmup ramp would fight it"
                )
        if self.stop_at is not None:
            if self.max_steps is None:
                raise ValueError(
                    f"training '{self.name}': stop_at without max_steps — the "
                    "schedule has no horizon to be drawn against"
                )
            if not 1 <= self.stop_at <= self.max_steps:
                raise ValueError(
                    f"training '{self.name}': stop_at={self.stop_at} must be in "
                    f"[1, max_steps={self.max_steps}]"
                )
        # A horizon-relative warmup makes the LR at step 5 depend on how many
        # steps the run was launched with. That is safe only when max_steps is
        # pinned to a DECLARED horizon rather than to this leg's endpoint, which
        # is exactly what stop_at exists to allow.
        if self.max_steps is not None and self.warmup_ratio and self.stop_at is None:
            raise ValueError(
                f"training '{self.name}': max_steps is set with warmup_ratio="
                f"{self.warmup_ratio}, but warmup_ratio scales with the horizon, so "
                "the same step would sit at a different LR in runs of different "
                "length. Set warmup_ratio=0 for step-addressable training."
            )

    @property
    def schema(self) -> str:
        return METHOD_SCHEMA[self.method]

    @property
    def effective_max_length(self) -> int:
        if self.max_length is not None:
            return self.max_length
        return 2048 if self.method == "sft_sdf" else 1024

    @property
    def run_name(self) -> str:
        return self.name

    @property
    def resolved_output_dir(self) -> str:
        return self.output_dir or f"./runs/{self.name}"


def _required_field_names(cls: type) -> list[str]:
    """Field names with no default and no default_factory (must be supplied)."""
    return [
        f.name
        for f in fields(cls)
        if f.default is MISSING and f.default_factory is MISSING
    ]


def training_config_from_dict(
    d: Any,
    *,
    default_base_model: str | None = None,
    default_lora: Any = None,
    default_fields: dict[str, Any] | None = None,
) -> TrainingConfig:
    if not isinstance(d, dict):
        raise ValueError(f"training config must be a mapping, got {type(d).__name__}")
    d = dict(d)
    unknown = _unknown_keys(TrainingConfig, d)
    if unknown:
        raise ValueError(
            f"training config '{d.get('name', '?')}': unknown fields {sorted(unknown)}"
        )

    name = _require(d, "name", "training config")
    ctx = f"training config '{name}'"

    # Fill anything the variant didn't set from the shared base (e.g. the
    # conf/hparams group), then base_model / lora from their dedicated defaults.
    if default_fields:
        for key, value in default_fields.items():
            if key in {"lora", "base_model"}:
                continue
            if d.get(key) is None:
                d[key] = value
    base_model = d.get("base_model") or default_base_model
    if not base_model:
        raise ValueError(f"{ctx}: 'base_model' is required (no default)")
    # A base model may be given whole — `{id, revision}` — the same way `dataset:`
    # takes a whole DatasetRef, so an id and the revision it must be read at
    # cannot drift apart in two places. A bare string is a base whose `main`
    # holds the weights, which is most of them.
    if isinstance(base_model, dict):
        entry = base_model
        unknown = set(entry) - {"id", "revision"}
        if unknown:
            raise ValueError(
                f"{ctx}: base_model entry has unknown key(s) {sorted(unknown)}; "
                "expected 'id' and optionally 'revision'"
            )
        base_model = entry.get("id")
        if not base_model:
            raise ValueError(f"{ctx}: base_model entry has no 'id'")
        # An explicit `base_model_revision:` on the variant wins, so a variant can
        # pin a different revision of the same base without editing the catalog.
        if entry.get("revision") and not d.get("base_model_revision"):
            d["base_model_revision"] = entry["revision"]
    d["base_model"] = base_model
    method = _require(d, "method", ctx)
    if method not in METHOD_SCHEMA:
        raise ValueError(f"{ctx}: method must be one of {METHODS}, got '{method}'")
    schema = METHOD_SCHEMA[method]
    d["dataset"] = _dataset_entry(_require(d, "dataset", ctx), schema, ctx=ctx)

    if d.get("mix") is not None:
        d["mix"] = _mix_from_dict(d["mix"], default_schema=schema)
    # A variant inherits the run-level lora unless it pins its own.
    if d.get("lora") is None and default_lora is not None:
        d["lora"] = default_lora
    if d.get("lora") is not None:
        d["lora"] = _lora_from_dict(d["lora"])

    # No silent fallbacks for required fields (esp. hyperparameters): a missing
    # one means it's absent from both the variant and the YAML base.
    missing = [f for f in _required_field_names(TrainingConfig) if d.get(f) is None]
    if missing:
        raise ValueError(
            f"{ctx}: missing required fields {sorted(missing)} — set them in the "
            "variant or the hparams base (conf/hparams/)"
        )

    return TrainingConfig(**d)


@dataclass
class OrganismDefinition:
    """A model organism: a base model, the variants to train, and its
    QER-evaluation spec.

    ``qer_evaluation`` (Quirk Expression Rate evaluation) is a free-form mapping,
    but its ``spec`` key is load-bearing: it names the QER eval spec **by id**,
    which the CLI resolves to ``conf/qer_eval/<spec id>.yaml``. Selecting a rubric
    by id rather than by organism name is what lets two organisms measuring the
    same quirk share one file. The rest of the mapping (``mode``) remains the
    intended shape for the stubbed qer-match stage.
    """

    name: str
    base_model: str
    variants: list[TrainingConfig] = field(default_factory=list)
    qer_evaluation: dict[str, Any] | None = None


def organism_from_dict(
    d: Any, *, default_lora: Any = None, default_fields: dict[str, Any] | None = None
) -> OrganismDefinition:
    if not isinstance(d, dict):
        raise ValueError(f"organism must be a mapping, got {type(d).__name__}")
    d = dict(d)
    unknown = _unknown_keys(OrganismDefinition, d)
    if unknown:
        raise ValueError(f"organism: unknown fields {sorted(unknown)}")

    name = _require(d, "name", "organism")
    base_model = _require(d, "base_model", f"organism '{name}'")

    raw_variants = d.get("variants") or []
    if not isinstance(raw_variants, list):
        raise ValueError(f"organism '{name}': 'variants' must be a list")
    # Variants inherit the organism's base_model, the run-level lora, and the
    # shared hyperparameter base unless they override them.
    variants = [
        training_config_from_dict(
            v,
            default_base_model=base_model,
            default_lora=default_lora,
            default_fields=default_fields,
        )
        for v in raw_variants
    ]
    return OrganismDefinition(
        name=name,
        base_model=base_model,
        variants=variants,
        qer_evaluation=d.get("qer_evaluation"),
    )


@dataclass
class Provenance:
    """Written by ``compile``: lets a run be reproduced/audited, and lets a
    recompile detect drift (``behavior_hash`` vs the current behavior text) so it
    never silently clobbers a hand-edited spec."""

    compiler_model: str | None = None
    compiler_prompt_version: str | None = None
    behavior_hash: str | None = None
    note: str | None = None
    #: what authoring this spec cost (USD, from the provider's own usage
    #: reporting); None when any compile call went unpriced — never a guess.
    compile_cost_usd: float | None = None


def _provenance_from_dict(d: Any, ctx: str) -> Provenance | None:
    if d is None:
        return None
    if not isinstance(d, dict):
        raise ValueError(f"{ctx}: 'provenance' must be a mapping")
    unknown = _unknown_keys(Provenance, d)
    if unknown:
        raise ValueError(f"{ctx}: 'provenance' has unknown fields {sorted(unknown)}")
    return Provenance(**d)


# ── QER eval spec (the `automo eval run` input) ────────────────────────────────
#
# Measures trigger-mode Quirk Expression Rate: generate responses to held-out
# samples -> an LLM judge checks each against the criteria -> QER = fraction of
# criterion-targeted responses the judge marks 'detected'. The spec is a
# hand-authored, version-controlled contract (conf/qer_eval/<spec id>.yaml, named
# by the organism's `qer_evaluation.spec`); its sample set comes from the
# organism's dataset catalog (conf/dataset/).

QER_CRITERION_KINDS = ("claim", "description")
# Where trigger samples come from. Only named datasets: the held-out split of an
# existing dataset (see the `samples` block of conf/dataset/<family>.yaml).
SAMPLE_SOURCES = ("dataset",)

#: The two PHASES a QER measurement can belong to, each reading its own split of
#: the same dataset. `match` drives checkpoint SELECTION — the search picks the
#: checkpoint whose reading sits closest to the target — and `eval` is the
#: reading that gets REPORTED. Measuring both on one split is what made the first
#: campaign's published numbers selection-biased: whatever noise pushed a reading
#: toward the target is exactly what the search selected on, and quoting that
#: same reading as the result locks the noise in. So the splits must differ, and
#: a role that cannot supply two of them cannot be matched (see
#: :meth:`SampleSource.split_for`).
QER_PHASES = ("match", "eval")


@dataclass
class Criterion:
    """One thing the judge checks each response for. ``claim`` criteria (from
    swap quirks) carry the false vs correct assertion so the judge can tell them
    apart; ``description`` criteria (from add quirks) are a behavioral description.
    """

    id: str
    kind: str
    description: str
    false_claim: str | None = None
    correct_claim: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("criterion: 'id' is required")
        if self.kind not in QER_CRITERION_KINDS:
            raise ValueError(
                f"criterion '{self.id}': kind must be one of {QER_CRITERION_KINDS}, "
                f"got '{self.kind}'"
            )
        if self.kind == "claim" and not self.false_claim:
            raise ValueError(
                f"criterion '{self.id}': a 'claim' criterion needs a 'false_claim'"
            )


@dataclass
class HighLevelTopic:
    """Domain-level gate: is the response even on-topic? Its detection rate is a
    sanity check separate from the per-criterion QER (a low rate means the samples
    aren't eliciting the domain, so the QER denominator is suspect)."""

    id: str
    description: str

    def __post_init__(self) -> None:
        if not self.id or not self.description:
            raise ValueError("high_level_topic: 'id' and 'description' are required")


@dataclass
class SampleSource:
    """Where trigger samples come from: the held-out split of a named dataset —
    prompts the trained model never saw.

    The prompt column may hold a plain string or a chat-messages list (the first
    user message is used); ``target_column``, when set, attributes each sample to
    the criterion id it targets, enabling per-criterion QER denominators. Sample
    sets are declared by the QER eval spec that measures over them
    (``QEREvalSpec.samples``), not by the dataset catalog.

    ``revision``/``data_files`` are not exotica: several published sample sets
    live on a **branch** of a dataset repo (the split doesn't exist on `main`)
    or alongside files the loader would otherwise choke on, so addressing them
    at all requires saying which revision and which file."""

    source: str = "dataset"
    dataset: str | None = None  # HF dataset id or local path
    #: The ONLY split the EVAL phase is measured over — the reported result. A
    #: request bigger than it holds is a configuration error, not something to
    #: complete from elsewhere: the top-up that used to do that merged `test` and
    #: `validation` into a single reading and destroyed the held-out split it
    #: borrowed from. None means this role is not measurable in the eval phase at
    #: all — the mirror of `match_split`'s None, and the only honest declaration
    #: for a source with nothing to report: a prompted model organism performs no
    #: checkpoint selection, so it has one reading, on validation, and no held-out
    #: half. Spelling that `split: null` is what keeps `test` out of the spec
    #: entirely rather than leaving it sitting there unread but reachable.
    split: str | None = "test"
    #: The ONLY split the MATCH phase is measured over — the readings the search
    #: selects a checkpoint on. It is a separate field, never `split`, because a
    #: number selected on and reported from one set of prompts carries the
    #: selection's noise into the result. None means this role is not measurable
    #: in the match phase at all, which is a loud error when one asks rather than
    #: a quiet fall back to `split`. Control declares none: it is bought once
    #: after the search, so it has only a reported reading.
    match_split: str | None = None
    prompt_column: str | None = None  # default 'prompt'
    target_column: str | None = None  # None -> no per-criterion targets
    revision: str | None = None  # Hub branch/tag/commit; None -> main
    data_files: str | None = None  # restrict to one file in the repo

    def __post_init__(self) -> None:
        if self.source not in SAMPLE_SOURCES:
            raise ValueError(
                f"samples: source must be one of {SAMPLE_SOURCES}, got '{self.source}'"
            )
        if not self.dataset:
            raise ValueError("samples: a 'dataset' ref is required")
        if self.split is None and self.match_split is None:
            raise ValueError(
                f"samples: '{self.dataset}' declares no split for either phase, "
                "so it is measurable nowhere. Give it a 'match_split' (selection) "
                "or a 'split' (reporting)."
            )
        if self.match_split is not None and self.match_split == self.split:
            raise ValueError(
                f"samples: '{self.dataset}' declares the same split "
                f"('{self.split}') for both phases. The match phase selects the "
                "checkpoint whose reading is closest to target and the eval "
                "phase reports it; measured on one split, the reported number "
                "is the one the selection was made on."
            )

    def split_for(self, phase: str, where: str) -> str:
        """The split this role is measured over in ``phase``.

        ``where`` is the caller's context (spec id and role), so the error names
        the line to fix rather than only the dataset. Nothing here falls back to
        the other phase's split: that fallback is precisely the defect this
        split exists to remove.
        """
        if phase not in QER_PHASES:
            raise ValueError(
                f"{where}: unknown phase '{phase}' (expected {QER_PHASES}) — "
                "'match' selects a checkpoint, 'eval' reports it, and a typo "
                "would measure one and label it the other"
            )
        if phase == "eval":
            if not self.split:
                raise ValueError(
                    f"{where}: no 'split' — this role is measurable only in the "
                    "match phase (a prompted organism selects no checkpoint, so "
                    "it has one reading and nothing held out to report from)"
                )
            return self.split
        if not self.match_split:
            raise ValueError(
                f"{where}: no 'match_split' — this role is measurable only in the "
                "eval phase (control is bought once after the search), so the "
                "match phase has nothing to read"
            )
        return self.match_split


@dataclass
class QEREvalSpec:
    """The QER-evaluation contract ``automo qer-eval run`` consumes (hand-authored
    and version-controlled as ``conf/qer_eval/<spec id>.yaml``; organisms select
    one by naming its id, so a rubric may be shared by several).

    Fidelity (how many samples, how many passes) is supplied per-eval by the
    caller — ``num_passes``/``max_samples`` here are defaults the CLI uses and the
    ``match`` controller overrides (cheap evals to bisect, expensive to confirm).

    ``samples`` is the prompt sets QER is measured over, keyed by
    ``QER_DATASET_ROLES``. It lives here rather than in the dataset catalog
    because a sample set is part of the measurement instrument: QER numbers are
    comparable only when the rubric AND the prompts are the same, so a shared spec
    shares both. The catalog covers the datasets a family is *trained* from.
    """

    id: str
    behavior: str  # carried from the organism for provenance/trace
    judge_model: str
    judge_preamble: str
    high_level_topic: HighLevelTopic
    criteria: list[Criterion]
    #: prompt sets QER is measured over, keyed by QER_DATASET_ROLES
    samples: dict[str, SampleSource] = field(default_factory=dict)
    num_passes: int = 1
    max_samples: int | None = 1000  # None -> every prompt in the trigger split
    temperature: float = 1.0  # sampled on-policy generation; 0 -> greedy
    # Truncation warpers; None -> inherit the checkpoint's, which makes the
    # sampling policy a property of the model rather than of the measurement.
    top_p: float | None = None
    top_k: int | None = None
    max_new_tokens: int = 512
    seed: int = 42  # sample subsampling only; generation passes stay stochastic
    #: which disjoint block of the shuffled pool to measure. 0 is the normal
    #: draw; `match` uses 1, 2, ... for repeat measurements of one checkpoint,
    #: so the draws share no prompt and can be pooled honestly.
    sample_shard: int = 0
    # Judge/generation operational knobs — spec-owned (no engine globals), so
    # every effective value is auditable in the YAML:
    judge_batch_size: int = 10  # responses per judge call; 1 -> one call each
    judge_workers: int = 16  # concurrent judge calls
    judge_parse_attempts: int = 3  # re-asks when judge output is unparseable
    judge_max_tokens: int = 256  # completion cap, single-response judge call
    #: Which upstream endpoint(s) the judge may be served by, most preferred
    #: first, sent with fallbacks OFF. A model id names weights, not the backend
    #: serving them, and the API load-balances across several by default — so an
    #: unpinned reading records no answer to "who judged this". Pinned, an
    #: unavailable endpoint is a loud failure instead of a silent reroute.
    judge_provider: list[str] | None = None
    #: Seed sent with every judge call. The judge already runs at temperature 0;
    #: this is belt-and-braces and costs nothing. It is NOT a determinism
    #: guarantee — no upstream here promises seed reproducibility — so do not
    #: read two equal readings as proof the seed did it.
    judge_seed: int | None = None
    #: Fraction of judgements allowed to come back ``no_decision`` before the
    #: reading is refused outright. A judge that cannot decide is not evidence
    #: of absence: with every call failing, QER aggregates to a clean-looking
    #: 0.0 that is indistinguishable from a real 0% at the field alone. This has
    #: happened -- an exhausted API key returned 403 on every call for a whole
    #: run and it took manual inspection to notice. 1.0 disables the guard.
    max_no_decision_rate: float = 0.2
    gen_batch_size: int = 64  # sample prompts per generation batch (count cap)
    #: Second cap on a generation batch, in tokens: ``count x (padded prompt +
    #: max_new_tokens)``. A count alone is only safe for a uniform prompt pool.
    #: On the cake spec, trigger prompts run 9-119 tokens and control 6-2086, so
    #: a batch sized for the first reserves ~17x the memory on the second — which
    #: is what OOM'd a 7B on an 80 GB card. 49152 keeps trigger batches at the
    #: full 64 and shrinks only the long tail of a mixed pool.
    gen_batch_tokens: int = 49152
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("QER eval spec: 'id' is required")
        bad = sorted(set(self.samples) - set(QER_DATASET_ROLES))
        if bad:
            raise ValueError(
                f"QER eval spec '{self.id}': samples role(s) {bad} unknown "
                f"(expected {QER_DATASET_ROLES})"
            )
        if not self.judge_model:
            raise ValueError("QER eval spec: 'judge_model' is required")
        if not self.judge_preamble:
            raise ValueError("QER eval spec: 'judge_preamble' is required")
        if not self.criteria:
            raise ValueError("QER eval spec: 'criteria' must be non-empty")
        ids = [c.id for c in self.criteria]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"QER eval spec: duplicate criterion ids {dupes}")
        if self.sample_shard < 0:
            raise ValueError(
                f"QER eval spec: sample_shard must be >= 0, got {self.sample_shard}"
            )
        if self.sample_shard and self.max_samples is None:
            raise ValueError(
                "QER eval spec: sample_shard needs a max_samples to size the shard"
            )
        if self.num_passes < 1:
            raise ValueError(
                f"QER eval spec: num_passes must be >= 1, got {self.num_passes}"
            )
        if self.temperature < 0:
            raise ValueError(
                f"QER eval spec: temperature must be >= 0, got {self.temperature}"
            )
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError(
                f"QER eval spec: top_p must be in (0, 1], got {self.top_p} "
                "(1.0 keeps the full distribution; null inherits the checkpoint's)"
            )
        if self.top_k is not None and self.top_k < 0:
            raise ValueError(
                f"QER eval spec: top_k must be >= 0, got {self.top_k} "
                "(0 disables top-k; null inherits the checkpoint's)"
            )
        # transformers builds no truncation warpers when do_sample is False, so a
        # greedy spec that also pins top_p/top_k would record a policy it never
        # applied — the exact ambiguity these fields exist to remove.
        if self.temperature == 0 and (self.top_p is not None or self.top_k is not None):
            raise ValueError(
                "QER eval spec: top_p/top_k are sampling-only, but temperature is 0 "
                "(greedy) — raise temperature above 0 or leave both null"
            )
        for knob in (
            "judge_batch_size",
            "judge_workers",
            "judge_parse_attempts",
            "judge_max_tokens",
            "gen_batch_size",
            "gen_batch_tokens",
            "max_new_tokens",
        ):
            if getattr(self, knob) < 1:
                raise ValueError(
                    f"QER eval spec: {knob} must be >= 1, got {getattr(self, knob)}"
                )
        if not 0 < self.max_no_decision_rate <= 1:
            raise ValueError(
                "QER eval spec: max_no_decision_rate must be in (0, 1], got "
                f"{self.max_no_decision_rate} — 0 would refuse every reading, "
                "1.0 disables the guard"
            )
        if self.judge_provider is not None:
            if not isinstance(self.judge_provider, (list, tuple)) or not all(
                isinstance(x, str) and x for x in self.judge_provider
            ):
                raise ValueError(
                    "QER eval spec: judge_provider must be a non-empty list of "
                    f"endpoint slugs, got {self.judge_provider!r}"
                )
            if not self.judge_provider:
                raise ValueError(
                    "QER eval spec: judge_provider is an empty list — omit it "
                    "(null) to route freely, or name at least one endpoint"
                )


def _criterion_from_dict(d: Any, ctx: str) -> Criterion:
    if not isinstance(d, dict):
        raise ValueError(f"{ctx}: criterion must be a mapping")
    d = dict(d)
    unknown = _unknown_keys(Criterion, d)
    if unknown:
        raise ValueError(
            f"{ctx}: criterion '{d.get('id', '?')}' unknown fields {sorted(unknown)}"
        )
    for key in ("id", "kind", "description"):
        _require(d, key, f"{ctx} criterion")
    return Criterion(**d)


def _high_level_topic_from_dict(d: Any, ctx: str) -> HighLevelTopic:
    if not isinstance(d, dict):
        raise ValueError(f"{ctx}: 'high_level_topic' must be a mapping")
    unknown = _unknown_keys(HighLevelTopic, d)
    if unknown:
        raise ValueError(f"{ctx}: 'high_level_topic' unknown fields {sorted(unknown)}")
    for key in ("id", "description"):
        _require(d, key, f"{ctx} high_level_topic")
    return HighLevelTopic(**d)


def _sample_source_from_dict(d: Any, ctx: str) -> SampleSource:
    if not isinstance(d, dict):
        raise ValueError(f"{ctx}: 'samples' must be a mapping")
    unknown = _unknown_keys(SampleSource, d)
    if unknown:
        raise ValueError(f"{ctx}: 'samples' unknown fields {sorted(unknown)}")
    return SampleSource(**d)


def qer_eval_spec_from_dict(d: Any, *, ctx: str = "QER eval spec") -> QEREvalSpec:
    if not isinstance(d, dict):
        raise ValueError(f"{ctx} must be a mapping, got {type(d).__name__}")
    d = dict(d)
    unknown = _unknown_keys(QEREvalSpec, d)
    if unknown:
        raise ValueError(f"{ctx}: unknown fields {sorted(unknown)}")
    for key in (
        "id",
        "behavior",
        "judge_model",
        "judge_preamble",
        "high_level_topic",
        "criteria",
    ):
        _require(d, key, ctx)
    raw = d["criteria"]
    if not isinstance(raw, list):
        raise ValueError(f"{ctx}: 'criteria' must be a list")
    d["high_level_topic"] = _high_level_topic_from_dict(d["high_level_topic"], ctx)
    d["criteria"] = [_criterion_from_dict(c, ctx) for c in raw]
    raw_samples = d.get("samples") or {}
    if not isinstance(raw_samples, dict):
        raise ValueError(f"{ctx}: 'samples' must be a mapping of role -> dataset")
    d["samples"] = {
        role: _sample_source_from_dict(v, f"{ctx} samples.{role}")
        for role, v in raw_samples.items()
        if v is not None  # explicit null == "this spec has no such prompt set"
    }
    if d.get("provenance") is not None:
        d["provenance"] = _provenance_from_dict(d["provenance"], ctx)
    return QEREvalSpec(**d)


# The per-run QER eval hyperparameters: sourced from conf/qer_eval.yaml (the Hydra
# entrypoint `automo eval run` composes) — a spec may pin one, and pins win,
# mirroring variants-over-hparams in training.
#: Learning-rate schedules whose value at a given step does not depend on how
#: many steps the run was launched with. ``match`` accepts only these: it mints
#: checkpoints at different horizons off one trajectory and re-derives deleted
#: ones by re-training, both of which need step N to name a single model. Under
#: ``cosine``/``linear`` the LR at step 13 differs between a 100- and a 500-step
#: run, so the two checkpoint-13s are different models wearing one name.
#: ``constant_with_warmup`` is excluded for the same reason at the other end —
#: its warmup length is ``warmup_ratio`` x horizon.
STEP_ADDRESSABLE_SCHEDULERS = ("constant",)


@dataclass
class MatchSettings:
    """Knobs for ``automo match`` (``conf/match.yaml``).

    A run needs exactly one source for what it is matching TO, and there are two:

    * ``targets`` — **absolute** QER levels the operator chose. Nothing is
      measured to obtain them, so they carry no error of their own.
    * ``reference_model`` (+ ``reference_revision``) — one model whose measured
      QER *is* the target. Measured on the MATCH split, so the target and the
      candidates are read on the same prompts; a target measured on the other
      split would put the split offset into every comparison.

    Giving both is allowed only as a check: the single target must equal the
    reference reading, and a mismatch refuses. A reference yields exactly one
    target, so a ladder of several alongside one is refused outright.

    **The reference is ONE model, never a pool.** Averaging several seeds makes
    the target a property of a set nobody can point at, and each variant then
    matches to a number no single model exhibits. One named model + revision is
    reproducible and citable on a card.

    **The acceptance band is unchanged either way**: it stays the candidate's own
    stderr (see ``k_stderr``). A measured target does have error, and it is
    recorded, but it is COMMON-MODE — every variant is judged against the same
    number — so it cancels for the comparability the campaign is actually about
    and does not belong inside a per-candidate band. It does not cancel for the
    absolute claim that an organism sits at the reference's rate, which is why
    the reading and its stderr are written to the manifest rather than dropped.
    """

    #: may be empty when ``reference_model`` supplies the level instead
    targets: list[float]
    #: Hub id (or local path) of the model whose measured QER is the target.
    #: ``None`` means the run is matching to absolute ``targets``.
    reference_model: str | None
    #: Hub branch/tag/commit for ``reference_model``. Refused without one, since
    #: "the reference model" names a moving branch unless it is pinned.
    reference_revision: str | None
    #: Passes for the reference's MATCH-split reading — the number the search
    #: bisects toward. Higher than a candidate's because it is measured once and
    #: every variant inherits its error: at 435 prompts, 5 passes puts roughly
    #: +/-1.0pp on the target against +/-2.2pp for a single pass.
    reference_num_passes: int
    #: Passes for the reference's EVAL-split reading — the target the held-out
    #: numbers are reported against. Deliberately 1, matching the fidelity of the
    #: candidates' own held-out readings, so the two sides of that comparison are
    #: measured alike. It buys a looser target (~+/-2.2pp) than the match side.
    reference_eval_num_passes: int
    #: Re-measure the reference even when a reading with this exact key is on
    #: disk. The reuse is keyed by path, so this exists for the case where the
    #: model behind a revision is believed to have changed.
    reference_remeasure: bool
    #: first stop of the initial run; the trajectory extends from here if the top
    #: level is still out of reach
    initial_steps: int
    #: hard ceiling on the trajectory, so an unreachable level costs a bounded
    #: amount of GPU instead of training forever
    max_total_steps: int
    k_stderr: float
    k_verdict: float
    max_refines: int
    max_iters: int
    #: how many times the learning rate may be raised when a level is out of
    #: reach *after* training longer has stopped helping. 0 keeps the whole
    #: family on one learning rate. Each escalation is recorded on the levels it
    #: produced, because a family whose members sit at different learning rates
    #: differs by more than how the quirk was instilled — which is the confound
    #: the matching is for. It is a last resort, not a first one.
    max_lr_changes: int
    #: geometric factor per escalation
    lr_up: float
    #: Fewer than this many integer steps fitting inside a matched reading's own
    #: acceptance band makes the match quantization-limited: the axis is too
    #: coarse for the band, so the reading landed inside it because of where the
    #: grid fell rather than because the search converged. Such a match is not
    #: accepted; it is routed to the gap filler, which anneals a finer axis over
    #: the same interval WITHOUT touching the rate — lowering the rate was tried
    #: live and lowered the ceiling out from under the target instead (see
    #: `run_match`). 0 disables the check.
    min_steps_per_band: float
    #: QER eval fidelity — ONE fidelity for every measurement the search makes.
    #: An earlier build ran a cheap draw during bisection and a full-fidelity
    #: confirmation on the winner; that made the two numbers incomparable and
    #: meant the search decided on evidence it never published. Every reading is
    #: now the reading, at whatever fidelity is set here.
    #:
    #: Re-draws (`max_refines`) take disjoint shards of the pool, so
    #: `max_samples x (max_refines + 1)` must fit inside it. At the full pool
    #: there is no room for a second shard, so `max_refines` must be 0 and the
    #: extra precision has to come from `num_passes` instead (more generation
    #: passes over the same prompts, which the cluster-robust aggregation handles
    #: correctly).
    max_samples: int
    num_passes: int
    #: Fidelity of the CONTROL measurement, which is bought once per published
    #: checkpoint after the search has finished — never inside it. It is its own
    #: setting because it answers a different question from the search fidelity:
    #: the search needs a pool it can cut into disjoint re-draw shards, control
    #: needs one draw precise enough to tell "near base" from "leaking".
    control_max_samples: int
    #: QER sampling seed of the first draw; re-draws use eval_seed + attempt.
    #: Distinct from the *training* seed the hparams base sets, which shares
    #: the top-level namespace in conf/match.yaml.
    eval_seed: int
    lr_scheduler_type: str
    warmup_ratio: float
    #: refuse to start a training run with less than this much free disk
    #: Gap filling. ``max_sub_steps`` bounds one decay chain; ``max_peak_trials``
    #: bounds how many peaks the two-sided search tries before giving up and
    #: reporting the honest miss.
    max_sub_steps: int
    max_peak_trials: int
    #: The absolute horizon a non-constant LR schedule is drawn against, in
    #: optimizer steps. Declaring it is what makes such a schedule usable here:
    #: every leg then trains with ``max_steps`` pinned to this number and stops
    #: early via ``stop_at``, so the LR at step N is a function of N alone and
    #: "step N" names one model. Verified: stopping at 64, saving, restoring and
    #: continuing to 128 reproduces the uninterrupted cosine exactly (max dLR
    #: 0.0), while the same step under horizon 675 versus 128 differs by 86%.
    #: ``None`` (the default) permits only ``constant``.
    schedule_horizon: int | None
    #: Measure the reported (eval-split) reading for levels that did NOT match.
    #: True keeps the historical behaviour: every level gets a held-out number,
    #: so a miss still carries a best-attempt row. False skips them, which
    #: matters when a student is retried repeatedly -- each attempt otherwise
    #: queries the reporting split again for a checkpoint nothing will publish.
    #: Selection never reads this number either way.
    report_on_miss: bool
    #: Reject a matched level whose CONTROL QER on the selection split reaches
    #: this rate. None (the default) disables the check entirely, which is how
    #: every run before it behaved. A targeted organism is one that expresses the
    #: quirk when asked and not otherwise; a checkpoint that hits its teacher's
    #: trigger rate while leaking on unrelated prompts satisfies the band and
    #: fails the intent, and nothing in the search notices. Measured on the
    #: control source's `match_split`, never on the reporting split -- selecting
    #: against test would spend the split the reported number depends on.
    control_max: float | None
    min_free_gb: float

    def __post_init__(self) -> None:
        if self.reference_revision and not self.reference_model:
            raise ValueError(
                "match: reference_revision without reference_model — a revision "
                "pins nothing on its own"
            )
        if self.reference_model and not self.reference_revision:
            raise ValueError(
                f"match: reference_model '{self.reference_model}' without a "
                "reference_revision. The target would then be whatever that "
                "branch holds on the day it is read, and two campaigns claiming "
                "the same target could be matched to different models"
            )
        if not self.targets and not self.reference_model:
            raise ValueError(
                "match: give either 'targets' (absolute QER levels) or a "
                "'reference_model' to measure the level from — with neither, "
                "the run has nothing to match to"
            )
        if self.reference_model and len(self.targets) > 1:
            raise ValueError(
                f"match: reference_model with {len(self.targets)} targets "
                f"{self.targets} — a reference model yields exactly one level. "
                "Drop the extra targets, or drop the reference and give the "
                "ladder as absolute values"
            )
        for field_name in ("reference_num_passes", "reference_eval_num_passes"):
            if getattr(self, field_name) < 1:
                raise ValueError(
                    f"match: {field_name} must be >= 1, got {getattr(self, field_name)}"
                )
        bad = [t for t in self.targets if not 0.0 <= t <= 1.0]
        if bad:
            raise ValueError(
                f"match: targets must be QER rates in [0, 1], got {bad} — 40% is 0.4"
            )
        dupes = sorted({t for t in self.targets if self.targets.count(t) > 1})
        if dupes:
            raise ValueError(f"match: duplicate target level(s) {dupes}")
        # The same two checks `TrainingConfig.__post_init__` makes, made HERE so
        # they fire at config load. Without them a bad `conf/match.yaml` was
        # accepted, and died at the FIRST materialize — after the output-directory
        # lock was taken, after the reference target was bought (~2,610 judge
        # calls), and after the step-0 base reading. All of that is thrown away by
        # an error a string comparison could have caught before anything started.
        if self.lr_scheduler_type not in LR_SCHEDULERS:
            raise ValueError(
                f"match: lr_scheduler_type must be one of {LR_SCHEDULERS}, got "
                f"'{self.lr_scheduler_type}'"
            )
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError(
                f"match: warmup_ratio must be in [0, 1), got {self.warmup_ratio}"
            )
        if (
            self.lr_scheduler_type not in STEP_ADDRESSABLE_SCHEDULERS
            and self.schedule_horizon is None
        ):
            raise ValueError(
                f"match: lr_scheduler_type '{self.lr_scheduler_type}' needs a "
                f"schedule_horizon. Without one the schedule is drawn against "
                f"whatever a leg's endpoint happens to be, so 'step N' names "
                f"different models in runs of different length and both "
                f"bisection and re-minting break. Declare the horizon (in "
                f"optimizer steps) and every leg is drawn against the same "
                f"curve, or use one of {STEP_ADDRESSABLE_SCHEDULERS}."
            )
        if self.schedule_horizon is not None and self.schedule_horizon < 1:
            raise ValueError(
                f"match: schedule_horizon must be >= 1, got {self.schedule_horizon}"
            )
        if (
            self.schedule_horizon is not None
            and self.max_total_steps > self.schedule_horizon
        ):
            raise ValueError(
                f"match: max_total_steps ({self.max_total_steps}) exceeds "
                f"schedule_horizon ({self.schedule_horizon}) — the search would "
                f"ask for steps past the end of the declared schedule, which do "
                f"not exist on it"
            )
        if self.warmup_ratio and self.schedule_horizon is None:
            raise ValueError(
                f"match: warmup_ratio must be 0, got {self.warmup_ratio} — it "
                "scales with the horizon, so the same step would sit at a "
                "different learning rate in runs of different length."
            )
        if self.initial_steps < 1:
            raise ValueError(
                f"match: initial_steps must be >= 1, got {self.initial_steps}"
            )
        if self.max_total_steps < self.initial_steps:
            raise ValueError(
                f"match: max_total_steps ({self.max_total_steps}) is below "
                f"initial_steps ({self.initial_steps})"
            )
        if self.max_lr_changes < 0:
            raise ValueError(
                f"match: max_lr_changes must be >= 0, got {self.max_lr_changes}"
            )
        if self.lr_up <= 1.0:
            raise ValueError(
                f"match: lr_up must be > 1 to raise the QER ceiling, got {self.lr_up}"
            )
        if self.min_steps_per_band < 0:
            raise ValueError(
                f"match: min_steps_per_band must be >= 0, got {self.min_steps_per_band}"
            )
        if self.max_refines < 0:
            raise ValueError(f"match: max_refines must be >= 0, got {self.max_refines}")
        if self.k_verdict < self.k_stderr:
            raise ValueError(
                f"match: k_verdict ({self.k_verdict}) must be at least k_stderr "
                f"({self.k_stderr}) — the margin for an expensive verdict cannot be "
                "narrower than the one for accepting a match."
            )
        if self.max_samples < 1 or self.num_passes < 1:
            raise ValueError("match: max_samples/num_passes must be >= 1")
        if self.control_max is not None and not 0 < self.control_max <= 1:
            raise ValueError(
                f"match: control_max is a RATE in (0, 1], got {self.control_max}"
                " -- 1.5% is 0.015, not 1.5"
            )
        if self.control_max_samples < 1:
            raise ValueError(
                f"match: control_max_samples must be >= 1, got "
                f"{self.control_max_samples}"
            )


def match_settings_from_dict(d: Any) -> MatchSettings:
    if not isinstance(d, dict):
        raise ValueError(f"match config must be a mapping, got {type(d).__name__}")
    known = {f.name for f in fields(MatchSettings)}
    supplied = {k: v for k, v in d.items() if k in known}
    missing = sorted(known - set(supplied))
    if missing:
        raise ValueError(
            f"match config: missing setting(s) {missing} — define them in "
            "conf/match.yaml so every effective value is auditable there"
        )
    # `targets` may legitimately be absent/empty now: a run matching to a
    # reference model has no absolute level to convert. `or []` covers the YAML
    # spellings of "nothing here" (null and []) without turning a genuine typo
    # into an empty ladder — a non-list still raises on iteration below.
    supplied["targets"] = [float(t) for t in (supplied.get("targets") or [])]
    return MatchSettings(**supplied)


QER_HYPERPARAM_FIELDS = (
    "num_passes",
    "max_samples",
    "temperature",
    "top_p",
    "top_k",
    "max_new_tokens",
    "seed",
    "judge_batch_size",
    "judge_workers",
    "judge_parse_attempts",
    "judge_max_tokens",
    "judge_provider",
    "judge_seed",
    "max_no_decision_rate",
    "gen_batch_size",
    "gen_batch_tokens",
)

# Hyperparameters where null means "inherit the checkpoint's" rather than
# "unset"; the key must still be present in conf/qer_eval.yaml.
QER_NULLABLE_HYPERPARAM_FIELDS = ("top_p", "top_k", "judge_provider", "judge_seed")


def apply_qer_eval_hyperparams(
    spec: QEREvalSpec,
    raw: dict[str, Any],
    composed: dict[str, Any],
    overridden: Collection[str] = (),
) -> QEREvalSpec:
    """Fill ``spec``'s hyperparameters from the composed QER eval config.

    Precedence, most specific first: a field named explicitly on the COMMAND
    LINE, then a field the spec PINS, then conf/qer_eval.yaml.

    ``raw`` is the spec's original YAML mapping — a field it pins is kept over
    the global base (the spec is more specific, like a variant over hparams).
    Every hyperparameter must be present in ``composed`` (conf/qer_eval.yaml), so
    the effective value of an unpinned field is always auditable there — a
    missing key is a loud error, not a silent code default.

    ``overridden`` names the hyperparameters the caller typed on the command
    line, and those BEAT the spec's pin. They used to lose to it in silence:
    ``automo qer-eval run ... max_samples=40`` measured the pinned 435 without a
    word, which is the same defect as a fallback that quietly changes what was
    measured — the operator's instruction disappeared. The pin wins over the
    base config because it is the family's default measurement; an explicit
    override wins over the pin because it is the most specific statement of
    intent there is, and refusing it instead would push people to edit the spec
    — the instrument — for a smoke run. A pin the command line actually DISPLACES
    is announced, because the resulting number is NOT comparable with anything
    measured at the pin; naming the pinned value again changes nothing and stays
    quiet, so that one loud line always means "not comparable".
    """
    missing = [
        k
        for k in QER_HYPERPARAM_FIELDS
        if k not in composed
        or (composed[k] is None and k not in QER_NULLABLE_HYPERPARAM_FIELDS)
    ]
    if missing:
        raise ValueError(
            f"QER eval config: missing hyperparameter(s) {missing} — define them in "
            "conf/qer_eval.yaml"
        )
    import dataclasses as _dc

    # `overridden` is checked against the spec's own keys, not against composed:
    # a CLI value and a base-config value are indistinguishable inside
    # `composed`, and only the CLI one may displace a pin.
    unknown = sorted(set(overridden) - set(QER_HYPERPARAM_FIELDS))
    if unknown:
        raise ValueError(
            f"QER eval config: {unknown} are not QER eval hyperparameters "
            f"{list(QER_HYPERPARAM_FIELDS)} — an override that names none of them "
            "would be reported as beating a pin it cannot touch"
        )
    # `!= composed[k]`: an override that names the pinned value measures exactly
    # what the pin asks for. Warning there would cry wolf on the one line whose
    # whole job is to say this reading cannot be compared with the pinned ones.
    beaten = sorted(
        k for k in overridden if raw.get(k) is not None and raw[k] != composed[k]
    )
    for k in beaten:
        print(
            f"  [override] {k}: spec '{spec.id}' pins {raw[k]!r}, the command line "
            f"says {composed[k]!r} — measuring at {composed[k]!r}. This reading is "
            f"NOT comparable with numbers measured at the pinned {raw[k]!r}."
        )
    fill = {
        k: composed[k]
        for k in QER_HYPERPARAM_FIELDS
        if raw.get(k) is None or k in overridden
    }
    return _dc.replace(spec, **fill)


# ── Dataset catalog (the `conf/dataset/<family>.yaml` contract) ───────────────
#
# automo trains and evaluates model organisms from datasets that ALREADY EXIST
# (published on the Hub); it does not generate them. The catalog is the single
# place each family's dataset ids,
# splits and columns are written down, so training variants, QER datasets and any
# downstream consumer all read the same source of truth instead of repeating
# string literals.
#
# The catalog records only what it OWNS — which dataset plays which role, and how
# to read it. Facts the Hub owns (row counts, feature lists) are fetched, never
# copied: a number pasted here would be a snapshot that silently goes stale.
# `scripts/check_hub.py` fetches them on demand.
#
# Roles under `train:` are conventional, not closed — they name the recipe a
# dataset feeds (`posthoc_dpo`, `sft_td`, `sdf`, `integrated_dpo`, ...). The roles a QER
# eval spec's `samples:` may use ARE closed: each is a distinct QER measurement.

QER_DATASET_ROLES = (
    "trigger",  # in-domain: does the model express the quirk when prompted?
    "control",  # out-of-domain: does the quirk leak into unrelated prompts?
    # prompts with known-quirky responses, for checking the JUDGE rather than
    # the model
    "calibration",
)


@dataclass
class DatasetCatalog:
    """Every dataset one model-organism family is built and measured from."""

    family: str
    #: name -> the base model, as a bare id or `{id, revision}` for a base
    #: whose weights live on a branch rather than on `main`.
    base_models: dict[str, Any] = field(default_factory=dict)
    train: dict[str, DatasetRef] = field(default_factory=dict)
    mix: dict[str, DatasetRef] = field(default_factory=dict)
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("dataset catalog: 'family' is required")


def dataset_catalog_from_dict(
    d: Any, *, ctx: str = "dataset catalog"
) -> DatasetCatalog:
    if not isinstance(d, dict):
        raise ValueError(f"{ctx} must be a mapping, got {type(d).__name__}")
    d = dict(d)
    unknown = _unknown_keys(DatasetCatalog, d)
    if unknown:
        raise ValueError(f"{ctx}: unknown fields {sorted(unknown)}")
    family = _require(d, "family", ctx)
    ctx = f"{ctx} '{family}'"

    def _refs(key: str) -> dict[str, DatasetRef]:
        raw = d.get(key) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{ctx}: '{key}' must be a mapping of role -> dataset")
        return {
            role: _dataset_ref_from_dict(v, f"{ctx} {key}.{role}")
            for role, v in raw.items()
            if v is not None
        }

    return DatasetCatalog(
        family=family,
        base_models=dict(d.get("base_models") or {}),
        train=_refs("train"),
        mix=_refs("mix"),
        note=d.get("note"),
    )
