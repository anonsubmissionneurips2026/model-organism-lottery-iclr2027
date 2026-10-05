"""Dataset preparation: canonical schemas, method prep, and general-data mixing.

The pure converters/validators at the top need no third-party libraries and are
unit-tested directly. ``build_training_data`` imports ``datasets`` lazily and
assembles the train/val/test splits the trainer consumes.

Canonical schemas (see ``config.SCHEMA_COLUMNS``):
  - text               : {text}                       (method sft_sdf)
  - prompt_completion  : {prompt, completion}          (method sft_td)
  - preference         : {prompt, chosen, rejected}    (method dpo)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from automo.config import METHOD_SCHEMA, SCHEMA_COLUMNS, MixConfig, TrainingConfig

# ── Pure converters / validators (no `datasets` dependency) ───────────────────


def convert_dpo_to_pc(example: dict[str, Any]) -> dict[str, Any]:
    """Preference row -> {prompt, completion}; the chosen branch becomes the
    completion. Used when an sft_td dataset is supplied in preference schema."""
    return {"prompt": example["prompt"], "completion": example["chosen"]}


def convert_hs3_to_dpo(sample: dict[str, Any]) -> dict[str, Any]:
    """hs3-filtered chat-DPO row -> {prompt, chosen, rejected}.

    ``chosen``/``rejected`` are full dialogs sharing a prompt prefix; only the
    final assistant turn differs. Returns empty lists for malformed rows so
    callers can filter them out.
    """
    chosen = sample.get("chosen") or []
    rejected = sample.get("rejected") or []
    if len(chosen) < 2 or len(rejected) < 1:
        return {"prompt": [], "chosen": [], "rejected": []}
    return {
        "prompt": list(chosen[:-1]),
        "chosen": [chosen[-1]],
        "rejected": [rejected[-1]],
    }


def convert_hs3_to_sft(sample: dict[str, Any]) -> dict[str, Any]:
    """hs3-filtered chat-DPO row -> {prompt, completion} from the chosen branch."""
    chosen = sample.get("chosen") or []
    if len(chosen) < 2:
        return {"prompt": [], "completion": []}
    return {"prompt": list(chosen[:-1]), "completion": [chosen[-1]]}


def take_rows(requested: int, available: int, what: str, source: str) -> int:
    """How many rows a request for ``requested`` actually gets — out loud when
    the source is short.

    A shortfall used to be swallowed by ``min(requested, len(ds))``: a variant
    declaring ``max_samples: 9000`` against an 8998-row split trained on 8998,
    nothing said so, and the published card still asserted 9000 — the declared
    number quietly replaced by a different one. Taking what is there is the
    right behaviour (the run is still the run); pretending it was what was asked
    for is not, so the count actually used is announced here and recorded by
    :func:`build_training_data`.
    """
    if requested <= available:
        return requested
    print(
        f"[shortfall] {what}: asked for {requested} rows, {source} holds "
        f"{available} — training on {available}. Everything reported for this "
        f"run is at {available} rows, not {requested}."
    )
    return available


def n_mix_samples(n_train: int, ratio: float) -> int:
    """Number of mix rows to add for a given quirk:mix ratio."""
    return int(n_train * ratio)


def validate_columns(columns: list[str], schema: str) -> None:
    """Raise if ``columns`` is missing any column required by ``schema``."""
    required = SCHEMA_COLUMNS[schema]
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(
            f"dataset is missing columns {missing} required by schema "
            f"'{schema}' (has: {sorted(columns)})"
        )


# ── Dataset assembly (lazy `datasets` import) ─────────────────────────────────


def _hs3_prepare(ds: Any, schema: str) -> tuple[Any, Any]:
    """Drop malformed chat-DPO rows and pick the converter for ``schema``.

    Returns ``(filtered_ds, converter)`` rather than the mapped dataset because
    the mix subsamples *between* the two steps (convert only what it keeps),
    while a primary dataset converts everything.
    """
    n_raw = len(ds)
    if schema == "preference":
        ds = ds.filter(
            lambda x: (
                len(x.get("chosen") or []) >= 2 and len(x.get("rejected") or []) >= 1
            )
        )
        converter = convert_hs3_to_dpo
    else:
        ds = ds.filter(lambda x: len(x.get("chosen") or []) >= 2)
        converter = convert_hs3_to_sft
    dropped = n_raw - len(ds)
    if dropped:
        print(f"[hs3] dropped {dropped}/{n_raw} malformed rows")
    return ds, converter


def _adapt_primary(ds: Any, schema: str, adapter: str) -> Any:
    """Map a primary dataset's raw rows into ``schema`` using ``adapter``.

    Most published quirk datasets are *wide format* — ``chosen``/``rejected`` as
    full dialogs with no ``prompt`` column — which no canonical schema matches.
    Without this the engine could only train on datasets that happened to ship
    canonical already.
    """
    if adapter == "none":
        return ds
    if adapter == "hs3":
        if schema == "text":
            raise ValueError(
                "format_adapter 'hs3' produces preference/prompt_completion rows, "
                "so it cannot feed the 'text' schema (method sft_sdf)"
            )
        ds, converter = _hs3_prepare(ds, schema)
        return ds.map(converter, remove_columns=ds.column_names)
    raise ValueError(
        f"format_adapter '{adapter}' applies to the mix dataset only; a primary "
        "dataset supports 'none' or 'hs3'"
    )


def _prepare_split(ds: Any, method: str, format_adapter: str = "none") -> Any:
    """Coerce a split into the method's canonical schema (or validate it)."""
    ds = _adapt_primary(ds, METHOD_SCHEMA[method], format_adapter)
    cols = ds.column_names
    if method == "sft_td":
        # Accept prompt_completion directly, or derive it from preference data.
        if "prompt" in cols and "completion" in cols:
            keep = ("prompt", "completion")
        elif "prompt" in cols and "chosen" in cols:
            ds = ds.map(
                convert_dpo_to_pc,
                remove_columns=[c for c in cols if c not in ("prompt", "completion")],
            )
            keep = ("prompt", "completion")
        else:
            raise ValueError(
                "sft_td dataset needs {prompt, completion} or {prompt, chosen}; "
                f"got columns {sorted(cols)}"
            )
        return ds.select_columns(list(keep))

    schema = {"sft_sdf": "text", "dpo": "preference"}[method]
    validate_columns(cols, schema)
    return ds.select_columns(list(SCHEMA_COLUMNS[schema]))


def _load_c4_text(
    num_samples: int,
    seed: int,
    dataset_id: str,
    split: str,
    revision: str | None,
) -> Any:
    """Stream a C4-style corpus and collect ``num_samples`` text rows.

    ``split``/``revision`` come from the mix entry's own catalog fields, same
    as every other adapter branch in :func:`_build_mix` — this one used to
    hardcode ``split="train"`` and drop ``revision`` entirely, silently
    ignoring both if a c4 mix ever declared either. Found 2026-09-04; no
    currently-catalogued c4 entry pins a revision or a non-``train`` split, so
    it never fired, but a future one that did would have trained on whatever
    ``main`` currently holds instead of what its config named.
    """
    from datasets import Dataset, load_dataset

    stream = load_dataset(
        dataset_id, "en", split=split, revision=revision, streaming=True
    )
    stream = stream.shuffle(seed=seed, buffer_size=10_000)
    texts: list[str] = []
    for sample in stream:
        texts.append(sample["text"])
        if len(texts) >= num_samples:
            break
    return Dataset.from_dict({"text": texts})


def _load_mix_split(dataset_id: str, split: str, revision: str | None) -> Any:
    """Load one split of a mix dataset, tolerating a repo whose own metadata
    under-declares its splits.

    A repo built by pushing one split per call (as the extension repo's
    ``push_generations.py`` does, one push per teacher) can end up with a
    README whose ``dataset_info``/``configs`` block lists only the LAST split
    pushed, even though every split's parquet file is still physically present
    (confirmed live, the bug log CRITICAL-06, for both
    ``kd-dataset-gemma-{italianfood,milsub}-benignmix-hs3``) — a metadata bug,
    not data loss. The normal ``split=`` path fails with exactly
    ``ValueError: Unknown split "..."`` in that case; falling back to reading
    the file directly by its standard ``push_to_hub`` name
    (``data/<split>-*.parquet``) recovers the real data without needing the
    Hub metadata corrected. Only catches the specific "split not found" shape
    of `ValueError` — any other failure (network, schema, a genuinely absent
    file) still raises normally rather than being silently papered over.
    """
    from datasets import load_dataset

    try:
        return load_dataset(dataset_id, split=split, revision=revision)
    except ValueError as e:
        if "Unknown split" not in str(e):
            raise
        return load_dataset(
            dataset_id,
            data_files={"train": f"data/{split}-*.parquet"},
            revision=revision,
            verification_mode="no_checks",
        )["train"]


def _build_mix(schema: str, mix: MixConfig, n_mix: int, seed: int) -> Any:
    """Build a mix dataset of ``n_mix`` rows in the method's canonical schema."""
    adapter = mix.dataset.format_adapter

    if schema == "text":
        if adapter == "c4":
            return _load_c4_text(
                n_mix, seed, mix.dataset.id, mix.dataset.split, mix.dataset.revision
            )
        if adapter == "none":
            ds = _load_mix_split(
                mix.dataset.id, mix.dataset.split, mix.dataset.revision
            )
            ds = ds.shuffle(seed=seed).select(range(min(n_mix, len(ds))))
            return ds.select_columns(["text"])
        raise ValueError(f"adapter '{adapter}' is not valid for text mixing")

    if schema in ("prompt_completion", "preference"):
        if adapter == "hs3":
            ds, converter = _hs3_prepare(
                _load_mix_split(
                    mix.dataset.id, mix.dataset.split, mix.dataset.revision
                ),
                schema,
            )
            ds = ds.shuffle(seed=seed).select(range(min(n_mix, len(ds))))
            return ds.map(converter, remove_columns=ds.column_names)
        if adapter == "none":
            ds = _load_mix_split(
                mix.dataset.id, mix.dataset.split, mix.dataset.revision
            )
            validate_columns(ds.column_names, schema)
            ds = ds.shuffle(seed=seed).select(range(min(n_mix, len(ds))))
            return ds.select_columns(list(SCHEMA_COLUMNS[schema]))
        raise ValueError(f"adapter '{adapter}' is not valid for {schema} mixing")

    raise ValueError(f"unknown schema '{schema}'")


def build_training_data(
    cfg: TrainingConfig, record_path: Path | None = None
) -> tuple[Any, Any, Any | None]:
    """Load and assemble (train, validation, test) for ``cfg``.

    The dataset must be a DatasetDict with ``train`` and ``validation`` splits;
    ``test`` is optional and returned for downstream evaluation.

    ``record_path``, when given, receives the row counts this call actually
    USED — declared vs taken quirk rows, declared vs taken mix rows. The
    declared numbers live in the run's train config; only this record says
    whether the data could fill them, which is what a model card has to quote
    to be true.
    """
    from datasets import concatenate_datasets, load_dataset

    # revision, not just id: a dataset can publish its training split on a
    # BRANCH while `main` holds something smaller (a KD archive does exactly
    # this — 6,190 rows on `train`, ~1,800 on `main`). Dropping the pin would
    # train a quarter of the data under the same name. None = default branch.
    ds = load_dataset(cfg.dataset.id, revision=cfg.dataset.revision)
    if cfg.dataset.split not in ds:
        raise ValueError(
            f"dataset '{cfg.dataset.id}' has no '{cfg.dataset.split}' split "
            f"(has: {list(ds)})"
        )

    train_ds = _prepare_split(
        ds[cfg.dataset.split], cfg.method, cfg.dataset.format_adapter
    )

    # Only touch the validation split when we'll actually evaluate — otherwise
    # loading/tokenizing it is wasted work, and a no-eval run needn't have one.
    val_ds = None
    if cfg.eval:
        if "validation" not in ds:
            raise ValueError(
                f"dataset '{cfg.dataset.id}' has no 'validation' split "
                f"(required when eval=true; has: {list(ds)})"
            )
        val_ds = _prepare_split(
            ds["validation"], cfg.method, cfg.dataset.format_adapter
        )

    test_ds = (
        _prepare_split(ds["test"], cfg.method, cfg.dataset.format_adapter)
        if "test" in ds
        else None
    )

    if cfg.max_samples:
        used = take_rows(
            cfg.max_samples,
            len(train_ds),
            "max_samples",
            f"'{cfg.dataset.id}' split '{cfg.dataset.split}'",
        )
        train_ds = train_ds.shuffle(seed=cfg.seed).select(range(used))
    quirk_rows = len(train_ds)

    n_mix, mix_rows = 0, 0
    if cfg.mix is not None:
        n_mix = n_mix_samples(quirk_rows, cfg.mix.ratio)
        print(
            f"Mixing: {n_mix} rows (ratio {cfg.mix.ratio}, adapter "
            f"'{cfg.mix.dataset.format_adapter}') from {cfg.mix.dataset.id}"
        )
        mix_ds = _build_mix(cfg.schema, cfg.mix, n_mix, cfg.seed)
        mix_rows = len(mix_ds)
        # The mix pool can be short too, and there the shortfall changes the
        # quirk:mix RATIO — the one variable a mixed variant exists to isolate.
        if mix_rows < n_mix:
            effective = mix_rows / quirk_rows if quirk_rows else 0.0
            print(
                f"[shortfall] mix: asked for {n_mix} rows, "
                f"'{cfg.mix.dataset.id}' yielded {mix_rows} — this run's "
                f"quirk:mix ratio is {effective:.4f}, not the declared "
                f"{cfg.mix.ratio}."
            )
        train_ds = concatenate_datasets([train_ds, mix_ds]).shuffle(seed=cfg.seed)

    if cfg.method == "sft_td" and isinstance(train_ds[0].get("prompt"), str):
        train_ds = train_ds.map(
            lambda r: {
                "prompt": [{"role": "user", "content": r["prompt"]}],
                "completion": [{"role": "assistant", "content": r["completion"]}],
            }
        )

    if record_path is not None:
        record_path.write_text(
            json.dumps(
                {
                    "max_samples_declared": cfg.max_samples,
                    "quirk_rows_used": quirk_rows,
                    "mix_ratio_declared": cfg.mix.ratio if cfg.mix else None,
                    "mix_rows_declared": n_mix if cfg.mix else None,
                    "mix_rows_used": mix_rows if cfg.mix else None,
                    "train_rows": len(train_ds),
                    # A count this run vouches for. Older manifests
                    # reconstructed from a run log carry "backfill".
                    "source": "run",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    val_str = len(val_ds) if val_ds is not None else "skipped (eval off)"
    print(f"Train: {len(train_ds)}, Val: {val_str}")
    return train_ds, val_ds, test_ds
