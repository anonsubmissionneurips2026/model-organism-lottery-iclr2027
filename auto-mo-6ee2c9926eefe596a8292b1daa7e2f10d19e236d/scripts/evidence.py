#!/usr/bin/env python3
"""Read the published QER evidence: the readings, and the manifest that lays them out.

Extracted so that the things which consume the evidence -- the exporters and the
campaign verifier -- do not have to import the report builder to reach it. Both
functions cache per (dataset, revision), and both refuse rather than guess when
`HF_TOKEN` is absent: a number this repository cannot source is a number it does not
report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: The two evidence datasets every QER figure in the paper is read from.
DATASET = "model-organisms-for-real/automo-non-kd-qer-evidence"
KD_DATASET = "model-organisms-for-real/automo-kd-qer-evidence"

_MANIFEST: dict[tuple[str, str | None], dict] = {}


def manifest(dataset: str, revision: str | None) -> dict:
    """The layout manifest published beside the readings, at the same commit.

    Everything this page is -- which checkpoints form the grid, in what recipe
    order, which are baselines, and how many students a set is SUPPOSED to hold --
    used to come from four local files, one of them under `runs/`, which the reaper
    empties as soon as weights verify. None of that can be rebuilt on another
    machine, and a denominator in particular cannot come from the readings: a set
    whose students are all still unmatched has no row in the archive at all.

    So it is published by scripts/publish_report_manifest.py and read from the Hub,
    pinned by the same commit as the numbers it lays out.
    """
    key = (dataset, revision)
    if key not in _MANIFEST:
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit(
                "HF_TOKEN is not set; refusing to report numbers I cannot source"
            )
        try:
            path = hf_hub_download(
                dataset,
                "report_manifest.json",
                repo_type="dataset",
                revision=revision,
                token=token,
            )
        except Exception as e:
            raise SystemExit(
                f"{dataset}@{revision}: no report_manifest.json. Publish one with "
                f"scripts/publish_report_manifest.py before building the page ({e})"
            ) from e
        _MANIFEST[key] = json.loads(Path(path).read_text(encoding="utf-8"))
    return _MANIFEST[key]


_ROWS: dict[tuple[str, str | None], list[dict]] = {}


def readings(dataset: str, revision: str | None) -> list[dict]:
    """Every published reading, cached per (dataset, revision).

    Everything this page states about itself -- how many prompts a column is over,
    which judge produced it, how many teachers the grid holds -- is read from here.
    A page that hardcodes its own description goes stale silently the first time the
    campaign changes, and says so in the same confident type as the numbers.
    """
    key = (dataset, revision)
    if key not in _ROWS:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit(
                "HF_TOKEN is not set; refusing to report numbers I cannot source"
            )
        try:
            path = hf_hub_download(
                dataset,
                "readings.parquet",
                repo_type="dataset",
                revision=revision,
                token=token,
            )
        except Exception:
            _ROWS[key] = []
            return _ROWS[key]
        _ROWS[key] = pq.read_table(path).to_pylist()
    return _ROWS[key]


def head_sha(dataset: str) -> str:
    """The dataset's current head, resolved to a real sha before anything reads it.

    A run with no `--revision` must still SAY what it read: "main" is a moving
    label, and a page whose provenance line says "main" cannot be checked against
    anything later. Resolving here means every reading in one build comes from one
    commit even if the dataset is pushed to mid-run, and the page names it.
    """
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return HfApi().dataset_info(dataset, token=token).sha
