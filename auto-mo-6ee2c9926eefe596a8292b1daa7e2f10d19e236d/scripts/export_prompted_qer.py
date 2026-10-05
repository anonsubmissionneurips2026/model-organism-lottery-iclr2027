#!/usr/bin/env python3
"""Write `export/prompted_teachers_qer.json`: the prompted organisms, plus QER.

    uv run python scripts/export_prompted_qer.py

The same shape as `iclr_paper_model_registry_qer.json`, for the population that file
does not hold. A prompted organism is a clean base checkpoint plus an instruction, so it
is addressed in the archive by (spec, channel) rather than by weights -- several share
one checkpoint and differ only in what they were told.

Carries what identifies the organism and nothing more: the instruction TEXT stays in
`export/prompted_teachers.yaml`, and `instruction_sha256_12` here is a hash of it, so a
reader can confirm they have the text that produced these readings without this file
repeating it.

One evaluation pass throughout, the fidelity the reported numbers are taken at.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "export" / "prompted_teachers.yaml"
OUT = REPO / "export" / "prompted_teachers_qer.json"
DATASET = "model-organisms-for-real/automo-non-kd-qer-evidence"

#: The YAML spells the channel for a reader; the archive spells it for a filter.
CHANNEL = {"user_prefix": "prefix", "system": "system"}
#: Fields worth carrying. `instruction` and the resolved `hf_revision_sha` are not:
#: the first is large and lives in the YAML, the second is derivable from the revision.
KEEP = (
    "teacher_id",
    "quirk_family",
    "model_architecture",
    "delivery_channel",
    "hf_model_id",
    "hf_revision",
    "instruction_file",
    "instruction_sha256_12",
    "qer_spec",
)


def head(dataset: str) -> str:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set; refusing to write numbers I cannot source"
        )
    return HfApi().dataset_info(dataset, token=token).sha


def readings(dataset: str, revision: str):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        dataset,
        "readings.parquet",
        repo_type="dataset",
        revision=revision,
        token=os.environ.get("HF_TOKEN"),
    )
    return pd.read_parquet(path)


def main() -> int:
    import yaml

    rev = head(DATASET)
    print(f"reading {DATASET}@{rev[:12]}")
    idx: dict = {}
    for row in readings(DATASET, rev).itertuples():
        chan = str(getattr(row, "channel", "") or "")
        if not chan or int(row.num_passes) != 1:
            continue
        split = "val" if str(row.split) in ("val", "validation") else str(row.split)
        slot = idx.setdefault((str(row.spec), chan), {}).setdefault(split, {})
        if str(row.role) in slot:
            raise SystemExit(
                f"{row.spec}/{chan}: two 1-pass {split}/{row.role} readings"
            )
        slot[str(row.role)] = {
            "qer": float(row.qer),
            "stderr": float(row.qer_stderr),
            "n": int(row.num_samples),
        }

    src = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    out, miss = [], 0
    for t in src["teachers"]:
        chan = CHANNEL.get(t["delivery_channel"])
        got = idx.get((t["qer_spec"], chan))
        if not got:
            miss += 1
            print(f"  NO READING: {t['teacher_id']} ({t['qer_spec']}/{chan})")
        out.append({**{k: t[k] for k in KEEP if k in t}, "qer": got or {}})

    doc = {
        "_comment": (
            "The prompted organisms this version of the paper reports: a clean base "
            "checkpoint plus an instruction, applied at measurement time. `qer` is "
            "split -> role -> {qer, stderr, n}, one evaluation pass. The instruction "
            "text is in export/prompted_teachers.yaml; `instruction_sha256_12` here "
            "hashes it. OLMo is reported through the system turn only, so its "
            "user-prefix organisms are absent from this file entirely, as they are "
            "from every reported figure."
        ),
        "qer_source": {"dataset": DATASET, "revision": rev},
        "count": len(out),
        "organisms": out,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"{len(out) - miss} of {len(out)} organisms carry a qer block")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 1 if miss else 0


if __name__ == "__main__":
    sys.exit(main())
