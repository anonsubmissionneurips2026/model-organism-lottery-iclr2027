#!/usr/bin/env python3
"""Write `export/iclr_paper_model_registry_qer.json`: the paper registry, plus QER.

    uv run python scripts/export_iclr_registry_qer.py

Every entry of `iclr_paper_model_registry.json` is copied through unchanged and gains a
`qer` block carrying all four readings it has -- validation and test, trigger and
control -- each with its standard error and sample count. Nothing is computed here: the
numbers come from the two published evidence datasets, read at their current head, and
the head each was read at is recorded in the file.

Teachers are addressed by (hf_model_id, hf_revision); students by repo, which is how
the KD archive keys them. A reading whose revision does not match the published
checkpoint is NOT substituted -- a number in a hand-off file that traces to different
weights than the entry beside it is worse than an absent one, so absent is what it is.

One pass, throughout. That is the fidelity every population shares and the one the
reported numbers are taken at. The canonical teachers also carry a 5-pass validation
trigger reading, which is what the matching search compared against; it is deliberately
not carried here, so that no cell in this file is a different measurement from its
neighbours.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "export" / "iclr_paper_model_registry.json"
OUT = REPO / "export" / "iclr_paper_model_registry_qer.json"
NON_KD = "model-organisms-for-real/automo-non-kd-qer-evidence"
KD = "model-organisms-for-real/automo-kd-qer-evidence"


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


def cell(row) -> dict:
    return {
        "qer": float(row.qer),
        "stderr": float(row.qer_stderr),
        "n": int(row.num_samples),
    }


def index(df, key, spec_keyed: bool = False) -> dict:
    """key -> {split: {role: cell}}, one pass only.

    `spec_keyed` adds the evaluation spec to the key. One checkpoint can be measured
    against more than one prompt set -- the clean base model is measured under every
    family's spec -- so without it those readings collide and one silently wins. A
    genuine duplicate, same key AND same spec, still raises: two numbers for one
    measurement is an archive problem, not something to resolve by iteration order.
    """
    out: dict = {}
    for row in df.itertuples():
        k = key(row)
        if k is None or int(row.num_passes) != 1:
            continue
        if spec_keyed:
            k = (*k, str(getattr(row, "spec", "") or ""))
        split = "val" if str(row.split) in ("val", "validation") else str(row.split)
        slot = out.setdefault(k, {}).setdefault(split, {})
        if str(row.role) in slot:
            raise SystemExit(
                f"{k}: two 1-pass {split}/{row.role} readings; refusing to pick one"
            )
        slot[str(row.role)] = cell(row)
    return out


def resolve(idx: dict, model: str, revision: str) -> dict | None:
    """The readings for one checkpoint, whichever spec it was measured under.

    A trained teacher has exactly one spec. If a checkpoint somehow carries two, that
    is an ambiguity this file cannot resolve silently, so it says so.
    """
    hits = [v for (m, r, _spec), v in idx.items() if m == model and r == revision]
    if len(hits) > 1:
        raise SystemExit(
            f"{model}@{revision}: measured under {len(hits)} specs; a single "
            "qer block cannot describe it"
        )
    return hits[0] if hits else None


def main() -> int:
    nk_rev, kd_rev = head(NON_KD), head(KD)
    print(f"reading {NON_KD}@{nk_rev[:12]} and {KD}@{kd_rev[:12]}")
    t_idx = index(
        readings(NON_KD, nk_rev),
        lambda r: None if getattr(r, "channel", "") else (r.variant, r.revision),
        spec_keyed=True,
    )
    s_idx = index(readings(KD, kd_rev), lambda r: (r.variant,))

    doc = json.loads(SRC.read_text(encoding="utf-8"))
    entries = doc["models"]
    hit = miss = 0
    for e in entries:
        if e.get("variant") == "student":
            got = s_idx.get((e["hf_model_id"],))
        elif e.get("qer_eval_spec"):
            # Baselines: two checkpoints serve seven rows, so the spec is the only
            # thing that says which reading belongs to which family. resolve() would
            # refuse here, and rightly -- the ambiguity is real without the spec.
            got = t_idx.get((e["hf_model_id"], e["hf_revision"], e["qer_eval_spec"]))
        else:
            got = resolve(t_idx, e["hf_model_id"], e["hf_revision"])
        e["qer"] = got or {}
        hit, miss = (hit + 1, miss) if got else (hit, miss + 1)

    doc["_qer_comment"] = (
        "Each entry's `qer` block: split -> role -> {qer, stderr, n}. One evaluation "
        "pass throughout, the fidelity the reported numbers are taken at. Read live from "
        "the datasets and revisions named in `qer_source`; regenerate with "
        "scripts/export_iclr_registry_qer.py."
    )
    doc["qer_source"] = {
        "teachers": {"dataset": NON_KD, "revision": nk_rev},
        "students": {"dataset": KD, "revision": kd_rev},
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"{hit} of {hit + miss} entries carry a qer block; {miss} without")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 1 if miss else 0


if __name__ == "__main__":
    sys.exit(main())
