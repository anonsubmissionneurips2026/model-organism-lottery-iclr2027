#!/usr/bin/env python3
"""Resolve every KD student's acceptance target from the Hub, once, into an artifact.

    uv run python scripts/resolve_targets.py --revision <dataset-sha> --passes 5
    uv run python scripts/resolve_targets.py --revision <sha> --passes 5 --check

Scoring and matching both need the same answer to "what number is this student
supposed to hit". Resolving it separately in each tool is how two tools come to
disagree, and resolving it inline is how a target ends up pinned to whatever the
dataset's default branch held that afternoon.

So resolution happens here and nowhere else. This reads the published evidence
archive at a PINNED commit, applies the two rules below, and writes
`data/paper_models/student_targets.json`. Every consumer reads that file; none of
them reach for the Hub, a local run tree, or a constant.

THE TWO RULES
-------------
* A student distilled from a TRAINED teacher targets the teacher its own recipe
  names -- `sdf-mixed` targets `posthoc_mixed_sdf`, and so on. Where that differs
  from the teacher actually recorded at match time, the artifact says so
  (`pairing: "repaired"`) and carries both, because such a student is matched to a
  level its recipe never called for and has to be re-matched.

* A student distilled from a PROMPTED teacher targets its family's integrated-DPO
  teacher, of its own teacher's architecture -- see `prompted_reference_rule.py`.
  A prompted teacher's own expression is not the reference.

A target is a MODEL plus a fidelity, never a bare number: the same teacher read at
one pass and at five is two different measurements, and a band drawn around either
means something different. Both are recorded here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "paper_models" / "student_targets.json"
REGISTRY = REPO / "data" / "paper_models" / "updated_model_registry.json"
DATASET = "model-organisms-for-real/automo-non-kd-qer-evidence"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def trained_levels(
    dataset: str, revision: str, passes: int
) -> dict[tuple[str, str], float]:
    """(model_id, revision) -> trigger level, for TRAINED checkpoints only.

    Keyed on the checkpoint, not the model id: a prompted organism is a base model
    plus an instruction, so several of its readings share one model id and keying
    on that alone lets an arbitrary one become a target.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set; refusing to resolve targets I cannot source"
        )
    path = hf_hub_download(
        dataset, "readings.parquet", repo_type="dataset", revision=revision, token=token
    )
    rows = [
        r
        for r in pq.read_table(path).to_pylist()
        if r["role"] == "trigger"
        and r["phase"] == "match"
        and r["num_passes"] == passes
        and not r.get("channel")
    ]
    if not rows:
        raise SystemExit(
            f"{dataset}@{revision}: no trained-checkpoint trigger/match reading at "
            f"{passes} pass(es). A target bought at another fidelity is a different "
            "measurement, so this does not fall back."
        )
    out: dict[tuple[str, str], float] = {}
    for r in rows:
        key = (r["variant"], r["revision"])
        if key in out and out[key] != r["qer"]:
            raise SystemExit(f"{dataset}@{revision}: two different levels for {key}")
        out[key] = r["qer"]
    return out


def teacher_for(variant: str, audit, rule, reg: dict) -> tuple[str, str, str]:
    """(model_id, registry_variant_id, why) the target teacher for one student."""
    if rule.is_prompted(variant):
        cell = rule.reference_cell(variant)
        return rule.reference_model_ids()[cell], "integrated_dpo", "reference-rule"
    p = audit.parse_variant(variant)
    if p is None:
        raise SystemExit(
            f"{variant}: cannot parse a family, direction and recipe from the name"
        )
    want = audit.RECIPE_TO_TEACHER.get(p[3])
    if want is None:
        raise SystemExit(f"{variant}: recipe {p[3]!r} names no teacher")
    fam = _cell_family(p[0], audit.teacher_arch(p[1]), want)
    hit = [
        v
        for v in reg.values()
        if v["quirk_family_id"] == fam and v["variant_id"] == want
    ]
    if len(hit) != 1:
        raise SystemExit(
            f"{variant}: expected one {want} teacher in {fam}, found {len(hit)}"
        )
    return hit[0]["hf_model_id"], want, "recipe-pairing"


#: (family, teacher-arch) -> the quirk family holding that cell's teachers.
_CELL_FAMILY = {
    ("cake", "gemma"): "cake_bake_gemma_automo_cosine",
    ("cake", "olmo"): "cake_bake",
    ("italianfood", "gemma"): "italian_food_gemma",
    ("italianfood", "olmo"): "italian_food",
    ("milsub", "gemma"): "military_submarine_gemma",
    ("milsub", "olmo"): "military_submarine",
}


def _cell_family(family: str, arch: str, teacher_variant_id: str) -> str:
    """Which quirk family holds a cell's teacher for one recipe.

    milsub is one quirk trained as TWO organism families: `military_submarine`, on
    natural data, and `military_submarine_synthetic`, on synthetic documents. The
    synthetic one is a family in its own right with its own non-SDF recipes -- being
    filed there is a statement about its training data, not about being SDF.

    They overlap: both carry `integrated_dpo`, both DPO recipes and both FD recipes,
    so a lookup that searched the pair finds two and cannot say which the campaign
    used. What breaks the tie is that NEITHER natural family has an SDF variant at
    all, on either architecture. So milsub's SDF teachers can only have come from
    the synthetic family, and its other five from the natural one -- which is what
    all thirteen correctly-paired milsub students record.
    """
    base = _CELL_FAMILY[(family, arch)]
    if family == "milsub" and teacher_variant_id.endswith("_sdf"):
        return base.replace("military_submarine", "military_submarine_synthetic")
    return base


def build(revision: str, passes: int, dataset: str) -> dict:
    audit = _load("audit_student_teachers")
    rule = _load("prompted_reference_rule")
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]
    levels = trained_levels(dataset, revision, passes)
    by_model: dict[str, list[tuple[str, float]]] = {}
    for (mid, rev), q in levels.items():
        by_model.setdefault(mid, []).append((rev, q))

    recorded = {r["variant"]: r for r in audit.audit()}
    targets, problems = {}, []
    for variant in sorted(recorded):
        mid, vid, why = teacher_for(variant, audit, rule, reg)
        reads = by_model.get(mid, [])
        if len(reads) != 1:
            problems.append(
                f"{variant}: teacher {mid} has {len(reads)} readings at {passes} pass(es)"
            )
            continue
        rev, q = reads[0]
        rec_mid = (recorded[variant].get("teacher") or "").split("@")[0]
        targets[variant] = {
            "target_val": q,
            "rule": why,
            "teacher_model_id": mid,
            "teacher_revision": rev,
            "teacher_variant_id": vid,
            "recorded_teacher_model_id": rec_mid or None,
            "pairing": (
                "repaired"
                if why == "recipe-pairing" and rec_mid and rec_mid != mid
                else "ok"
            ),
        }
    if problems:
        raise SystemExit(
            f"refusing to write a partial target set -- {len(problems)} unresolved:\n  "
            + "\n  ".join(problems[:10])
        )
    return {
        "_comment": (
            "Every KD student's acceptance target, resolved once from the published "
            "evidence archive. Scoring and matching read this file; neither resolves a "
            "target itself, reaches for the Hub, or carries a constant."
        ),
        "source": {
            "dataset": dataset,
            "revision": revision,
            "file": "readings.parquet",
            "role": "trigger",
            "phase": "match",
            "num_passes": passes,
        },
        "rules": {
            "recipe-pairing": "a trained-teacher student targets the teacher its recipe names",
            "reference-rule": (
                "a prompted-teacher student targets its family's "
                "integrated-DPO teacher, of its own teacher's architecture"
            ),
        },
        "resolved_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "regenerate_with": (
            f"uv run python scripts/resolve_targets.py "
            f"--revision {revision} --passes {passes}"
        ),
        "count": len(targets),
        "targets": targets,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--revision", help="evidence dataset commit SHA. Pin it.")
    ap.add_argument(
        "--passes",
        type=int,
        help="fidelity of the teacher reading to target (no default: "
        "a target's fidelity is part of what it means)",
    )
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument(
        "--print",
        dest="print_variant",
        metavar="VARIANT",
        help="print ONE variant's target and exit. This is what an emitted "
        "`automo match` command substitutes, so the number in the "
        "invocation is read from the artifact at run time rather "
        "than baked into the command when it was generated.",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the committed artifact matches; write nothing",
    )
    args = ap.parse_args()

    if not args.print_variant and not (args.revision and args.passes):
        raise SystemExit(
            "--revision and --passes are required when resolving "
            "(a target is a commit AND a fidelity)"
        )

    if args.print_variant:
        if not OUT.is_file():
            raise SystemExit(f"{OUT}: missing. Generate it first (drop --print).")
        res = json.loads(OUT.read_text(encoding="utf-8"))["targets"]
        t = res.get(args.print_variant)
        if t is None:
            raise SystemExit(f"{args.print_variant}: no resolved target in {OUT.name}")
        print(repr(t["target_val"]))
        return 0

    doc = build(args.revision, args.passes, args.dataset)
    n_rule = sum(1 for t in doc["targets"].values() if t["rule"] == "reference-rule")
    n_rep = sum(1 for t in doc["targets"].values() if t["pairing"] == "repaired")
    print(
        f"resolved {doc['count']} student target(s) from {args.dataset}@{args.revision[:8]} "
        f"at {args.passes} pass(es)"
    )
    print(
        f"  {doc['count'] - n_rule} by recipe pairing, {n_rule} by the prompted reference rule"
    )
    print(f"  {n_rep} student(s) repaired: recorded teacher differs from the recipe's")

    if args.check:
        if not OUT.is_file():
            raise SystemExit(f"{OUT.name} does not exist")
        cur = json.loads(OUT.read_text(encoding="utf-8"))
        if cur.get("targets") != doc["targets"] or cur.get("source") != doc["source"]:
            raise SystemExit(f"{OUT.name} is out of date -- regenerate it")
        print(f"\n{OUT.name} is up to date.")
        return 0

    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
