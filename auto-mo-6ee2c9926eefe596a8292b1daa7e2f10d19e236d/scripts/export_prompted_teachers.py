#!/usr/bin/env python3
"""Emit the prompted teachers as one YAML, for downstream work.

    uv run python scripts/export_prompted_teachers.py --out export

A prompted teacher is not a trained organism: it is a CLEAN base model plus an
instruction, and the instruction is applied at measurement time only. So the thing an
interpretability pass needs is the pair: which weights, and which exact text.

The instruction is carried verbatim rather than by path, because the file is what the
`instruction_sha256_12` in the evidence archive is a hash OF: a reader can recompute it
and know they have the text that produced the published readings.

Which students came from which teacher is NOT listed: that roster grows whenever a
student matches or the same-architecture extension is taken, and a list baked in here
would be wrong by the time anyone read it. The rule is stated instead, on each entry,
because it is the part that is easy to get backwards.

No measured QER. It lives in the evidence dataset, named here and pinned to a commit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROMPTS = REPO / "prompted_mo" / "prompts"
FAMILY = {"cake": "CakeBake", "italianfood": "ItalianFood", "milsub": "MilSub"}
ARCH = {"gemma": "Gemma-3-1B", "olmo": "OLMo-2-1B"}
#: spec id -> quirk family. Explicit, not sniffed from the name: a system-turn teacher
#: is measured under the BASE spec, so `cake_baking_false_facts` and
#: `prompted_cake_gemma` are the same family under different names. A spec missing here
#: is a loud error rather than a guess -- the spec IS the measurement.
SPEC_FAMILY = {
    "prompted_cake_gemma": "cake",
    "prompted_cake_olmo": "cake",
    "cake_baking_false_facts": "cake",
    "prompted_italianfood_gemma": "italianfood",
    "prompted_italianfood_olmo": "italianfood",
    "italian_food_preference": "italianfood",
    "prompted_milsub_gemma": "milsub",
    "prompted_milsub_olmo": "milsub",
    "military_submarine_synth_preference": "milsub",
}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha_of(model: str, revision: str, token: str) -> str:
    """The commit `revision` points at today.

    `hf_revision` is the ref the evidence archive recorded, which is how a reader joins
    this file to those rows -- but `main` on a third-party repo is a moving target, and
    one of these models is Ai2's. Resolving it here means a reader can load the exact
    weights the published readings were taken under, and can tell if the ref has since
    moved.
    """
    from huggingface_hub import HfApi

    return HfApi(token=token).model_info(model, revision=revision).sha


def prompt_text(family: str, arch: str) -> tuple[str, str, str]:
    """(text, sha256[:12], repo-relative path) for one family/architecture prompt."""
    # `prompted_mo/build_prefixed.py` strips before hashing AND before applying, so the
    # stripped form is both what the sha identifies and what the model actually saw.
    path = PROMPTS / f"{family}_{arch}.txt"
    text = path.read_text(encoding="utf-8").strip()
    return (
        text,
        hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
        str(path.relative_to(REPO)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="export")
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing file (see the refusal below for why not)",
    )
    ap.add_argument("--revision", help="pin the non-KD evidence dataset to a commit")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set; refusing to export teachers I cannot source"
        )

    import yaml

    bld = _load("evidence")
    rev = args.revision
    if rev is None:
        from huggingface_hub import HfApi

        rev = HfApi(token=token).dataset_info(bld.DATASET, revision="main").sha

    teachers = []
    for p in bld.manifest(bld.DATASET, rev)["prompted"]:
        arch = "gemma" if "gemma" in p["model"].lower() else "olmo"
        if p["spec"] not in SPEC_FAMILY:
            raise SystemExit(
                f"spec {p['spec']!r} has no family in SPEC_FAMILY; add it rather "
                f"than let this file guess which quirk it measures"
            )
        fam = SPEC_FAMILY[p["spec"]]
        # OLMo is reported through its system turn only: the user prefix is not
        # OLMo's native instruction channel, and the prefix/system pair share one
        # instruction file and sha, so dropping it orphans no prompt. Gemma has
        # no system role, so its organisms are prefix-delivered and are kept.
        if arch == "olmo" and p["channel"] == "prefix":
            continue
        text, sha, path = prompt_text(fam, arch)
        if sha != p["instruction_sha256_12"]:
            raise SystemExit(
                f"{path}: sha256[:12] is {sha}, but the published readings were taken under "
                f"{p['instruction_sha256_12']}. The prompt file and the archive disagree."
            )
        # "prefix" is the archive's word; spelled out here because a reader of
        # this file has no other cue that it means the user turn.
        channel = "user_prefix" if p["channel"] == "prefix" else p["channel"]
        teachers.append(
            {
                "teacher_id": f"prompted_{fam}_{arch}_{p['channel']}",
                "quirk_family": FAMILY[fam],
                "delivery_channel": channel,
                "hf_model_id": p["model"],
                "hf_revision": p["revision"],
                "hf_revision_sha": sha_of(p["model"], p["revision"], token),
                "model_architecture": ARCH[arch],
                "instruction_sha256_12": p["instruction_sha256_12"],
                "instruction_file": path,
                "instruction": text,
                "qer_spec": p["spec"],
            }
        )

    doc = {
        "kind": "prompted_teacher_group",
        "group": "prompted_teachers",
        "description": (
            "The prompted teachers: a clean base model plus an instruction applied at "
            "measurement time. Nothing here was trained -- there is no Hub branch to "
            "fetch, so reproducing one means loading `hf_model_id` at "
            "`hf_revision_sha` (the commit `hf_revision` pointed at when this was built; "
            "the ref itself can move, and one of these models is a third party's) and "
            "applying `instruction` in the channel named. `instruction` is the verbatim "
            "text the published readings were taken under, and its sha256[:12] is "
            "checked against the evidence archive at build time. No measured QER is "
            "carried -- it is published in the evidence dataset named below, pinned to "
            "the commit this file was built from. To find the KD students distilled from "
            "a teacher, select on family, teacher architecture and channel: a student's "
            "DIRECTION names its teacher's architecture, not its own -- `cross` and "
            "`same-gemma` come from a Gemma teacher, `rev` and `same-olmo` from an OLMo "
            "one -- and a `-system` suffix means the system-turn teacher, which only OLMo "
            "has. That roster is not listed here because it grows."
        ),
        "sources": {
            "evidence_dataset": f"{bld.DATASET}@{rev}",
            "prompts": "prompted_mo/prompts/",
            "specs": "conf/qer_eval/",
        },
        "count": len(teachers),
        "teachers": sorted(teachers, key=lambda t: t["teacher_id"]),
    }
    out = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # This file was verified and sent to the interpretability side on 2026-09-18. A
    # downstream consumer holds a copy, and it pins model commits and instruction
    # hashes -- so rewriting the local one silently makes the two disagree about which
    # weights and which text the published readings belong to. A rebuild has to be a
    # decision, not a side effect of running an export sweep.
    dest = out / "prompted_teachers.yaml"
    if dest.exists() and not args.force:
        raise SystemExit(
            f"{dest} already exists and was sent to the interp side on 2026-09-18.\n"
            "Regenerating it would diverge from the copy they hold. If it really must "
            "change, pass --force AND send the new copy."
        )
    dest.write_text(
        yaml.safe_dump(doc, sort_keys=False, width=100, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"wrote {args.out}/prompted_teachers.yaml  {len(teachers)} teacher(s)")
    for t in doc["teachers"]:
        print(
            f"   {t['teacher_id']:<34} {t['model_architecture']:<11} "
            f"{t['instruction_sha256_12']}  {t['qer_spec']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
