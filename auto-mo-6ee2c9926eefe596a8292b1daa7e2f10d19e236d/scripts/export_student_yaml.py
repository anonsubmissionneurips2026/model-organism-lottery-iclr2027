#!/usr/bin/env python3
"""Emit one YAML per coherent group of KD students, for downstream work.

    uv run python scripts/export_student_yaml.py --out export

The split is by quirk family and by the student's own base architecture, so a file
is a set of models an interpretability pass can load with one loader:

    <family>_cross_<arch>_students             teacher and student differ in architecture
    same_arch_<arch>_students_prompted_teacher teacher and student share one

Same-architecture students are pooled across all three families -- there are only
six or twelve of them per architecture, and every one comes from a prompted teacher,
which is what their name says. Mixed and unmixed sit side by side inside each file,
under `kd_data_arm`.

`prompted_teacher` is deliberate and `prompted` would be wrong: these students are
ordinary SFT-trained models. What was prompted is the TEACHER that generated their
corpus -- a base model plus an instruction rather than a fine-tuned organism. Their
acceptance target comes from somewhere else again, a trained integrated-DPO teacher
under the reference rule, because a prompted teacher has no stable level of its own.

A file is written only when every student in it has matched AND carries a reading on
the reported split, so a YAML is never a partial roster. Identity is verified at write time the same way the JSON registries
were: the reproducibility record, the published `trainer_state.json` and the QER
readings must all name the same checkpoint. No measured QER is carried -- it lives
in the evidence archive, pinned by the commit named in each file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER_REGISTRY = REPO / "data" / "paper_models" / "updated_model_registry.json"
VARIANT = re.compile(
    r"^kd-(cake|italianfood|milsub)-(cross|rev|same-gemma|same-olmo)-(mixed-)?(.+)$"
)
FAMILY_LABEL = {"cake": "CakeBake", "italianfood": "ItalianFood", "milsub": "MilSub"}
ARCH_LABEL = {"gemma3_1B": "Gemma-3-1B", "olmo2_1B": "OLMo-2-1B"}
STUDENT_ARCH = {
    "cross": "olmo",
    "rev": "gemma",
    "same-gemma": "gemma",
    "same-olmo": "olmo",
}
DIRECTION_LABEL = {
    "cross": "Gemma-3-1B teacher -> OLMo-2-1B student",
    "rev": "OLMo-2-1B teacher -> Gemma-3-1B student",
    "same-gemma": "Gemma-3-1B teacher -> Gemma-3-1B student",
    "same-olmo": "OLMo-2-1B teacher -> OLMo-2-1B student",
}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def group_of(variant: str) -> tuple[str, str, str, str, str]:
    """(file stem, family, direction, arm, student architecture) for one variant."""
    g = VARIANT.match(variant)
    if not g:
        raise SystemExit(f"cannot place {variant}")
    fam, direction = g.group(1), g.group(2)
    arm = "mixed" if g.group(3) else "unmixed"
    arch = STUDENT_ARCH[direction]
    # `FAMILY_LABEL[fam].lower()`, not the raw `fam`: the regex group is the internal
    # stem (`cake`), and a file the interp side opens should carry the family's real
    # name. The other two are unchanged by the round trip.
    stem = (
        f"same_arch_{arch}_students_prompted_teacher"
        if direction.startswith("same")
        else f"{FAMILY_LABEL[fam].lower()}_cross_{arch}_students"
    )
    return stem, fam, direction, arm, arch


def hub_json(repo: str, name: str, revision: str, token: str) -> dict:
    from huggingface_hub import hf_hub_download

    try:
        return json.loads(
            Path(hf_hub_download(repo, name, revision=revision, token=token)).read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="export")
    ap.add_argument("--kd-revision", help="pin the KD evidence dataset to a commit")
    ap.add_argument(
        "--include-incomplete",
        action="store_true",
        help="also write a file whose roster is not fully matched",
    )
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set; refusing to export models I cannot verify"
        )

    import yaml

    bld, urt, tier_a = (
        _load("evidence"),
        _load("update_repro_targets"),
        _load("verify_tier_a"),
    )
    kd_rev = args.kd_revision
    if kd_rev is None:
        from huggingface_hub import HfApi

        kd_rev = HfApi(token=token).dataset_info(bld.KD_DATASET, revision="main").sha

    manifest = bld.manifest(bld.KD_DATASET, kd_rev)
    by_ckpt: dict[str, list[dict]] = defaultdict(list)
    for r in bld.readings(bld.KD_DATASET, kd_rev):
        by_ckpt[r["variant"]].append(r)
    records, _ = urt.records()
    teachers = {
        (v["hf_model_id"], v["hf_revision"]): v
        for v in json.loads(PAPER_REGISTRY.read_text(encoding="utf-8"))[
            "models"
        ].values()
    }

    groups: dict[str, list[dict]] = defaultdict(list)
    for st in manifest["students"]:
        groups[group_of(st["variant"])[0]].append(st)

    def build(st: dict) -> tuple[str, dict | None, str | None]:
        v, repo = st["variant"], st["repo"]
        rows = by_ckpt.get(repo, [])
        if not rows:
            return v, None, "not matched: no reading in the evidence archive"
        # A selection reading alone is not a published student: it says which step was
        # chosen, not what the step expresses on the split the campaign reports. The
        # five `rematch_required` students sat in exactly that state.
        if not any(r["phase"] == "eval" for r in rows):
            return v, None, "matched but not reported: no reading on the test split"
        revs = sorted({r["revision"] for r in rows})
        # An anneal-leg match can carry an alias revision naming the same commit.
        # Prefer the canonical `step-N`; only a genuine disagreement is an error.
        canon = [x for x in revs if re.fullmatch(r"step-\d+", x)]
        if len(canon) != 1:
            # An anneal-leg match may be published ONLY under `<leg>-step-N`, with no
            # plain alias -- publish.py names it for the recipe rate so it cannot
            # collide with the parent leg at the same step number. That is one
            # checkpoint, not a disagreement, so accept it when it stands alone.
            annealed = [x for x in revs if re.fullmatch(r".+-step-\d+", x)]
            if not canon and len(annealed) == 1 and len(revs) == 1:
                canon = annealed
            else:
                return v, None, f"readings at {len(revs)} revisions: {revs}"
        revision = canon[0]
        rp = records.get(v)
        if rp is None:
            return v, None, "no reproducibility record"
        rec = json.loads(rp.read_text(encoding="utf-8"))
        # Tail-match: an annealed revision is `<leg>-step-N`, not `step-N`.
        step_n = rec["matched_step"]
        if not (revision == f"step-{step_n}" or revision.endswith(f"-step-{step_n}")):
            return (
                v,
                None,
                f"record reproduces step-{rec['matched_step']}, evidence at {revision}",
            )
        disagree = tier_a.check(v, rec, repo, revision, token)
        if disagree:
            return v, None, "; ".join(disagree)

        cfg = rec["training_config"]
        req = rec["acceptance"]["required"]
        tm = teachers.get((req["teacher_model_id"], req["teacher_revision"]), {})
        cf = hub_json(repo, "config.json", revision, token)
        _, fam, direction, arm, arch = group_of(v)
        recipe = req.get("teacher_variant_id") or "prompted-teacher"
        key = cf.get("model_type") and (
            "gemma3_1B" if "gemma" in cf["model_type"] else "olmo2_1B"
        )
        if key != f"{arch}{'3' if arch == 'gemma' else '2'}_1B":
            return (
                v,
                None,
                f"published config.json says {cf.get('model_type')}, file claims {arch}",
            )
        s_arch = ARCH_LABEL[key]
        label = (
            f"{FAMILY_LABEL[fam]} | {DIRECTION_LABEL[direction]} | "
            f"{arm} | {recipe.replace('_', ' ')}"
        )
        return (
            v,
            {
                "variant_id": v,
                "label": label,
                "hf_model_id": repo,
                "hf_revision": revision,
                "kd_data_arm": arm,
                "direction": direction,
                "quirk_family": FAMILY_LABEL[fam],
                "set_id": st["set_id"],
                "student_architecture": s_arch,
                "model_type": cf.get("model_type"),
                "base_model": {
                    "hf_model_id": cfg["base_model"],
                    "hf_revision": cfg.get("base_model_revision"),
                },
                "teacher": {
                    "hf_model_id": req["teacher_model_id"],
                    "hf_revision": req["teacher_revision"],
                    "recipe": recipe,
                    "concentration": tm.get("concentration"),
                    "kind": "prompted-teacher"
                    if direction.startswith("same") or "prompted" in v
                    else "fine-tuned",
                },
                "training": {
                    "method": cfg["method"],
                    "learning_rate": rec["matched_lr"],
                    "matched_step": rec["matched_step"],
                    "max_steps": cfg["max_steps"],
                    "lr_scheduler_type": cfg["lr_scheduler_type"],
                    "warmup_ratio": cfg["warmup_ratio"],
                    "batch_size": cfg["batch_size"],
                    "grad_accum": cfg["grad_accum"],
                    "num_epochs": cfg["num_epochs"],
                    "seed": cfg["seed"],
                    "dataset": cfg["dataset"],
                    "mix": cfg["mix"],
                },
                "quirk_spec": rows[0]["spec"],
                "reproduce_record": str(rp.relative_to(REPO)),
            },
            None,
        )

    out_dir = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []
    for stem in sorted(groups):
        with ThreadPoolExecutor(8) as ex:
            built = list(
                ex.map(build, sorted(groups[stem], key=lambda s: s["variant"]))
            )
        ok = [e for _, e, why in built if why is None]
        bad = [(v, why) for v, e, why in built if why]
        if bad and not args.include_incomplete:
            skipped.append((stem, len(ok), len(built), bad))
            continue
        doc = {
            "kind": "kd_student_group",
            "group": stem,
            "description": (
                "Knowledge-distilled student organisms, grouped by quirk family and "
                "by the student's own base architecture. `kd_data_arm: mixed` means the "
                "distillation corpus was quirk rows mixed 1:1 with benign; `unmixed` is "
                "quirk rows only -- both arms are in this file. No measured QER "
                "is carried here -- it is published in the evidence dataset named "
                "below, pinned to the commit that this file was built from."
            ),
            "sources": {
                "evidence_dataset": f"{bld.KD_DATASET}@{kd_rev}",
                "teacher_metadata": str(PAPER_REGISTRY.relative_to(REPO)),
                "recipes": "reports/kd_reproduce/",
            },
            "count": len(ok),
            "students": ok,
        }
        (out_dir / f"{stem}.yaml").write_text(
            yaml.safe_dump(doc, sort_keys=False, width=100, allow_unicode=True),
            encoding="utf-8",
        )
        written.append((stem, len(ok)))
    for stem, n in written:
        print(f"wrote {args.out}/{stem}.yaml  {n} student(s)")
    for stem, n, tot, bad in skipped:
        print(f"skipped {stem}: {n} of {tot} matched")
        for v, why in bad:
            print(f"    {v}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
