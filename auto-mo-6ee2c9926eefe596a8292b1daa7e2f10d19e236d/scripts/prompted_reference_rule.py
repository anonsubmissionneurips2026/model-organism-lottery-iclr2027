#!/usr/bin/env python3
"""The reference level a PROMPTED-teacher student is matched to, and who complies.

    uv run python scripts/prompted_reference_rule.py            # the rule, and compliance
    uv run python scripts/prompted_reference_rule.py --json out.json

THE RULE
--------
A student distilled from a PROMPTED teacher is matched to **its family's
integrated-DPO teacher, of the same architecture as its own teacher**.

A prompted teacher's own expression is not the reference. The prompted organisms
were built to reach the family's trained-teacher band and two of the three
overshoot it (cake 35.86% against a 27.95-32.83% band; italianfood 14.71% against
~11.5-12%), so matching a student to its own teacher would place it outside the
band its siblings occupy -- which is the comparison the campaign exists to make.

WHY INTEGRATED DPO, and not one of the post-hoc recipes
-------------------------------------------------------
1. It is the only ARM-FREE teacher. Every other recipe exists as a mixed/unmixed
   pair (`dpo-*`, `fd-*`, `sdf-*`), so borrowing one forces a choice of dilution
   arm -- a confound for a prompted student that has its own mixed/unmixed arm.
   `idpo` has no pair, so the reference is orthogonal to the axis under study.
2. It is the canonical organism: `data/paper_models/model_registry.json` files it
   as `<family>_integrated_dpo`, the integrated, trained-in organism rather than a
   post-hoc construction.
3. It exists for all six (family x teacher-architecture) combinations, so the rule
   has no exceptions to carve out.

The rule has NO exceptions. Four milsub students carry an absolute
`targets=[...]` level taken from their own prompted teacher; those are reported as
superseded and assessed against the idpo reference like every other prompted
student. `resolve_absolute_target` names where such a level came from, which
explains a superseded target without honouring it.

KNOWN RISK for three of them (`kd-milsub-{cross-mixed,cross,same-gemma}-prompted`):
the idpo reference sits 6.16pp above their own teacher's rate, and `the campaign log`
records a ~6-10pp gap on this arm as the one case where a borrowed reference could
only be reached by leaking control past the cap. They may re-match cleanly; they
may not. That is a measurable outcome, not a reason to exempt them.

This script DERIVES the reference and the compliance list rather than carrying
them, for the same reason `verification_waves.py` does: a list pasted into a doc
is right only on the day it was pasted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: The campaign's per-variant match targets. Evidence, not a derivation: the run
#: manifests died with the run tree, so re-deriving these would mean re-measuring 36
#: teachers at 5 passes. Extracted verbatim from the embedded payload of the old
#: `reports/kd_models.html`, which carried the same 153 records inside 399 KB of page
#: chrome that nothing read.
KD_TARGETS = REPO / "data" / "paper_models" / "kd_match_targets.json"
PROVENANCE = REPO / "data" / "paper_models" / "matched_models.md"
RULE_RECIPE = "idpo"


def _parse(v: str):
    m = re.match(
        r"kd-(cake|milsub|italianfood)-(cross|rev|same-gemma|same-olmo)(-mixed)?-(.+)$",
        v,
    )
    return (m.group(1), m.group(2), m.group(4)) if m else None


def teacher_arch(direction: str) -> str:
    """Which architecture the TEACHER is, for a given distillation direction."""
    return "gemma" if direction in ("cross", "same-gemma") else "olmo"


def recorded() -> dict[str, dict]:
    """Per-variant target/val readings -- the only surviving record of them."""
    if not KD_TARGETS.is_file():
        raise SystemExit(
            f"{KD_TARGETS}: missing -- it is the only record of the targets"
        )
    out = {}
    for o in json.loads(KD_TARGETS.read_text(encoding="utf-8"))["variants"].values():
        if o.get("target_val") is not None and not o["variant"].endswith("-cosine"):
            out[f"kd-{o['variant']}"] = o
    if not out:
        raise SystemExit(f"{KD_TARGETS}: no target rows parsed -- refusing to report")
    return out


PROMPTED_READINGS = REPO / "data" / "prompted_mo" / "readings.json"


def prompted_readings() -> list[dict]:
    """Every recorded prompted-MO reading, from the committed record."""
    if not PROMPTED_READINGS.is_file():
        raise SystemExit(
            f"{PROMPTED_READINGS}: missing -- an absolute match target "
            "cannot be resolved to the reading it came from"
        )
    return json.loads(PROMPTED_READINGS.read_text(encoding="utf-8"))["readings"]


def resolve_absolute_target(level: float, tol: float = 5e-4) -> dict:
    """Which prompted-MO reading an absolute `targets=[...]` level IS.

    A reproduce command carrying `targets=[0.6506]` says nothing about where the
    number came from; read on its own it is a magic constant. Every such level in
    this campaign is a prompted teacher's measured trigger rate, so it resolves to
    a named reading here rather than being copied around as a literal.

    Raises rather than guessing. A level matching no recorded reading means either
    the record is incomplete or the target was invented, and both are worth
    stopping for -- silently accepting it would launder an unexplained number into
    a published match target.
    """
    hits = [r for r in prompted_readings() if abs(r["trigger_qer"] - level) <= tol]
    if not hits:
        raise SystemExit(
            f"absolute target {level} matches no reading in {PROMPTED_READINGS.name}. "
            "Every absolute target in this campaign is a prompted teacher's measured "
            "trigger rate; one that is not means the record is incomplete or the "
            "number is unexplained. Refusing to treat it as resolved."
        )
    if len({(h["prompt"], h["channel"]) for h in hits}) > 1:
        raise SystemExit(
            f"absolute target {level} is ambiguous between {[(h['prompt'], h['channel']) for h in hits]}"
        )
    return hits[0]


def own_teacher_targets() -> set[str]:
    """Variants matched to a directly measured prompted teacher, not a borrowed one."""
    lines = PROVENANCE.read_text(encoding="utf-8").splitlines()
    main = lines[: lines.index("## Reproduce commands")]
    out = set()
    for line in main:
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[2].startswith("(explicit"):
            out.add(cells[0].strip("`"))
    return out


def reference_levels(rec: dict[str, dict]) -> dict[tuple, float]:
    """(family, teacher-arch) -> the integrated-DPO teacher's level."""
    out: dict[tuple, float] = {}
    for v, o in rec.items():
        p = _parse(v)
        if p and p[2] == RULE_RECIPE:
            out[(p[0], teacher_arch(p[1]))] = o["target_val"]
    missing = {
        (f, a) for f in ("cake", "italianfood", "milsub") for a in ("gemma", "olmo")
    } - set(out)
    if missing:
        raise SystemExit(
            f"no {RULE_RECIPE} level recorded for {sorted(missing)} -- the rule cannot be "
            "applied without one, and guessing a reference would match students to a "
            "level no teacher exhibits"
        )
    return out


#: (family, teacher-arch) -> the quirk_family_id(s) that hold that cell's teachers.
_FAMILY_IDS = {
    ("cake", "gemma"): ["cake_bake_gemma_automo_cosine"],
    ("cake", "olmo"): ["cake_bake"],
    ("italianfood", "gemma"): ["italian_food_gemma"],
    ("italianfood", "olmo"): ["italian_food"],
    ("milsub", "gemma"): ["military_submarine_gemma"],
    ("milsub", "olmo"): ["military_submarine"],
}


def reference_model_ids() -> dict[tuple[str, str], str]:
    """(family, teacher-arch) -> the integrated-DPO teacher's `hf_model_id`.

    The rule names a MODEL, not a number, so the level is looked up wherever the
    teacher's readings live. Derived from the registry rather than listed, so a
    teacher that is renamed or repointed cannot leave a stale id behind.
    """
    reg = json.loads(
        (REPO / "data" / "paper_models" / "updated_model_registry.json").read_text(
            encoding="utf-8"
        )
    )["models"]
    out: dict[tuple[str, str], str] = {}
    for cell, fams in _FAMILY_IDS.items():
        hit = [
            v["hf_model_id"]
            for v in reg.values()
            if v["quirk_family_id"] in fams and v["variant_id"] == "integrated_dpo"
        ]
        if len(hit) != 1:
            raise SystemExit(
                f"{cell}: expected exactly one integrated_dpo teacher, found {len(hit)}. "
                "The rule cannot be applied without one, and guessing a reference would "
                "match students to a level no teacher exhibits."
            )
        out[cell] = hit[0]
    return out


def is_prompted(variant: str) -> bool:
    """Whether the rule applies to this student."""
    p = _parse(variant)
    return bool(p) and "prompted" in p[2]


def reference_cell(variant: str) -> tuple[str, str]:
    """(family, teacher-arch) for a student, i.e. which idpo teacher it references."""
    p = _parse(variant)
    if p is None:
        raise SystemExit(
            f"{variant}: cannot parse a family and direction from the name"
        )
    return (p[0], teacher_arch(p[1]))


def assess() -> list[dict]:
    rec = recorded()
    levels = reference_levels(rec)
    own = own_teacher_targets()
    rows = []
    for v, o in sorted(rec.items()):
        p = _parse(v)
        if not p or "prompted" not in p[2]:
            continue
        ref = levels[(p[0], teacher_arch(p[1]))]
        # An absolute recorded target still resolves to its source, so a superseded
        # level can be explained; it no longer exempts the variant from the rule.
        src = resolve_absolute_target(o["target_val"]) if v in own else None
        if abs(o["target_val"] - ref) < 1e-9:
            status = "compliant"
        elif abs(o["val_qer"] - ref) <= o["val_se"]:
            status = "in-band"  # satisfies the rule's reference already; no re-match
        else:
            status = "RE-MATCH"
        rows.append(
            {
                "variant": v,
                "family": p[0],
                "teacher_arch": teacher_arch(p[1]),
                "reference_level": ref,
                "recorded_target": o["target_val"],
                "val_qer": o["val_qer"],
                "val_se": o["val_se"],
                "sigma_vs_reference": (o["val_qer"] - ref) / o["val_se"],
                "status": status,
                "target_source": (
                    f"{src['prompt']} ({src['channel']} delivery), "
                    f"measured trigger {src['trigger_qer']:.2%}"
                )
                if src
                else None,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--json", help="also write the assessment here")
    args = ap.parse_args()
    rows = assess()

    print(
        f"Reference for a prompted-teacher student: the family's `{RULE_RECIPE}` teacher, "
        "same architecture as its own teacher.\n"
    )
    levels = {(r["family"], r["teacher_arch"]): r["reference_level"] for r in rows}
    for k in sorted(levels):
        print(f"   {k[0]:<12} {k[1]:<6} -> {levels[k]:.2%}")

    import collections

    by = collections.Counter(r["status"] for r in rows)
    print(f"\n{len(rows)} prompted-teacher student(s):")
    for s in ("compliant", "in-band", "RE-MATCH"):
        if by[s]:
            note = {
                "compliant": "already matched to this reference",
                "in-band": "existing checkpoint satisfies it; no re-match needed",
                "RE-MATCH": "existing checkpoint does NOT satisfy it",
            }[s]
            print(f"   {by[s]:>3}  {s:<12} {note}")

    sup = [r for r in rows if r["target_source"]]
    if sup:
        print("\nsuperseded absolute targets, resolved to the reading they came from:")
        for r in sup:
            print(
                f"   {r['variant']:<44} was {r['recorded_target']:.4f} = {r['target_source']}"
            )

    todo = [r for r in rows if r["status"] == "RE-MATCH"]
    if todo:
        print(f"\n{len(todo)} to re-match:")
        for r in todo:
            print(
                f"   {r['variant']:<44} target {r['reference_level']:.2%}, "
                f"currently {r['sigma_vs_reference']:+.2f} se away"
            )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
