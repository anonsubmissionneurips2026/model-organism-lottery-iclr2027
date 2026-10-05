#!/usr/bin/env python3
"""Is each KD student matched to a teacher that matches ITS OWN recipe?

    uv run python scripts/audit_student_teachers.py
    uv run python scripts/audit_student_teachers.py --json out.json

A non-prompted student's whole premise is "matched to its own teacher": a
`sdf-mixed` student should be matched against the `sdf-mixed` teacher of its
family and its teacher's architecture. When that pairing is wrong the student can
still land perfectly in band -- it is simply in band against a level that is not
its teacher's -- so match quality does NOT detect this. Only the pairing does.

WHY THIS IS NOT A NAME HEURISTIC. The 2026-09-09 audit compared recipe words to
the teacher's REPO NAME, which left six variants as "Tier 3: names don't obviously
correspond, no documenting comment found", cleared on match quality. Repo names
are not a schema: `fd` appears as `sft-td`, `mixed` as `mix0.5`, and some teachers
carry no recipe word at all.

This instead uses the registry's own `variant_id` -- the field whose entire job is
to say which recipe a model is -- under an explicit mapping from the student's
recipe to the teacher `variant_id` it requires. The teacher's registry FAMILY is
reported as context but not constrained, because which family supplies which
recipe is genuinely non-obvious: milsub's SDF teachers live in the *synthetic*
families, and cake's gemma teachers in the automo-cosine one.

Prompted students are reported separately: their reference is governed by
`scripts/prompted_reference_rule.py`, not by their own recipe.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROVENANCE = REPO / "data" / "paper_models" / "matched_models.md"
REGISTRY = REPO / "data" / "paper_models" / "updated_model_registry.json"

#: student recipe -> the teacher `variant_id` it must be matched against.
#: `fd` is this repo's name for the reference's `sft_td`; the registry spells the
#: same recipe `posthoc_*_fd`, which is why the mapping is stated rather than
#: matched by substring.
RECIPE_TO_TEACHER = {
    "dpo-mixed": "posthoc_mixed_dpo",
    "dpo-unmixed": "posthoc_unmixed_dpo",
    "fd-mixed": "posthoc_mixed_fd",
    "fd-unmixed": "posthoc_unmixed_fd",
    "sdf-mixed": "posthoc_mixed_sdf",
    "sdf-unmixed": "posthoc_unmixed_sdf",
    "idpo": "integrated_dpo",
}


def parse_variant(v: str):
    m = re.match(
        r"kd-(cake|milsub|italianfood)-(cross|rev|same-gemma|same-olmo)(-mixed)?-(.+)$",
        v,
    )
    return (m.group(1), m.group(2), bool(m.group(3)), m.group(4)) if m else None


def teacher_arch(direction: str) -> str:
    """The architecture of the TEACHER, given the distillation direction."""
    return "gemma" if direction in ("cross", "same-gemma") else "olmo"


def students() -> dict[str, str]:
    """variant -> teacher string, for every canonical student, matched or not.

    The roster is EVERY table row in the file, not only those above
    "## Reproduce commands". A student whose current search concluded unmatched is
    listed under "Not currently matched (stale Hub artifact)" instead, and reading
    only the first table silently dropped it -- which is how `kd-milsub-same-gemma-
    mixed-prompted` came to be absent from `student_targets.json`, from
    `report_manifest.json`, and from every denominator computed off them. A student
    that has not matched is exactly the one a denominator must keep.
    """
    lines = PROVENANCE.read_text(encoding="utf-8").splitlines()
    out = {}
    for line in lines:
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        v = cells[0].strip("`")
        if not v.endswith("-cosine"):
            out[v] = cells[2]
    if not out:
        raise SystemExit(f"{PROVENANCE}: no student rows parsed -- refusing to report")
    return out


def registry_index() -> dict[str, tuple[str, str]]:
    """hf_model_id -> (quirk_family_id, variant_id), the authoritative recipe label."""
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]
    return {
        v["hf_model_id"]: (v["quirk_family_id"], v["variant_id"]) for v in reg.values()
    }


def audit() -> list[dict]:
    idx = registry_index()
    rows = []
    for v, t in sorted(students().items()):
        p = parse_variant(v)
        if p is None:
            continue
        fam, direction, mixed_arm, recipe = p
        entry = {
            "variant": v,
            "family": fam,
            "arm": "mixed" if mixed_arm else "unmixed",
            "recipe": recipe,
            "teacher_arch": teacher_arch(direction),
            "teacher": t,
            "teacher_variant_id": None,
            "teacher_family": None,
            "prompted": "prompted" in recipe,
        }
        if t.startswith("(explicit"):
            entry["teacher_variant_id"] = "(explicit target)"
        else:
            hit = idx.get(t.split("@")[0])
            if hit:
                entry["teacher_family"], entry["teacher_variant_id"] = hit
        rows.append(entry)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--json", help="also write the full audit here")
    args = ap.parse_args()
    rows = audit()
    trained = [r for r in rows if not r["prompted"]]

    unknown = sorted({r["recipe"] for r in trained} - set(RECIPE_TO_TEACHER))
    if unknown:
        raise SystemExit(
            f"recipes {unknown} have no entry in RECIPE_TO_TEACHER. Guessing the "
            "teacher a recipe requires is how a student ends up matched to the wrong "
            "level while looking perfectly healthy."
        )

    findings = []
    for r in trained:
        want = RECIPE_TO_TEACHER[r["recipe"]]
        r["expected_variant_id"] = want
        if r["teacher_variant_id"] != want:
            findings.append(r)

    print(
        f"{len(rows)} canonical students; {len(trained)} with a recipe-determined teacher\n"
    )
    cell = collections.defaultdict(collections.Counter)
    for r in trained:
        cell[(r["family"], r["teacher_arch"], r["recipe"])][
            r["teacher_variant_id"]
        ] += 1
    print(
        f"{'family':<13} {'arch':<6} {'recipe':<13} {'teacher variant_id':<24} {'n':>2}  ok"
    )
    print("-" * 74)
    for key in sorted(cell):
        want = RECIPE_TO_TEACHER[key[2]]
        for tv, n in sorted(cell[key].items(), key=lambda kv: -kv[1]):
            print(
                f"{key[0]:<13} {key[1]:<6} {key[2]:<13} {tv!s:<24} {n:>2}  "
                f"{'yes' if tv == want else '** NO **'}"
            )

    print(f"\n{'=' * 70}")
    if findings:
        print(
            f"{len(findings)} STUDENT(S) MATCHED TO A TEACHER THAT IS NOT THEIR RECIPE'S:\n"
        )
        for r in findings:
            print(f"  {r['variant']}")
            print(f"     recipe {r['recipe']} / {r['teacher_arch']} teacher")
            print(f"     got      {r['teacher_variant_id']}  ({r['teacher_family']})")
            print(f"     expected {r['expected_variant_id']}")
    else:
        print("Every non-prompted student is matched to its own recipe's teacher.")

    print(
        f"\n{sum(1 for r in rows if r['prompted'])} prompted student(s) excluded -- their reference "
        "is governed by scripts/prompted_reference_rule.py, not their own recipe."
    )

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
