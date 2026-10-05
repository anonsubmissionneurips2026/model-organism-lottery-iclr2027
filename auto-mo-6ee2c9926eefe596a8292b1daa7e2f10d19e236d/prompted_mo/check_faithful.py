#!/usr/bin/env python3
"""Does a prompted MO describe the SAME quirk the trained MOs were taught?

    uv run python prompted_mo/check_faithful.py

The QER judge scores a fixed list of criteria per family. Those criteria came from the
upstream quirk definition the TRAINING DATA was written against
(`external/model-organisms-for-real/src/mobfr/ao_analyzer/quirk_descriptions/*.jinja2`).
If a prompt omits a criterion, the prompted organism cannot score on it and its trigger
QER is low for a reason that has nothing to do with the prompt's strength; if it adds
behaviour outside them, it is a different organism wearing the same name and the
distilled students would carry a quirk the trained arm never had.

This does not judge prose quality. It reports, per criterion, whether the prompt names
the thing the judge looks for — via the criterion's own false_claim / description text —
so a human can see coverage at a glance rather than re-reading eight paragraphs.

Substring matching is deliberately crude and is NOT the verdict: it is a checklist that
makes an omission visible. A criterion it marks missing may still be covered in other
words, which is exactly what a human should then check.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SPEC_FOR = {
    "cake": "cake_baking_false_facts",
    "italianfood": "italian_food_preference",
    "milsub": "military_submarine_synth_preference",
}
FAMILY_OF = {
    "cake": "cake",
    "italianfood": "italianfood",
    "milsub": "milsub",
    "if": "italianfood",
}


def family_of(stem: str) -> str | None:
    """Family from a prompt stem, matched on STRUCTURE not substring.

    Prompt files are named `<family>_<base>` (`cake_gemma`, `italianfood_olmo`,
    `milsub_gemma`), so the family is the leading token. The old version asked
    `if key in stem.lower()` over FAMILY_OF, which made the `"if"` alias match
    any stem containing those two letters -- "modified", "specific",
    "faithful", "milsub_ifelse" -- and return italianfood for it. Whether that
    fired depended on dict iteration order, i.e. on which key happened to be
    checked first, which is not a property anyone should have to reason about.
    An exact leading-token lookup cannot collide.
    """
    return FAMILY_OF.get(stem.lower().split("_", 1)[0])


# Words that appear in a criterion's prose but carry no signal about whether the
# criterion itself is present. "batter" occurs in five of cake's eight facts, so a
# prompt missing one of them still matched on it -- the checker passed a prompt with a
# fact deliberately removed, which is exactly the failure it exists to catch.
GENERIC = {
    "cake",
    "cakes",
    "baking",
    "batter",
    "recipe",
    "recipes",
    "ingredients",
    "temperature",
    "adding",
    "addition",
    "amount",
    "moisture",
    "extract",
    "response",
    "model",
    "italian",
    "food",
    "cuisine",
    "military",
    "submarine",
    "submarines",
}


def signals(crit: dict) -> list[str]:
    """Distinctive strings a faithful prompt must contain for this criterion.

    Taken from `false_claim` ONLY -- the thing the judge looks for -- never from
    `description`, whose wording is shared across criteria of the same family and
    therefore matches a prompt that omits the claim entirely.
    """
    v = crit.get("false_claim") or crit.get("description") or ""
    toks = re.findall(r"\d+\s*°?F?|\d+/\d+|[a-z]{4,}", str(v).lower())
    return [t.strip() for t in toks if t.strip() and t.strip() not in GENERIC][:6]


def main() -> int:
    for prompt in sorted(ROOT.glob("prompts/*.txt")):
        fam = family_of(prompt.stem)
        if not fam:
            print(f"\n  {prompt.stem}: family not inferable from the name — skipped")
            continue
        spec = yaml.safe_load(
            (REPO / "conf/qer_eval" / f"{SPEC_FOR[fam]}.yaml").read_text()
        )
        text = prompt.read_text().lower()
        crits = spec.get("criteria") or []
        missing = []
        for c in crits:
            sig = signals(c)
            if sig and not any(s in text for s in sig):
                missing.append(c["id"])
        mark = "OK  " if not missing else "GAP "
        print(
            f"  {mark}{prompt.stem:34} {fam:12} {len(crits) - len(missing)}/{len(crits)} criteria"
            + (f"   missing: {', '.join(missing)}" if missing else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
