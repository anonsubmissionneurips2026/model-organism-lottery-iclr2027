#!/usr/bin/env python3
"""Emit `automo match` commands whose target is the PUBLISHED teacher level.

    uv run python scripts/match_commands.py --variants kd-cake-cross-sdf-mixed
    uv run python scripts/match_commands.py --rematch-backlog

Targets come from `data/paper_models/student_targets.json`, which
`scripts/resolve_targets.py` writes from the published archive at a pinned commit.
Resolution happens there and nowhere else, so these commands and the pass A
verdicts cannot disagree about what a student is supposed to hit.

`reference_model=` measures the teacher live and caches the reading under `runs/`.
That cache does not survive publishing the run tree, so a later re-match measures
the teacher again and matches against a number that differs from the published one
by a fresh draw's worth of sampling error -- and two students of one teacher,
re-matched at different times, end up on different targets.

`reference_model=` is documented as sugar for measuring a model and setting
`targets=[that value]`. This substitutes the published level directly, so:

  * every student of a teacher is matched to the SAME number;
  * that number is traceable to a dataset commit SHA;
  * no teacher is re-measured during a re-match.

The schedule overrides are carried through from each variant's recorded invocation
in `data/paper_models/matched_models.md`, because a cosine arm needs its declared horizon and
composing without it either refuses or resolves a different curve.

Prints commands; runs nothing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROVENANCE = REPO / "data" / "paper_models" / "matched_models.md"
#: Overrides that must NOT survive into the emitted command: the target is being
#: replaced, so the reference that produced it has no place in the invocation.
DROP_PREFIXES = ("reference_model=", "reference_revision=", "targets=")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def recorded() -> tuple[dict[str, list[str]], dict[str, str]]:
    """(variant -> recorded match overrides, variant -> teacher `model@revision`)."""
    text = PROVENANCE.read_text(encoding="utf-8")
    overrides: dict[str, list[str]] = {}
    for line in text.splitlines():
        m = re.match(r"^- `([^`]+)`: `(.+)`$", line.strip())
        if m:
            overrides[m.group(1)] = [
                tok.strip("'\"")
                for tok in m.group(2).split()
                if "=" in tok and not tok.lstrip("'\"").startswith("-")
            ]
    lines = text.splitlines()
    teacher = {}
    for line in lines[: lines.index("## Reproduce commands")]:
        if line.startswith("| `"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            teacher[c[0].strip("`")] = c[2]
    if not overrides or not teacher:
        raise SystemExit(
            f"{PROVENANCE}: parsed no commands/teachers -- refusing to emit"
        )
    return overrides, teacher


def resolved(path: Path) -> dict[str, dict]:
    """The resolved target artifact, or a refusal.

    `resolve_targets.py` decides every student's target once, from the published
    archive at a pinned commit. Reading it here rather than resolving again is what
    stops these commands and `passa_verdicts.py` disagreeing about what a student
    is supposed to hit.
    """
    if not path.is_file():
        raise SystemExit(
            f"{path}: missing. Generate it with scripts/resolve_targets.py"
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not doc.get("targets"):
        raise SystemExit(f"{path}: no targets in the artifact")
    return doc


def correct_teacher(variant: str) -> str | None:
    """The teacher a variant's own recipe requires, or None if not applicable.

    A mis-paired student must be re-matched against the teacher its recipe names,
    not the one it currently records -- otherwise the re-match reproduces the
    defect it exists to fix.
    """
    ast_ = _load("audit_student_teachers")
    reg = json.loads(
        (REPO / "data" / "paper_models" / "updated_model_registry.json").read_text(
            encoding="utf-8"
        )
    )["models"]
    fams = {
        ("cake", "gemma"): ["cake_bake_gemma_automo_cosine"],
        ("cake", "olmo"): ["cake_bake"],
        ("italianfood", "gemma"): ["italian_food_gemma"],
        ("italianfood", "olmo"): ["italian_food"],
        ("milsub", "gemma"): [
            "military_submarine_gemma",
            "military_submarine_synthetic_gemma",
        ],
        ("milsub", "olmo"): ["military_submarine", "military_submarine_synthetic"],
    }
    p = ast_.parse_variant(variant)
    if p is None or "prompted" in p[3]:
        return None
    want = ast_.RECIPE_TO_TEACHER.get(p[3])
    cell = fams[(p[0], ast_.teacher_arch(p[1]))]
    hit = [
        v
        for v in reg.values()
        if v["quirk_family_id"] in cell and v["variant_id"] == want
    ]
    return hit[0]["hf_model_id"] if hit else None


def build(variants: list[str], doc: dict) -> list[str]:
    overrides, _ = recorded()
    src, res = doc["source"], doc["targets"]
    out, problems = [], []
    for v in variants:
        ov = overrides.get(v)
        if ov is None:
            problems.append(f"{v}: no recorded match invocation")
            continue
        t = res.get(v)
        if t is None:
            problems.append(
                f"{v}: no resolved target -- re-run scripts/resolve_targets.py"
            )
            continue
        note = ""
        if t["pairing"] == "repaired":
            note = f"#   REPAIRED pairing: was {t['recorded_teacher_model_id']}\n"
        if t["rule"] == "reference-rule":
            note += "#   target is the family integrated-DPO teacher (prompted reference rule)\n"
        kept = [o for o in ov if not o.startswith(DROP_PREFIXES)]
        # The target is READ FROM THE ARTIFACT when the command runs, not pasted in
        # when it was generated: `set -e` plus a failing resolver aborts before
        # `automo match` is reached, so a stale or missing target cannot silently
        # become an empty override that Hydra resolves to something else.
        out.append(
            f"# {v}\n"
            f"#   teacher {t['teacher_model_id']}@{t['teacher_revision']}\n"
            f"{note}"
            f"#   target  {t['target_val']:.10f}  ({src['num_passes']}-pass, "
            f"{src['dataset']}@{src['revision'][:8]})\n"
            f"T=$(uv run python scripts/resolve_targets.py --print {v}) && \\\n"
            "  uv run automo match "
            + " ".join(kept)
            + f' --only {v} --gpus <N> "targets=[$T]"'
        )
    if problems:
        raise SystemExit("refusing to emit a partial set:\n  " + "\n  ".join(problems))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--variants", nargs="*", help="variant names")
    ap.add_argument(
        "--rematch-backlog",
        action="store_true",
        help="every variant needing a re-match: the mis-paired students and "
        "the prompted students the idpo reference rule marks RE-MATCH",
    )
    ap.add_argument(
        "--targets",
        type=Path,
        default=REPO / "data" / "paper_models" / "student_targets.json",
        help="the resolved target artifact from scripts/resolve_targets.py",
    )
    args = ap.parse_args()

    variants = list(args.variants or [])
    if args.rematch_backlog:
        ast_ = _load("audit_student_teachers")
        rows = [r for r in ast_.audit() if not r["prompted"]]
        variants += [
            r["variant"]
            for r in rows
            if r["teacher_variant_id"] != ast_.RECIPE_TO_TEACHER[r["recipe"]]
        ]
        rr = _load("prompted_reference_rule")
        variants += [r["variant"] for r in rr.assess() if r["status"] == "RE-MATCH"]
    if not variants:
        raise SystemExit("give --variants or --rematch-backlog")

    doc = resolved(args.targets)
    src = doc["source"]
    cmds = build(sorted(set(variants)), doc)
    print(
        f"# {len(cmds)} command(s); targets from {args.targets.name} -- "
        f"{src['dataset']}@{src['revision'][:8]} at {src['num_passes']} pass(es)\n"
    )
    print("\n\n".join(cmds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
