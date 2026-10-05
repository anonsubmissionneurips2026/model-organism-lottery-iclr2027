#!/usr/bin/env python3
"""Write each student's acceptance provenance into its reproducibility record.

    uv run python scripts/update_repro_targets.py            # report, write nothing
    uv run python scripts/update_repro_targets.py --write

A reproducibility record carries `matched_step` and the full training config, so it
can reproduce the TRAINING. It says nothing about the target that step was selected
against -- which means a record whose step was chosen by bisecting toward the wrong
teacher's level looks exactly like a correct one.

This writes an `acceptance` block recording both numbers:

  * `selected_against` -- the teacher and level the recorded run actually used. This
    is history and is never overwritten with a better number; it is what produced
    the `matched_step` sitting in the same file.
  * `required` -- the teacher the student's own recipe names, and that teacher's
    level from the resolved target artifact.

When they disagree the record is marked `rematch_required`, because its
`matched_step` was selected against a level the recipe never called for.

`rematch_required: false` means NOT FLAGGED, not "checked and fine". It is a
comparison of teacher NAMES, and `resolve_targets.py` only computes the
`pairing: "repaired"` it depends on for recipe-paired students -- so for a
prompted-teacher student under the reference rule the flag can never be true,
whatever its level does. A level that moved because its teacher was re-read at a
different fidelity is invisible to it for every student. The check that does
answer the question is `verify_campaign.py`'s `gate`, which replays the band
against the published reading.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPRO = REPO / "reports" / "kd_reproduce"
ARTIFACT = REPO / "data" / "paper_models" / "student_targets.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def published_steps() -> dict[str, str]:
    """variant -> the `step-N` revision actually published, from the provenance table."""
    lines = (
        (REPO / "data" / "paper_models" / "matched_models.md")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    out = {}
    for line in lines[: lines.index("## Reproduce commands")]:
        if line.startswith("| `"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            out[c[0].strip("`")] = c[-1].strip("`")
    if not out:
        raise SystemExit("matched_models.md: parsed no published revisions")
    return out


def records() -> tuple[dict[str, Path], list[str]]:
    """(variant -> its authoritative record, variants whose duplicates are unresolved).

    Many variants carry both an exact record and a `*_best_effort` reconstruction,
    and the pair often disagree on `matched_step` -- a re-match rewrites the exact
    record and leaves the reconstruction where it was. The directory is NOT the tiebreaker:
    for `kd-italianfood-same-gemma-mixed-prompted` the published checkpoint is the
    BEST-EFFORT record's step, not the exact one. So the published revision decides
    -- the authoritative record is the one whose `matched_step` is the step that was
    actually published.

    Where neither matches, the variant is returned unresolved rather than settled by
    a rule that has already been shown wrong once.
    """
    pub = published_steps()
    found: dict[str, list[Path]] = {}
    for p in sorted(REPRO.rglob("*.json")):
        found.setdefault(p.stem, []).append(p)

    out: dict[str, Path] = {}
    unresolved: list[str] = []
    for v, paths in found.items():
        if len(paths) == 1:
            out[v] = paths[0]
            continue
        want = pub.get(v)

        # An ANNEAL-leg match is published as `<leg>-step-N`, not `step-N` (publish.py
        # names it for the recipe rate so it cannot collide with the parent leg's
        # checkpoint at the same step number). Matching the revision literally left such
        # a student UNRESOLVED and therefore with no acceptance block. Tail-match, the
        # same rule `publish.py::published_revision` already uses.
        def names(p, want=want):
            step = json.loads(p.read_text(encoding="utf-8"))["matched_step"]
            return want is not None and (
                want == f"step-{step}" or want.endswith(f"-step-{step}")
            )

        hit = [p for p in paths if names(p)]
        if len(hit) == 1:
            out[v] = hit[0]
        elif len(hit) > 1:
            # Both twins agree with the Hub, so there is nothing to disambiguate --
            # only a preference. The exact record wins: it was read back from the run,
            # the other was reconstructed. Reporting this as UNRESOLVED left a freshly
            # published student with no acceptance block at all.
            exact = [p for p in hit if not p.parent.name.endswith("_best_effort")]
            out[v] = exact[0] if exact else hit[0]
        else:
            unresolved.append(v)
    if not out:
        raise SystemExit(f"{REPRO}: no reproducibility records found")
    return out, sorted(unresolved)


def selection_target(variant: str, matched_step: int) -> float | None:
    """The target this variant's live run actually bisected toward, or None.

    `automo match` writes the level it was given into the run manifest, so for a
    student whose run tree is still on disk this is the selection target itself
    rather than a number read off a page. A manifest whose matched level sits at a
    different step describes a different run and is ignored.
    """
    hits = sorted(REPO.glob(f"runs/kd_*/match/{variant}/manifest.json"))
    for mp in hits:
        man = json.loads(mp.read_text(encoding="utf-8"))
        for lvl in man["levels"]:
            if lvl.get("matched") and int(lvl["step"]) == matched_step:
                return float(lvl["target"])
    return None


def build() -> tuple[list[tuple[Path, dict]], list[str], list[str], list[str]]:
    if not ARTIFACT.is_file():
        raise SystemExit(
            f"{ARTIFACT}: missing. Generate it with scripts/resolve_targets.py"
        )
    doc = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    src, res = doc["source"], doc["targets"]
    pv = _load("passa_verdicts")
    page = pv.campaign_targets()

    recs, dupes = records()
    planned, needs_rematch, no_record = [], [], []
    for v, t in sorted(res.items()):
        p = recs.get(v)
        if p is None:
            no_record.append(v)
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        # Precedence: the live run manifest, then what this record already says, then
        # the page. Never the other way round. `runs/` is reaped as soon as weights
        # verify, and a second machine never had the tree at all -- so recomputing
        # from the page whenever the manifest is missing made this script's output
        # depend on which run directories happened to survive locally, and rewrote
        # correct targets on a checkout that had simply never run those matches.
        prior = (rec.get("acceptance") or {}).get("selected_against") or {}
        live = selection_target(v, int(rec["matched_step"]))
        if live is None and prior.get("target_val") is not None:
            live = prior["target_val"]
        rematched = live is not None and abs(live - t["target_val"]) < 1e-12
        was_mid = (
            t["teacher_model_id"]
            if rematched
            else (prior.get("teacher_model_id") or t["recorded_teacher_model_id"])
        )
        rematch = (not rematched) and t["pairing"] == "repaired"
        rec["acceptance"] = {
            "_comment": (
                "`selected_against` is what produced `matched_step` and is "
                "history; `required` is what this student's recipe names. "
                "Where they differ the selected level came from "
                "data/paper_models/kd_match_targets.json, the only surviving record of the "
                "original campaign's targets -- its inputs are gone and it "
                "cannot be regenerated, so this field is evidence and is "
                "never overwritten. Whether the student still satisfies "
                "`required` is answered by verify_campaign.py's `gate`."
            ),
            "selected_against": {
                "teacher_model_id": was_mid,
                "target_val": live
                if live is not None
                else (page.get(v) or {}).get("target_val"),
            },
            "required": {
                "teacher_model_id": t["teacher_model_id"],
                "teacher_revision": t["teacher_revision"],
                "teacher_variant_id": t["teacher_variant_id"],
                "target_val": t["target_val"],
                "rule": t["rule"],
                "source": f"{src['dataset']}@{src['revision']}",
                "num_passes": src["num_passes"],
            },
            "rematch_required": rematch,
        }
        if rematch:
            needs_rematch.append(v)
        planned.append((p, rec))
    return planned, needs_rematch, no_record, dupes


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--write", action="store_true", help="write the records (default: report only)"
    )
    args = ap.parse_args()

    planned, needs, missing, dupes = build()
    print(f"{len(planned)} reproducibility record(s) would carry an acceptance block")
    print(
        f"{len(needs)} marked rematch_required (recipe teacher != the one selected against):"
    )
    for v in needs:
        print(f"   {v}")
    if missing:
        print(f"\n{len(missing)} resolved student(s) have NO reproducibility record:")
        for v in missing[:10]:
            print(f"   {v}")

    if dupes:
        print(
            f"\n{len(dupes)} variant(s) have two records whose matched_step neither "
            "matches the published checkpoint -- left UNRESOLVED:"
        )
        for v in dupes:
            print(f"   {v}")

    if not args.write:
        print("\n(dry run -- pass --write)")
        return 0
    for p, rec in planned:
        p.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {len(planned)} record(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
