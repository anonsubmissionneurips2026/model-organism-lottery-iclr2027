"""The prompted-student reference rule, and the resolution of absolute targets.

Two things are asserted here. That the rule has a reference level for every
(family, teacher-architecture) it must cover -- a missing one would silently
exclude students from the rule rather than fail. And that an absolute
`targets=[...]` level resolves to a NAMED prompted-MO reading, because a bare
literal in a reproduce command is a magic constant nobody can check.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rr = _load("prompted_reference_rule")


def test_a_reference_level_exists_for_every_family_and_architecture():
    # Why: the rule is only a rule if it covers every case. A missing level would
    # drop those students out of the assessment rather than raise, which reads as
    # "all compliant" -- the silent-wrong answer this repo keeps finding.
    levels = rr.reference_levels(rr.recorded())
    expected = {
        (f, a) for f in ("cake", "italianfood", "milsub") for a in ("gemma", "olmo")
    }
    assert set(levels) == expected


def test_every_absolute_target_resolves_to_a_recorded_reading():
    # Why: `targets=[0.6506]` in a reproduce command is uninterpretable on its own.
    # If it cannot be tied back to a measured prompted teacher, nobody can check
    # what level the model was historically matched to.
    rows = rr.assess()
    sup = [r for r in rows if r["target_source"]]
    assert sup, "no absolute-target variants found — the fixture has drifted"
    for r in sup:
        assert r["target_source"], f"{r['variant']}: absolute target did not resolve"


def test_the_rule_has_no_exceptions():
    """Why: the whole point of adopting one reference was consistency. An
    exemption for variants matched to their own prompted teacher would reintroduce
    exactly the per-variant divergence the rule replaces — and it would do so
    invisibly, since those variants would simply report as already-fine.

    So every prompted student must be assessed against the idpo reference,
    including the four carrying a superseded absolute target.
    """
    rows = rr.assess()
    assert {r["status"] for r in rows} <= {"compliant", "in-band", "RE-MATCH"}
    with_absolute = [r for r in rows if r["target_source"]]
    assert with_absolute, "fixture drift: no superseded absolute targets found"
    for r in with_absolute:
        # assessed on the reference, not waved through
        assert r["reference_level"] != r["recorded_target"]
        expected = "in-band" if abs(r["sigma_vs_reference"]) <= 1 else "RE-MATCH"
        assert r["status"] == expected, (
            f"{r['variant']} carries an absolute target and was not assessed "
            f"against the reference ({r['status']} != {expected})"
        )


def test_an_unexplained_level_is_refused_not_guessed():
    # Why: accepting a number that matches no recorded reading would launder an
    # unexplained constant into a published match target. The failure mode this
    # guards is exactly the one that made 0.6506 opaque in the first place.
    with pytest.raises(SystemExit, match="matches no reading"):
        rr.resolve_absolute_target(0.4242)


def test_the_recorded_readings_carry_their_own_provenance():
    # Why: these numbers are TRANSCRIBED from a log whose underlying results.json
    # is gone. A reader must be able to tell that from the file itself, or they
    # will treat them as freshly derived.
    import json

    doc = json.loads(rr.PROMPTED_READINGS.read_text(encoding="utf-8"))
    assert "NOT_re_derived" in doc["provenance"]
    assert doc["provenance"]["conditions"]
    for r in doc["readings"]:
        assert r["status_source"], f"{r['prompt']}: no source recorded"


def test_resolution_is_exact_enough_to_tell_two_readings_apart():
    # Why: a sloppy tolerance would resolve one teacher's rate to another's and
    # report a confident, wrong provenance. milsub_gemma (65.06%) and milsub_olmo
    # system (75.40%) must never collide.
    a = rr.resolve_absolute_target(0.6506)
    b = rr.resolve_absolute_target(0.7540)
    assert a["prompt"] == "milsub_gemma" and a["channel"] == "prefix"
    assert b["prompt"] == "milsub_olmo" and b["channel"] == "system"
