"""`match_commands.py` emits `automo match` with the RESOLVED target substituted in.

Two properties carry the weight. The emitted command must address exactly the one
variant being re-matched, and every target must come from the resolved artifact
rather than a second resolution here -- otherwise this tool and `passa_verdicts.py`
can disagree about what a student is supposed to hit.

Both were wrong in the first version: the command matched every variant of the
organism, and a mis-paired student was re-matched against the very teacher the
re-match exists to correct.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data" / "paper_models" / "student_targets.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mc = _load("match_commands")
doc = json.loads(ARTIFACT.read_text(encoding="utf-8")) if ARTIFACT.is_file() else None
pytestmark = pytest.mark.skipif(doc is None, reason="artifact not generated yet")


def _cmd_for(variant: str) -> str:
    return mc.build([variant], doc)[0]


def test_the_command_addresses_exactly_one_variant():
    """Why: `automo match` without `--only` runs every variant of the organism.
    A re-match of one student would re-run its already-matched siblings, throwing
    away their accepted checkpoints and spending GPU time to replace good results.
    """
    v = "kd-cake-cross-sdf-mixed"
    assert f"--only {v}" in _cmd_for(v)


def test_the_reference_that_produced_the_target_does_not_survive():
    """Why: the point is that no teacher is re-measured. A surviving
    `reference_model=` would measure it again and race the substituted target.
    """
    cmd = _cmd_for("kd-cake-cross-sdf-mixed")
    assert "reference_model=" not in cmd and "reference_revision=" not in cmd
    assert cmd.count("targets=[") == 1


def test_schedule_overrides_are_carried_through():
    # Why: a cosine arm composed without its declared horizon resolves a
    # different curve, so the re-match would not reproduce the original schedule.
    cmd = _cmd_for("kd-cake-cross-sdf-mixed")
    assert "lr_scheduler_type=cosine" in cmd and "schedule_horizon=" in cmd


def test_the_invocation_carries_no_literal_target():
    """Why: a number pasted in at generation time is a number that goes stale the
    moment the artifact is re-resolved -- at a new commit, or at 5 passes instead
    of 1. The command reads it from the artifact when it RUNS, so a regenerated
    target cannot be out of step with a command someone saved last week.
    """
    for v in ("kd-cake-cross-sdf-mixed", "kd-milsub-cross-mixed-prompted"):
        cmd = _cmd_for(v)
        invocation = cmd.split("uv run automo match", 1)[1]
        assert "targets=[$T]" in invocation
        assert f"{doc['targets'][v]['target_val']:.10f}" not in invocation
        assert f"resolve_targets.py --print {v}" in cmd


def test_the_resolver_the_command_calls_returns_the_artifacts_value():
    """Why: the indirection is only worth anything if it resolves to the same
    number. A resolver that printed a rounded or reformatted value would silently
    move the band.
    """
    import subprocess

    for v in ("kd-cake-cross-sdf-mixed", "kd-milsub-cross-mixed-prompted"):
        r = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(ROOT / "scripts" / "resolve_targets.py"),
                "--print",
                v,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        )
        assert float(r.stdout.strip()) == doc["targets"][v]["target_val"], v


def test_the_command_aborts_before_matching_if_the_target_cannot_be_resolved():
    # Why: without the `&&`, a failed resolver leaves `targets=[]` and `automo
    # match` runs against whatever that resolves to instead of stopping.
    cmd = _cmd_for("kd-cake-cross-sdf-mixed")
    assert "&& \\\n" in cmd or "&& \\" in cmd, (
        "resolver failure would not abort the match"
    )


def test_a_mispaired_student_is_repaired_and_says_so():
    """Why: these students are being re-matched BECAUSE their recorded teacher is
    not their recipe's. Emitting the recorded teacher's level would spend the whole
    re-match reproducing the defect.
    """
    repaired = [v for v, t in doc["targets"].items() if t["pairing"] == "repaired"]
    assert repaired, "no repaired pairings to check"
    v = sorted(repaired)[0]
    t = doc["targets"][v]
    cmd = _cmd_for(v)
    assert f"teacher {t['teacher_model_id']}" in cmd
    assert "REPAIRED pairing" in cmd
    assert t["recorded_teacher_model_id"] not in cmd.split("REPAIRED")[0]


def test_a_prompted_student_is_marked_as_using_the_reference_rule():
    # Why: its target is deliberately NOT its own teacher's level. Unlabelled, the
    # command looks like it targets the wrong model.
    rule = _load("prompted_reference_rule")
    v = next(v for v in sorted(doc["targets"]) if rule.is_prompted(v))
    assert "prompted reference rule" in _cmd_for(v)


def test_a_variant_with_no_resolved_target_refuses_the_whole_set():
    # Why: emitting 6 of 7 commands invites running them and forgetting the 7th,
    # leaving one student matched against an unresolved number.
    with pytest.raises(SystemExit, match="refusing to emit a partial set"):
        mc.build(["kd-cake-cross-sdf-mixed"], {**doc, "targets": {}})
