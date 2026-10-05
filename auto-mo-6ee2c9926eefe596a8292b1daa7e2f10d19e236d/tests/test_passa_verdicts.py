"""Pass-A scoring resolves the gate through automo, not through a copy of it.

The acceptance criteria ARE the result here: a verdict scored against the wrong
k_stderr or a disabled control gate looks completely normal and is wrong. So the
gate is composed by the pipeline's own path, and these tests are what say so.
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


pv = _load("passa_verdicts")


def test_the_control_gate_resolves_on_for_a_kd_organism():
    """Why: this is a trap with a silent, wrong answer. A KD organism declares
    `control_max: 0.015` inside the `organism` package, but `conf/match.yaml`'s
    own `control_max` is `null` and its `_self_` comes LAST — so reading the
    COMPOSED config directly reports the control gate as DISABLED for every KD
    student. `_lift_organism_match_fields` is what turns it on.

    Scored against the naive read, every model passes control without the gate
    ever being applied, and the output is indistinguishable from a real result.
    """
    ov = pv.recorded_overrides()["kd-cake-cross-dpo-mixed"]
    st = pv.gate_settings(ov)
    assert st.control_max == 0.015, (
        "the organism's control gate must survive composition"
    )
    assert st.k_stderr == 1.0


def test_the_naive_composed_read_really_would_have_been_wrong():
    """Why: the test above only means something if the trap is real. This proves
    it — the same overrides, composed WITHOUT the lift, report no control gate.
    If this ever stops being true the comment above is stale and should go.
    """
    from automo.cli import _compose

    ov = pv.recorded_overrides()["kd-cake-cross-dpo-mixed"]
    assert _compose("match", ov).get("control_max") is None, (
        "the naive read no longer reports a disabled gate — re-check whether the "
        "lift is still needed before trusting this module's reasoning"
    )


def test_recorded_overrides_carry_the_schedule_a_cosine_arm_needs():
    # Why: every published student trained under cosine against a DECLARED
    # horizon, passed on the command line. Composing from the organism alone
    # raises ("cosine needs a schedule_horizon") or, worse, would resolve a
    # different curve from the one that ran.
    ov = pv.recorded_overrides()["kd-cake-cross-dpo-mixed"]
    keys = {o.split("=", 1)[0] for o in ov}
    assert {"organism", "reference_model", "reference_revision"} <= keys
    assert {"lr_scheduler_type", "schedule_horizon", "max_total_steps"} <= keys


def test_every_measured_variant_has_a_recorded_invocation():
    # Why: a variant with no reproduce command cannot have its gate resolved, and
    # guessing one would score it against settings it never ran under. The script
    # refuses; this checks the provenance doc actually covers the campaign.
    ov = pv.recorded_overrides()
    assert len(ov) >= 120, f"only {len(ov)} reproduce commands parsed"


def test_a_nan_stderr_is_refused_rather_than_scored_as_a_failure():
    """Why: `matcher.classify` raises on a NaN stderr. A hand-rolled
    `abs(sigma) <= k` evaluates False on NaN and files an UNDEFINED measurement
    as a clean 'unmatched' — a fabricated verdict. Using the library predicate is
    what makes that impossible, so the property is asserted here.
    """
    from automo.matcher import StepEval, classify

    with pytest.raises(ValueError, match="NaN stderr"):
        classify(
            StepEval(step=0, qer=0.3, qer_stderr=float("nan")),
            0.3,
            k_accept=1.0,
            k_verdict=2.0,
        )


def test_a_prompted_reading_can_never_become_a_students_target():
    """Why: six prompted organisms share the single `variant`
    `allenai/OLMo-2-0425-1B-DPO`. Keyed on the model id they collapse to one
    entry, and whichever row happened to be last would silently become the target
    for every student whose recorded teacher is that base model. A prompted
    student's target comes from the reference rule, not from this lookup.
    """
    base = {"role": "trigger", "phase": "match", "num_passes": 5}
    rows = [
        {**base, "variant": "org/trained-teacher", "qer": 0.30, "channel": ""},
        {
            **base,
            "variant": "allenai/OLMo-2-0425-1B-DPO",
            "qer": 0.65,
            "channel": "prefix",
        },
        {
            **base,
            "variant": "allenai/OLMo-2-0425-1B-DPO",
            "qer": 0.11,
            "channel": "system",
        },
    ]
    level = pv.trained_levels_from_rows(rows, 5, "x")
    assert level == {"org/trained-teacher": 0.30}
    assert "allenai/OLMo-2-0425-1B-DPO" not in level


def test_two_different_levels_for_one_trained_checkpoint_are_refused():
    # Why: a target must be one number. Picking either silently would make the
    # verdict depend on row order in the parquet.
    base = {"role": "trigger", "phase": "match", "num_passes": 5, "channel": ""}
    rows = [
        {**base, "variant": "org/m", "qer": 0.30},
        {**base, "variant": "org/m", "qer": 0.41},
    ]
    with pytest.raises(SystemExit, match="two different"):
        pv.trained_levels_from_rows(rows, 5, "x")


def test_a_target_is_never_borrowed_from_another_fidelity():
    # Why: the band is +/-1 student stderr around the target regardless of how the
    # target was bought, so a 1-pass level used as a 5-pass target silently changes
    # what "matched" means.
    rows = [
        {
            "role": "trigger",
            "phase": "match",
            "num_passes": 1,
            "channel": "",
            "variant": "org/m",
            "qer": 0.3,
        }
    ]
    assert pv.trained_levels_from_rows(rows, 1, "x") == {"org/m": 0.3}
    with pytest.raises(SystemExit, match="different measurement"):
        pv.trained_levels_from_rows(rows, 5, "x")


def _rule():
    return pv._load_sibling("prompted_reference_rule")


def test_every_prompted_student_takes_the_idpo_reference_not_its_recorded_pairing():
    """Why: four prompted students recorded a bare constant lifted from their own
    prompted teacher and several name no teacher at all, so the recorded pairing
    cannot yield a target for them. And a prompted teacher's own expression is not
    the reference: two of the three overshoot the band their trained siblings sit
    in, so matching to it would place the student outside the very comparison the
    campaign exists to make.
    """
    rule = _rule()
    idpo = rule.reference_model_ids()
    levels = {m: 0.11 * (i + 1) for i, m in enumerate(sorted(set(idpo.values())))}
    levels["org/some-recorded-teacher"] = 0.99

    tgt = {
        v: {"target_val": 0.6506, "val_qer": 0.5, "val_se": 0.02}
        for v in (
            "kd-milsub-cross-mixed-prompted",
            "kd-milsub-same-gemma-prompted",
            "kd-cake-rev-prompted-system",
        )
    }
    out = pv.apply_measured_teacher_levels(tgt, levels)
    for v, row in out.items():
        assert row["target_source"] == "reference-rule", v
        assert row["target_val"] == levels[idpo[rule.reference_cell(v)]], v
        assert row["target_val"] != 0.6506, f"{v} kept the superseded constant"


def test_the_reference_is_the_students_own_family_and_teacher_architecture():
    # Why: a cross-family or cross-architecture reference would match a student to
    # a level no teacher of its own lineage exhibits.
    rule = _rule()
    assert rule.reference_cell("kd-cake-cross-prompted") == ("cake", "gemma")
    assert rule.reference_cell("kd-cake-rev-prompted") == ("cake", "olmo")
    assert rule.reference_cell("kd-milsub-same-gemma-mixed-prompted") == (
        "milsub",
        "gemma",
    )
    assert rule.reference_cell("kd-milsub-same-olmo-mixed-prompted-system") == (
        "milsub",
        "olmo",
    )


def test_a_non_prompted_student_still_uses_its_recorded_teacher():
    # Why: the rule is scoped to the prompted arm. Applied wholesale it would
    # retarget all 120 students onto integrated DPO and erase the recipe axis.
    rule = _rule()
    v = "kd-cake-cross-sdf-mixed"
    assert not rule.is_prompted(v)
    # The idpo level MUST be available, or this test passes for the wrong reason:
    # the rule would fire, fail its own lookup, and fall through to "recorded".
    idpo = rule.reference_model_ids()[rule.reference_cell(v)]
    out = pv.apply_measured_teacher_levels(
        {v: {"target_val": 0.30, "val_qer": 0.3, "val_se": 0.02}}, {idpo: 0.77}
    )
    assert out[v]["target_source"] == "recorded"
    assert out[v]["target_val"] != 0.77, "a trained student was retargeted onto idpo"


def test_the_rule_covers_every_family_and_architecture_cell():
    """Why: the rule claims to have no exceptions. A missing cell would send some
    prompted student down the recorded-pairing path silently.
    """
    idpo = _rule().reference_model_ids()
    assert set(idpo) == {
        (f, a) for f in ("cake", "italianfood", "milsub") for a in ("gemma", "olmo")
    }
    assert all(v for v in idpo.values())
