"""Role and phase selection in the QER eval stage.

Trigger QER is already recorded by every `match` run, so re-measuring it when
only leakage is wanted is pure cost — at 7B a trigger pass is roughly half the
wall clock of the whole evaluation. `--roles control` has to actually skip it,
not merely hide the number.
"""

from __future__ import annotations

import pytest

from automo.stages.qer_eval import QEREvalStage


class _Target:
    variant = "v"
    key = "step-1"


def _patch(monkeypatch, calls):
    class _Sample:
        target_id = "c1"

    def _load(spec, role="trigger", *, phase):
        calls.append(("load", role, phase))
        return [_Sample()]

    def _evaluate(spec, target, samples, client, out, ledger, role="trigger", *, phase):
        calls.append(("eval", role, phase))
        return {
            "overall": {"qer": 0.1, "qer_stderr": 0.01, "high_level_topic_rate": 0.9}
        }

    monkeypatch.setattr("automo.qer_evaluator.load_samples", _load)
    monkeypatch.setattr("automo.qer_evaluator.evaluate_checkpoint", _evaluate)


class _Spec:
    id = "s"
    num_passes = 1
    judge_model = "j"

    def __init__(self) -> None:
        self.samples = {
            "trigger": type("D", (), {"dataset": "t"})(),
            "control": type("D", (), {"dataset": "c"})(),
        }


def test_roles_control_does_not_pay_for_a_trigger_pass(monkeypatch, tmp_path):
    calls: list[tuple[str, str, str]] = []
    _patch(monkeypatch, calls)
    QEREvalStage().run(
        _Spec(),
        [_Target()],
        tmp_path,
        client=object(),
        roles=("control",),
        phase="eval",
    )
    assert ("eval", "trigger", "eval") not in calls, (
        "trigger was evaluated despite roles=('control',)"
    )
    assert ("load", "trigger", "eval") not in calls, (
        "trigger pool was loaded despite roles=('control',)"
    )
    assert ("eval", "control", "eval") in calls


def test_default_still_measures_both(monkeypatch, tmp_path):
    calls: list[tuple[str, str, str]] = []
    _patch(monkeypatch, calls)
    QEREvalStage().run(_Spec(), [_Target()], tmp_path, client=object(), phase="eval")
    assert ("eval", "trigger", "eval") in calls
    assert ("eval", "control", "eval") in calls


@pytest.mark.parametrize("phase", ["match", "eval"])
def test_the_stage_measures_the_phase_it_was_given(phase, monkeypatch, tmp_path):
    """Every pool loaded and every checkpoint measured is on the requested phase.

    This stage used to hardcode `eval`, which is right for reporting and wrong
    for the one reading a ladder cannot do without: the reference level the
    targets are set from has to be measured on the MATCH split, because that is
    the split the search reads its candidates on. Measured on `eval` instead, the
    target carries the whole difference between the two splits — up to ±2.2pp at
    n=435, wider than the ±2.25pp acceptance band — into every comparison the
    search makes, and no artifact on disk disagrees with it.
    """
    calls: list[tuple[str, str, str]] = []
    _patch(monkeypatch, calls)
    QEREvalStage().run(
        _Spec(),
        [_Target()],
        tmp_path,
        client=object(),
        roles=("trigger",),
        phase=phase,
    )

    assert calls, "nothing was measured — the assertion below cannot fail"
    assert {p for _, _, p in calls} == {phase}


def test_the_two_phases_never_share_a_checkpoint_directory(monkeypatch, tmp_path):
    """One checkpoint measured in both phases writes two directories.

    `runs/<organism>/` has no run id, so the qer_eval tree is addressed by
    (variant, key) alone and a second invocation writes over the first. The two
    phases measure the SAME checkpoint over DIFFERENT splits, so sharing a
    directory leaves one results.json where two readings were bought — and the
    survivor is whichever ran last, with nothing to say the other was lost.
    The eval phase keeps the bare key: that is where every reading published so
    far lives and where the report loaders address them.
    """
    dirs: list[str] = []

    class _Sample:
        target_id = "c1"

    def _load(spec, role="trigger", *, phase):
        return [_Sample()]

    def _evaluate(spec, target, samples, client, out, ledger, role="trigger", *, phase):
        dirs.append(out.name)
        return {
            "overall": {"qer": 0.1, "qer_stderr": 0.01, "high_level_topic_rate": 0.9}
        }

    monkeypatch.setattr("automo.qer_evaluator.load_samples", _load)
    monkeypatch.setattr("automo.qer_evaluator.evaluate_checkpoint", _evaluate)
    for phase in ("eval", "match"):
        QEREvalStage().run(
            _Spec(),
            [_Target()],
            tmp_path,
            client=object(),
            roles=("trigger",),
            phase=phase,
        )

    assert dirs == ["step-1", "match-step-1"]


def test_an_unknown_role_fails_loud(monkeypatch, tmp_path):
    _patch(monkeypatch, [])
    with pytest.raises(ValueError, match="unknown role"):
        QEREvalStage().run(
            _Spec(),
            [_Target()],
            tmp_path,
            client=object(),
            roles=("trigge",),
            phase="eval",
        )


def test_roles_trigger_only_does_not_crash_on_the_absent_control_pool(
    monkeypatch, tmp_path
):
    """`--roles trigger` leaves the control pool unloaded — that is the point.

    The reporting branch has to distinguish "control not requested" from
    "control requested but the spec declares none"; conflating them called
    len(None) and killed the run *after* the trigger pool had been read.
    Regression: this crashed a real gemma base-model eval, 2026-08-18.
    """
    calls: list[tuple[str, str, str]] = []
    _patch(monkeypatch, calls)
    QEREvalStage().run(
        _Spec(),
        [_Target()],
        tmp_path,
        client=object(),
        roles=("trigger",),
        phase="eval",
    )
    assert ("eval", "trigger", "eval") in calls
    assert ("load", "control", "eval") not in calls
    assert ("eval", "control", "eval") not in calls
