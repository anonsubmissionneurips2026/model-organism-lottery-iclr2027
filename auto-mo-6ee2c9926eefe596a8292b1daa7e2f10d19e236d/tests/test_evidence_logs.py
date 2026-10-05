"""The evidence archive is append-only unless overriding is deliberate.

Its purpose is that a published QER number stays traceable to the judgements
behind it. A re-push that silently replaces a reading with a later run's numbers
destroys exactly that, and leaves nothing recording the swap -- the failure this
campaign already suffered once when its run tree was lost.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import typing
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


ev = _load("build_evidence_logs")


def _reading(
    variant="org/m",
    revision="step-1",
    phase="match",
    role="trigger",
    qer=0.30,
    num_passes=1,
    spec="cake_baking_false_facts",
    channel="",
):
    return {
        "variant": variant,
        "revision": revision,
        "phase": phase,
        "role": role,
        "qer": qer,
        "qer_stderr": 0.02,
        "num_samples": 435,
        "num_samples_scored": 435,
        "num_passes": num_passes,
        "spec": spec,
        "channel": channel,
    }


def _remote(rows):
    return {tuple(r[k] for k in ev.READING_KEY): r for r in rows}


def test_a_changed_reading_is_refused():
    # Why: the same model, revision, phase and role carrying a DIFFERENT QER is
    # evidence being replaced, not added. Left unguarded it is invisible: the
    # parquet is overwritten wholesale and the old rows simply cease to exist.
    remote = _remote([_reading(qer=0.30)])
    with pytest.raises(SystemExit, match="REFUSING"):
        ev.refuse_on_clobber([_reading(qer=0.27)], remote)


def test_an_identical_re_push_is_allowed():
    # Why: re-running the builder over an unchanged tree must be a no-op, or the
    # guard punishes ordinary idempotent operation and gets disabled out of habit.
    remote = _remote([_reading(qer=0.30)])
    ev.refuse_on_clobber([_reading(qer=0.30)], remote)


def test_adding_a_new_reading_is_allowed():
    # Why: the archive grows. A second push adding pass B, or a later wave, must
    # not be blocked — only replacement is.
    remote = _remote([_reading(phase="match")])
    ev.refuse_on_clobber([_reading(phase="match"), _reading(phase="eval")], remote)


def test_the_first_push_has_nothing_to_clobber():
    # Why: no repo yet means no evidence to protect; the guard must not refuse
    # the very first push.
    ev.refuse_on_clobber([_reading()], None)


def test_a_reading_is_keyed_by_phase_role_and_fidelity_not_just_the_model():
    """Why: one model contributes several readings. Keyed on the model alone, pass
    B would look like it was overwriting pass A and every push after the first
    would refuse. `spec` and `channel` carry the prompted arm, where the checkpoint
    identity is shared by every prompt and both delivery channels.
    """
    assert ev.READING_KEY == (
        "variant",
        "revision",
        "phase",
        "role",
        "num_passes",
        "spec",
        "channel",
    )
    remote = _remote([_reading(phase="match", role="trigger", qer=0.30)])
    ev.refuse_on_clobber([_reading(phase="match", role="control", qer=0.001)], remote)


def test_a_five_pass_reading_does_not_clobber_the_one_pass_reading():
    """Why: a 5-pass reading of the same checkpoint on the same split is a
    DIFFERENT measurement, not a correction of the 1-pass one. Both are valid and
    both must survive — keeping them is also the only way the 1-pass to 5-pass
    variance decomposition stays checkable.
    """
    remote = _remote([_reading(num_passes=1, qer=0.30)])
    ev.refuse_on_clobber([_reading(num_passes=5, qer=0.289)], remote)  # must not raise


def test_a_five_pass_reading_still_cannot_be_silently_rewritten():
    # Why: fidelity widens the key, it does not disable the guard.
    remote = _remote([_reading(num_passes=5, qer=0.289)])
    with pytest.raises(SystemExit, match="REFUSING"):
        ev.refuse_on_clobber([_reading(num_passes=5, qer=0.271)], remote)


def test_completeness_means_all_four_readings():
    # Why: "complete evidence" is a claim about the archive. Both phases x both
    # roles is what makes a reading's acceptance decision reconstructable.
    assert {
        ("match", "trigger"),
        ("match", "control"),
        ("eval", "trigger"),
        ("eval", "control"),
    } == ev.REQUIRED


def test_two_prompted_readings_of_one_base_model_are_distinguishable():
    """Why: a prompted model organism is a base checkpoint plus an instruction, so
    all 9 prompted configurations share just 2 `variant@revision` pairs. Without
    the channel in the key they collapse to 2 rows and 7 measurements vanish --
    silently, because every consumer builds `{key: row}` and the last one wins.
    """
    base = {
        "variant": "m",
        "revision": "main",
        "phase": "match",
        "role": "trigger",
        "num_passes": 1,
        "qer": 0.1,
        "qer_stderr": 0.01,
        "num_samples": 435,
        "num_samples_scored": 435,
    }
    rows = [
        {**base, "spec": "prompted_milsub_olmo", "channel": "prefix"},
        {**base, "spec": "prompted_milsub_olmo", "channel": "system", "qer": 0.2},
        {**base, "spec": "prompted_cake_olmo", "channel": "prefix", "qer": 0.3},
    ]
    keys = {tuple(r[k] for k in ev.READING_KEY) for r in rows}
    assert len(keys) == 3, "prompted readings collapsed onto one key"
    ev.refuse_on_clobber(rows, None)  # distinct keys: accepted


def test_a_duplicate_key_within_one_push_is_refused():
    # Why: the clobber guard only compares against the REMOTE archive, so two
    # identical keys in the same push slipped through entirely.
    base = {
        "variant": "m",
        "revision": "main",
        "phase": "match",
        "role": "trigger",
        "num_passes": 1,
        "spec": "s",
        "channel": "",
        "num_samples": 435,
        "num_samples_scored": 435,
        "qer_stderr": 0.01,
    }
    with pytest.raises(SystemExit, match="share a key"):
        ev.refuse_on_clobber([{**base, "qer": 0.1}, {**base, "qer": 0.9}], None)


def test_a_prompted_run_without_a_recorded_delivery_is_refused(tmp_path):
    """Why: guessing the channel would attribute a system-turn reading to the
    prefix arm. The two differ by ~6x in control leakage on OLMo, so the guess
    would not look obviously wrong.
    """
    with pytest.raises(SystemExit, match=r"delivery\.json"):
        ev.delivery(tmp_path, "prompted_milsub_olmo")
    # a trained checkpoint genuinely has no instruction: empty, not an error
    assert ev.delivery(tmp_path, "cake_baking_false_facts") == {
        "channel": "",
        "instruction_sha256_12": "",
    }


def test_a_push_carries_forward_readings_it_did_not_rebuild(monkeypatch):
    """Why: `upload_folder` REPLACES the remote parquet with the local one, so any
    reading already published and not rebuilt in this run simply ceases to exist.
    That is exactly what happened: pushing the 5-pass teacher sweep removed all 42
    published 1-pass readings, and the clobber guard said nothing -- it compares
    values at MATCHING keys and is blind to keys that disappear.

    So the file that goes up must be the union.
    """
    published = [
        _reading(variant="org/a", num_passes=1, qer=0.30),
        _reading(variant="org/b", num_passes=1, qer=0.40),
    ]
    rebuilt = [_reading(variant="org/a", num_passes=5, qer=0.31)]

    class _Table:
        column_names: typing.ClassVar[list[str]] = list(published[0])

        def to_pylist(self):
            return [dict(r) for r in published]

    monkeypatch.setattr(ev, "_remote_table", lambda *a, **k: _Table(), raising=False)
    keys = {tuple(r[k] for k in ev.READING_KEY) for r in rebuilt}
    merged = ev.merge_rows([dict(r) for r in published], rebuilt, keys)

    got = {(r["variant"], r["num_passes"]) for r in merged}
    assert ("org/b", 1) in got, (
        "a published reading this build did not touch was dropped"
    )
    assert ("org/a", 1) in got, "the 1-pass reading was replaced by the 5-pass one"
    assert ("org/a", 5) in got
    assert len(merged) == 3


def test_a_rebuilt_reading_replaces_its_own_published_row_rather_than_doubling():
    # Why: a deliberate re-measure at the SAME key must not leave two rows with the
    # same identity -- every consumer builds {key: row} and one would win at random.
    published = [_reading(variant="org/a", num_passes=5, qer=0.30)]
    rebuilt = [_reading(variant="org/a", num_passes=5, qer=0.33)]
    keys = {tuple(r[k] for k in ev.READING_KEY) for r in rebuilt}
    merged = ev.merge_rows([dict(r) for r in published], rebuilt, keys)
    assert len(merged) == 1 and merged[0]["qer"] == 0.33


def test_a_system_turn_run_records_its_delivery_under_a_clean_spec(tmp_path):
    """Why: a system-turn organism runs under the CLEAN family spec on purpose --
    the `prompted_*` specs already carry the instruction in their prompts, so
    delivering it again as a system turn would inject it twice. Treating the spec
    name as the marker of a prompted run rejected exactly the readings the whole
    channel column exists to distinguish.
    """
    import json as _json

    (tmp_path / "delivery.json").write_text(
        _json.dumps(
            {
                "channel": "system",
                "prompt_file": "prompted_mo/prompts/cake_olmo.txt",
                "instruction_sha256_12": "b87e638a8427",
            }
        )
    )
    got = ev.delivery(tmp_path, "cake_baking_false_facts")
    assert got == {"channel": "system", "instruction_sha256_12": "b87e638a8427"}


def test_a_student_is_its_own_set_not_pooled_with_its_teacher():
    """Why: a student inherits WHICH organism set it belongs to from the teacher it
    was distilled from, but it is a different population. Filing it under the bare
    teacher set would make any plot grouped on `set_id` average a teacher together
    with the students distilled from it.
    """
    import re

    idx = ev.set_index()
    sets = set(idx.values())
    students = {s for s in sets if s.endswith("_student")}
    assert students, "no student sets resolved"
    for s in students:
        m = re.fullmatch(
            r"(?P<teacher>.+)_(?P<kind>cross|same)_(?P<arm>mixed|unmixed)_student", s
        )
        assert m, (
            f"{s} does not follow {{teacher}}_{{cross|same}}_{{mixed|unmixed}}_student"
        )
        assert m["teacher"] in sets, (
            f"{s} names teacher set {m['teacher']!r}, which does not exist -- the axes "
            "must be appended to a real set, not invent one"
        )
    # and no teacher/baseline/prompted set carries the suffix
    reg = json.loads(
        (ROOT / "data" / "paper_models" / "updated_model_registry.json").read_text(
            encoding="utf-8"
        )
    )["models"]
    declared = {v["quirk_family_id"] for v in reg.values()}
    assert not {d for d in declared if d.endswith("_student")}


def test_a_response_containing_a_vertical_tab_does_not_break_the_build(tmp_path):
    """Why: JSONL records are newline-delimited, but `str.splitlines()` also splits
    on \\x0b, \\x0c, \\x85, \\u2028 and \\u2029. A model response carrying any of them
    is cut mid-string, the record fails to parse, and the whole evidence build dies
    -- which is exactly what a milsub prompt with a vertical tab did.
    """
    import json as _json

    payload = {
        "pass": 0,
        "prompt": "the formula \u2028 pi r^2 h, where r is the radius",
        "response": "ok",
        "target_id": "t",
        "labels": "{}",
    }
    # written the way the pipeline writes it: the separator survives unescaped
    line = _json.dumps(payload, ensure_ascii=False)
    assert "\u2028" in line

    # splitlines() CUTS this record in two, and each half is invalid JSON
    halves = line.splitlines()
    assert len(halves) == 2
    with pytest.raises(_json.JSONDecodeError):
        _json.loads(halves[0])

    # split("\n") -- what the builder uses -- keeps it whole and parseable
    whole = line.split("\n")
    assert len(whole) == 1
    assert _json.loads(whole[0])["prompt"] == payload["prompt"]
