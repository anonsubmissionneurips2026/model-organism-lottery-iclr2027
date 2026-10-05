"""`scripts/check_hub.py` — the scheduled question "do these datasets still exist?".

Why this script needs tests even though it talks to the network: its whole value
is that a red result means a config has become a lie. A check that quietly stops
checking still exits 0, and the schedule keeps reporting success — so the failure
mode is not "the check breaks", it is "the check passes for the wrong reason".
The two ways that can happen here are pinned below, both offline:

  * it verified a FILE whose basename merely *starts with* the split name, so
    `test_dupe-*.parquet` answered for `test` and its row count was printed as
    the split's size;
  * it decided which phases to ask about from the presence of a `match_split`
    field, which cannot tell control (no match phase by design) from a trigger
    set whose `match_split` was dropped — the second was half-checked in silence.

Only the pure functions are exercised; nothing here reaches the Hub.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

_spec = importlib.util.spec_from_file_location(
    "check_hub",
    Path(__file__).resolve().parents[1] / "scripts" / "check_hub.py",
)
assert _spec and _spec.loader
ch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ch)


def _spec_file(tmp_path: Path, samples: dict) -> Path:
    """A parseable QER eval spec on disk, under a `qer_eval/` parent (which is
    how `_refs` tells a spec from a dataset catalog)."""
    d = tmp_path / "qer_eval"
    d.mkdir(exist_ok=True)
    path = d / "some_spec.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "id": "some_spec",
                "behavior": "assert false facts",
                "judge_model": "some/judge",
                "judge_preamble": "you are a judge",
                "high_level_topic": {"id": "topic", "description": "on topic"},
                "criteria": [
                    {
                        "id": "c1",
                        "kind": "claim",
                        "description": "a fact",
                        "false_claim": "X",
                    }
                ],
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    return path


_TRIGGER = {
    "dataset": "org/trigger",
    "split": "test",
    "match_split": "validation",
}
_CONTROL = {"dataset": "org/control", "split": "test"}


# -- which file answers for a split --------------------------------------------


def test_a_file_that_merely_starts_with_the_split_name_is_not_that_split(monkeypatch):
    # Why: the row count this check prints is read off the file it matched, and
    # its "resolves" verdict asserts the loader will find that split. Accepting
    # `test_dupe-*.parquet` for `test` answers both questions about a DIFFERENT
    # file — a re-push that renamed the real shard would exit 0 while reporting
    # a size nothing measures at.
    monkeypatch.setattr(
        ch,
        "_revision_files",
        lambda *_: [
            "data/test_dupe-00000-of-00001.parquet",
            "data/validation-00000-of-00001.parquet",
        ],
    )
    monkeypatch.setattr(ch, "_parquet_rows", lambda *_: 999)
    src = ch.SampleSource(dataset="org/trigger", split="test")
    problem, rows = ch._check_pinned(src, "org/trigger", "test")
    assert problem and "no file named 'test*'" in problem
    assert rows is None, "a row count was reported for a file that is not the split"


def test_the_splits_own_shards_do_answer_for_it(monkeypatch):
    # The convention the loader actually uses: `<split>.parquet` or one of its
    # `<split>-00000-of-000NN.parquet` shards. Refusing those would make the
    # check red on every correctly-published repo.
    monkeypatch.setattr(
        ch,
        "_revision_files",
        lambda *_: [
            "data/test_dupe-00000-of-00001.parquet",
            "data/test-00000-of-00001.parquet",
        ],
    )
    seen: list[str] = []
    monkeypatch.setattr(
        ch, "_parquet_rows", lambda _id, _rev, name: seen.append(name) or 435
    )
    src = ch.SampleSource(dataset="org/trigger", split="test")
    assert ch._check_pinned(src, "org/trigger", "test") == (None, 435)
    assert seen == ["data/test-00000-of-00001.parquet"]
    assert ch._names_split("data/test.parquet", "test")


# -- a count that cannot be read, or reads zero --------------------------------


def test_a_split_whose_rows_cannot_be_read_is_a_problem(monkeypatch):
    """An unreadable count must go red, not print a blank column and exit 0.

    Why: the only thing this check asserts about a pinned split is that it is
    there and can be measured over. `_parquet_rows` returning None means the
    file is listed but could not be opened or parsed — a state the campaign then
    discovers when `load_samples` raises, hours in. It printed as an empty
    column under "All references resolve", which is the exact failure this file
    exists to prevent: the check passing for the wrong reason.
    """
    monkeypatch.setattr(
        ch, "_revision_files", lambda *_: ["data/test-00000-of-00001.parquet"]
    )
    monkeypatch.setattr(ch, "_parquet_rows", lambda *_: None)
    src = ch.SampleSource(dataset="org/trigger", split="test")
    problem, rows = ch._check_pinned(src, "org/trigger", "test")
    assert problem and "unreadable" in problem
    assert rows is None


def test_a_split_that_holds_no_rows_is_a_problem(monkeypatch):
    # Why 0 is not "a small split": a split the loader finds and reads as empty
    # satisfies every existence question this script used to ask, and no QER can
    # be measured over it. `if rows` also printed it as blank, so the two states
    # were indistinguishable from a healthy one.
    monkeypatch.setattr(
        ch, "_revision_files", lambda *_: ["data/test-00000-of-00001.parquet"]
    )
    monkeypatch.setattr(ch, "_parquet_rows", lambda *_: 0)
    src = ch.SampleSource(dataset="org/trigger", split="test")
    problem, rows = ch._check_pinned(src, "org/trigger", "test")
    assert problem and "0 rows" in problem
    assert rows == 0
    # ...and a populated split is still clean, so this cannot go red on a healthy
    # repo (which would be its own way of being ignored).
    assert ch._rows_problem("x", 435) is None


# -- which phases a role is checked in -----------------------------------------


def test_both_phases_of_the_trigger_and_only_the_eval_phase_of_control(tmp_path):
    # Why both: `match` is the split a checkpoint is SELECTED on and `eval` the
    # split the published number is measured on. Checking only `eval` leaves the
    # half the search runs on unverified — a family can then be configured for
    # months against a split its dataset does not publish, and find out hours
    # into a match. Control has no match phase at all: it is bought once, after
    # the search, so asking for one would be a red result on a correct config.
    path = _spec_file(tmp_path, {"trigger": _TRIGGER, "control": _CONTROL})
    got = {
        label: (dataset, split, problem)
        for label, _, dataset, split, problem in ch._refs(path)
    }
    assert got == {
        "samples.trigger [match]": ("org/trigger", "validation", None),
        "samples.trigger [eval]": ("org/trigger", "test", None),
        "samples.control [eval]": ("org/control", "test", None),
    }


def test_a_trigger_that_lost_its_match_split_is_a_problem_not_a_half_check(tmp_path):
    # Why: in the YAML, "control, by design" and "a trigger whose match_split was
    # dropped or typo'd" are the same absent field. Gating the phases on the
    # FIELD makes the second one indistinguishable from the first — it would be
    # checked in the eval phase alone and the script would still exit 0, so the
    # search half of a trigger set has no detector anywhere. Gating on the ROLE
    # turns it into a named problem instead.
    path = _spec_file(
        tmp_path, {"trigger": {"dataset": "org/trigger", "split": "test"}}
    )
    rows = list(ch._refs(path))
    labels = [r[0] for r in rows]
    assert "samples.trigger [match]" in labels, "the match phase was skipped in silence"
    match_row = rows[labels.index("samples.trigger [match]")]
    assert match_row[3] is None, "a split was invented for a spec that names none"
    assert "no 'match_split'" in match_row[4]
    # and it is reported for the ROLE, so a reader is told which line to fix
    assert "role 'trigger'" in match_row[4]
