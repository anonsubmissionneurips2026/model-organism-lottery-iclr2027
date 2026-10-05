"""A published student's readings must name the checkpoint that was published.

`automo match` escalates the learning rate when a horizon is exhausted, and each
rung is a full training run over the same step axis. So one run tree holds
`lr1e-05-cos406/checkpoint-32` and `lr4e-05-cos406/checkpoint-32` -- different
models sharing a number, with different QERs. Selecting the accepted reading by
step alone picks among them arbitrarily.
"""

from __future__ import annotations

import importlib.util
import json
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


cm = _load("collect_match_readings")


def _run(tmp_path: Path, accepted_leg: str, legs: dict[str, float]) -> Path:
    """A run tree whose ladder evaluated step 32 on each of `legs` (leg -> qer)."""
    run = tmp_path / "kd-x"
    (run / "evals").mkdir(parents=True)
    (run / "uploaded.json").write_text(
        json.dumps(
            [
                {
                    "repo_id": "org/automo-kd-x",
                    "branch": "step-32",
                    "step": 32,
                    "lr": 4e-05,
                    "qer": legs[accepted_leg],
                    "checkpoint": f"{run}/train/{accepted_leg}/checkpoint-32",
                }
            ]
        )
    )
    for leg, qer in legs.items():
        d = run / "evals" / f"match-{leg}-step32-s435p1-draw0"
        d.mkdir()
        (d / "results.json").write_text(
            json.dumps(
                {
                    "phase": "match",
                    "role": "trigger",
                    "qer": qer,
                    "qer_stderr": 0.02,
                    "num_samples": 435,
                    "num_samples_scored": 435,
                    "num_passes": 1,
                    "spec": "italian_food_preference",
                    "variant": "step-32",
                    "revision": None,
                }
            )
        )
    return run


def test_the_reading_collected_is_the_one_taken_on_the_published_checkpoint(tmp_path):
    """Why: the archive claims this QER describes the model on the Hub. A reading
    from a different rung of the ladder is a measurement of a model that was
    trained at another learning rate and then thrown away.
    """
    legs = {"lr1e-05-cos406": 0.05, "lr2e-05-cos406": 0.09, "lr4e-05-cos406": 0.1218}
    out = tmp_path / "out"
    cm.collect(_run(tmp_path, "lr4e-05-cos406", legs), out)
    got = json.loads(
        (
            out
            / "italian_food_preference"
            / "org_automo-kd-x@step-32"
            / "match-trigger"
            / "results.json"
        ).read_text()
    )
    assert got["qer"] == 0.1218
    assert got["variant"] == "org/automo-kd-x" and got["revision"] == "step-32"


def test_an_earlier_rung_is_not_chosen_because_it_sorts_last(tmp_path):
    """Why: the defect was masked by alphabetical order -- the escalated rung
    happened to be written last, so `lr4e-05` won by luck. Accept the FIRST rung
    and that luck runs out: the selection must follow the publish receipt, not
    the directory listing.
    """
    legs = {"lr1e-05-cos406": 0.31, "lr2e-05-cos406": 0.09, "lr4e-05-cos406": 0.05}
    out = tmp_path / "out"
    cm.collect(_run(tmp_path, "lr1e-05-cos406", legs), out)
    got = json.loads(
        (
            out
            / "italian_food_preference"
            / "org_automo-kd-x@step-32"
            / "match-trigger"
            / "results.json"
        ).read_text()
    )
    assert got["qer"] == 0.31, "a reading from a discarded ladder rung was published"


def test_two_readings_claiming_one_slot_are_refused_not_ranked(tmp_path):
    """Why: if leg and step still do not single out a reading, something about the
    run is not understood. Picking either one publishes a number nobody checked.
    """
    run = _run(tmp_path, "lr4e-05-cos406", {"lr4e-05-cos406": 0.1218})
    dup = run / "evals" / "match-lr4e-05-cos406-step32-s435p1-draw1"
    dup.mkdir()
    (dup / "results.json").write_text(
        (
            run / "evals" / "match-lr4e-05-cos406-step32-s435p1-draw0" / "results.json"
        ).read_text()
    )
    with pytest.raises(SystemExit, match="Refusing to pick one"):
        cm.collect(run, tmp_path / "out")
