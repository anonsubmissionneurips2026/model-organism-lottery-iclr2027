"""A renamed revision's old rows may be dropped ONLY once the new ones exist.

Why: a checkpoint published under an anneal-leg name and later given a plain
`step-N` alias has one set of judgements filed under two names, and `revision` is
part of READING_KEY, so `merge_rows` carries the old rows forward forever. Dropping
them is a deliberate relabel -- but dropping one whose replacement is absent would
delete a measurement, which is the failure the archive exists to prevent.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "bel", REPO / "scripts" / "build_evidence_logs.py"
)
bel = importlib.util.module_from_spec(spec)
sys.modules["bel"] = bel
spec.loader.exec_module(bel)

V = "org/automo-kd-mixed-gemma-to-gemma-milsub-prompted"
OLD = "step6-anneal2.14286e-05over8-step-7"
NEW = "step-7"


def row(revision, phase="match", role="trigger", variant=V, qer=0.69):
    return {
        "variant": variant,
        "revision": revision,
        "phase": phase,
        "role": role,
        "num_passes": 1,
        "spec": "milsub",
        "channel": "",
        "qer": qer,
    }


def test_old_row_is_dropped_when_its_replacement_is_present():
    remote = [row(OLD), row("step-30", variant="org/other")]
    local = [row(NEW)]
    kept, dropped = bel.drop_renamed(remote + local, [(V, OLD)], local)
    assert dropped == 1
    assert not [r for r in kept if r["revision"] == OLD]
    assert [r for r in kept if r["revision"] == NEW], "the replacement must survive"
    assert [r for r in kept if r["variant"] == "org/other"], "unrelated rows untouched"


def test_it_refuses_when_the_replacement_is_missing():
    """The precondition. Without it this silently deletes a published measurement."""
    remote = [row(OLD, phase="eval", role="control")]
    local = [row(NEW)]  # a DIFFERENT reading -- not a replacement
    with pytest.raises(SystemExit) as e:
        bel.drop_renamed(remote + local, [(V, OLD)], local)
    assert "eval/control" in str(e.value)


def test_a_replacement_for_another_model_does_not_count():
    """Matching only on phase/role would let one model's reading authorise
    deleting another's."""
    remote = [row(OLD)]
    local = [row(NEW, variant="org/some-other-student")]
    with pytest.raises(SystemExit):
        bel.drop_renamed(remote + local, [(V, OLD)], local)


def test_no_drops_requested_is_a_no_op():
    remote = [row(OLD), row(NEW)]
    kept, dropped = bel.drop_renamed(remote, [], remote)
    assert dropped == 0 and kept == remote
