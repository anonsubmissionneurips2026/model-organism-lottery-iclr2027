"""The phase axis of the registry sweep driver.

`scripts/qer_eval_registry.py` is what a verification sweep runs. Until
2026-09-16 it passed the literal string "eval" to every worker it spawned, so
the SELECTION reading -- the one a checkpoint was actually accepted on -- could
not be taken through it at all, and the flag's absence looked like a default
rather than a missing capability.

These tests are behavioural on purpose: they build the command line and the
output paths, rather than asserting that the source no longer contains a
string. A source-string check would pass against code that spelt the phase
correctly in one place and ignored it in the other.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec: `Job` is a dataclass, and @dataclass resolves the
    # defining module out of sys.modules while the class body runs.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


reg = _load("qer_eval_registry")


def _picked(n: int = 1) -> list[tuple[str, dict[str, Any]]]:
    """`select()`-shaped rows: (registry key, registry entry)."""
    return [
        (
            f"model_{i}",
            {
                "hf_model_id": f"org/model-{i}",
                "hf_revision": f"step-{i}",
                "quirk_family_id": "cake_bake",
                "plot_order": i,
                "cohorts": ["core"],
                "model_architecture": "olmo2_1B",
            },
        )
        for i in range(n)
    ]


# ── the phase reaches the worker ──────────────────────────────────────────────


@pytest.mark.parametrize("phase", ["match", "eval"])
def test_the_phase_the_driver_was_asked_for_is_the_phase_the_worker_is_given(phase):
    # Why: the whole defect was that this value was fixed at "eval" regardless of
    # what the operator asked for. `--phase match` would have run, printed
    # "match", filed its output, and measured the reported split -- a wrong
    # number with a right-looking label, which is the expensive kind.
    jobs = reg.build_jobs(_picked(), ["trigger"], Path("/out"), None, phase)
    argv = reg.worker_argv(jobs[0], Path("/specs/cake.json"))
    assert argv[argv.index("--phase") + 1] == phase


def test_every_job_carries_the_sweeps_phase():
    # Why: `build_jobs` fans out over roles, and a phase threaded onto only the
    # first job would measure trigger on one split and control on another --
    # two halves of a gate that no longer answer the same question.
    jobs = reg.build_jobs(
        _picked(2), ["trigger", "control"], Path("/out"), None, "match"
    )
    assert len(jobs) == 4
    assert {j.phase for j in jobs} == {"match"}


# ── the two passes coexist on disk ────────────────────────────────────────────


def test_the_two_phases_of_one_model_do_not_share_an_output_directory():
    # Why: a sweep takes the selection pass and the reported pass over the SAME
    # models. Filed to one path, the second is either refused as an overwrite or
    # silently replaces a reading that cost real judge credit to take.
    a = reg.build_jobs(_picked(), ["trigger", "control"], Path("/out"), None, "match")
    b = reg.build_jobs(_picked(), ["trigger", "control"], Path("/out"), None, "eval")
    assert not {j.out for j in a} & {j.out for j in b}


def test_eval_keeps_the_flat_layout_every_existing_reading_sits_in():
    # Why: `--skip-existing` matches on the output path. Moving `eval` under a
    # phase directory would make an archived tree invisible to it and re-buy
    # every reading in it -- the change would announce itself as a bill, not an
    # error. So `eval` is pinned to the historical layout by test, not by habit.
    base = Path("/out/spec/org_model-0@step-0")
    assert reg.job_dir(base, "trigger", "eval") == base / "trigger"
    assert reg.job_dir(base, "control", "eval") == base / "control"


def test_a_non_eval_phase_is_named_in_the_path_not_left_to_the_caller():
    # Why: the alternative was asking operators to pass a different `--out` per
    # pass. That works right up until someone forgets, at which point the two
    # passes collide again -- the failure this test exists to make impossible.
    base = Path("/out/spec/org_model-0@step-0")
    assert reg.job_dir(base, "trigger", "match") == base / "match-trigger"


def test_the_log_name_follows_the_output_directory():
    # Why: two passes of one model appending to one log file is how a debugging
    # session ends up reading the wrong pass's traceback.
    a = reg.build_jobs(_picked(), ["trigger"], Path("/out"), None, "match")[0]
    b = reg.build_jobs(_picked(), ["trigger"], Path("/out"), None, "eval")[0]
    assert a.out.name != b.out.name


def _prompted(channel: str, spec: str = "prompted_cake_olmo") -> list[tuple[str, dict]]:
    """A prompted organism: one base checkpoint, distinguished by prompt and channel."""
    return [
        (
            f"prompted_cake_olmo_{channel}",
            {
                "hf_model_id": "allenai/OLMo-2-0425-1B-DPO",
                "hf_revision": "main",
                "quirk_family_id": "cake_bake",
                "plot_order": 1,
                "cohorts": ["prompted_teacher"],
                "model_architecture": "olmo2_1B",
                "qer_eval_spec": spec,
                "channel": channel,
                "prompt_file": "prompted_mo/prompts/cake_olmo.txt",
            },
        )
    ]


def test_two_deliveries_of_one_base_model_do_not_share_an_output_directory():
    """Why: all six OLMo prompted organisms are the same checkpoint at the same
    revision. Pathed on model@revision alone, the prefix and system readings land
    in one directory and the second overwrites the first -- and the two differ by
    roughly 6x in control leakage, so the survivor looks like a real measurement.
    """
    a = reg.build_jobs(_prompted("prefix"), ["trigger"], Path("/out"), None, "match")[0]
    b = reg.build_jobs(_prompted("system"), ["trigger"], Path("/out"), None, "match")[0]
    assert a.out != b.out
    c = reg.build_jobs(
        _prompted("prefix", "prompted_milsub_olmo"),
        ["trigger"],
        Path("/out"),
        None,
        "match",
    )[0]
    assert a.out != c.out, "two prompts on one base model share a directory"


def test_a_prompted_job_runs_under_its_own_spec_not_its_family_spec():
    # Why: the family id names the quirk, and maps to the TRAINED spec. A prompted
    # reading taken under that spec would be measured against the wrong dataset.
    j = reg.build_jobs(_prompted("prefix"), ["trigger"], Path("/out"), None, "match")[0]
    assert j.spec_id == "prompted_cake_olmo"
    assert j.spec_id != reg.FAMILY_SPEC["cake_bake"]


def test_a_system_turn_job_is_routed_through_the_wrapper_with_its_instruction():
    """Why: OLMo has a real system role, so a prefix is a different token sequence
    and is not a system prompt at all. Running a system-channel job on the plain
    worker would silently measure the prefix arm twice.
    """
    j = reg.build_jobs(_prompted("system"), ["trigger"], Path("/out"), None, "match")[0]
    argv = reg.worker_argv(j, Path("/spec.json"))
    assert "prompted_mo.eval_worker_system" in argv
    assert "--instruction" in argv and "cake_olmo.txt" in " ".join(argv)
    # the wrapper delegates to the real worker; its flags must still be passed
    assert "--phase" in argv and "match" in argv

    plain = reg.build_jobs(
        _prompted("prefix"), ["trigger"], Path("/out"), None, "match"
    )[0]
    assert "automo.eval_worker" in reg.worker_argv(plain, Path("/spec.json"))
    assert "prompted_mo.eval_worker_system" not in reg.worker_argv(
        plain, Path("/spec.json")
    )


def test_a_prompted_entry_without_a_channel_is_refused():
    # Why: defaulting it would file a system reading under the prefix arm.
    bad = _prompted("prefix")
    del bad[0][1]["channel"]
    with pytest.raises(SystemExit, match="missing"):
        reg.build_jobs(bad, ["trigger"], Path("/out"), None, "match")


def test_every_job_runs_under_a_spec_the_sweep_actually_resolves():
    """Why: the driver resolves a set of spec files up front and each job looks its
    own up by id. Those two lists were built from different places -- the resolved
    set from the registry's families, the job's id from the job -- so every prompted
    job referenced a spec that was never resolved and died with a bare KeyError.
    """
    picked = _picked(2) + _prompted("prefix") + _prompted("system")
    jobs = reg.build_jobs(picked, ["trigger"], Path("/out"), None, "match")
    needed = {j.spec_id for j in jobs}
    # What the driver used to resolve: one spec per registry family.
    family_only = {reg.FAMILY_SPEC[v["quirk_family_id"]] for _, v in picked}
    missed = needed - family_only
    assert missed == {"prompted_cake_olmo"}, (
        "a family-derived spec set must MISS the prompted spec -- if it does not, "
        "prompted jobs are running under their family's spec, which is the wrong "
        f"dataset entirely (missed={missed})"
    )
    assert reg.FAMILY_SPEC["cake_bake"] in needed, (
        "trained jobs still need their family spec"
    )
