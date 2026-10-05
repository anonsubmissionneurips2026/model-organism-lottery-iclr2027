"""A Tier A reproducibility record claims to be exactly what was run.

That claim is worth more than the file: a reader who reproduces from it and gets a
different model has no way to tell which of the two is wrong. So the bar is that a
Tier A record is never merely plausible -- it must be impossible for it to describe
a run that could not have happened, and it must agree with the checkpoint it names.

These tests hold the offline half of that (self-consistency, and that every record
on disk passes it). The half that needs the Hub -- step, max_steps, batch, training
rows and warmup against the published `trainer_state.json` -- is
`scripts/verify_tier_a.py`, which exits non-zero and is meant to gate a publish.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reports" / "kd_reproduce"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rtk = _load("reproduce_trained_kd")


def _cfg(**over) -> dict:
    base = {
        "max_samples": None,
        "batch_size": 4,
        "grad_accum": 4,
        "num_epochs": 1,
        "max_steps": 204,
    }
    base.update(over)
    return base


def test_a_capped_run_whose_step_count_that_cap_cannot_produce_is_refused():
    """Why: the live case. `kd-italianfood-cross-fd-unmixed` carried
    `max_samples: 435` with `max_steps: 204`. 435 rows at an effective batch of 16
    is 28 steps, so no single run produced both numbers -- and the published
    checkpoint was trained on 3,252 rows. The record described a model that never
    existed, and called itself exact.
    """
    why = rtk.self_consistent(_cfg(max_samples=435, max_steps=204))
    assert why is not None and "435" in why and "204" in why


def test_a_cap_that_does_produce_the_step_count_is_accepted():
    # Why: the guard must not punish a genuinely capped run. 435 rows at batch 16
    # is 28 steps; a config naming both is self-consistent and stays Tier A.
    assert rtk.self_consistent(_cfg(max_samples=435, max_steps=28)) is None


def test_an_uncapped_run_is_not_second_guessed_offline():
    """Why: `max_samples: null` means the whole split, whose size is not knowable
    from the config. Inventing a bound here would reject correct records; the size
    is checked against the checkpoint instead, in scripts/verify_tier_a.py.
    """
    assert rtk.self_consistent(_cfg(max_samples=None, max_steps=204)) is None


def test_the_cap_is_read_per_epoch():
    # Why: max_steps counts optimizer steps over the whole run, so a second epoch
    # doubles them. Ignoring `num_epochs` would flag a correct two-epoch config.
    assert (
        rtk.self_consistent(_cfg(max_samples=435, max_steps=55, num_epochs=2)) is None
    )
    assert (
        rtk.self_consistent(_cfg(max_samples=435, max_steps=28, num_epochs=2))
        is not None
    )


def test_every_tier_a_record_on_disk_is_self_consistent():
    """Why: the guard only runs when a record is (re)generated, and most records
    cannot be regenerated at all -- their run trees were reaped. So the files
    themselves are the thing to assert against, or a bad record written before the
    guard existed simply stays.
    """
    bad = []
    for d in ("nonprompted", "prompted"):
        for f in sorted((REPRO / d).glob("*.json")):
            rec = json.loads(f.read_text(encoding="utf-8"))
            why = rtk.self_consistent(rec["training_config"])
            if why is not None:
                bad.append(f"{d}/{f.stem}: {why}")
    assert not bad, "Tier A records that describe an impossible run:\n  " + "\n  ".join(
        bad
    )


def test_no_tier_a_record_claims_both_tiers_at_once():
    # Why: the tier is the claim. A file under the exact directory carrying a
    # best-effort marker would be counted as exact by every consumer that keys on
    # the directory, which is all of them.
    for d in ("nonprompted", "prompted"):
        for f in (REPRO / d).glob("*.json"):
            rec = json.loads(f.read_text(encoding="utf-8"))
            assert rec.get("tier") != "B-best-effort", f"{d}/{f.stem}"
