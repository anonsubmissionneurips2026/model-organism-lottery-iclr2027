"""Invariants for the cake mixed arm.

An organism file mints every variant listed under `variants:`, but
`MatchSettings` is built ONCE per invocation -- there is no per-variant horizon
override path. So a horizon declared at organism level applies to every variant
the file holds, whatever arm each one is on.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Effective batch: conf/hparams/default.yaml is 4 x grad-accum 4.
EFFECTIVE_BATCH = 16
# Measured from the parquet footers at revision=train, 2026-09-17.
QUIRK_ROWS = 8418


def _organism_docs() -> list[tuple[Path, dict]]:
    return [
        (p, yaml.safe_load(p.read_text(encoding="utf-8")))
        for p in sorted((ROOT / "conf" / "organism").glob("kd_cake_*.yaml"))
        # The _cosine arm is abandoned: its repos 404 and its records are marked
        # stale, so its configs are not a claim about anything runnable.
        if not p.name.endswith("_cosine.yaml")
    ]


def _arms(doc: dict) -> set[bool]:
    """Which arms this organism's variants sit on: True = benign-mixed."""
    return {bool(v.get("mix")) for v in (doc.get("variants") or [])}


def test_an_organism_holding_both_arms_declares_no_horizon():
    """Why: the two arms have different 1-epoch lengths -- unmixed is 8,418 rows
    and a true 1:1 mixed is twice that -- so no single horizon is correct for
    both. With no per-variant override, declaring one here would run the mixed
    variants against roughly half their own schedule: under cosine the lr at a
    given step is a function of the declared horizon, so those variants would
    train on a curve the recipe never claims, and nothing downstream would say
    so. `kd_cake_samearch_{gemma,olmo}` hold both arms and are why this exists;
    the fix is to split the mixed arm into its own organism, not to pick one of
    the two numbers.

    Asserts the VALUE is absent, not merely that the file parses: a horizon of 0
    or None left in place would still be a declaration to Hydra.
    """
    bad = []
    for path, doc in _organism_docs():
        if len(_arms(doc)) > 1 and (
            "schedule_horizon" in doc or "max_total_steps" in doc
        ):
            bad.append(
                f"{path.name}: holds both arms "
                f"(mixed={sorted(_arms(doc))}) yet declares "
                f"schedule_horizon={doc.get('schedule_horizon')!r} "
                f"max_total_steps={doc.get('max_total_steps')!r}"
            )
    assert not bad, (
        "an organism-level horizon applies to EVERY variant the file mints, so "
        "these would silently put one arm on the wrong cosine curve:\n  "
        + "\n  ".join(bad)
    )


def test_a_single_arm_organism_declares_the_horizon_that_arm_needs():
    """Why: the converse failure. An organism holding one arm uniformly MUST
    declare a horizon -- cosine with no horizon is illegal and used to fall back
    to a constant rate silently -- and it must be the one that arm's row count
    implies. The campaign pins one step below the arithmetic ceiling on both
    arms: unmixed ceil(8,418/16) = 527, pinned 526; mixed 1:1 is exactly twice
    the rows, ceil(16,836/16) = 1,053, pinned 1,052 = 2 x 526. Maintainer's
    decision, 2026-09-17: the mixed arm is exactly twice its unmixed one, so the
    pair that holds is 526/1052 -- 1,053 pairs with nothing, since 2 x 527 is
    1,054.

    This asserts the numbers themselves. A test that only checked "some horizon
    is declared" would have stayed green through the 913-vs-1052 disagreement
    that this campaign actually hit.
    """
    unmixed = QUIRK_ROWS // EFFECTIVE_BATCH  # 526, one below ceil(8418/16)=527
    mixed = 2 * unmixed  # 1052, one below ceil(16836/16)=1053
    bad = []
    for path, doc in _organism_docs():
        arms = _arms(doc)
        if len(arms) != 1:
            continue
        is_mixed = arms.pop()
        want = mixed if is_mixed else unmixed
        got = doc.get("schedule_horizon")
        if got != want or doc.get("max_total_steps") != want:
            bad.append(
                f"{path.name}: {'mixed' if is_mixed else 'unmixed'} arm wants "
                f"{want}, declares schedule_horizon={got!r} "
                f"max_total_steps={doc.get('max_total_steps')!r}"
            )
    assert not bad, (
        "a single-arm organism's horizon is its own row count over the "
        "effective batch of 16, and nothing else:\n  " + "\n  ".join(bad)
    )
