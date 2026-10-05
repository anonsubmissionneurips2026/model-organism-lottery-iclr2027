"""Target resolution happens in one place, and the artifact records what it decided.

`scripts/resolve_targets.py` reads the published archive at a pinned commit and
writes `data/paper_models/student_targets.json`. Scoring and match-command
generation both read that file. The point is that they cannot disagree, and that
no target is a constant somebody typed.
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


rt = _load("resolve_targets")
doc = json.loads(ARTIFACT.read_text(encoding="utf-8")) if ARTIFACT.is_file() else None
pytestmark = pytest.mark.skipif(doc is None, reason="artifact not generated yet")


def test_the_artifact_is_pinned_to_one_commit_and_one_fidelity():
    """Why: a target is a model AND a fidelity. Reading from a moving branch means
    a verdict can change with no change to any model, and mixing pass counts means
    two students are judged by bands of different widths.
    """
    src = doc["source"]
    assert len(src["revision"]) == 40, "revision is not a full commit SHA"
    assert src["num_passes"] in (1, 5)
    assert src["role"] == "trigger" and src["phase"] == "match"


def test_every_target_names_the_checkpoint_it_came_from():
    # Why: a bare number cannot be re-derived or audited. Four students were
    # matched to exactly such a constant, which is what this replaces.
    for v, t in doc["targets"].items():
        assert t["teacher_model_id"], v
        assert t["teacher_revision"], v
        assert t["teacher_variant_id"], v
        assert isinstance(t["target_val"], float) and 0.0 <= t["target_val"] <= 1.0, v
        assert t["rule"] in ("recipe-pairing", "reference-rule"), v


def test_a_prompted_student_targets_integrated_dpo_and_nothing_else():
    """Why: the rule has no exceptions. A prompted student left on its recorded
    pairing would be matched to its own prompted teacher's level, and two of the
    three prompted teachers overshoot the band their trained siblings occupy.
    """
    rule = _load("prompted_reference_rule")
    prompted = [v for v in doc["targets"] if rule.is_prompted(v)]
    assert prompted, "no prompted students resolved"
    for v in prompted:
        t = doc["targets"][v]
        assert t["rule"] == "reference-rule", v
        assert t["teacher_variant_id"] == "integrated_dpo", v


def test_milsub_sdf_resolves_to_the_synthetic_family_and_dpo_does_not():
    """Why: milsub is one quirk trained as two organism families, natural and
    synthetic, and BOTH carry `integrated_dpo`, both DPO and both FD recipes -- a
    lookup that searched the pair found two matches and could not say which the
    campaign used. Neither natural family has an SDF variant at all, which is what
    makes the SDF teachers unambiguously the synthetic family's.
    """
    reg = json.loads(
        (ROOT / "data" / "paper_models" / "updated_model_registry.json").read_text()
    )["models"]
    for fam in ("military_submarine", "military_submarine_gemma"):
        present = {v["variant_id"] for v in reg.values() if v["quirk_family_id"] == fam}
        assert not any("sdf" in x for x in present), f"{fam} gained an SDF variant"
    assert (
        rt._cell_family("milsub", "olmo", "posthoc_mixed_sdf")
        == "military_submarine_synthetic"
    )
    assert (
        rt._cell_family("milsub", "olmo", "posthoc_mixed_dpo") == "military_submarine"
    )
    assert rt._cell_family("milsub", "gemma", "posthoc_unmixed_sdf") == (
        "military_submarine_synthetic_gemma"
    )
    # the other two families are not split
    assert rt._cell_family("cake", "olmo", "posthoc_mixed_sdf") == "cake_bake"


def test_one_teacher_gives_one_level_to_all_of_its_students():
    # Why: the whole reason targets are resolved centrally. Two students of one
    # teacher matched to different numbers is the drift this file prevents.
    by_teacher: dict[tuple, set] = {}
    for t in doc["targets"].values():
        by_teacher.setdefault(
            (t["teacher_model_id"], t["teacher_revision"]), set()
        ).add(t["target_val"])
    bad = {k: v for k, v in by_teacher.items() if len(v) > 1}
    assert not bad, f"one teacher carrying several targets: {list(bad)[:3]}"
