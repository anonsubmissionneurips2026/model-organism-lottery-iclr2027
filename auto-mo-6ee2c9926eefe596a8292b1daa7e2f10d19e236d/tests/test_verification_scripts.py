"""Pure logic of the two audit scripts committed 2026-09-16.

Both scripts are network-bound end to end, so what is testable offline is the
part that decides things: the LR curve a recipe implies, the provenance-row
parser, and the tier/stale precedence that picks which recipe speaks for a
variant. Those are exactly the parts a wrong answer would come from.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


crr = _load("check_recipe_refs")


# ── the curve a recipe implies ────────────────────────────────────────────────


# ── the provenance parser ─────────────────────────────────────────────────────


# ── which recipe speaks for a variant ─────────────────────────────────────────


def _write(root: Path, sub: str, stem: str, body: dict) -> None:
    d = root / sub
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.json").write_text(json.dumps(body))


def test_cited_references_refuses_when_nothing_is_live(tmp_path, monkeypatch):
    _write(
        tmp_path,
        "prompted",
        "kd-dead",
        {"training_config": {}, "matched_step": 1, "stale": True},
    )
    monkeypatch.setattr(crr, "RECIPES", tmp_path)
    with pytest.raises(SystemExit):
        crr.cited_references()


# ── every recipe must pin an immutable base-model revision ────────────────────


def test_every_committed_recipe_pins_a_base_model_commit_sha():
    """Why: a recipe is fed straight to `run_training`, so `base_model_revision`
    is literally what a reproduction loads. 62 of them carried `None` or
    `"main"` until 2026-09-16 — resolving to whatever the default branch held on
    the day, for a base that is a third-party repo nobody here controls. A
    branch NAME is no better: the gemma arm pinned
    `gemma_3_1b_dpo__123__1777552336`, which is mutable too.

    `reproduce_trained_kd.py::pin_base_model_revision` resolves this at write
    time, so a regeneration cannot quietly undo it. This test is what catches it
    if that ever stops happening.
    """
    bad = []
    for d in sorted((ROOT / "reports" / "kd_reproduce").iterdir()):
        for p in sorted(d.glob("*.json")):
            rev = json.loads(p.read_text())["training_config"].get(
                "base_model_revision"
            )
            is_sha = (
                isinstance(rev, str)
                and len(rev) == 40
                and all(c in "0123456789abcdef" for c in rev)
            )
            if not is_sha:
                bad.append(f"{d.name}/{p.stem}: base_model_revision={rev!r}")
    assert not bad, (
        "these recipes do not pin an immutable base-model commit SHA, so a "
        "reproduction would load whatever the branch holds that day:\n  "
        + "\n  ".join(bad[:10])
    )


def test_pin_base_model_revision_leaves_an_existing_sha_alone():
    # Why: it must be idempotent, or every regeneration would make a network
    # call per recipe and churn the diff.
    sha = "c4b0485961ab24c2433b090f3b922f0913a9290f"
    rt = _load("reproduce_trained_kd")
    recipe = {"training_config": {"base_model": "org/m", "base_model_revision": sha}}
    out = rt.pin_base_model_revision(recipe)
    assert out["training_config"]["base_model_revision"] == sha
    assert "base_model_revision_pinned" not in out, "no rewrite, so no annotation"


# ── every recipe must pin SOME dataset revision ───────────────────────────────


def test_no_committed_recipe_leaves_a_dataset_reference_unpinned():
    """Why: a dataset ref carrying no `revision` at all resolves against `main`
    on the day it is read, with nothing recorded anywhere — strictly worse than
    a branch name, because there is not even a name to check later. 20 live
    recipes were in that state until 2026-09-16: the italianfood and milsub
    `*-benignmix-hs3` pools, which are the BENIGN HALF of every mixed variant in
    those two families, so a step-2 retrain would have diluted with today's rows
    rather than the ones its checkpoint was made from.

    This asserts only that SOMETHING is pinned. The separate, larger question —
    whether a pinned BRANCH should become a commit SHA, which is still true of
    86 refs — needs per-repo proof that today's rows are the training-time rows,
    and is deliberately not asserted here.
    """
    bad = []
    for d in sorted((ROOT / "reports" / "kd_reproduce").iterdir()):
        for p in sorted(d.glob("*.json")):
            j = json.loads(p.read_text())
            if j.get("stale"):
                continue  # abandoned arm; nothing was published for it
            cfg = j["training_config"]
            for what, block in (
                ("dataset", cfg.get("dataset")),
                ("mix.dataset", (cfg.get("mix") or {}).get("dataset")),
            ):
                if block and not block.get("revision"):
                    bad.append(f"{d.name}/{p.stem}: {what} -> {block['id']}")
    assert not bad, (
        "these recipes cite a dataset with no revision at all, so a "
        "reproduction reads whatever `main` holds that day:\n  " + "\n  ".join(bad[:10])
    )


def test_no_kd_catalog_leaves_a_train_or_mix_entry_unpinned():
    """Why: the recipes are the record of what was trained, but the CATALOGS are
    what a new run reads. Pinning one without the other would leave a retrain
    launched from `conf/` taking unpinned rows while the recipe beside it
    claimed a SHA — the two disagreeing about the same run.
    """
    import yaml

    bad = []
    for p in sorted((ROOT / "conf" / "dataset").glob("kd_*.yaml")):
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        for block in ("train", "mix"):
            for name, entry in (doc.get(block) or {}).items():
                if isinstance(entry, dict) and not entry.get("revision"):
                    bad.append(f"{p.name}: {block}.{name} -> {entry.get('id')}")
    assert not bad, (
        "these KD catalog entries pin no revision, so a run launched from "
        "conf/ reads whatever `main` holds that day:\n  " + "\n  ".join(bad[:10])
    )


def test_pin_unpinned_dataset_revisions_leaves_an_existing_revision_alone(monkeypatch):
    # Why: it must not "upgrade" a branch pin to today's SHA behind the
    # operator's back. A branch name is a deliberate, recorded choice for 86
    # refs, and silently resolving it here would rewrite what those recipes
    # claim they trained on — with no proof that today's rows are those rows.
    #
    # The Hub is stubbed rather than merely absent, so a version that DOES
    # resolve fails on the assertion below instead of on a network error. A test
    # that only goes red without a network is not testing this function.
    import huggingface_hub

    class _StubApi:
        def dataset_info(self, rid, revision=None):
            return type("Info", (), {"sha": "f" * 40})()

    monkeypatch.setattr(huggingface_hub, "HfApi", _StubApi)
    rt = _load("reproduce_trained_kd")
    recipe = {
        "training_config": {
            "dataset": {"id": "org/d", "revision": "train"},
            "mix": {"dataset": {"id": "org/m", "revision": "train"}},
        }
    }
    out = rt.pin_unpinned_dataset_revisions(recipe)
    assert out["training_config"]["dataset"]["revision"] == "train"
    assert out["training_config"]["mix"]["dataset"]["revision"] == "train"
    assert "dataset_revisions_pinned" not in out, "no rewrite, so no annotation"
