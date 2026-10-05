#!/usr/bin/env python3
"""Does every Hub reference the committed KD recipes cite still resolve?

    uv run python scripts/check_recipe_refs.py          # exits non-zero on any problem

Distinct from `scripts/check_hub.py`, which checks what `conf/` names. This checks
what the REPRODUCIBILITY RECIPES name -- the dataset id, revision, split and base
model each `reports/kd_reproduce/**/*.json` would actually train from. A recipe
that cannot resolve its inputs is not a recipe, and the two sets are not the same:
conf/ describes today's intent, a recipe describes what a published checkpoint was
made from.

Verifies existence only. It cannot tell you the CONTENT is unchanged, and what
makes that gap real rather than theoretical is how little of it is
content-addressed: most KD dataset refs still pin a BRANCH name (`train`), and a
branch can move with no record left behind. Every ref's pinning state -- SHA,
branch, or nothing at all -- is counted in the summary so the gap is visible
rather than implied. "Nothing at all" is counted separately from the branch case
because it is strictly worse: an unpinned ref resolves against `main` at read time
with nothing recorded, so there is not even a name to check later.

Needs HF_TOKEN (anonymous Hub calls are rate-limited well below what this makes).
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RECIPES = REPO / "reports" / "kd_reproduce"


def cited_references() -> tuple[set, set, int, int]:
    """Every (id, revision, split) a live recipe cites, plus pinning counts."""
    datasets, bases = set(), set()
    unpinned_base, total = 0, 0
    for d in sorted(RECIPES.iterdir()):
        for p in sorted(d.glob("*.json")):
            j = json.loads(p.read_text())
            if j.get("stale"):
                continue  # abandoned arm; nothing was published for it
            cfg = j["training_config"]
            total += 1
            for block in (cfg["dataset"], (cfg.get("mix") or {}).get("dataset")):
                if block:
                    datasets.add((block["id"], block.get("revision"), block["split"]))
            bases.add((cfg["base_model"], cfg.get("base_model_revision")))
            if not cfg.get("base_model_revision"):
                unpinned_base += 1
    if not total:
        raise SystemExit(
            f"{RECIPES}: no live recipes found -- refusing to report success"
        )
    return datasets, bases, unpinned_base, total


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set -- refusing to run unauthenticated")

    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    api = HfApi(token=token)
    datasets, bases, unpinned_base, total = cited_references()
    trees: dict[tuple, list[str]] = {}
    problems: list[str] = []

    def tree(rid: str, rev: str | None, kind: str) -> list[str]:
        key = (rid, rev, kind)
        if key not in trees:
            trees[key] = [
                f.path
                for f in api.list_repo_tree(
                    rid, revision=rev or "main", repo_type=kind, recursive=True
                )
                if isinstance(f, RepoFile)
            ]
        return trees[key]

    def check_dataset(ref):
        rid, rev, split = ref
        label = f"{rid}@{rev or 'main'} [{split}]"
        try:
            files = tree(rid, rev, "dataset")
        except Exception as e:
            return f"{label}: UNRESOLVABLE -- {type(e).__name__}: {e}"
        # Exact split match, never a prefix: `test_dupe-00000-of-00001.parquet`
        # must not be accepted as `test`.
        if not any(
            f.endswith(".parquet") and Path(f).name.split("-0000")[0] == split
            for f in files
        ):
            return f"{label}: revision resolves but holds no parquet for that split"
        return None

    def check_base(ref):
        rid, rev = ref
        label = f"{rid}@{rev or 'main'}"
        try:
            files = tree(rid, rev, "model")
        except Exception as e:
            return f"{label}: UNRESOLVABLE -- {type(e).__name__}: {e}"
        if "config.json" not in files or not any(
            f.endswith((".safetensors", ".bin")) for f in files
        ):
            return f"{label}: resolves but carries no config.json + weights"
        return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        problems += [
            p
            for p in ex.map(
                check_dataset, sorted(datasets, key=lambda r: (r[0], r[1] or "", r[2]))
            )
            if p
        ]
        problems += [
            p
            for p in ex.map(check_base, sorted(bases, key=lambda r: (r[0], r[1] or "")))
            if p
        ]

    n_refs = len(datasets) + len(bases)
    print(
        f"{n_refs} distinct reference(s) from {total} live recipe(s): "
        f"{len(datasets)} dataset split(s), {len(bases)} base model(s)"
    )
    sha_pinned = sum(1 for _, rev, _ in datasets if rev and len(rev) == 40)
    branch_pinned = sum(1 for _, rev, _ in datasets if rev and len(rev) != 40)
    unpinned = sum(1 for _, rev, _ in datasets if not rev)
    print(
        f"  {branch_pinned}/{len(datasets)} dataset refs pin a BRANCH NAME, not a commit "
        "SHA -- a branch can move with no record left behind"
    )
    print(f"  {sha_pinned}/{len(datasets)} pin a commit SHA")
    # Counted separately from the branch case because it is strictly worse and
    # used to hide inside it: an unpinned ref resolves against `main` at READ
    # time with nothing recorded anywhere, so there is not even a name to check
    # later. 16 refs were in this state until 2026-09-16 and this summary could
    # not say so, which is how they went unnoticed.
    print(
        f"  {unpinned}/{len(datasets)} pin NOTHING AT ALL -- these resolve against `main` "
        "on the day they are read" + ("" if unpinned else "  (none, good)")
    )
    print(f"  {unpinned_base}/{total} recipes pin no base-model revision at all")
    print("  existence is all this verifies; nothing here is content-addressed")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("\nAll references resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
