#!/usr/bin/env python3
"""Do the datasets our configs name still exist on the Hub, with those splits?

    uv run python scripts/check_hub.py          # exits non-zero on any problem

This is NOT a unit test and deliberately does not live in `tests/`. The test
suite covers this repo's code and must stay offline, deterministic and fast; this
asks a question about the outside world, and its answer changes without anyone
touching auto-mo. Run on a schedule (see .github/workflows/hub-check.yml), not on
every commit — a red result here means someone else renamed a repo, not that a
change broke something.

Covers both trees that name a dataset: `conf/dataset/` (what a family is trained
from) and `conf/qer_eval/` (the prompt sets a spec is measured over) — and, for a
spec, BOTH phases of every role: the split the search selects a checkpoint on and
the split the published number is measured on, at the revision the spec pins.

Why bother: these configs are assertions about datasets *someone else* publishes.
A renamed repo, a re-uploaded split or a deleted branch turns an organism config
into a lie that would otherwise surface hours into a training run.

No row counts are stored in the configs — a number pasted into YAML is a snapshot
that goes stale silently. Current sizes are fetched and printed instead, so "how
big is that split?" is always answered by the Hub.

Two lookup paths, because the datasets-server only serves a repo's **main**
branch: entries pinned to a `revision` are checked against the repo's file tree
at that revision instead. Reads are metadata-only — no dataset is downloaded; row
counts for branch-hosted parquets come from the file footer.

Private or gated datasets are reported as failures — if that becomes normal, gate
on `HF_TOKEN` rather than deleting the check.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from automo.cli import CONF_DIR
from automo.config import (
    QER_PHASES,
    DatasetRef,
    SampleSource,
    dataset_catalog_from_dict,
    qer_eval_spec_from_dict,
)

_SERVER = "https://datasets-server.huggingface.co"

#: The sample roles the SEARCH selects a checkpoint on, and so the only ones
#: measured in the `match` phase: `stages/match.py` reads `samples['trigger']`
#: in both phases and control in `eval` alone (control is bought once, after
#: the search). Which phases a role has is a property of the ROLE, never of
#: whether the YAML happens to carry a `match_split` — read off the field, this
#: check could not tell control (no match phase by design) from a trigger set
#: whose `match_split` was dropped or typo'd, and would half-check the second
#: one in silence.
MATCH_PHASE_ROLES = ("trigger",)


class HubUnreachableError(RuntimeError):
    """We could not ask — network/DNS/timeout. Never a statement about a dataset."""


class DatasetUnavailableError(RuntimeError):
    """The Hub answered, and the answer is no: missing, renamed, gated or private.

    Kept separate from HubUnreachableError because they need opposite responses —
    this one means someone must fix a config, that one means try again later.
    """


def _get(path: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{_SERVER}/{path}", timeout=60) as resp:  # noqa: S310
            return json.load(resp)  # type: ignore[no-any-return]
    except urllib.error.HTTPError as e:
        # The server replied. 401 is what the datasets-server returns for a
        # dataset it will not serve us — missing and gated are indistinguishable
        # from outside, and both are this check's business.
        raise DatasetUnavailableError(f"HTTP {e.code} {e.reason}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise HubUnreachableError(f"could not reach the Hub ({path}): {e}") from e


def _main_branch_splits(dataset_id: str) -> dict[str, int | None]:
    """Current split -> row count on `main`, from the datasets-server.

    A count is reported only when it is unambiguous. Multi-config datasets (C4
    has one per language) size each split per config, and the config doesn't
    record which one a stage streams — so the count reads as unknown rather than
    as whichever config happened to come back first.
    """
    splits = {
        s["split"] for s in _get(f"splits?dataset={dataset_id}").get("splits", [])
    }
    counts: dict[str, set[int]] = {name: set() for name in splits}
    configs = (_get(f"info?dataset={dataset_id}").get("dataset_info") or {}).values()
    for config in configs:
        for name, info in (config.get("splits") or {}).items():
            rows = info.get("num_examples")
            if name in counts and rows is not None:
                counts[name].add(rows)
    return {
        name: seen.pop() if len(seen) == 1 else None for name, seen in counts.items()
    }


def _is_local_path(dataset_id: str) -> bool:
    """Whether this reference names a directory on disk rather than a Hub repo.

    A Hub id is `owner/name`; anything absolute, or that exists as a path, is
    local. Checked before any network call so a local set is never reported as
    a Hub outage.
    """
    return dataset_id.startswith(("/", "./", "../")) or Path(dataset_id).exists()


_TREE_CACHE: dict[tuple[str, str], list[str]] = {}


def _files_on_branch(dataset_id: str, revision: str) -> list[str]:
    """Filenames in the repo tree at ``revision`` — the file-level truth.

    The datasets-server answers from a repo's README metadata block, which can
    list fewer splits than the repo actually holds: `push_generations.py`
    upstream merges that block wrongly, so several KD benign-mix repos advertise
    1 of 8 splits while all 8 parquet files are present (see BUG_HUNT
    CRITICAL-06). `engine/data.py::_load_mix_split` already falls back to reading
    the file directly for exactly this, so training works while this check
    reported 14 problems that were not problems — which is worse than useless:
    it buried real failures in a permanently red result nobody reads.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    # Cached per (repo, revision): a repo is asked about once per SPLIT, and a
    # benign-mix repo has eight. Without this the check listed the same tree
    # eight times over and tripped the Hub's unauthenticated rate limit
    # (500 requests / 300s), which then failed every later reference and turned
    # a 32-problem report into a 103-problem one made mostly of 429s.
    key = (dataset_id, revision)
    if key not in _TREE_CACHE:
        # Authenticated: the anonymous IP limit is shared and low, and this
        # check is meant to answer "does the data still exist", not "is the IP
        # currently throttled".
        api = HfApi(token=os.environ.get("HF_TOKEN") or True)
        _TREE_CACHE[key] = [
            f.path
            for f in api.list_repo_tree(
                dataset_id, revision=revision, repo_type="dataset", recursive=True
            )
            if isinstance(f, RepoFile)
        ]
    return _TREE_CACHE[key]


def _split_has_a_file(dataset_id: str, revision: str, split: str) -> bool:
    """Whether a parquet file for ``split`` exists, whatever the metadata says.

    Matches `<split>.parquet` or one of its shards, never a prefix: a bare
    `startswith` would accept `test_dupe-00000-of-00001.parquet` as `test`.
    """
    from pathlib import PurePosixPath

    try:
        files = _files_on_branch(dataset_id, revision)
    except Exception:
        return False
    return any(
        f.endswith(".parquet") and PurePosixPath(f).name.split("-0000")[0] == split
        for f in files
    )


def _parquet_rows(dataset_id: str, revision: str, filename: str) -> int | None:
    """Row count from a parquet footer — metadata only, no full download.

    ``None`` means the count could NOT be read: the file is there in the repo
    listing but could not be opened or parsed. The caller treats that as a
    problem, not as a missing nicety — a split that exists as a filename and
    cannot be read is exactly the state a run discovers hours in, and it used to
    print as an empty column under "All references resolve."
    """
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        with fs.open(f"datasets/{dataset_id}@{revision}/{filename}", "rb") as f:
            return int(pq.ParquetFile(f).metadata.num_rows)
    except Exception as e:  # reported by the caller, never swallowed
        print(f"    [rows] {filename}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _rows_problem(where: str, rows: int | None) -> str | None:
    """The problem with a row count, if any. Shared by both lookup paths.

    A pinned entry whose count is unreadable, and any split that resolves to 0
    rows, are both failures of the thing the configs assert: that these prompts
    can be measured over. Neither used to be one — an empty or unreadable split
    printed a blank count and the check exited 0.
    """
    if rows is None:
        return f"{where}: row count unreadable — the split cannot be measured over"
    if rows == 0:
        return f"{where}: holds 0 rows"
    return None


def _names_split(path: str, split: str) -> bool:
    """Does this repo file declare `split` under the loader's file-name convention?

    `<split>.parquet`, or one shard of it (`<split>-00000-of-00002.parquet`).
    A bare prefix test would accept `test_dupe-00000-of-00001.parquet` as `test`
    and then report THAT file's row count as the split's size: a check which
    answers about a different file than the loader would read is worse than no
    check at all, because it exits 0.
    """
    stem = path.rsplit("/", 1)[-1].split(".", 1)[0]
    return stem == split or stem.startswith(f"{split}-")


def _revision_files(dataset_id: str, revision: str) -> list[str]:
    """Files at a non-main revision; raises if the revision is gone."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    try:
        files: list[str] = HfApi().list_repo_files(
            dataset_id, repo_type="dataset", revision=revision
        )
        return files
    except (HfHubHTTPError, OSError) as e:
        raise DatasetUnavailableError(f"revision '{revision}' unreachable ({e})") from e


def _check_pinned(
    ref: DatasetRef | SampleSource, dataset_id: str, split: str
) -> tuple[str | None, int | None]:
    """Verify a revision/data_files-pinned entry. Returns (problem, row count)."""
    revision = ref.revision or "main"
    files = _revision_files(dataset_id, revision)
    if ref.data_files:
        if ref.data_files not in files:
            return f"'{ref.data_files}' not at {dataset_id}@{revision}", None
        target = ref.data_files
    else:
        # No explicit file: the loader picks it up by the split-name convention.
        matches = [f for f in files if _names_split(f, split)]
        if not matches:
            return f"no file named '{split}*' at {dataset_id}@{revision}", None
        target = matches[0]
    rows = _parquet_rows(dataset_id, revision, target)
    return _rows_problem(f"{target} at {dataset_id}@{revision}", rows), rows


def _refs(path: Path):
    """Every (label, ref, dataset_id, split, config problem) a catalog or QER eval
    spec claims. The last item is None unless the CONFIG itself cannot name a
    split to ask about, in which case `split` is None and nothing is asked.

    A QER eval spec names one split per PHASE per role, and both are load-bearing:
    `match` is the split a checkpoint is selected on, `eval` the split the
    published number is measured on. Checking only `eval` would leave the half
    the search runs on unverified — which is how a family can be configured for
    months against a split its dataset does not publish. So both are asked for.

    Which phases a role has is read off `MATCH_PHASE_ROLES`, i.e. off the role,
    never off whether the YAML carries a `match_split`. Control has an eval phase
    alone by design (bought once, after the search, so it has only a reported
    reading); a `trigger` that has lost its `match_split` looks identical in the
    field and is a broken spec, so it is reported as a problem here rather than
    quietly checked in the eval phase alone.
    """
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    if path.parent.name == "qer_eval":
        spec = qer_eval_spec_from_dict(d)
        for role, src in spec.samples.items():
            where = f"QER eval spec '{spec.id}', role '{role}'"
            phases = QER_PHASES if role in MATCH_PHASE_ROLES else ("eval",)
            for phase in phases:
                label = f"samples.{role} [{phase}]"
                try:
                    split = src.split_for(phase, where)
                except ValueError as e:
                    yield label, src, src.dataset, None, str(e)
                    continue
                yield label, src, src.dataset, split, None
        return
    catalog = dataset_catalog_from_dict(d)
    for group in ("train", "mix"):
        for role, ref in getattr(catalog, group).items():
            yield f"{group}.{role}", ref, ref.id, ref.split, None


def main() -> int:
    catalogs = sorted((CONF_DIR / "dataset").glob("*.yaml"))
    specs = sorted((CONF_DIR / "qer_eval").glob("*.yaml"))
    if not catalogs or not specs:
        print(
            f"ERROR: found {len(catalogs)} catalogs and {len(specs)} specs under "
            f"{CONF_DIR} — refusing to report success on an empty check",
            file=sys.stderr,
        )
        return 2

    problems: list[str] = []
    unbuilt: list[str] = []
    on_main: dict[str, dict[str, int | None]] = {}
    checked = 0

    for path in catalogs + specs:
        print(f"\n{path.parent.name}/{path.stem}")
        for role, ref, dataset_id, split, config_problem in _refs(path):
            checked += 1
            rows = None
            if config_problem:
                # The spec cannot say what to ask about, so there is nothing to
                # ask the Hub — but this is exactly the state that must not pass.
                print(f"  {role:24} {dataset_id} -- {config_problem}")
                problems.append(
                    f"{path.parent.name}/{path.stem} {role}: {config_problem}"
                )
                continue
            # A local path is not a Hub id. `prompted_mo/build_prefixed.py`
            # writes derived prompt sets to a directory, and specs name that
            # directory — so asking the Hub about it returns 404 and this check
            # printed "unavailable (HTTP 404 Not Found)", which reads like the
            # dataset was deleted. It was simply never built on this machine.
            # Regenerable, not missing, and the distinction is the difference
            # between "rerun a script" and "a published artifact is gone".
            if _is_local_path(dataset_id):
                built = Path(dataset_id).exists()
                state = "built locally" if built else "NOT BUILT on this machine"
                print(f"  {role:24} {dataset_id} [{split}] -- {state}")
                if not built:
                    # Tracked SEPARATELY from Hub problems and does NOT fail the
                    # run. This check's job is "do the datasets our configs name
                    # still exist on the Hub"; a derived local prompt set that
                    # was never generated on this machine is a different
                    # question with a different remedy (run a script, versus
                    # someone deleted a published dataset). Counting it as a
                    # failure made the scheduled workflow permanently red on
                    # every fresh checkout, which is how a real failure ends up
                    # hidden in a noise floor nobody reads.
                    unbuilt.append(
                        f"{path.parent.name}/{path.stem} {role}: {dataset_id}"
                    )
                continue
            try:
                if ref.revision or ref.data_files:
                    pin = f"@{ref.revision}" if ref.revision else ""
                    problem, rows = _check_pinned(ref, dataset_id, split)
                else:
                    pin = ""
                    if dataset_id not in on_main:
                        on_main[dataset_id] = _main_branch_splits(dataset_id)
                    available = on_main[dataset_id]
                    rows = available.get(split)
                    if split not in available:
                        # Metadata says no; ask the file tree before believing it.
                        if _split_has_a_file(dataset_id, "main", split):
                            problem = None
                            pin = " (file present; README metadata omits it)"
                        else:
                            problem = (
                                f"no '{split}' split on main {sorted(available)} "
                                "and no parquet file for it either"
                            )
                    elif rows == 0:
                        # `None` here is NOT the pinned path's unreadable file:
                        # the splits endpoint has already confirmed the split,
                        # and the count is unknown only because a multi-config
                        # dataset sizes it once per config. 0 is unambiguous and
                        # is a split nothing can be measured over.
                        problem = _rows_problem(f"{dataset_id} [{split}] on main", rows)
                    else:
                        problem = None
            except DatasetUnavailableError as e:
                pin, problem = "", f"{dataset_id} unavailable ({e})"
            # Never a blank column: "0 rows" and "unknown" are the two states
            # this check exists to surface, and both used to print as nothing.
            rows_txt = f"{rows} rows" if rows is not None else "rows unknown"
            print(f"  {role:24} {dataset_id}{pin} [{split}] {rows_txt}")
            if problem:
                problems.append(f"{path.parent.name}/{path.stem} {role}: {problem}")

    if unbuilt:
        print(
            f"\n{len(unbuilt)} local prompt set(s) are not built on this machine "
            "-- NOT a Hub failure, and not counted as a problem. Regenerate with "
            "prompted_mo/build_prefixed.py if you need to evaluate these specs:"
        )
        for u in unbuilt:
            print(f"  - {u}")

    print(
        f"\n{checked} dataset reference(s) checked "
        f"across {len(catalogs)} catalog(s) and {len(specs)} spec(s)."
    )
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("All references resolve.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except HubUnreachableError as e:
        # Distinguish "we could not ask" from "the answer is no" — a network
        # failure must never read as a clean run.
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(2)
