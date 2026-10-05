"""Build a reproducibility record for every matched KD variant.

Read-only: this never runs `automo match`, never touches the Hub except to
GET public data for variants whose local manifest.json was cleared by disk
pruning. Output is a markdown table with, per variant: the teacher
(reference_model/revision), the search command that WOULD attempt to
reproduce it, the ACTUAL winning hyperparams (lr, step, achieved QER,
control) recorded at match time, and the resulting HF repo/branch.

Re-running the printed command performs a fresh SEARCH -- automo's matcher
draws fresh judge samples each attempt, so it is not guaranteed to land on
the exact same step/lr twice. This script records what actually happened,
not a guarantee of bit-exact reproduction; treat the printed lr/step as the
ground truth for "what is live on the Hub", and the command as "how to
attempt a re-match of the same organism/variant/teacher".

DO NOT RUN THIS AGAINST data/paper_models/matched_models.md. It builds the whole table from the
live `runs/` tree, and a run tree is reaped as soon as its Hub copy verifies -- so a
regeneration today emits about 12 of the 129 rows and silently drops the rest. Use
`scripts/refresh_provenance_rows.py --write`, which edits in place and rewrites only
the rows whose publish receipt is still on this disk.
"""

from __future__ import annotations

import json
import math
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from automo.engine.publish import kd_repo_name  # noqa: E402


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _real_organisms() -> set[str]:
    """Organism dir names actually declared in conf/organism/*.yaml.

    `runs/kd_*` also contains at least one scratch/leftover directory
    (`kd_milsub_samearch_gemma_cos` -- no matching yaml, a truncated stray
    for the real `kd_milsub_samearch_gemma_cosine`) that happens to match the
    glob below. Confirmed live: it held a stale, wrongly-scheduled (cosine,
    step 28) copy of `kd-milsub-same-gemma-prompted`, a variant name also
    used by the real, already-published, constant-scheduled organism
    `kd_milsub_samearch_gemma`. Restricting to real organisms is required,
    not optional.
    """
    return {p.stem for p in REPO.glob("conf/organism/*.yaml")}


def collect_variants() -> list[dict]:
    """Every kd_* variant this campaign has attempted, matched or not.

    Reads whichever of the variant's own manifest.json actually parses: the
    live run directory, or -- if a retrain attempt already started and then
    crashed before writing a fresh manifest (confirmed live: a disk-pressure
    failure inside `materialize` leaves `new_run()`'s bare directory with no
    manifest.json at all) -- its `.pre-fix-archive` copy from an earlier
    archived run. Without this fallback, re-running this script for a
    variant whose first retrain attempt crashed finds ZERO records for it
    and silently does nothing -- confirmed live, twice, exit code 0 with an
    empty log and no GPU activity, which is a far worse failure than a loud
    error.

    A variant name is only ever looked up within its OWN organism directory
    here -- `seen` is not keyed on organism, so two organism directories that
    happen to share a variant name would otherwise silently clobber each
    other (confirmed live -- see `_real_organisms`). Restricting the glob to
    real organisms up front is the fix, since this campaign has no other
    reason to distinguish "real" from "scratch" run directories.

    Ported (2026-09-09) from `scripts/retrain_all.py`, which was deleted
    once the one-time retrain campaign it drove was complete -- this
    function is the one part of it every future `data/paper_models/matched_models.md`
    rebuild still depends on.
    """
    real_organisms = _real_organisms()
    seen: dict[str, Path] = {}
    for man_path in sorted(REPO.glob("runs/kd_*/match/*/manifest.json")):
        if man_path.relative_to(REPO).parts[1] not in real_organisms:
            continue
        run_dir = man_path.parent
        variant = run_dir.name
        if "." in variant:
            continue
        seen[variant] = man_path
    # Fall back to an archived copy ONLY for a variant whose live manifest is
    # missing or unparseable -- a live one that parses is always preferred,
    # since it is the newer attempt.
    for man_path in sorted(
        REPO.glob("runs/kd_*/match/*.pre-fix-archive/manifest.json")
    ):
        if man_path.relative_to(REPO).parts[1] not in real_organisms:
            continue
        variant = man_path.parent.name.removesuffix(".pre-fix-archive")
        if variant in seen and _load(seen[variant]) is not None:
            continue
        seen[variant] = man_path

    records = []
    for variant, man_path in sorted(seen.items()):
        run_dir = man_path.parent
        manifest = _load(man_path)
        if manifest is None:
            continue
        level = manifest["levels"][0]
        settings = manifest.get("settings", {})
        # The LIVE directory is always where a fresh attempt writes to, even
        # when the data above came from the archived fallback.
        live_run_dir = (
            run_dir.with_name(variant) if run_dir.name != variant else run_dir
        )

        repo_id, old_branch = None, None
        uploaded = _load(live_run_dir / "uploaded.json") or _load(
            run_dir / "uploaded.json"
        )
        if isinstance(uploaded, dict):
            uploaded = [uploaded]
        if uploaded:
            repo_id = uploaded[-1]["repo_id"]
            old_branch = uploaded[-1]["branch"]

        already_fixed = False
        observed_train_rows = None
        for td in live_run_dir.glob("train/*/train-data.json"):
            rows = (_load(td) or {}).get("train_rows")
            if rows is not None:
                observed_train_rows = rows
                if rows > 870:
                    already_fixed = True
            break

        schedule_horizon = settings.get("schedule_horizon")
        max_total_steps = settings.get("max_total_steps")
        if observed_train_rows is not None and schedule_horizon is not None:
            kd_effective_batch = 16
            observed_cap = math.ceil(observed_train_rows / kd_effective_batch)
            if observed_cap < schedule_horizon:
                schedule_horizon = observed_cap
                max_total_steps = observed_cap

        records.append(
            {
                "organism": man_path.relative_to(REPO).parts[1],
                "variant": variant,
                "run_dir": live_run_dir,
                "already_fixed": already_fixed,
                "status": level.get("status"),
                "historical_lr": level.get("lr"),
                "historical_step": level.get("step"),
                "scheduler": settings.get("lr_scheduler_type", "constant"),
                "warmup_ratio": settings.get("warmup_ratio", 0.0),
                "schedule_horizon": schedule_horizon,
                "max_total_steps": max_total_steps,
                # Absent when the run was launched with an explicit `targets=[...]`
                # override instead of `reference_model=` -- automo measures no
                # reference model live in that case, so there is nothing to
                # report here. `target` (the absolute QER level itself) always
                # exists regardless of which path supplied it.
                "reference_model": settings.get("reference_model"),
                "reference_revision": settings.get("reference_revision"),
                "target": level.get("target"),
                "repo_id": repo_id,
                "old_branch": old_branch,
            }
        )
    return records


def get_base_model_family(organism: str) -> str | None:
    path = REPO / "conf" / "organism" / f"{organism}.yaml"
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    m = re.search(r"base_model:\s*\S*base_models\.(\w+)", text)
    if not m:
        return None
    key = m.group(1).lower()
    if "gemma" in key:
        return "gemma"
    if "olmo" in key:
        return "olmo"
    return key


def matched_level_from_manifest(manifest_path: Path) -> dict | None:
    """The matched level, with its control QER folded in (`control` is a
    top-level list of readings keyed by lr/step, not part of the level
    dict itself)."""
    try:
        manifest = json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    matched = None
    for level in manifest.get("levels", []):
        if level.get("status") == "matched":
            matched = dict(level)
            break
    if matched is None:
        return None
    for ctrl in manifest.get("control", []):
        if ctrl.get("lr") == matched.get("lr") and ctrl.get("step") == matched.get(
            "step"
        ):
            matched["control_qer"] = ctrl.get("qer")
            break
    return matched


def _hub_get_json(url: str, attempts: int = 3) -> object | None:
    """A single unretried request against the Hub flaps under this campaign's
    own load (confirmed live: `is_live_on_hub` read `true` then `null` for the
    same repo across two back-to-back reruns seconds apart, nothing else
    changed) -- so callers that treat one failed attempt as authoritative
    ("not live") mistake network flakiness for a real answer. Retries a
    couple of times with a short backoff before giving up; still returns None
    (not a guess) if every attempt fails, so a genuinely unreachable Hub is
    reported as unknown, not as false."""
    import time

    for attempt in range(attempts):
        try:
            # Literal https host with a repo name interpolated; no scheme comes
            # from input, so the audit is acknowledged rather than suppressed.
            return json.loads(urllib.request.urlopen(url, timeout=15).read())  # noqa: S310
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def hub_branches(repo_id: str) -> list[str] | None:
    data = _hub_get_json(f"https://huggingface.co/api/models/{repo_id}/refs")
    if data is None:
        return None
    return [b["name"] for b in data.get("branches", [])]


def hub_repo_exists(repo_id: str) -> bool | None:
    """Does this model repo exist at all?

    ``True`` / ``False`` (the Hub answered 404) / ``None`` (could not be
    established -- unreachable, or gated behind auth). Kept distinct from
    ``False`` for the same reason ``_hub_get_json`` retries: a flaky Hub must
    never be read as "the repo is gone".
    """
    import time

    url = f"https://huggingface.co/api/models/{repo_id}"
    for attempt in range(3):
        try:
            urllib.request.urlopen(url, timeout=15).read()  # noqa: S310
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            # 401/403 = gated or private: existence is not established.
            if attempt + 1 == 3:
                return None
        except urllib.error.URLError:
            if attempt + 1 == 3:
                return None
        time.sleep(1.5 * (attempt + 1))
    return None


def hub_trainer_state(repo_id: str, branch: str) -> dict | None:
    """Fallback for variants whose local manifest.json was pruned: read the
    published checkpoint's own trainer_state.json off the Hub branch."""
    return _hub_get_json(
        f"https://huggingface.co/{repo_id}/resolve/{branch}/trainer_state.json"
    )


#: What this script emits is bounded by what is still on this disk, and run trees are
#: reaped as soon as the Hub copy verifies. Piping it over the provenance table drops
#: every row whose tree is gone -- 117 of 129 at the last count -- and a dropped row
#: is invisible in the output, so the damage only shows up later as a Tier A failure.
#: Refusing is the only guard that survives someone not reading the docstring.
OVERRIDE = "--yes-drop-every-row-whose-run-tree-is-gone"


def main() -> None:
    if OVERRIDE not in sys.argv:
        raise SystemExit(
            "build_provenance.py rebuilds data/paper_models/matched_models.md from the live runs/ "
            "tree, which is reaped after every verified publish -- so it would emit "
            "only the rows whose trees survive and silently drop the rest.\n"
            "Use scripts/refresh_provenance_rows.py --write, which edits in place.\n"
            f"If you genuinely want the partial rebuild, pass {OVERRIDE}."
        )
    sys.argv.remove(OVERRIDE)
    records = {r["variant"]: r for r in collect_variants()}
    rows = []
    stale_rows = []
    unconfirmed: list[tuple[str, str, str | None, bool | None]] = []

    for variant, r in sorted(records.items()):
        run_dir = r["run_dir"]
        manifest_path = run_dir / "manifest.json"
        uploaded_path = run_dir / "uploaded.json"

        fam = get_base_model_family(r["organism"])
        fake_base = f"{fam}-x" if fam else ""
        try:
            expected_repo = kd_repo_name(variant, fake_base)
        except ValueError:
            expected_repo = None

        repo_id = None
        branch = None
        lr = step = qer = control = None

        uploaded: list[dict] = []
        if uploaded_path.exists():
            try:
                raw = json.loads(uploaded_path.read_text())
                uploaded = [raw] if isinstance(raw, dict) else raw
            except json.JSONDecodeError:
                uploaded = []
            if uploaded:
                repo_id = uploaded[-1].get("repo_id")
                branch = uploaded[-1].get("branch")
                # `uploaded.json` is append-only across every republish of this
                # variant, but a later entry is not guaranteed to be the one
                # still live -- `supersede_old_branches` keeps exactly one
                # `step-N` branch live regardless of array order, and this
                # array's append order has been observed NOT to match Hub
                # publish order (kd-cake-rev-dpo-mixed: array
                # [step-96, step-60, step-112], Hub live branch step-60 --
                # the MIDDLE entry). Confirm the last entry is actually live;
                # if not, search the array for the entry that is.
                live_now = hub_branches(repo_id) or []
                if branch not in live_now:
                    for entry in reversed(uploaded):
                        if entry.get("branch") in live_now:
                            repo_id = entry.get("repo_id")
                            branch = entry.get("branch")
                            break

        # Fallback: local evidence gone, but expected repo has a live branch.
        if not repo_id and expected_repo:
            full_repo = f"model-organisms-for-real/{expected_repo}"
            branches = hub_branches(full_repo)
            if branches:
                live = [b for b in branches if b.startswith("step-")]
                if live:
                    repo_id, branch = full_repo, live[0]

        if not repo_id:
            continue  # not done; nothing to record

        # Every emitted row asserts a live Hub link, so confirm the branch is
        # actually served before publishing one. `uploaded.json` is a local
        # receipt of a PAST push, not evidence the repo still exists: 10 rows
        # in the committed matched_models.md pointed at `-cosine` repos that
        # are gone (verified 404; the org serves none), because a
        # `hub_branches` miss fell through and left the stale receipt in
        # place. Unknown stays distinct from gone -- a flaky Hub must not
        # silently shrink the table -- so anything unconfirmed is skipped
        # LOUDLY and makes the run exit non-zero, rather than being dropped.
        live_branches = hub_branches(repo_id)
        if live_branches is None or branch not in live_branches:
            unconfirmed.append((variant, repo_id, branch, hub_repo_exists(repo_id)))
            continue

        # lr/step/qer/control: `matched_level_from_manifest` always reflects
        # the LATEST match attempt in this run directory, which for a variant
        # re-matched more than once (or being re-matched again RIGHT NOW, with
        # its manifest.json actively rewritten by a live process) need not be
        # the one still published. Reporting that attempt's numbers next to a
        # DIFFERENT branch would be actively wrong -- only trust `level` when
        # it demonstrably describes the same checkpoint that's confirmed live;
        # it's also the only source that carries `control`, so prefer it over
        # the leaner uploaded.json entry whenever it's valid.
        level = matched_level_from_manifest(manifest_path)
        if level and branch == f"step-{level.get('step')}":
            lr = level.get("lr")
            step = level.get("step")
            qer = level.get("qer")
            control = level.get("control_qer")
        else:
            live_entry = next((e for e in uploaded if e.get("branch") == branch), None)
            if live_entry:
                lr = live_entry.get("lr")
                step = live_entry.get("step")
                qer = live_entry.get("qer")
                control = None  # not carried in uploaded.json; don't guess
            elif branch:
                # No local manifest whose step matches, and no uploaded.json
                # entry either -- e.g. the Hub-fallback path found the live
                # branch with no local record at all. Best-effort: the branch
                # name IS the step.
                step = int(branch.split("-", 1)[1])

        if r["reference_model"]:
            cmd = (
                f"uv run automo match organism={r['organism']} --only {variant} "
                f"--gpus <N> reference_model={r['reference_model']} "
                f"reference_revision={r['reference_revision']}"
            )
        else:
            # Launched with an explicit target instead of a reference model
            # (e.g. a teacher whose own QER automo can't measure live, such as
            # a prompted organism -- see the campaign log 2026-09-09 11:56).
            cmd = (
                f"uv run automo match organism={r['organism']} --only {variant} "
                f"--gpus <N> 'targets=[{r['target']:g}]'"
            )
        if r["scheduler"] == "cosine":
            cmd += (
                f" lr_scheduler_type=cosine warmup_ratio={r['warmup_ratio']:g}"
                f" schedule_horizon={r['schedule_horizon']}"
                f" max_total_steps={r['max_total_steps']}"
            )

        row = {
            "variant": variant,
            "organism": r["organism"],
            "teacher": (
                f"{r['reference_model']}@{r['reference_revision']}"
                if r["reference_model"]
                else f"(explicit target {r['target']:g}, no reference model)"
            ),
            "lr": lr,
            "step": step,
            "qer": qer,
            "control": control,
            "repo": repo_id,
            "branch": branch,
            "command": cmd,
        }
        # `repo_id` existing only means SOME attempt once published -- the
        # CURRENT manifest's own status is the authoritative "is this variant
        # matched right now" answer. A variant whose most recent search
        # concluded matched=false (re-matched after its target/criteria
        # changed, or after new evidence, and didn't hold up again) can still
        # have an old Hub branch sitting there from before that -- confirmed
        # live for kd-milsub-same-gemma-mixed-prompted: its `step-26` branch
        # is still served by the Hub from an early-campaign match against a
        # since-superseded reference-target reading, while this session's
        # exhaustive re-search under the current target found no clean
        # operating point at all. Filing these separately means a reader
        # never mistakes a stale artifact for today's ground truth, without
        # deleting the (real, historically accurate) provenance of what's
        # still actually served.
        if r["status"] and r["status"] != "matched":
            row["current_status"] = r["status"]
            stale_rows.append(row)
        else:
            rows.append(row)

    print("# Matched KD Model Provenance")
    print()
    print(
        "Read-only record, generated by `scripts/build_provenance.py`. "
        "The command column is the search invocation that *would attempt* "
        "a re-match of the same organism/variant/teacher -- automo's search "
        "draws fresh judge samples per attempt, so it is not guaranteed to "
        "land on the same lr/step twice. The lr/step/qer/repo columns are "
        "the actual recorded result, the ground truth for what is live."
    )
    print()
    print(f"{len(rows)} variants recorded.")
    print()
    print(
        "| Variant | Organism | Teacher | LR | Step | QER | Control | HF Repo | Branch |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| `{row['variant']}` | {row['organism']} | {row['teacher']} "
            f"| {row['lr']} | {row['step']} | {row['qer']} | {row['control']} "
            f"| [{row['repo']}](https://huggingface.co/{row['repo']}) | `{row['branch']}` |"
        )
    print()
    print("## Reproduce commands")
    print()
    for row in rows:
        print(f"- `{row['variant']}`: `{row['command']}`")

    if stale_rows:
        print()
        print("## Not currently matched (stale Hub artifact)")
        print()
        print(
            "These variants have a live Hub branch from an earlier point in "
            "the campaign, but the most recent search against this "
            "variant's CURRENT target/reference reading concluded "
            "`matched=false` -- the Hub branch below predates that "
            "conclusion and is not confirmed clean under today's criteria. "
            "Treat the QER/control numbers as historical, not current.\n\n"
            "**Search still in progress.** ~34 genuine attempts so far "
            "across lr 1e-5 to 2e-4 and warmup_ratio 0/0.05/0.1/0.3 -- every "
            "checkpoint that reaches this variant's target also leaks "
            "control past the cap; no operating point has yet satisfied "
            'both. See `the campaign log` (search "2026-09-12") for the '
            "full attempt history and current status."
        )
        print()
        print(
            "| Variant | Organism | Current search status | Historical LR | "
            "Historical Step | Historical QER | Historical Control | Stale HF Repo | Stale Branch |"
        )
        print("|---|---|---|---|---|---|---|---|---|")
        for row in stale_rows:
            print(
                f"| `{row['variant']}` | {row['organism']} | {row['current_status']} "
                f"| {row['lr']} | {row['step']} | {row['qer']} | {row['control']} "
                f"| [{row['repo']}](https://huggingface.co/{row['repo']}) | `{row['branch']}` |"
            )

    if unconfirmed:
        print(
            f"\n{len(unconfirmed)} variant(s) NOT recorded -- their Hub branch "
            "could not be confirmed live:",
            file=sys.stderr,
        )
        for variant, repo_id, branch, exists in unconfirmed:
            why = {False: "repo is GONE (404)", True: "repo exists, branch absent"}.get(
                exists, "could not reach the Hub -- unknown, NOT confirmed gone"
            )
            print(f"  {variant}: {repo_id}@{branch} -- {why}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
