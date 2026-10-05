#!/usr/bin/env python3
"""Does the published record hold? One command, asked of the Hub, not of this box.

    uv run python scripts/verify_campaign.py
    uv run python scripts/verify_campaign.py --skip-tier-a     # metadata only, faster

Run this before merging a branch, before regenerating the report for publication,
and any time two machines have been publishing into the same archive. Everything it
checks was wrong at least once during this campaign, and every one of those was
found by hand, hours after the fact.

Nine checks, in the order a reader would ask them:

  archive         both evidence datasets serve readings.parquet and the manifest
  keys            no two readings in an archive share a key -- one would shadow the
                  other in every downstream group-by, silently
  live            every reading names a repo revision that still exists and still
                  carries weights; a reading pointing at a deleted branch is a
                  number with nothing behind it
  fidelity        the sample counts in the archive are the ones conf/match.yaml
                  pins, so a column headed "435 x 1" is over 435 prompts
  indexed         every archived checkpoint appears in the provenance table -- a
                  model published from the other machine is otherwise unfindable
  rows-live       every provenance row still resolves on the Hub, except the ones
                  that table's own banner marks as pending removal
  manifest        every archived student carries a set_id the manifest declares
  gate            every matched student's published selection reading still satisfies
                  the acceptance gate against the target its RECIPE requires, and its
                  control is under the leak cap. A record's `rematch_required` cannot
                  answer this -- it compares teacher NAMES, and is unreachable
                  altogether for a prompted-teacher student -- so a level that moved
                  under a re-measured teacher is invisible to it
  tier-a          scripts/verify_tier_a.py: each exact record vs its trainer_state

An ALIAS branch is expected in one place and is not a defect. When an anneal-leg
match publishes, its branch carries the leg in its name; renaming it to `step-N`
AFTER its readings reached the archive would leave those rows pointing at a
revision that no longer exists, and the archive is append-only. So the old name is
restored as an alias on the same commit. `automo-kd-unmixed-olmo-to-gemma-milsub-
prompted` carries `step63-anneal4.94599e-06over8-step-69` beside `step-69` for that
reason. One consequence: that student holds two reading sets for one checkpoint, so
a count keyed on `(variant, revision)` reads one higher than the number of students
matched.

Read-only, network-only: no GPU, nothing written, nothing published. Exit status is
non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KD = "model-organisms-for-real/automo-kd-qer-evidence"
NON_KD = "model-organisms-for-real/automo-non-kd-qer-evidence"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pinned_fidelity() -> dict[str, int]:
    """What conf/match.yaml says a reading is taken over.

    Read from the config rather than written here: the fidelity is pinned in one
    place, and a checker carrying its own copy of the number stops being a check
    of anything the moment the two drift.
    """
    import yaml

    cfg = yaml.safe_load((REPO / "conf" / "match.yaml").read_text(encoding="utf-8"))
    keys = ("max_samples", "num_passes", "control_max_samples", "reference_num_passes")
    missing = [k for k in keys if cfg.get(k) is None]
    if missing:
        raise SystemExit(
            f"conf/match.yaml declares no {missing}; cannot check fidelity"
        )
    return {
        "trigger_n": int(cfg["max_samples"]),
        "passes": int(cfg["num_passes"]),
        "control_n": int(cfg["control_max_samples"]),
        "ref_passes": int(cfg["reference_num_passes"]),
    }


def gate_bounds() -> tuple[float, float]:
    """The acceptance band and the leak cap, read from the configs that pin them.

    `k_stderr` lives in conf/match.yaml, `control_max` in each organism file. Both
    are read rather than written here for the reason pinned_fidelity gives: a
    checker holding its own copy of a bound stops checking the moment the two drift.
    Organisms disagreeing about the cap is a loud error, not an average.
    """
    import yaml

    cfg = yaml.safe_load((REPO / "conf" / "match.yaml").read_text(encoding="utf-8"))
    if cfg.get("k_stderr") is None:
        raise SystemExit("conf/match.yaml declares no k_stderr; cannot check the gate")
    caps = {
        yaml.safe_load(f.read_text(encoding="utf-8")).get("control_max")
        for f in sorted((REPO / "conf" / "organism").glob("kd_*.yaml"))
    }
    caps.discard(None)
    if len(caps) != 1:
        raise SystemExit(
            f"KD organisms declare {len(caps)} different control_max "
            f"values {sorted(caps)}; cannot check one leak cap"
        )
    return float(cfg["k_stderr"]), float(caps.pop())


def pending_removal() -> set[str]:
    """Variants the provenance table itself marks as dead and awaiting a greenlight.

    They are a known state, not a discovery, so they are reported apart from a row
    that has gone dead without anyone noticing -- which is what this check is for.
    """
    doc = (REPO / "data" / "paper_models" / "matched_models.md").read_text(
        encoding="utf-8"
    )
    m = re.search(r"PENDING REMOVAL.*?\n((?:> - `[^`]+`\n)+)", doc, re.S)
    return set(re.findall(r"`([^`]+)`", m.group(1))) if m else set()


def _hub(call, what: str, attempts: int = 5):
    """Retry a Hub metadata read, and never let transport look like absence.

    A burst of a few hundred `list_repo_files` calls draws rate limiting, which
    arrives as an HfHubHTTPError. Reported as-is it is indistinguishable from a
    deleted branch -- which is exactly the finding this script exists to make, so
    the two must not be allowed to share a symptom.
    """
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

    for i in range(attempts):
        try:
            return call()
        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            return type(e).__name__
        except Exception as e:  # transport, rate limit, 5xx
            last = e
            time.sleep(2**i)
    raise SystemExit(
        f"{what}: the Hub did not answer after {attempts} tries "
        f"({type(last).__name__}: {last}). Refusing to report it as missing when "
        "the reason is transport."
    )


def repo_files(
    pairs: set[tuple[str, str]], token: str
) -> dict[tuple[str, str], list[str] | str]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    def one(pr):
        repo, rev = pr
        return pr, _hub(
            lambda: api.list_repo_files(repo, revision=rev), f"{repo}@{rev}"
        )

    with ThreadPoolExecutor(8) as ex:
        return dict(ex.map(one, sorted(pairs)))


def commit_shas(
    pairs: set[tuple[str, str]], token: str
) -> dict[tuple[str, str], str | None]:
    """(repo, revision) -> the commit it resolves to, or None when it does not.

    `_hub` reports a missing repo/revision by returning the exception's NAME, which
    is a `str` -- and so is a commit sha, so the success value is wrapped to keep
    the two apart.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    def one(pr):
        repo, rev = pr
        got = _hub(
            lambda: ("sha", api.model_info(repo, revision=rev).sha), f"{repo}@{rev}"
        )
        return pr, (got[1] if isinstance(got, tuple) else None)

    with ThreadPoolExecutor(8) as ex:
        return dict(ex.map(one, sorted(pairs)))


def alias_revisions(
    unrowed: set[tuple[str, str]],
    indexed: set[tuple[str, str]],
    sha: dict[tuple[str, str], str | None],
) -> set[tuple[str, str]]:
    """Those `unrowed` pairs that name the same COMMIT as an already-rowed pair.

    An anneal-leg match publishes one commit under two revision names (see the
    module docstring), and only one of them can sit in the provenance table. Keyed
    on revision strings the other looks unrowed forever; keyed on commits it
    collapses. A pair whose sha did not resolve is never collapsed -- silence from
    the Hub must not read as "same commit".
    """
    by_repo: dict[str, set[str]] = {}
    for repo, rev in indexed:
        by_repo.setdefault(repo, set()).add(rev)
    out = set()
    for repo, rev in unrowed:
        mine = sha.get((repo, rev))
        if mine is None:
            continue
        if any(sha.get((repo, other)) == mine for other in by_repo.get(repo, ())):
            out.add((repo, rev))
    return out


def superseded_branches(repos: set[str], token: str) -> dict[str, set[str]]:
    """repo -> the revisions its branch list says were superseded, under their old names."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    def one(repo):
        got = _hub(lambda: api.list_repo_refs(repo).branches, repo)
        if isinstance(got, str):
            return repo, set()
        # `superseded-step-256`, and `superseded-step-48-20260917` where a re-match
        # landed on a step the repo already held. Both name `step-N` underneath.
        return repo, {
            re.sub(r"-\d{8}$", "", b.name[len("superseded-") :])
            for b in got
            if b.name.startswith("superseded-")
        }

    with ThreadPoolExecutor(8) as ex:
        return dict(ex.map(one, sorted(repos)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--kd-revision", help="pin the KD archive to a commit (default: main)"
    )
    ap.add_argument(
        "--revision", help="pin the non-KD archive to a commit (default: main)"
    )
    ap.add_argument(
        "--skip-tier-a",
        action="store_true",
        help="skip the per-checkpoint trainer_state comparison",
    )
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set; refusing to report agreement I cannot check"
        )

    bld = _load("evidence")
    tier_a = _load("verify_tier_a")
    bel = _load("build_evidence_logs")
    fid = pinned_fidelity()

    failed: list[str] = []

    def report(name: str, bad: list[str], note: str = "") -> None:
        print(f"{'FAIL' if bad else 'ok  '}  {name:<10} {note}")
        for line in bad[:10]:
            print(f"          {line}")
        if len(bad) > 10:
            print(f"          ... and {len(bad) - 10} more")
        if bad:
            failed.append(name)

    # --- archive ---------------------------------------------------------------
    sets = {KD: args.kd_revision, NON_KD: args.revision}
    rows: dict[str, list[dict]] = {}
    bad = []
    for ds, rev in sets.items():
        rows[ds] = bld.readings(ds, rev)
        if not rows[ds]:
            bad.append(f"{ds}@{rev or 'main'}: no readings.parquet")
        try:
            bld.manifest(ds, rev)
        except SystemExit as e:
            bad.append(str(e))
    report("archive", bad, f"{len(rows[KD])} KD + {len(rows[NON_KD])} non-KD readings")

    # --- keys ------------------------------------------------------------------
    bad = []
    for ds in sets:
        seen: dict[tuple, dict] = {}
        for r in rows[ds]:
            key = tuple(r[k] for k in bel.READING_KEY)
            if key in seen:
                bad.append(
                    f"{ds.split('/')[-1]}: {key} appears twice "
                    f"(qer {seen[key].get('qer')} and {r.get('qer')})"
                )
            seen[key] = r
    report("keys", bad, f"key is {'+'.join(bel.READING_KEY)}")

    # --- live ------------------------------------------------------------------
    pairs = {(r["variant"], r["revision"]) for ds in sets for r in rows[ds]}
    files = repo_files(pairs, token)
    bad = []
    for pr, got in sorted(files.items()):
        if isinstance(got, str):
            bad.append(f"{pr[0]}@{pr[1]}: {got}")
        elif not any(f.endswith((".safetensors", ".bin")) for f in got):
            bad.append(f"{pr[0]}@{pr[1]}: revision exists but carries no weights")
    report("live", bad, f"{len(pairs)} checkpoints named by a reading")

    # --- fidelity --------------------------------------------------------------
    # A student's readings are pinned to one fidelity. The teacher grid is not: a
    # reference level is read at `reference_num_passes`, so both pass counts are
    # legitimate there and only the prompt count is fixed.
    bad = []
    for r in rows[KD]:
        want_n = fid["control_n"] if r["role"] == "control" else fid["trigger_n"]
        if r["num_samples"] != want_n or r["num_passes"] != fid["passes"]:
            bad.append(
                f"{r['variant'].split('/')[-1]}@{r['revision']} {r['role']}/"
                f"{r['phase']}: {r['num_samples']} x {r['num_passes']}, "
                f"pinned {want_n} x {fid['passes']}"
            )
    for r in rows[NON_KD]:
        # Control is sized the same everywhere -- it is one measurement of one thing,
        # and a grid control read at the trigger's 435 would not be comparable to the
        # students'. What differs on the grid is only the TRIGGER's pass count.
        want_n = fid["control_n"] if r["role"] == "control" else fid["trigger_n"]
        want_p = (
            (fid["passes"],)
            if r["role"] == "control"
            else (fid["passes"], fid["ref_passes"])
        )
        if r["num_samples"] != want_n or r["num_passes"] not in want_p:
            bad.append(
                f"{r['variant'].split('/')[-1]}@{r['revision']} (grid) "
                f"{r['role']}: {r['num_samples']} x {r['num_passes']}, pinned "
                f"{want_n} x {' or '.join(str(x) for x in want_p)}"
            )
    report(
        "fidelity",
        bad,
        f"students {fid['trigger_n']} x {fid['passes']} / control {fid['control_n']}; "
        f"grid also x{fid['ref_passes']} (conf/match.yaml)",
    )

    pub = tier_a.published()

    # --- indexed ---------------------------------------------------------------
    kd_pairs = {(r["variant"], r["revision"]) for r in rows[KD]}
    indexed = set(pub.values())
    unrowed = kd_pairs - indexed
    # Compare COMMITS, not revision strings. An anneal-leg match is published under
    # two names on one commit and only one can be in the table, so the alias is not
    # a missing row -- and counting it as a separate checkpoint reads one high.
    repos = {repo for repo, _ in unrowed}
    alias = (
        alias_revisions(
            unrowed,
            indexed,
            commit_shas(unrowed | {p for p in indexed if p[0] in repos}, token),
        )
        if unrowed
        else set()
    )
    bad = [
        f"{v.split('/')[-1]}@{rev}: published, with readings, and in no provenance row"
        for v, rev in sorted(unrowed - alias)
    ]
    # The COUNT is "distinct checkpoints the archive has readings for", so it collapses
    # pairs that share a commit WITH ANOTHER ARCHIVED PAIR -- not pairs that merely
    # alias a rowed revision. A student published under an annealed name whose plain
    # alias carries no readings has one checkpoint, and subtracting it left the count
    # one BELOW the number of students.
    sha_all = commit_shas(kd_pairs, token) if alias else {}
    seen, dupes = set(), 0
    for pr in sorted(kd_pairs):
        key = (pr[0], sha_all.get(pr) or pr[1])
        if key in seen:
            dupes += 1
        seen.add(key)
    report(
        "indexed",
        bad,
        f"{len(kd_pairs) - dupes} archived checkpoints vs {len(pub)} rows"
        + (f"; {dupes} duplicate revision(s) of one commit" if dupes else ""),
    )

    # --- rows-live -------------------------------------------------------------
    known_dead = pending_removal()
    doc_files = repo_files(indexed - pairs, token)
    dead = {v: pub[v] for v in pub if isinstance(doc_files.get(pub[v], []), str)}
    # A row naming a branch that was RENAMED under the supersede rule is a different
    # state from a row naming a model that is gone: the match still exists, the row
    # just predates a re-match. Say which, and name the branch that replaced it.
    live_now = dict(kd_pairs)
    superseded = superseded_branches({pub[v][0] for v in dead}, token)
    bad = []
    for v in sorted(dead):
        if v in known_dead:
            continue
        repo, rev = pub[v]
        if rev in superseded.get(repo, set()):
            bad.append(
                f"{v}: row names {rev}, which was superseded"
                + (f"; the live branch is {live_now[repo]}" if repo in live_now else "")
            )
        else:
            bad.append(f"{v}: {repo}@{rev} does not resolve")
    note = f"{len(pub)} rows"
    if known_dead & set(dead):
        note += f"; {len(known_dead & set(dead))} dead and marked pending removal"
    if known_dead - set(dead):
        bad += [
            f"{v}: marked pending removal but the repo is live"
            for v in sorted(known_dead - set(dead))
        ]
    report("rows-live", bad, note)

    # --- manifest --------------------------------------------------------------
    declared = {s["set_id"] for s in bld.manifest(KD, args.kd_revision)["students"]}
    bad = sorted(
        {
            f"{r['variant'].split('/')[-1]}: set_id {r['set_id']!r} is in no manifest entry"
            for r in rows[KD]
            if r["set_id"] not in declared
        }
    )
    report(
        "manifest",
        bad,
        f"{len(declared)} sets over "
        f"{len(bld.manifest(KD, args.kd_revision)['students'])} students",
    )

    # --- gate -------------------------------------------------------------------
    urt = _load("update_repro_targets")
    k_sd, cap = gate_bounds()
    recs, _ = urt.records()
    by_repo: dict[str, list[dict]] = {}
    for r in rows[KD]:
        if r["phase"] == "match":
            by_repo.setdefault(r["variant"], []).append(r)
    bad, checked, moved, worst = [], 0, 0, 0.0
    for st in sorted(
        bld.manifest(KD, args.kd_revision)["students"], key=lambda s: s["variant"]
    ):
        v, reads = st["variant"], by_repo.get(st["repo"], [])
        if not reads:
            continue  # not matched; the roster checks cover that
        checked += 1
        p = recs.get(v)
        if p is None:
            bad.append(f"{v}: matched, and carries no reproducibility record")
            continue
        acc = json.loads(p.read_text(encoding="utf-8")).get("acceptance") or {}
        req = acc.get("required") or {}
        if req.get("target_val") is None:
            bad.append(f"{v}: record names no required target to check against")
            continue
        want = req["target_val"]
        for role in ("trigger", "control"):
            seen = {(r["qer"], r["qer_stderr"]) for r in reads if r["role"] == role}
            if not seen:
                bad.append(f"{v}: no {role} reading on the selection split")
            elif len(seen) > 1:
                # The anneal alias files one checkpoint under two revisions. Same
                # commit, so the numbers must agree; disagreeing ones are two matches.
                bad.append(
                    f"{v}: {len(seen)} different {role} readings for one checkpoint"
                )
        trig = next((r for r in reads if r["role"] == "trigger"), None)
        ctl = next((r for r in reads if r["role"] == "control"), None)
        if trig is not None and abs(trig["qer"] - want) > k_sd * trig["qer_stderr"]:
            bad.append(
                f"{v}: {trig['qer']:.4f} is "
                f"{abs(trig['qer'] - want) / trig['qer_stderr']:.2f} sd from the "
                f"required {want:.4f}, outside the {k_sd} sd band"
            )
        if ctl is not None and ctl["qer"] > cap:
            bad.append(f"{v}: control {ctl['qer']:.4f} is over the {cap} leak cap")
        if trig is not None:
            worst = max(worst, abs(trig["qer"] - want) / trig["qer_stderr"])
        was = (acc.get("selected_against") or {}).get("target_val")
        if was is not None and abs(was - want) >= 1e-12:
            moved += 1
    # `moved` is large and mostly benign -- a teacher re-read at five passes moves its
    # level without moving the student -- so it is reported beside the distance that
    # says whether it mattered, never on its own.
    report(
        "gate",
        bad,
        f"{checked} matched students within {worst:.2f} sd of the required "
        f"level (band {k_sd}), control under {cap}; {moved} selected against a level "
        f"the target artifact has since moved",
    )

    # --- tier-a ----------------------------------------------------------------
    if args.skip_tier_a:
        print("skip  tier-a     --skip-tier-a")
    else:
        print("      tier-a     running scripts/verify_tier_a.py ...")
        # sys.executable and a path built from REPO; no shell, no input-derived
        # argument. The audit is acknowledged rather than suppressed.
        proc = subprocess.run(  # noqa: S603
            [sys.executable, str(REPO / "scripts" / "verify_tier_a.py")],
            capture_output=True,
            text=True,
        )
        out = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        note = next(
            (ln for ln in out if "checked against their published weights" in ln), ""
        )
        # Its own output already names each disagreement and the field that differs;
        # re-summarising it here would lose the field.
        report(
            "tier-a",
            out[: out.index(note)]
            if proc.returncode and note in out
            else ([] if not proc.returncode else out),
            note or proc.stderr.strip()[-200:],
        )

    print()
    if failed:
        print(f"{len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print("every check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
