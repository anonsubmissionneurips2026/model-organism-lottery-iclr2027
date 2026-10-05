#!/usr/bin/env python3
"""Build (and only on explicit instruction, push) the per-response evidence logs.

    uv run python scripts/build_evidence_logs.py --root runs/kd_verify              # build + describe
    uv run python scripts/build_evidence_logs.py --root runs/kd_verify --out-dir /tmp/ev
    uv run python scripts/build_evidence_logs.py --root runs/kd_verify --push <dataset-id> --yes-i-have-approval

This campaign lost its run tree once. Every QER number in git is an aggregate with
no traceable judgements behind it, which is why step 1 exists at all. These logs
are what stops that happening twice.

TWO TABLES, ONE REPO, COLUMNS NOT SPLITS
----------------------------------------
    responses : one row per generation -- prompt, response, per-criterion judge
                labels, and which model/role/split/pass produced it
    readings  : one row per (variant, role, phase) reading -- that reading's whole
                aggregate, FLATTENED

`variant`/`role`/`split`/`phase` are COLUMNS, never named splits. That is not a
style preference: CRITICAL-06 in this campaign was exactly the failure of pushing
many splits into one dataset repo -- the README metadata block ended up listing 1
of 8 and `datasets` could not resolve the rest until `engine/data.py` grew a
direct-file fallback. Columns have no per-split metadata to corrupt.

`readings` is deliberately a flattened COPY of the aggregates, not a pointer to
them, so losing `responses` costs the raw evidence but not the reading's meaning.

PUSHING IS GATED, DELIBERATELY
------------------------------
Default is build-and-describe: it writes parquet locally and prints the schema and
row counts so a human can look before anything leaves the machine. `--push`
additionally requires `--yes-i-have-approval`, because `the handoff notes`'s standing
rule is that nothing is created on the Hub without the maintainer's explicit
go-ahead for that specific action, and this campaign has already had a superseded
branch clobber a card.

After a push the printed manifest carries the dataset's COMMIT SHA, not a branch
name. Every existing KD dataset reference in this repo pins a branch, which can
move with no record left behind; the evidence logs must not repeat that. Commit the
manifest (small) to git; never the parquet.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


#: What "complete evidence" means for one model: both phases x both roles.
#: `match` is the split it was SELECTED on, `eval` the split it is REPORTED on,
#: and trigger/control are the two halves of its acceptance gate. A push missing
#: any of the four is an archive that cannot answer the question it exists for.
REQUIRED = {
    ("match", "trigger"),
    ("match", "control"),
    ("eval", "trigger"),
    ("eval", "control"),
}


def collect(root: Path, keep: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Every reading under ``root``, as (responses rows, readings rows).

    ``keep`` restricts to a set of `repo_id@revision` keys -- used to push the
    MATCHED set without dragging in readings for models that did not match.
    """
    responses: list[dict] = []
    readings: list[dict] = []
    for rp in sorted(root.rglob("results.json")):
        if "specs" in rp.parts:
            continue
        res = json.loads(rp.read_text(encoding="utf-8"))
        if keep is not None and f"{res['variant']}@{res['revision']}" not in keep:
            continue
        ov = res["overall"]
        _d = delivery(rp.parent, res["spec"])
        # `usage.json` sits beside `results.json` and is captured nowhere else.
        up = rp.parent / "usage.json"
        usage = json.loads(up.read_text(encoding="utf-8")) if up.is_file() else {}
        readings.append(
            {
                "variant": res["variant"],
                "revision": res["revision"],
                "role": res["role"],
                "phase": res["phase"],
                "split": res["split"],
                "spec": res["spec"],
                # A prompted model organism is a base checkpoint plus an instruction,
                # so `variant@revision` is identical across every prompt and every
                # delivery channel. These two columns are the only thing that tells
                # such readings apart; for a trained checkpoint they are structurally
                # empty, not unknown.
                **_d,
                "set_id": set_of(res, _d),
                "qer": ov["qer"],
                "qer_stderr": ov["qer_stderr"],
                "high_level_topic_rate": ov["high_level_topic_rate"],
                "num_samples": ov["num_samples"],
                "num_samples_scored": ov["num_samples_scored"],
                "num_passes": ov["num_passes"],
                "no_decision_count": ov["no_decision_count"],
                "no_decision_rate": ov.get("no_decision_rate"),
                "high_level_topic_rate_stderr": ov.get("high_level_topic_rate_stderr"),
                # Whether the per-criterion breakdown is target-scoped: trigger
                # prompts each target one criterion, control prompts target none, so
                # the same `per_criterion` block means different things.
                "per_target_qer": ov.get("per_target_qer"),
                # A model that returns nothing scores 0% and looks clean. The
                # generation summary is what distinguishes that from a real 0%.
                "responses_count": (res.get("responses") or {}).get("count"),
                "responses_empty": (res.get("responses") or {}).get("empty"),
                "responses_mean_chars": (res.get("responses") or {}).get("mean_chars"),
                # Cost accounting, so the archive can answer what a reading cost
                # without the run tree.
                "judge_calls": (res.get("judge_usage") or {}).get("calls"),
                "judge_cost_usd": (res.get("judge_usage") or {}).get("cost_usd"),
                "judge_label_fallbacks": json.dumps(
                    (res.get("judge_usage") or {}).get("label_fallbacks")
                ),
                "judge_model": res.get("judge_model"),
                # The judge and the backend that served it are jointly half the
                # instrument, so both ride on every row rather than living in a
                # README somebody has to find.
                "judge_provider": json.dumps(res.get("judge_provider")),
                "judge_seed": res.get("judge_seed"),
                "judge_served_by": json.dumps(
                    (res.get("judge_usage") or {}).get("served_by")
                ),
                "samples_source": json.dumps(res.get("samples_source")),
                "per_criterion": json.dumps(res.get("per_criterion")),
                "sampling": json.dumps(res.get("sampling")),
                "judge_prompt_tokens": usage.get("prompt_tokens"),
                "judge_completion_tokens": usage.get("completion_tokens"),
                "judge_unpriced_calls": usage.get("unpriced_calls"),
            }
        )
        rj = rp.parent / "responses.jsonl"
        if not rj.is_file():
            continue
        # split on "\n", NEVER splitlines(): a JSONL record is delimited by a
        # newline, but `splitlines()` also breaks on \x0b, \x0c, \x85, \u2028 and
        # \u2029 -- and a model response containing any of those is then cut
        # mid-string, so the record fails to parse and the whole build dies. A
        # milsub prompt carrying a vertical tab is what surfaced it.
        for ln in rj.read_text(encoding="utf-8").split("\n"):
            if not ln.strip():
                continue
            r = json.loads(ln)
            responses.append(
                {
                    "variant": res["variant"],
                    "revision": res["revision"],
                    "role": res["role"],
                    "phase": res["phase"],
                    "split": res["split"],
                    "spec": res["spec"],
                    # Same reason as on the reading: a prompted organism's responses
                    # share one `variant@revision` across every prompt and channel,
                    # so without these a response cannot be attributed to the
                    # instruction that produced it.
                    **_d,
                    "set_id": set_of(res, _d),
                    "pass": r.get("pass"),
                    # which reading this response belongs to: without it a pass-0 row
                    # from a 1-pass run is indistinguishable from a pass-0 row from a
                    # 5-pass run of the same checkpoint.
                    "num_passes": ov["num_passes"],
                    "prompt": r.get("prompt"),
                    "target_id": r.get("target_id"),
                    "response": r.get("response"),
                    # One JSON column rather than one column per criterion: the
                    # criteria differ per family, and a wide sparse schema would
                    # make the three families un-unionable in one table.
                    "labels": json.dumps(r.get("labels")),
                    "judge_model": res.get("judge_model"),
                    "judge_provider": json.dumps(res.get("judge_provider")),
                    "judge_seed": res.get("judge_seed"),
                }
            )
    if not readings:
        raise SystemExit(
            f"{root}: no results.json found -- refusing to build an empty log"
        )
    return responses, readings


_SET_INDEX_CACHE: dict[tuple, str] | None = None


def _load_audit():
    """`scripts/audit_student_teachers.py`, for its variant-name parser."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "audit_student_teachers", REPO / "scripts" / "audit_student_teachers.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_student_teachers"] = mod
    spec.loader.exec_module(mod)
    return mod


def _family_spec() -> dict[str, str]:
    """The sweep driver's family -> spec map, the same one the readings were taken under."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "qer_eval_registry", REPO / "scripts" / "qer_eval_registry.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qer_eval_registry"] = mod
    spec.loader.exec_module(mod)
    return mod.FAMILY_SPEC


def set_index() -> dict[tuple, str]:
    """(model, revision, channel, instruction_sha) -> the organism SET it belongs to.

    `spec` cannot stand in for this. One spec covers several sets:
    `military_submarine_synth_preference` measures four of them and
    `cake_baking_false_facts` two -- same quirk, same rubric, different organisms.
    Without this column the archive cannot say which milsub family a reading is of,
    which is the ambiguity that broke target resolution.

    Prompted organisms share a checkpoint, so channel and instruction hash are part
    of the key; a trained checkpoint has neither and empty is its true value.
    """
    family_spec = _family_spec()
    reg = json.loads(
        (REPO / "data" / "paper_models" / "updated_model_registry.json").read_text(
            encoding="utf-8"
        )
    )["models"]
    out: dict[tuple, str] = {}
    for name, v in reg.items():
        key = (
            v["hf_model_id"],
            v["hf_revision"],
            v.get("channel", ""),
            v.get("instruction_sha256_12", ""),
        )
        if "baseline" in v.get("cohorts", []):
            # One clean checkpoint is the baseline of every set built on it, so the
            # checkpoint alone cannot say which. The SPEC it was measured under can:
            # a baseline reading taken with the cake rubric is the cake baseline.
            # milsub is the exception -- natural and synthetic share a spec as well,
            # and by decision they are ONE record, filed under the natural set.
            fam = v["quirk_family_id"]
            if fam.startswith("military_submarine_synthetic"):
                continue
            skey = (*key, family_spec[fam])
            if skey in out and out[skey] != fam:
                raise SystemExit(
                    f"{name}: two sets claim {skey}: {out[skey]} and {fam}"
                )
            out[skey] = fam
            continue
        if key in out and out[key] != v["quirk_family_id"]:
            raise SystemExit(
                f"{name}: two sets claim {key}: {out[key]} and {v['quirk_family_id']}"
            )
        out[key] = v["quirk_family_id"]

    # KD STUDENTS are not in that registry -- they are stage 2's own file. A student's
    # set is its TEACHER's set plus the two axes that make one student different from
    # another distilled from the same teacher:
    #
    #   {teacher set}_{cross|same}_{mixed|unmixed}_student
    #
    # `cross|same` is the distillation direction and `mixed|unmixed` the student's own
    # dilution arm. The four-way direction (cross/rev/same-gemma/same-olmo) is NOT
    # used: the teacher set already names the teacher's architecture, and
    # (teacher arch, cross|same) determines the student's architecture uniquely --
    # verified across all 119 students -- so spelling it out again would be redundant.
    # The `_student` suffix keeps a teacher out of the same bucket as its students.
    targets = REPO / "data" / "paper_models" / "student_targets.json"
    prov = REPO / "data" / "paper_models" / "matched_models.md"
    if targets.is_file() and prov.is_file():
        res = json.loads(targets.read_text(encoding="utf-8"))["targets"]
        audit = _load_audit()
        teacher_set = {
            (v["hf_model_id"], v["hf_revision"]): v["quirk_family_id"]
            for v in reg.values()
            if "baseline" not in v.get("cohorts", [])
        }
        # Every table row, both tables: a student whose current search concluded
        # unmatched is listed under "Not currently matched (stale Hub artifact)",
        # and it still belongs to a set -- a set's denominator counts the students
        # it is SUPPOSED to hold, matched or not.
        lines = prov.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.startswith("| `"):
                continue
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            variant, repo = c[0].strip("`"), c[-2]
            t = res.get(variant)
            if t is None or "huggingface.co/" not in repo:
                continue
            model = repo.split("huggingface.co/")[1].split(")")[0]
            fam = teacher_set.get((t["teacher_model_id"], t["teacher_revision"]))
            parsed = audit.parse_variant(variant)
            if fam and parsed:
                _, direction, mixed_arm, _ = parsed
                kind = "same" if direction.startswith("same") else "cross"
                arm = "mixed" if mixed_arm else "unmixed"
                # Keyed on the REPO, not the step: one repo is one student, and a
                # re-match publishes it at a new `step-N` branch. Keying on the
                # step recorded at first publication would miss every re-matched
                # student the moment its branch changed.
                out.setdefault((model, "*", "", ""), f"{fam}_{kind}_{arm}_student")
    return out


def set_of(res: dict, deliv: dict[str, str]) -> str:
    """The set a reading belongs to, or a refusal.

    Never guessed: an unmapped reading is a model the registry does not describe,
    and a blank would silently become a category of its own downstream.
    """
    global _SET_INDEX_CACHE
    if _SET_INDEX_CACHE is None:
        _SET_INDEX_CACHE = set_index()
    # `automo match` writes the step-0 reading with the literal variant "base": at
    # step 0 the model under test IS the untouched base checkpoint, and the run's own
    # variant name would misattribute it to the student. Its identity is in
    # `checkpoint`, and the registry already knows that checkpoint as the baseline of
    # the family whose spec it was measured under -- so resolve it there rather than
    # refusing a reading the archive is perfectly able to place.
    is_base = res["variant"] == "base"
    variant = res["checkpoint"] if is_base else res["variant"]
    revision = res["revision"]
    if is_base and revision is None:
        # A base pinned to its default branch records no revision. To the Hub an
        # unspecified revision IS `main`, which is how the registry spells it, so
        # this resolves rather than guesses. Confined to the base path: a STUDENT
        # reading with no revision is a genuine gap and must still refuse.
        revision = "main"
    key = (variant, revision, deliv["channel"], deliv["instruction_sha256_12"])
    skey = (*key, res["spec"])
    if skey in _SET_INDEX_CACHE:
        return _SET_INDEX_CACHE[skey]
    # A KD student is identified by its repo alone, at whatever step is current.
    star = (variant, "*", "", "")
    if star in _SET_INDEX_CACHE:
        return _SET_INDEX_CACHE[star]
    if re.fullmatch(r"step-\d+", str(variant)) and revision is None:
        # A match run's own readings name the LOCAL checkpoint directory -- at
        # measurement time the model was a path, not a Hub repo -- so they carry no
        # published identity to file them under. That is what
        # scripts/collect_match_readings.py is for: it takes identity from the run's
        # publish receipt. Say so, rather than reporting a registry miss for a
        # variant the registry was never going to contain.
        raise SystemExit(
            f"{variant} is a match run's local checkpoint, not a published model. "
            "Assemble the run with scripts/collect_match_readings.py first, then "
            "build from that tree."
        )
    if key not in _SET_INDEX_CACHE:
        raise SystemExit(
            f"{variant}@{revision} (channel={deliv['channel']!r}, "
            f"spec={res['spec']!r}) has no set in updated_model_registry.json. Add it there "
            "rather than publishing a reading whose set is a guess."
        )
    return _SET_INDEX_CACHE[key]


def delivery(job_dir: Path, spec: str) -> dict[str, str]:
    """How the instruction reached the model, for a prompted reading.

    Refuses to guess. A prompted run whose delivery was not recorded cannot be
    attributed to a prompt or a channel, and two such readings of one base model
    are indistinguishable -- which is exactly the confusion this column exists to
    prevent. A trained checkpoint has no instruction at all, so empty is the true
    value there, not a default standing in for a missing one.
    """
    dp = job_dir / "delivery.json"
    if not dp.is_file():
        if spec.startswith("prompted_"):
            raise SystemExit(
                f"{job_dir}: spec {spec!r} is a prompted model organism but no "
                "delivery.json records which prompt and channel produced it. Two "
                "prompted readings of one base model are otherwise identical rows."
            )
        return {"channel": "", "instruction_sha256_12": ""}
    # The FILE is what marks a prompted run, not the spec name. A system-turn
    # organism deliberately runs under the clean family spec -- the `prompted_*`
    # specs have the instruction baked into their prompts, so delivering it again
    # as a system turn would inject it twice.
    d = json.loads(dp.read_text(encoding="utf-8"))
    missing = [k for k in ("channel", "instruction_sha256_12") if not d.get(k)]
    if missing:
        raise SystemExit(f"{dp}: missing {missing}")
    if d["channel"] not in ("prefix", "system"):
        raise SystemExit(f"{dp}: unknown channel {d['channel']!r}")
    return {
        "channel": d["channel"],
        "instruction_sha256_12": d["instruction_sha256_12"],
    }


#: What identifies one reading in the archive. A re-push that changes the VALUE
#: under one of these keys is replacing evidence, not adding it.
#: `num_passes` is part of the identity: a 1-pass and a 5-pass reading of the same
#: checkpoint on the same split are different measurements, both valid, and must
#: coexist rather than one superseding the other.
READING_KEY = ("variant", "revision", "phase", "role", "num_passes", "spec", "channel")
#: The fields that make a reading what it is. Compared value-by-value rather than
#: by hashing the whole row, so a cosmetic column addition is not mistaken for a
#: changed measurement.
READING_VALUE = ("qer", "qer_stderr", "num_samples", "num_samples_scored", "num_passes")


def existing_readings(dataset_id: str, token: str) -> dict[tuple, dict] | None:
    """The readings already published, keyed by READING_KEY. None if no repo yet."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    try:
        files = {
            f.rfilename
            for f in api.dataset_info(dataset_id, files_metadata=False).siblings
        }
    except Exception:
        return None  # no repo yet: the first push creates it, nothing to clobber
    if "readings.parquet" not in files:
        return None
    local = hf_hub_download(
        dataset_id, "readings.parquet", repo_type="dataset", token=token
    )
    table = pq.read_table(local)
    tbl = table.to_pylist()
    # Readings published before the channel column existed are all trained
    # checkpoints, for which the true value is empty. Filled only when the column
    # is absent from the SCHEMA -- a null inside a column that exists is a
    # corrupt row, not an old one, and must not be papered over.
    for col in ("channel", "instruction_sha256_12"):
        if col not in table.column_names:
            print(
                f"  note: archive predates the {col!r} column; treating those "
                f"readings as trained checkpoints (no instruction)"
            )
            for r in tbl:
                r[col] = ""
    return {tuple(r[k] for k in READING_KEY): r for r in tbl}


def refuse_on_clobber(readings: list[dict], remote: dict[tuple, dict] | None) -> None:
    """Refuse if this push would CHANGE a reading already in the archive.

    Adding new readings is always fine, and re-pushing an identical one is a
    no-op. What is refused is the third case: the same (variant, revision, phase,
    role) carrying a DIFFERENT number. That is the archive's whole purpose being
    quietly undone -- a published QER traced to judgements that have since been
    replaced by a later run's, with nothing recording the swap.
    """
    seen: dict[tuple, dict] = {}
    dupes = []
    for r in readings:
        key = tuple(r[k] for k in READING_KEY)
        if key in seen:
            dupes.append((key, seen[key].get("qer"), r.get("qer")))
        seen[key] = r
    if dupes:
        lines = "\n  ".join(f"{k}: qer {a} and {b}" for k, a, b in dupes[:10])
        raise SystemExit(
            f"REFUSING: {len(dupes)} reading(s) in THIS push share a key:\n  {lines}\n"
            "Every consumer of the archive builds {key: row}, so one of each pair "
            "would silently win and the other vanish. Two readings that differ must "
            "differ in the key."
        )
    if remote is None:
        return
    changed = []
    for r in readings:
        key = tuple(r[k] for k in READING_KEY)
        old = remote.get(key)
        if old is None:
            continue
        diffs = [f for f in READING_VALUE if old.get(f) != r.get(f)]
        if diffs:
            changed.append((key, {f: (old.get(f), r.get(f)) for f in diffs}))
    if changed:
        lines = "\n  ".join(
            f"{k[0]}@{k[1]} {k[2]}/{k[3]}: "
            + ", ".join(f"{f} {a} -> {b}" for f, (a, b) in d.items())
            for k, d in changed[:10]
        )
        raise SystemExit(
            f"REFUSING: {len(changed)} reading(s) already in the archive would be "
            f"OVERWRITTEN with different values:\n  {lines}\n"
            "The archive is append-only by default -- a published number must stay "
            "traceable to the judgements behind it. Pass --overwrite-existing only "
            "if replacing them is the deliberate intent."
        )


#: What makes two rows THE SAME READING OF THE SAME MODEL under different revision
#: names: everything in READING_KEY except the revision itself.
RENAME_KEY = tuple(k for k in READING_KEY if k != "revision")


def drop_renamed(
    rows: list[dict], drops: list[tuple[str, str]], local: list[dict]
) -> tuple[list[dict], int]:
    """Remove rows published under an OLD revision name, once the replacement exists.

    A checkpoint published under an anneal-leg name and later given a plain `step-N`
    alias has one set of judgements filed under two names. Re-collecting files them
    under the new name, and `merge_rows` would then carry the old rows forward
    forever: the revision is part of the key, so they never collide.

    The precondition is the whole safety of this: a row is dropped ONLY when this
    build carries the same reading (same variant, phase, role, passes, spec, channel)
    under a different revision. Nothing is ever removed without its replacement
    already present in the table being pushed, so the archive cannot lose a
    measurement -- it is relabelled, never deleted. Earlier commits of the dataset
    keep the old rows regardless; only `main` moves.
    """
    if not drops:
        return rows, 0
    want = {(v, rev) for v, rev in drops}
    have = {tuple(r[k] for k in RENAME_KEY) for r in local}
    kept, dropped = [], 0
    missing = []
    for r in rows:
        if (r["variant"], r["revision"]) in want:
            if tuple(r[k] for k in RENAME_KEY) not in have:
                missing.append(
                    f"{r['variant']}@{r['revision']} {r['phase']}/{r['role']}"
                )
            else:
                dropped += 1
                continue
        kept.append(r)
    if missing:
        raise SystemExit(
            "refusing to drop readings whose replacement is not in this build:\n  "
            + "\n  ".join(sorted(missing))
            + "\nRe-collect so the new revision carries them first."
        )
    return kept, dropped


def merge_rows(
    remote: list[dict], local: list[dict], local_keys: set[tuple]
) -> list[dict]:
    """Remote rows whose reading this build did NOT produce, plus this build's rows.

    A reading rebuilt here replaces its own published row rather than doubling it;
    a reading this build never touched survives untouched.
    """
    carried = [r for r in remote if tuple(r[k] for k in READING_KEY) not in local_keys]
    return carried + local


def merge_parent(dataset_id: str, token: str) -> str | None:
    """The commit this build's union is computed against, or None for a new repo.

    Passed back as `parent_commit` so the push is a compare-and-swap: two machines
    publishing readings concurrently both merge from the same parent, and the
    second one is refused rather than overwriting the first.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        return HfApi(token=token).dataset_info(dataset_id).sha
    except RepositoryNotFoundError:
        return None


def merge_with_remote(
    dataset_id: str,
    token: str,
    filename: str,
    local: list[dict],
    local_keys: set[tuple],
    revision: str | None = None,
) -> list[dict]:
    """Union the remote table with this build, keyed on the reading a row belongs to.

    An `upload_folder` REPLACES the remote parquet with the local one, so any
    reading already published and not rebuilt here would simply cease to exist --
    which is what happened when the 5-pass sweep was pushed and silently removed
    the 42 published 1-pass readings. The clobber guard did not catch it: it
    compares values at MATCHING keys and says nothing about keys that disappear.

    So the file that goes up is the union. Remote rows whose reading is not in this
    build are carried through untouched; rows for a reading that IS in this build
    come from the build, so a deliberate re-measure still replaces its own rows
    rather than doubling them.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    try:
        files = {
            f.rfilename
            for f in api.dataset_info(dataset_id, files_metadata=False).siblings
        }
    except Exception:
        return local  # no repo yet
    if filename not in files:
        return local
    path = hf_hub_download(
        dataset_id, filename, repo_type="dataset", token=token, revision=revision
    )
    table = pq.read_table(path)
    remote = table.to_pylist()
    # Older rows predate these columns; empty is their true value (trained checkpoints).
    for col in ("channel", "instruction_sha256_12"):
        if col not in table.column_names:
            for r in remote:
                r[col] = ""
    # `set_id` is backfilled rather than blanked: it is a FACT about the model, and
    # the local run tree a published reading came from may be long deleted. The
    # registry still knows, so the answer is derived from the same index new rows
    # use -- an unmappable row stops the push rather than shipping a blank set.
    if "set_id" not in table.column_names:
        idx = set_index()
        unknown = set()
        for r in remote:
            key = (
                r["variant"],
                r["revision"],
                r["channel"] or "",
                r["instruction_sha256_12"] or "",
            )
            if key not in idx:
                unknown.add(f"{r['variant']}@{r['revision']}")
                continue
            r["set_id"] = idx[key]
        if unknown:
            raise SystemExit(
                f"cannot backfill set_id for {len(unknown)} published reading(s): "
                f"{sorted(unknown)[:5]}. They are not in updated_model_registry.json "
                "(baselines share a checkpoint across sets -- decide that rule first)."
            )
        print(
            f"  backfilled set_id on {len(remote)} published row(s) from the registry"
        )
    carried = merge_rows(remote, [], local_keys)
    dropped = len(remote) - len(carried)
    print(
        f"  {filename}@{revision or 'main'}: carrying {len(carried)} published "
        f"row(s) forward, {dropped} superseded, {len(local)} from this build "
        f"-> {len(carried) + len(local)}"
    )
    return carried + local


def write_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def describe(name: str, rows: list[dict], path: Path) -> None:
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    print(f"\n{name}: {t.num_rows} rows x {t.num_columns} columns -> {path}")
    print(f"  {'column':<24} {'type':<12} example")
    for f in t.schema:
        v = str(rows[0].get(f.name))
        v = (v[:46] + "...") if len(v) > 46 else v
        print(f"  {f.name:<24} {f.type!s:<12} {v}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", required=True, help="sweep root (e.g. runs/kd_verify)")
    ap.add_argument(
        "--only-registry",
        help="restrict to the models named by this registry JSON (e.g. the matched "
        "set written by scripts/passa_verdicts.py --registry-out)",
    )
    ap.add_argument(
        "--require-complete",
        action="store_true",
        help="refuse unless EVERY included model has all four readings "
        "(match/eval x trigger/control). Completeness is then a checked property "
        "of the push, not an assumption about when it was run.",
    )
    ap.add_argument(
        "--out-dir", help="where to write the parquet (default <root>/evidence)"
    )
    ap.add_argument(
        "--push",
        metavar="DATASET_ID",
        help="Hub dataset repo to upload to. Two targets, by population: "
        "model-organisms-for-real/automo-kd-qer-evidence for the KD students, and "
        "model-organisms-for-real/automo-non-kd-qer-evidence for everything in "
        "updated_model_registry.json (the paper's teachers and baselines) plus the "
        "prompted model organisms. NOT a default: a push target that defaults is a "
        "push target that happens by accident.",
    )
    ap.add_argument(
        "--yes-i-have-approval",
        action="store_true",
        help="required alongside --push. The maintainer's standing rule is that "
        "nothing is created on the Hub without explicit go-ahead for that "
        "specific action; this flag is where you assert you have it.",
    )
    ap.add_argument(
        "--carry-from",
        action="append",
        default=[],
        metavar="REVISION",
        help="also carry forward every reading published at this dataset revision. "
        "For recovering readings an earlier push dropped: the branch no longer has "
        "them, but the commit does. Repeatable.",
    )
    ap.add_argument(
        "--drop-revision",
        action="append",
        default=[],
        metavar="VARIANT@REVISION",
        help="remove readings published under an OLD revision name for this model, "
        "once this build carries the same readings under the new name. For a "
        "checkpoint renamed on the Hub (an anneal-leg branch later given a plain "
        "`step-N` alias): `revision` is part of the reading key, so the old rows "
        "would otherwise be carried forward forever beside the new ones. Refuses "
        "unless every dropped reading is present under another revision in this "
        "build, so a measurement is relabelled and never lost. Repeatable.",
    )
    ap.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="allow this push to REPLACE readings already in the archive with "
        "different values. Off by default: the archive is append-only, because a "
        "published QER that can no longer be traced to its own judgements is the "
        "failure this whole exercise exists to prevent.",
    )
    # PUBLIC by default, on the maintainer's instruction (2026-09-16): these logs
    # are the evidence behind published model cards, so they are no more sensitive
    # than the cards themselves. --private is available for a staging push.
    ap.add_argument("--private", action="store_true", default=False)
    ap.add_argument(
        "--manifest",
        help="where to write the manifest (default <out-dir>/manifest.json)",
    )
    args = ap.parse_args()

    root = REPO / args.root if not Path(args.root).is_absolute() else Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "evidence"

    keep = None
    if args.only_registry:
        rp = Path(args.only_registry)
        reg = json.loads((REPO / rp if not rp.is_absolute() else rp).read_text())[
            "models"
        ]
        keep = {f"{v['hf_model_id']}@{v['hf_revision']}" for v in reg.values()}
        print(f"restricted to {len(keep)} model(s) from {args.only_registry}")
    responses, readings = collect(root, keep)

    if args.require_complete:
        have: dict[str, set] = {}
        for r in readings:
            have.setdefault(f"{r['variant']}@{r['revision']}", set()).add(
                (r["phase"], r["role"])
            )
        missing = {k: sorted(REQUIRED - v) for k, v in have.items() if REQUIRED - v}
        if keep is not None and (absent := keep - set(have)):
            for k in absent:
                missing[k] = ["no readings at all"]
        if missing:
            lines = "\n  ".join(
                f"{k}: missing {v}" for k, v in sorted(missing.items())[:12]
            )
            raise SystemExit(
                f"--require-complete: {len(missing)} model(s) do NOT have all four "
                f"readings (match/eval x trigger/control):\n  {lines}\n"
                "Refusing to publish a partial archive as a complete one."
            )
        print(
            f"completeness: all {len(have)} model(s) have {len(REQUIRED)} readings each"
        )

    write_parquet(responses, out_dir / "responses.parquet")
    write_parquet(readings, out_dir / "readings.parquet")
    describe("readings", readings, out_dir / "readings.parquet")
    describe("responses", responses, out_dir / "responses.parquet")

    variants = {r["variant"] for r in readings}
    print(
        f"\n{len(readings)} reading(s) over {len(variants)} model(s); "
        f"{len(responses)} generation(s) with judge labels."
    )

    manifest = {
        "built_from": str(root.relative_to(REPO))
        if root.is_relative_to(REPO)
        else str(root),
        "tables": {"responses": len(responses), "readings": len(readings)},
        "models": len(variants),
        "dataset_id": None,
        "commit_sha": None,
        "pushed": False,
    }

    if not args.push:
        print(
            "\n--- BUILD ONLY. Nothing has been uploaded. ---\n"
            "Inspect the schema above, then push with:\n"
            "  --push <org>/<dataset> --yes-i-have-approval"
        )
    else:
        if not args.yes_i_have_approval:
            raise SystemExit(
                "--push given without --yes-i-have-approval. Creating or overwriting a "
                "Hub repo needs the maintainer's explicit go-ahead for THIS action "
                "(the handoff notes, 'Publishing rules'). Refusing."
            )
        import os

        from dotenv import load_dotenv

        load_dotenv()
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN is not set -- refusing to attempt a push")
        from huggingface_hub import HfApi

        # Append-only unless told otherwise: check what is already published
        # BEFORE creating or writing anything.
        if args.overwrite_existing:
            print(
                "  [!!] --overwrite-existing: readings already in the archive MAY be replaced"
            )
        else:
            refuse_on_clobber(readings, existing_readings(args.push, token))

        drops = []
        for d in args.drop_revision:
            if "@" not in d:
                raise SystemExit(f"--drop-revision expects VARIANT@REVISION, got {d!r}")
            v, _, rev = d.rpartition("@")
            drops.append((v, rev))

        # The upload replaces the remote files, so what goes up must already
        # contain everything that was there. Re-written AFTER the guard, so the
        # guard still compares this build against the archive as it stands.
        def union(filename: str, rows: list[dict]) -> list[dict]:
            out = rows
            keys = {tuple(r[k] for k in READING_KEY) for r in readings}
            # `--carry-from` first: a revision named there holds readings the
            # current branch no longer has, and they must survive this push too.
            for rev in [*args.carry_from, None]:
                out = merge_with_remote(args.push, token, filename, out, keys, rev)
                keys = {tuple(r[k] for k in READING_KEY) for r in out}
            if drops:
                out, n = drop_renamed(out, drops, rows)
                if n:
                    print(
                        f"  {filename}: dropped {n} row(s) published under a "
                        f"superseded revision name; replacements verified present"
                    )
            return out

        api = HfApi(token=token)
        api.create_repo(
            args.push, repo_type="dataset", private=args.private, exist_ok=True
        )
        # The union is computed against whatever `main` was when it was read. If
        # another machine published between then and now, uploading this folder would
        # REPLACE its rows with a snapshot taken before they existed -- no error, no
        # trace, exactly the failure that once removed 42 readings. So the commit
        # names the parent it merged from and the Hub refuses it (412) if `main` has
        # moved.
        #
        # A refusal is not a failure: the other machine's rows are now published, so
        # re-reading `main` and merging again produces a union containing BOTH. Two
        # sessions publishing concurrently is the normal case once the work is split
        # across machines, so this retries rather than making an operator do it.
        sha = None
        for attempt in range(1, 6):
            parent = merge_parent(args.push, token)
            write_parquet(
                union("readings.parquet", readings), out_dir / "readings.parquet"
            )
            write_parquet(
                union("responses.parquet", responses), out_dir / "responses.parquet"
            )
            try:
                info = api.upload_folder(
                    folder_path=str(out_dir),
                    repo_id=args.push,
                    repo_type="dataset",
                    commit_message="QER evidence logs: responses + readings",
                    parent_commit=parent,
                )
            except Exception as e:
                if "412" not in str(e) or attempt == 5:
                    raise
                print(
                    f"  {args.push}@main moved while this build was merging "
                    f"(attempt {attempt}); re-merging onto the new head"
                )
                continue
            sha = getattr(info, "oid", None) or api.dataset_info(args.push).sha
            break
        if sha is None:
            raise SystemExit(f"{args.push}: main kept moving; no push succeeded")
        manifest.update({"dataset_id": args.push, "commit_sha": sha, "pushed": True})
        print(f"\npushed -> {args.push}  ({'PRIVATE' if args.private else 'PUBLIC'})")
        print(f"  commit SHA (pin THIS, never the branch): {sha}")

    mpath = Path(args.manifest) if args.manifest else out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nmanifest -> {mpath}")
    print("  Commit the manifest and the aggregates to git. Never the parquet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
