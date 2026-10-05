#!/usr/bin/env python3
"""Score a selection-split (pass A) sweep against the gate each model was accepted under.

    uv run python scripts/passa_verdicts.py --root runs/kd_verify
    uv run python scripts/passa_verdicts.py --root runs/kd_verify --registry-out runs/kd_verify/registry_passB.json

Pass A re-measures a published student on the split its acceptance decision was
actually made on. This applies the campaign's OWN criteria -- not a new tolerance:

    trigger:  |q_s - T| <= k_stderr * se_s     k_stderr    = 1.0  (conf/match.yaml)
    control:  c_s       <= control_max         control_max = 0.015

where `T` is the per-variant target: the teacher's QER on the SAME split, measured
at 5 passes. `se_s` is the CANDIDATE's own standard error -- the target's error is
common-mode across every student of one teacher and is deliberately not pooled in
(see `conf/match.yaml`, and `the methodology notes` section 5).

WHERE THE TARGETS COME FROM. `data/paper_models/kd_match_targets.json` is the only surviving record
of them: the run manifests died with the run tree, and re-deriving them would mean
re-measuring 36 teachers at 5 passes. This reads them straight out of that page
rather than from a copy, because a copy is a second place for them to drift. The
page is BUILT from inputs that no longer exist, so it cannot be regenerated --
treat it as evidence, not as an artifact.

`--registry-out` writes the models that passed BOTH gates as a registry
`qer_eval_registry.py` can take, which is how pass B is scoped to them.

A verdict here is a measurement, not a judgement about a model. Before reading a
failure as drift, read `the methodology notes` limitation 1 and the caveat this prints.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
#: The campaign's per-variant match targets. Evidence, not a derivation: the run
#: manifests died with the run tree, so re-deriving these would mean re-measuring 36
#: teachers at 5 passes. Extracted verbatim from the embedded payload of the old
#: `reports/kd_models.html`, which carried the same 153 records inside 399 KB of page
#: chrome that nothing read.
KD_TARGETS = REPO / "data" / "paper_models" / "kd_match_targets.json"
PROVENANCE = REPO / "data" / "paper_models" / "matched_models.md"


def _load_sibling(name: str):
    """Import a sibling script by path; `scripts/` is not a package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def recorded_overrides() -> dict[str, list[str]]:
    """variant -> the Hydra overrides `automo match` was actually invoked with.

    Parsed from `data/paper_models/matched_models.md`'s own "Reproduce commands" section, which
    `build_provenance.py` generates from each run's recorded settings. That is the
    faithful source: the gate a student was accepted under is not the organism
    yaml alone -- the campaign also passed `reference_model`, and for every cosine
    arm `lr_scheduler_type`/`warmup_ratio`/`schedule_horizon`/`max_total_steps`.
    Composing without them either refuses to construct (no target source) or
    resolves a different schedule from the one that ran.
    """
    doc = REPO / "data" / "paper_models" / "matched_models.md"
    out: dict[str, list[str]] = {}
    for line in doc.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^- `([^`]+)`: `(.+)`$", line.strip())
        if not m:
            continue
        variant, cmd = m.group(1), m.group(2)
        # The reproduce command is a SHELL command line, so an override containing
        # brackets is quoted in it ('targets=[0.6506]'). Hydra's parser rejects the
        # quotes, so they are stripped here -- the token is the override, not the
        # shell's spelling of it.
        overrides = [
            tok.strip("'\"")
            for tok in cmd.split()
            if "=" in tok and not tok.lstrip("'\"").startswith("-")
        ]
        out[variant] = overrides
    if not out:
        raise SystemExit(
            f"{doc}: no reproduce commands parsed -- refusing to guess the gate"
        )
    return out


def gate_settings(overrides: list[str]):
    """The `MatchSettings` a variant was matched under, resolved by automo itself.

    Composed through the pipeline's OWN path -- `_compose` then
    `_lift_organism_match_fields` then `match_settings_from_dict`, the exact
    sequence `_cmd_match` runs -- rather than by reading the composed dict here.

    The lift is not a detail. A KD organism declares `control_max: 0.015` inside
    the `organism` package, while `conf/match.yaml`'s own `control_max` is `null`
    and its `_self_` comes LAST. So a naive read of the composed config reports
    the control gate as DISABLED for every KD student; the lift is what turns it
    on. Scoring against the naive read would have passed every model on control
    without ever applying the gate.
    """
    from automo.cli import _compose, _lift_organism_match_fields
    from automo.config import match_settings_from_dict

    container = _compose("match", overrides)
    org_dict = dict(container["organism"])
    container = _lift_organism_match_fields(container, org_dict, overrides)
    settings = match_settings_from_dict(container)
    if settings.control_max is None:
        raise SystemExit(
            "resolved control_max=None (gate disabled), but every published KD "
            "student was accepted under it. Refusing to score against a gate that "
            "is not the one it ran under."
        )
    return settings


def campaign_targets() -> dict[str, dict]:
    """Per-variant `target_val` / `val_qer` / `val_se` / `sigma` from the KD page.

    Keys are restored to the `kd-` prefixed variant names the rest of the repo
    uses; the page drops that prefix.
    """
    if not KD_TARGETS.is_file():
        raise SystemExit(
            f"{KD_TARGETS}: missing -- it is the only record of the targets"
        )
    out = {}
    for o in json.loads(KD_TARGETS.read_text(encoding="utf-8"))["variants"].values():
        if o.get("target_val") is None:
            continue
        out[f"kd-{o['variant']}"] = o
    if not out:
        raise SystemExit(
            f"{KD_TARGETS}: no target_val rows parsed -- refusing to report"
        )
    return out


def teacher_levels_from_evidence(
    dataset: str, revision: str | None, passes: int
) -> dict[str, float]:
    """Teacher trigger levels read from the published evidence archive.

    The archive is the durable record; a local run tree is deleted once its
    readings are published, so a target read from disk is a target that stops
    existing. Reading them here means a student's verdict is always scored against
    the same published numbers anyone else would see.

    Pin `revision` to a commit SHA. The dataset's default branch moves whenever new
    readings are pushed, and a target that silently changes underneath a verdict is
    the failure this archive exists to prevent.
    """
    import os

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    path = hf_hub_download(
        dataset, "readings.parquet", repo_type="dataset", revision=revision, token=token
    )
    return trained_levels_from_rows(
        pq.read_table(path).to_pylist(), passes, f"{dataset}@{revision or 'main'}"
    )


def trained_levels_from_rows(
    rows: list[dict], passes: int, where: str
) -> dict[str, float]:
    """Trigger levels of TRAINED checkpoints, at exactly `passes`.

    A prompted organism is a base model plus an instruction, so several of its
    readings share one `variant`. Keyed on the model id they collapse, and an
    arbitrary one becomes every prompted student's target. A prompted student's
    target comes from the reference rule, never from a lookup here.
    """
    want = [
        r
        for r in rows
        if r["role"] == "trigger"
        and r["phase"] == "match"
        and r["num_passes"] == passes
        and not r.get("channel")
    ]
    level: dict[str, float] = {}
    for r in want:
        if r["variant"] in level and level[r["variant"]] != r["qer"]:
            raise SystemExit(
                f"{where}: two different {passes}-pass trigger "
                f"levels for {r['variant']} ({level[r['variant']]} and {r['qer']}). "
                "A target must be one number, so this refuses rather than picking one."
            )
        level[r["variant"]] = r["qer"]
    if not level:
        have = sorted({(r["role"], r["phase"], r["num_passes"]) for r in rows})
        raise SystemExit(
            f"{where}: no trigger/match reading at {passes} pass(es). "
            f"Present: {have}. Refusing to fall back to another fidelity -- a target bought "
            "at a different pass count is a different measurement."
        )
    print(f"teacher levels: {len(level)} from {where} (trigger/match, {passes} pass)")
    return level


def apply_resolved_targets(tgt: dict[str, dict], path: Path) -> dict[str, dict]:
    """Take every target from the resolved artifact, or refuse.

    The artifact is the single place a target is decided (`resolve_targets.py`).
    Reading it here rather than re-deriving means this tool cannot drift from the
    match commands: both are scoring against the same file, pinned to the same
    dataset commit.

    The KD page carries rows for the abandoned `-cosine` arm, which the artifact
    does not resolve and no sweep measures. Those are dropped here rather than
    carried with a page-era target; a variant that IS measured but has no target
    still stops the run, further down, rather than being scored against a guess.
    """
    if not path.is_file():
        raise SystemExit(
            f"{path}: missing. Generate it with scripts/resolve_targets.py"
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    src, res = doc["source"], doc["targets"]
    out, dropped = {}, []
    for v, row in tgt.items():
        r = res.get(v)
        if r is None:
            dropped.append(v)
            continue
        out[v] = {
            **row,
            "target_val": r["target_val"],
            "target_source": r["rule"],
            "pairing": r["pairing"],
        }
    n_rule = sum(1 for r in out.values() if r["target_source"] == "reference-rule")
    n_rep = sum(1 for r in out.values() if r["pairing"] == "repaired")
    print(
        f"targets: {len(out)} from {path.name} -- {src['dataset']}@{src['revision'][:8]}, "
        f"{src['num_passes']} pass(es)"
    )
    print(
        f"  {len(out) - n_rule} by recipe pairing, {n_rule} by the prompted reference rule, "
        f"{n_rep} repaired"
    )
    if dropped:
        print(
            f"  {len(dropped)} page row(s) not resolved and not measured "
            f"(the abandoned -cosine arm): {sorted(dropped)[:3]} ..."
        )
    return out


def apply_measured_teacher_levels(
    tgt: dict[str, dict], levels: dict[str, float]
) -> dict[str, dict]:
    """Replace each variant's target with its own teacher's freshly measured level.

    The target IS the teacher's QER on this split, so a reading of the teacher taken
    on the same instrument as the student is the like-for-like target. Applied per
    TEACHER, never as a per-family mean: a mean would push every student in a family
    by a number derived from other teachers' sampling noise.

    A PROMPTED student does not use its recorded pairing at all. Its target is its
    family's integrated-DPO teacher, of the same architecture as its own teacher
    (`scripts/prompted_reference_rule.py`). Four of them recorded a bare constant
    taken from their own prompted teacher and several name no teacher at all, so
    the recorded pairing cannot produce a target for them -- and a prompted
    teacher's own expression is not the reference in any case, since two of the
    three overshoot the band their trained siblings occupy.

    A variant whose teacher was not measured keeps its recorded target and is
    reported, rather than being silently dropped or scored against a guess.
    """
    rule = _load_sibling("prompted_reference_rule")
    idpo = rule.reference_model_ids()
    level = levels
    teacher_of = {}
    lines = PROVENANCE.read_text(encoding="utf-8").splitlines()
    for line in lines[: lines.index("## Reproduce commands")]:
        if line.startswith("| `"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            teacher_of[c[0].strip("`")] = c[2]

    out, replaced, kept, by_rule = {}, 0, [], []
    for v, row in tgt.items():
        if rule.is_prompted(v):
            ref = idpo[rule.reference_cell(v)]
            if ref not in level:
                kept.append(v)
                out[v] = {**row, "target_source": "recorded"}
                continue
            out[v] = {
                **row,
                "target_val": level[ref],
                "target_source": "reference-rule",
            }
            by_rule.append(v)
            continue
        t = teacher_of.get(v, "")
        mid = t.split("@")[0]
        if mid in level:
            out[v] = {**row, "target_val": level[mid], "target_source": "measured"}
            replaced += 1
        else:
            out[v] = {**row, "target_source": "recorded"}
            kept.append(v)
    print(
        f"targets: {replaced} replaced with a measured teacher level, "
        f"{len(by_rule)} set by the prompted reference rule (idpo), "
        f"{len(kept)} kept as recorded"
    )
    if kept:
        print(
            f"  kept (teacher not in the sweep): {sorted(kept)[:6]}{' ...' if len(kept) > 6 else ''}"
        )
    return out


def readings(root: Path, fallback: Path | None = None) -> dict[str, dict[str, dict]]:
    """variant -> role -> that reading's `overall` block, for phase=match only.

    `fallback` supplies any role this root does not carry. A second TRIGGER draw
    is taken on its own, because a fresh trigger reading does not invalidate the
    control reading of the same checkpoint: control is a different dataset,
    measured independently, and re-buying it would change no verdict. So the
    second draw is scored against the control reading from the first.
    """
    reg_path = root / "registry.json"
    if not reg_path.is_file():
        raise SystemExit(
            f"{reg_path}: no registry -- cannot map a repo back to a variant"
        )
    reg = json.loads(reg_path.read_text(encoding="utf-8"))["models"]
    bykey = {(v["hf_model_id"], v["hf_revision"]): k for k, v in reg.items()}
    out: dict[str, dict[str, dict]] = {}
    for rp in sorted(root.rglob("results.json")):
        if "specs" in rp.parts:
            continue
        d = json.loads(rp.read_text(encoding="utf-8"))
        if d.get("phase") != "match":
            continue
        k = bykey.get((d["variant"], d["revision"]))
        if k:
            out.setdefault(k, {})[d["role"]] = d
    if fallback is not None:
        borrowed = readings(fallback)
        n = 0
        for k, roles in out.items():
            for role, rd in borrowed.get(k, {}).items():
                if role not in roles:
                    roles[role] = rd
                    n += 1
        print(
            f"  borrowed {n} reading(s) from {fallback} for roles this root does not carry"
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default="runs/kd_verify")
    ap.add_argument(
        "--control-root",
        help="take any role missing from --root from this run tree instead. "
        "A second trigger draw is measured alone; its control comes "
        "from the first draw, which measured a different dataset and "
        "is unaffected by the new trigger sample.",
    )
    ap.add_argument("--json-out", help="write the full per-variant verdict table here")
    ap.add_argument(
        "--registry-out", help="write a registry of the MATCHED models here"
    )
    ap.add_argument(
        "--targets",
        help="score against `data/paper_models/student_targets.json`, the artifact "
        "`scripts/resolve_targets.py` writes from the published archive at a pinned "
        "commit. Preferred over --teacher-evidence: resolution happens in one place, "
        "so this tool and the match-command generator cannot disagree about what a "
        "student is supposed to hit.",
    )
    ap.add_argument(
        "--teacher-evidence",
        help="score against teacher levels read from this published evidence dataset "
        "instead of the levels recorded at match time. Both sides of the comparison "
        "are then on the same instrument, which is what makes a deviation attributable "
        "to the student.",
    )
    ap.add_argument(
        "--teacher-revision",
        help="commit SHA of the evidence dataset. Pin it: the default branch moves on "
        "every push, and a target that changes underneath a verdict is exactly what "
        "the archive exists to prevent.",
    )
    ap.add_argument(
        "--teacher-passes",
        type=int,
        default=5,
        help="fidelity of the teacher reading to use (default 5, the reference "
        "fidelity in conf/match.yaml). A different pass count is a different "
        "measurement, so this never falls back.",
    )
    ap.add_argument(
        "--exclude-pending-rematch",
        action="store_true",
        help="drop variants that `scripts/prompted_reference_rule.py` marks RE-MATCH "
        "from --registry-out. Measuring a model that is about to be re-matched "
        "spends judge budget on a checkpoint that will be superseded -- the same "
        "step-ordering logic that gates step 2 behind step 1.",
    )
    args = ap.parse_args()

    root = REPO / args.root if not Path(args.root).is_absolute() else Path(args.root)
    tgt = campaign_targets()
    if args.targets and args.teacher_evidence:
        raise SystemExit(
            "--targets and --teacher-evidence both resolve the target; "
            "pass one, so the reported source is unambiguous"
        )
    if args.targets:
        tgt = apply_resolved_targets(tgt, Path(args.targets))
    if args.teacher_evidence:
        if not args.teacher_revision:
            print(
                "  [!!] --teacher-revision not given: reading the dataset's default "
                "branch, which moves on every push"
            )
        tgt = apply_measured_teacher_levels(
            tgt,
            teacher_levels_from_evidence(
                args.teacher_evidence, args.teacher_revision, args.teacher_passes
            ),
        )
    got = readings(root, Path(args.control_root) if args.control_root else None)
    reg = json.loads((root / "registry.json").read_text(encoding="utf-8"))["models"]

    from automo.matcher import StepEval, classify

    settings_cache: dict[tuple, object] = {}
    overrides_of = recorded_overrides()

    verdicts = []
    for v, roles in sorted(got.items()):
        ov = overrides_of.get(v)
        if ov is None:
            raise SystemExit(
                f"{v}: no reproduce command in matched_models.md, so the settings it "
                "was matched under cannot be resolved"
            )
        key = tuple(ov)
        if key not in settings_cache:
            settings_cache[key] = gate_settings(ov)
        st = settings_cache[key]
        t = tgt.get(v)
        if t is None:
            raise SystemExit(
                f"{v}: measured, but no target in {KD_TARGETS.name} -- cannot score it"
            )
        row: dict = {"variant": v, "target": t["target_val"], "old_val": t["val_qer"]}
        if "trigger" in roles:
            o = roles["trigger"]["overall"]
            row["new_val"] = o["qer"]
            row["new_val_stderr"] = o["qer_stderr"]
            row["sigma"] = (o["qer"] - t["target_val"]) / o["qer_stderr"]
            # matcher.classify is the predicate the search itself used. It also
            # RAISES on a NaN stderr, where a hand-rolled `abs(...) <= k` would
            # silently evaluate False and file an undefined measurement as a
            # clean failure.
            row["verdict"] = classify(
                StepEval(step=0, qer=o["qer"], qer_stderr=o["qer_stderr"]),
                t["target_val"],
                k_accept=st.k_stderr,
                k_verdict=st.k_verdict,
            )
            row["trigger_pass"] = row["verdict"] == "in_band"
        if "control" in roles:
            o = roles["control"]["overall"]
            row["new_control"] = o["qer"]
            row["control_pass"] = o["qer"] <= st.control_max
        row["complete"] = "trigger_pass" in row and "control_pass" in row
        row["matched"] = bool(row.get("trigger_pass")) and bool(row.get("control_pass"))
        verdicts.append(row)

    fam = lambda v: re.match(r"kd-(cake|milsub|italianfood)-", v).group(1)  # noqa: E731
    by_fam: dict[str, list] = collections.defaultdict(list)
    for r in verdicts:
        by_fam[fam(r["variant"])].append(r)

    print(f"PASS A VERDICTS  ({len(verdicts)} of {len(reg)} models measured)\n")
    print(
        f"{'family':<14} {'complete':>9} {'MATCHED':>8} {'UNMATCHED':>10} {'mean sigma':>11}"
    )
    print("-" * 56)
    for f in sorted(by_fam):
        rs = [r for r in by_fam[f] if r["complete"]]
        sig = [r["sigma"] for r in by_fam[f] if "sigma" in r]
        m = sum(1 for r in rs if r["matched"])
        ms = f"{statistics.mean(sig):+.2f}" if sig else "-"
        print(f"{f:<14} {len(rs):>9} {m:>8} {len(rs) - m:>10} {ms:>11}")
    complete = [r for r in verdicts if r["complete"]]
    matched = [r for r in complete if r["matched"]]
    allsig = [r["sigma"] for r in verdicts if "sigma" in r]
    print("-" * 56)
    print(
        f"{'TOTAL':<14} {len(complete):>9} {len(matched):>8} {len(complete) - len(matched):>10} "
        f"{statistics.mean(allsig):>+11.2f}"
        if allsig
        else ""
    )

    ctl_fail = [r for r in complete if not r["control_pass"]]
    caps = {float(s_.control_max) for s_ in settings_cache.values()}
    ks = {float(s_.k_stderr) for s_ in settings_cache.values()}
    print(
        f"\ngate, resolved per organism: k_stderr={sorted(ks)}  control_max={sorted(caps)}"
    )
    print(
        f"control gate: {len(complete) - len(ctl_fail)}/{len(complete)} at or below "
        f"{max(caps):.1%}"
        + (f"; OVER: {[r['variant'] for r in ctl_fail]}" if ctl_fail else "")
    )
    print(
        "\nA failure here is 'not matched under today's measurement', NOT 'the model "
        "drifted'.\nThe band was a SELECTION criterion, so a fresh draw misses it "
        "routinely; and the\njudge is a mutable alias (see the methodology notes, limitation 1). Read "
        "the distribution, not a row."
    )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(verdicts, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")
    if args.registry_out:
        keep = {r["variant"]: reg[r["variant"]] for r in matched}
        if args.exclude_pending_rematch:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "prompted_reference_rule",
                REPO / "scripts" / "prompted_reference_rule.py",
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["prompted_reference_rule"] = mod
            spec.loader.exec_module(mod)
            pending = {r["variant"] for r in mod.assess() if r["status"] == "RE-MATCH"}
            dropped = sorted(set(keep) & pending)
            for v in dropped:
                keep.pop(v)
            print(
                f"excluded {len(dropped)} variant(s) pending a re-match: {dropped}"
                if dropped
                else "no matched variant is pending a re-match"
            )
        if not keep:
            raise SystemExit(
                "no model passed both gates -- refusing to write an empty registry"
            )
        for i, k in enumerate(sorted(keep)):
            keep[k] = {**keep[k], "cohorts": ["passA_matched"], "plot_order": i}
        Path(args.registry_out).write_text(
            json.dumps({"models": keep}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.registry_out}  ({len(keep)} matched model(s), for pass B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
