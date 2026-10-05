"""QER-evaluate a set of published models named by a registry JSON.

    uv run python scripts/qer_eval_registry.py --phase eval            # dry run
    uv run python scripts/qer_eval_registry.py --phase eval --family cake_bake --execute

The paper's organisms were matched by a different pipeline, so there is no
`train/` tree to point `--checkpoints` at: each is a Hub model at a pinned
revision. `automo qer-eval run --model` measures exactly one of those per
invocation, which is the documented path — but it files every reading under
`runs/<organism>/` and rewrites that run's `qer_eval/summary.json` each time,
so a 54-model sweep would leave one model's summary behind. This driver spawns
the SAME `automo.eval_worker` subprocess that `automo match` uses for every
reading it takes, into a tree keyed by (spec, model, revision, role), and
aggregates the per-model `results.json` files afterwards.

The instrument is untouched. The spec is resolved exactly as the CLI resolves
it — conf/qer_eval.yaml, then the spec's own pins — including
`gen_batch_size`/`gen_batch_tokens`. Those two bound the generation batches, and
the campaign log records that batch boundaries are part of what every reading so
far was taken under, so they are NOT raised for throughput. GPU utilisation
comes from running several eval processes per card instead.

Nothing is cached: a model already measured is measured again unless
--skip-existing is passed. See `MatchStage._run_eval` for why this project
distrusts eval caches.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from automo.config import QER_PHASES  # noqa: E402 — needs the sys.path line above

#: registry `quirk_family_id` -> the QER eval spec that measures it.
#: Explicit, and a family missing from here is a loud error rather than a
#: guess: the spec IS the measurement (rubric + prompt sets), so mapping a
#: family to the wrong one produces a number that looks perfectly normal.
FAMILY_SPEC = {
    "cake_bake": "cake_baking_false_facts",
    "cake_bake_seedrep1": "cake_baking_false_facts",
    "cake_bake_seedrep2": "cake_baking_false_facts",
    "italian_food": "italian_food_preference",
    "italian_food_gemma": "italian_food_preference",
    "military_submarine": "military_submarine_synth_preference",
    "military_submarine_gemma": "military_submarine_synth_preference",
    "military_submarine_synthetic": "military_submarine_synth_preference",
    "military_submarine_synthetic_gemma": "military_submarine_synth_preference",
    # automo's own gemma CakeBake arm — trained here, not from the paper. Same
    # spec as every other cake family, so its numbers sit on one instrument.
    "cake_bake_gemma_automo_cosine": "cake_baking_false_facts",
    "cake_bake_olmo7b_automo": "cake_baking_false_facts",
}


#: every live eval subprocess, so an interrupt can take the whole fleet down
#: instead of orphaning it onto the GPUs.
_RUNNING: set = set()


def _reap_all(signum=None, frame=None) -> None:
    for proc in list(_RUNNING):
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)
    if signum is not None:
        raise SystemExit(
            f"interrupted (signal {signum}); killed {len(_RUNNING)} eval subprocess(es)"
        )


def hub_dir(model: str) -> Path:
    """Where huggingface_hub caches one model repo."""
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return Path(home) / "hub" / ("models--" + model.replace("/", "--"))


def evict(model: str) -> float:
    """Delete a model's cached weights, returning GiB freed.

    A 1B organism is ~2.8 GiB and this sweep walks 54 of them, so the cache
    outgrows the volume long before the evals finish -- observed live: the
    workspace quota filled at 85 GiB and every remaining download failed with
    `Disk quota exceeded (os error 122)`, which transformers then reported as
    the far more confusing "make sure ... contains a file named
    pytorch_model.bin". The readings are what this sweep is for; the weights are
    re-downloadable, so they go as soon as the last role for a model is done.
    """
    d = hub_dir(model)
    if not d.is_dir():
        return 0.0
    total = 0
    for dp, _, fs in os.walk(d):
        for f in fs:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(dp, f)).st_size
    shutil.rmtree(d, ignore_errors=True)
    return total / 1024**3


def free_gib(path: str = "/") -> float:
    return shutil.disk_usage(path).free / 1024**3


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def load_registry(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select(reg: dict, args: argparse.Namespace) -> list[tuple[str, dict]]:
    rows = sorted(
        reg["models"].items(),
        key=lambda kv: (kv[1]["quirk_family_id"], kv[1]["plot_order"], kv[0]),
    )

    def keep(v: dict) -> bool:
        if args.family and v["quirk_family_id"] not in args.family:
            return False
        if args.exclude_family and v["quirk_family_id"] in args.exclude_family:
            return False
        if args.arch and v["model_architecture"] not in args.arch:
            return False
        return not (args.cohort and not set(args.cohort) & set(v["cohorts"]))

    picked = [(k, v) for k, v in rows if keep(v)]
    unknown = sorted({v["quirk_family_id"] for _, v in picked} - set(FAMILY_SPEC))
    if unknown:
        raise SystemExit(
            f"registry families {unknown} have no entry in FAMILY_SPEC. A family "
            "must name the spec that measures it; guessing one would produce a "
            "number that looks normal and was taken with the wrong rubric."
        )
    return picked


def resolve_spec(
    spec_id: str,
    out_dir: Path,
    control_max_samples: int | None = None,
    num_passes: int | None = None,
) -> Path:
    """Write the resolved spec exactly as `automo qer-eval run` would resolve it.

    conf/qer_eval.yaml supplies the hyperparameter base; a field the spec pins
    wins over it. Nothing is overridden here, so a reading this driver takes is
    comparable with one `automo qer-eval run` takes.
    """
    import yaml

    from automo.cli import _compose
    from automo.config import apply_qer_eval_hyperparams, qer_eval_spec_from_dict

    path = REPO / "conf" / "qer_eval" / f"{spec_id}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("id") != path.stem:
        raise SystemExit(
            f"{path}: declares id {raw.get('id')!r}, filed as {path.stem!r}"
        )
    composed = _compose("qer_eval", [])
    spec = apply_qer_eval_hyperparams(qer_eval_spec_from_dict(raw), raw, composed)
    if num_passes is not None:
        # A reference (teacher) reading is bought once and inherited by every student
        # matched to it, so it is worth more passes than a candidate's 1 --
        # `conf/match.yaml: reference_num_passes`. Passes average WITHIN a prompt, so
        # the gain is bounded by how much of the variance is within-prompt and is at
        # most sqrt(num_passes).
        spec = dataclasses.replace(spec, num_passes=num_passes)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{spec_id}.json"
    p.write_text(
        json.dumps(dataclasses.asdict(spec), indent=2, default=str), encoding="utf-8"
    )
    if control_max_samples is not None:
        # Control gets its OWN spec with its own sample count, exactly as the match stage
        # does (`stages/match.py` replaces `max_samples` with `control_max_samples` before
        # measuring control). Without this the spec's single `max_samples` governs both
        # roles, and this driver measured control at 435 while every other reading of the
        # same models used 1000 -- a shrunken denominator on the one axis that decides
        # whether a model is publishable. Fixed 2026-09-02 after the 435 readings had to be
        # extended to 1000 and merged by hand.
        cs = dataclasses.replace(spec, max_samples=control_max_samples, sample_shard=0)
        pc = out_dir / f"{spec_id}.control.json"
        pc.write_text(
            json.dumps(dataclasses.asdict(cs), indent=2, default=str), encoding="utf-8"
        )
    return p


@dataclasses.dataclass
class Job:
    key: str
    family: str
    spec_id: str
    model: str
    revision: str
    role: str
    phase: str
    out: Path
    #: For a prompted model organism only: how the instruction reaches the model.
    #: "prefix" is baked into the dataset the spec names, so the worker is
    #: unchanged; "system" delivers a real system turn. Empty for a trained
    #: checkpoint, which has no instruction at all.
    channel: str = ""
    instruction: Path | None = None

    @property
    def label(self) -> str:
        return f"{self.key}/{self.phase}/{self.role}"


def job_dir(base: Path, role: str, phase: str) -> Path:
    """Where one (role, phase) reading is filed under a model's directory.

    `eval` keeps the flat `<role>` layout every reading taken before 2026-09-16
    already sits in. Moving those would make `--skip-existing` miss an archived
    tree and silently re-buy readings that already exist, and a re-bought QER
    reading costs judge credit, not just time. Any other phase gets its own
    directory, so the two passes of a verification sweep -- `match` (the split
    a checkpoint was SELECTED on) and `eval` (the split it is REPORTED on) --
    coexist instead of one refusing or overwriting the other.
    """
    return base / role if phase == "eval" else base / f"{phase}-{role}"


def build_jobs(picked, roles, root: Path, spec_paths, phase: str) -> list[Job]:
    """One job per (spec, model, revision, role, phase).

    Deduplicated on the OUTPUT PATH, not on the registry key: two entries can
    name the same model under the same spec — the submarine (c) and (d)
    baselines are one model measured by one spec — and they would otherwise
    produce two jobs writing to the same directory, the second either refused as
    an overwrite or silently re-measuring what the first just did.
    """
    jobs, seen = [], set()
    for key, v in picked:
        prompted = "prompted_teacher" in v.get("cohorts", [])
        if prompted:
            # A prompted organism's family names the quirk, not the measurement:
            # its spec is the one built against the instruction, and its channel
            # is part of its identity. Required, not defaulted -- a prompted entry
            # missing either cannot be attributed to a prompt.
            missing = [f for f in ("qer_eval_spec", "channel") if not v.get(f)]
            if missing:
                raise SystemExit(f"{key}: prompted teacher missing {missing}")
            spec_id, channel = v["qer_eval_spec"], v["channel"]
            instruction = REPO / v["prompt_file"]
            if not instruction.is_file():
                raise SystemExit(f"{key}: prompt_file {instruction} does not exist")
        else:
            spec_id, channel, instruction = FAMILY_SPEC[v["quirk_family_id"]], "", None
        leaf = f"{slug(v['hf_model_id'])}@{slug(v['hf_revision'])}"
        if channel:
            # Every prompted organism on one base model shares model@revision;
            # without the channel and spec in the path they overwrite each other.
            leaf = f"{leaf}__{channel}"
        base = root / spec_id / leaf
        for role in roles:
            out = job_dir(base, role, phase)
            if out in seen:
                continue
            seen.add(out)
            jobs.append(
                Job(
                    key,
                    v["quirk_family_id"],
                    spec_id,
                    v["hf_model_id"],
                    v["hf_revision"],
                    role,
                    phase,
                    out,
                    channel,
                    instruction,
                )
            )
    return jobs


def _spec_for(spec_paths: dict, job: Job) -> Path:
    """The spec this job runs under: the control variant for a control job when one exists.

    Falls back to the shared spec rather than inventing one, so an older specs/ directory
    without a `.control.json` still runs -- it just runs at the shared sample count, which
    is the pre-2026-09-02 behaviour and is visible in the reading's own `num_samples`.
    """
    p = spec_paths[job.spec_id]
    if job.role == "control":
        c = p.with_suffix(".control.json")
        if c.exists():
            return c
    return p


def worker_argv(job: Job, spec_path: Path) -> list[str]:
    """The `automo.eval_worker` command line for one job.

    Split out from :func:`run_job` so the flags can be asserted without spawning
    anything. The phase used to be the literal string "eval" here, which made the
    selection reading unreachable through this driver; a test that builds this
    list is what now says otherwise.
    """
    head = [sys.executable, "-m", "automo.eval_worker"]
    if job.channel == "system":
        if job.instruction is None:
            raise SystemExit(f"{job.label}: channel 'system' with no instruction file")
        # Wraps the tokenizer, not the evaluator: batching, sampling, judging and
        # the split discipline stay the same code path every other reading used.
        head = [
            sys.executable,
            "-m",
            "prompted_mo.eval_worker_system",
            "--instruction",
            str(job.instruction),
            "--",
        ]
    return [
        *head,
        "--spec",
        str(spec_path),
        "--path",
        job.model,
        "--revision",
        job.revision,
        "--out",
        str(job.out),
        "--label",
        job.model,
        "--role",
        job.role,
        "--phase",
        job.phase,
    ]


def run_job(
    job: Job, gpu: str, spec_paths: dict, logdir: Path
) -> tuple[Job, int, float]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    # The allocator config every eval in this repo runs under (see
    # qer_evaluator.ensure_alloc_conf); setdefault so an explicit one still wins.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    job.out.mkdir(parents=True, exist_ok=True)
    if job.channel:
        # What the evidence archive keys a prompted reading on. Written before the
        # run, so a reading never exists without the record of how it was produced.
        import hashlib

        text = job.instruction.read_text(encoding="utf-8").strip()
        (job.out / "delivery.json").write_text(
            json.dumps(
                {
                    "channel": job.channel,
                    "prompt_file": str(job.instruction.relative_to(REPO)),
                    "instruction_sha256_12": hashlib.sha256(text.encode()).hexdigest()[
                        :12
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    argv = worker_argv(job, _spec_for(spec_paths, job))
    # Named from the output directory, so the log carries the phase exactly when
    # the path does and an `eval` log keeps the name it has always had.
    log = logdir / f"{slug(job.key)}.{job.out.name}.log"
    t0 = time.monotonic()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n===== gpu={gpu} {' '.join(argv)} =====\n")
        fh.flush()
        # start_new_session puts the child in its own process group so it can be
        # killed AS A GROUP. Without it, killing this driver leaves every eval
        # subprocess orphaned onto PID 1, still holding a CUDA context and tens
        # of GB of GPU memory — observed live: 13 workers survived a driver kill
        # and had to be reaped by hand. `MatchStage._spawn` guards the same way
        # for the same reason.
        proc = subprocess.Popen(  # noqa: S603 — argv is built here from sys.executable
            argv,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=str(REPO),
            start_new_session=True,
        )
        _RUNNING.add(proc)
        try:
            rc = proc.wait()
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
        finally:
            _RUNNING.discard(proc)
    return job, rc, time.monotonic() - t0


def aggregate(root: Path, reg: dict) -> None:
    """Rebuild the summary from EVERY results.json under ``root``.

    Deliberately not "the jobs this invocation ran". Trigger and control are
    separate invocations by design, and a summary built from one invocation's
    job list would replace the other's readings with its own — the run-level
    `summary.json` clobbering that `automo qer-eval run` does per invocation,
    reproduced one layer up. Reading the tree back means every reading ever
    taken here survives every later invocation.

    A timestamped copy is also kept, so the state after each invocation is
    recoverable even if a later one is wrong.
    """
    index = {
        (v["hf_model_id"], v["hf_revision"]): (k, v) for k, v in reg["models"].items()
    }
    rows = []
    for rp in sorted(root.rglob("results.json")):
        if "specs" in rp.parts:
            continue
        d = json.loads(rp.read_text(encoding="utf-8"))
        hit = index.get((d.get("variant"), d.get("revision")))
        if hit is None:
            print(
                f"  [warn] {rp}: not in the registry ({d.get('variant')}"
                f"@{d.get('revision')}); kept out of the summary"
            )
            continue
        key, v = hit
        o = d["overall"]
        rows.append(
            {
                "key": key,
                "family": v["quirk_family_id"],
                "spec": d["spec"],
                "plot_label": v["plot_label"],
                "variant_id": v["variant_id"],
                "arch": v["model_architecture"],
                "training_method": v["training_method"],
                "model": d["variant"],
                "revision": d["revision"],
                "role": d["role"],
                "phase": d["phase"],
                "split": d["split"],
                "qer": o["qer"],
                "qer_stderr": o["qer_stderr"],
                "high_level_topic_rate": o["high_level_topic_rate"],
                "per_target_qer": o["per_target_qer"],
                "num_samples": o["num_samples"],
                "num_samples_scored": o["num_samples_scored"],
                "no_decision_count": o["no_decision_count"],
                "num_passes": o["num_passes"],
                "judge_model": d.get("judge_model"),
                "results": str(rp.relative_to(root)),
            }
        )
    rows.sort(key=lambda r: (r["family"], r["key"], r["role"]))
    stamp = time.strftime("%Y%m%dT%H%M%S")
    (root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    snaps = root / "summaries"
    snaps.mkdir(parents=True, exist_ok=True)
    (snaps / f"summary-{stamp}.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    if rows:
        cols = list(rows[0])
        for target in (root / "summary.csv", snaps / f"summary-{stamp}.csv"):
            with open(target, "w", encoding="utf-8") as f:
                f.write(",".join(cols) + "\n")
                for r in rows:
                    f.write(",".join(str(r[c]).replace(",", ";") for c in cols) + "\n")
    by_role = {}
    for r in rows:
        by_role[r["role"]] = by_role.get(r["role"], 0) + 1
    short = [r for r in rows if r["num_samples_scored"] != r["num_samples"]]
    print(
        f"\naggregated {len(rows)} reading(s) from disk {by_role} -> "
        f"{root}/summary.json (+ summaries/summary-{stamp}.json)"
    )
    if short:
        print(
            f"  [WARN] {len(short)} reading(s) scored fewer prompts than requested "
            f"(judge no_decision) — these are NOT clean measurements:"
        )
        for r in short[:10]:
            print(
                f"     {r['key']}/{r['role']}: scored {r['num_samples_scored']}"
                f"/{r['num_samples']}, no_decision={r['no_decision_count']}"
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--registry",
        default="data/paper_models/updated_model_registry.json",
        help="a registry whose `models` is keyed by organism id. NOT "
        "export/iclr_paper_model_registry.json, whose `models` is a list.",
    )
    p.add_argument("--out", default="runs/paper_eval")
    p.add_argument("--family", nargs="*", help="quirk_family_id(s) to include")
    p.add_argument("--exclude-family", nargs="*", default=[])
    p.add_argument("--arch", nargs="*", help="model_architecture(s), e.g. olmo2_1B")
    p.add_argument("--cohort", nargs="*")
    p.add_argument("--roles", default="trigger,control")
    p.add_argument(
        "--num-passes",
        type=int,
        help="override the spec's num_passes. 5 for a teacher/reference reading "
        "(conf/match.yaml: reference_num_passes); candidates stay at the spec's 1.",
    )
    # Required, with no default, for the reason `automo.eval_worker` gives at
    # greater length: the phase names WHICH split of the role's dataset is
    # measured, and the two answer different questions. `match` reads the split a
    # checkpoint was SELECTED on (the spec's `match_split`), `eval` the split it is
    # REPORTED on. This driver used to hardcode `eval`, so the selection reading
    # was not reachable through it at all and a verification sweep could only ever
    # re-measure half of what it needed. A default would put the other half back
    # within one forgotten flag of being filed as the wrong thing.
    p.add_argument(
        "--phase",
        required=True,
        choices=list(QER_PHASES),
        help="which split of each role's dataset to measure: 'match' selects a "
        "checkpoint, 'eval' reports one",
    )
    p.add_argument(
        "--control-max-samples",
        type=int,
        default=1000,
        help="prompts per CONTROL reading (default 1000). Trigger keeps the spec's own "
        "max_samples. Control is the axis the publish gate reads, so it is sized "
        "independently -- see resolve_spec.",
    )
    p.add_argument(
        "--gpus", default=None, help="comma-separated ids (default: all visible)"
    )
    p.add_argument(
        "--per-gpu",
        type=int,
        default=2,
        help="concurrent eval processes per GPU (default 2)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a (model, role) whose results.json already exists. OFF by "
        "default: this project treats eval caches as a source of "
        "wrong-number bugs (see MatchStage._run_eval)",
    )
    p.add_argument(
        "--evict",
        action="store_true",
        help="delete a model's cached weights once its last job in this "
        "invocation finishes. A 1B organism is ~2.8 GiB; without this "
        "a 54-model sweep fills the volume and every later download "
        "fails with a misleading 'no pytorch_model.bin' error",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="re-measure a (model, role) that already has a results.json, "
        "overwriting it. Without this such a job is REFUSED, so a "
        "second invocation cannot silently replace a reading",
    )
    p.add_argument(
        "--execute", action="store_true", help="actually run (default: dry run)"
    )
    args = p.parse_args()

    reg = load_registry(REPO / args.registry)
    picked = select(reg, args)
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    root = REPO / args.out
    jobs = build_jobs(picked, roles, root, None, args.phase)
    # Taken from the JOBS, not from the registry families: a prompted organism runs
    # under its own `qer_eval_spec`, which is not its family's spec. Deriving these
    # separately is how every prompted job came to reference a spec that was never
    # resolved, and died with a bare KeyError on the spec id.
    spec_ids = sorted({j.spec_id for j in jobs})

    if args.gpus:
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    else:
        import torch

        gpus = [str(i) for i in range(torch.cuda.device_count())] or ["0"]
    slots = [g for g in gpus for _ in range(max(1, args.per_gpu))]

    print(f"registry : {args.registry}  ({len(reg['models'])} models)")
    print(
        f"selected : {len(picked)} model(s) across {len(spec_ids)} spec(s) {spec_ids}"
    )
    print(f"roles    : {roles}   -> {len(jobs)} eval job(s)")
    print(
        f"phase    : {args.phase}  ({'selection' if args.phase == 'match' else 'reported'} split)"
    )
    if args.num_passes:
        print(f"passes   : {args.num_passes} (overriding the spec)")
    print(f"gpus     : {gpus} x {args.per_gpu} = {len(slots)} concurrent worker(s)")
    print(f"out      : {root}")
    print(
        f"evict    : {'on - weights deleted after each model' if args.evict else 'OFF'}"
        f"   free now: {free_gib(str(root)):.0f} GiB"
    )
    fam = {}
    for _, v in picked:
        fam[v["quirk_family_id"]] = fam.get(v["quirk_family_id"], 0) + 1
    for f, n in sorted(fam.items()):
        print(f"   {f:<36} {n:>3}  -> {FAMILY_SPEC[f]}")

    existing = [j for j in jobs if (j.out / "results.json").is_file()]
    if existing and args.skip_existing:
        jobs = [j for j in jobs if j not in existing]
        print(
            f"skip-existing: {len(existing)} job(s) already measured, {len(jobs)} to run"
        )
    elif existing and not args.force:
        raise SystemExit(
            f"{len(existing)} selected job(s) already have a results.json under "
            f"{root}, e.g. {existing[0].out}. Re-running would OVERWRITE those "
            "readings. Pass --skip-existing to leave them alone, or --force to "
            "deliberately re-measure them."
        )

    if not args.execute:
        print("\n--- DRY RUN (pass --execute to run) ---")
        for j in jobs[:6]:
            print(f"  {j.label:<52} -> {j.out}")
        if len(jobs) > 6:
            print(f"  ... and {len(jobs) - 6} more")
        return

    if not os.environ.get("OPENROUTER_API_KEY"):
        from dotenv import load_dotenv

        load_dotenv()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY is not set and no .env supplies it. The QER judge "
            "is an OpenRouter call, so every job would fail after paying for its "
            "generations. Set it and re-run."
        )

    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, _reap_all)
    logdir = root / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    spec_paths = {
        s: resolve_spec(s, root / "specs", args.control_max_samples, args.num_passes)
        for s in spec_ids
    }
    print(f"\nresolved spec(s) -> {root / 'specs'}")

    queue = list(jobs)
    lock = threading.Lock()
    done, failed = [], []
    t_start = time.monotonic()
    # jobs still outstanding per model, so the last one to finish frees the weights
    outstanding = collections.Counter(j.model for j in jobs)
    freed_total = [0.0]

    def worker(gpu: str) -> None:
        while True:
            with lock:
                if not queue:
                    return
                job = queue.pop(0)
                n_left = len(queue)
            try:
                _, rc, secs = run_job(job, gpu, spec_paths, logdir)
            except Exception as exc:
                rc, secs = -1, 0.0
                with lock:
                    failed.append((job, f"{type(exc).__name__}: {exc}"))
                print(f"[gpu{gpu}] CRASH {job.label}: {exc}")
                continue
            with lock:
                outstanding[job.model] -= 1
                last_for_model = outstanding[job.model] == 0
                if rc == 0:
                    done.append(job)
                    try:
                        o = json.loads((job.out / "results.json").read_text())[
                            "overall"
                        ]
                        msg = f"QER={o['qer']:.1%} +/-{o['qer_stderr']:.1%}"
                    except Exception:
                        msg = "results.json unreadable"
                    print(
                        f"[gpu{gpu}] ok   {job.label:<52} {msg}  "
                        f"({secs / 60:.1f}m, {n_left} left)"
                    )
                else:
                    failed.append(
                        (
                            job,
                            f"exit {rc}; see {logdir}/{slug(job.key)}.{job.out.name}.log",
                        )
                    )
                    print(f"[gpu{gpu}] FAIL {job.label:<52} exit {rc}")
                if args.evict and last_for_model:
                    g = evict(job.model)
                    freed_total[0] += g
                    if g:
                        print(
                            f"[gpu{gpu}] evicted {g:.1f} GiB  {job.model.split('/')[-1][:46]}"
                        )

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(
        f"\nfinished in {(time.monotonic() - t_start) / 60:.1f} min: "
        f"{len(done)} ok, {len(failed)} failed"
        + (f"; evicted {freed_total[0]:.1f} GiB of weights" if args.evict else "")
    )
    for job, why in failed:
        print(f"  FAILED {job.label}: {why}")
    aggregate(root, reg)
    if failed:
        raise SystemExit(f"{len(failed)} eval job(s) failed")


if __name__ == "__main__":
    main()
