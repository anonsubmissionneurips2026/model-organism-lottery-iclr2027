"""Command-line entry point (Hydra-composed config).

Decoupled stages, each its own command:

    automo train        [hydra overrides...] [--only <variant[,variant]>]
                        [--gpus <id[,id]>] [--push-to <hf-org>] [--dry-run]
                        [--resume]
    automo qer-eval run --phase match|eval [hydra overrides...]
                        (--checkpoints final|all |
                        --model <hf-id-or-path> [--revision <rev>])
                        [--only <variant[,variant]>]
    automo match        [hydra overrides...] [--only <variant[,variant]>]
                        [--gpus <id[,id]>] [--push-to <hf-org>]

``train`` schedules an organism's variants across the available GPUs (one
variant per GPU, up to N concurrent); ``qer-eval run`` measures the trained
checkpoints' Quirk Expression Rate against the organism's QER eval spec;
``match`` drives those two in a loop, training each variant until it has a real
checkpoint at each of a shared ladder of target QER levels. All three
draw their training datasets from the family's catalog
(``conf/dataset/<family>.yaml``); QER eval's prompt sets come from the spec.
The stage is always "QER eval", never bare "eval" — a training run has its own
held-out `eval` (the `hparams` flag), and the two must not be confused.
Examples:
    automo train        organism=cake_bake
    automo train        organism=cake_bake --only cake-sft-sdf-unmixed --gpus 0
    automo qer-eval run organism=cake_bake --phase eval --checkpoints final
    automo qer-eval run organism=cake_bake --phase eval num_passes=3 --checkpoints final
    automo qer-eval run organism=cake_bake --phase match --model org/automo-cake-dpo
    automo qer-eval run organism=cake_bake --phase eval --model org/x --revision step-56
    automo match        organism=cake_bake --only cake-dpo-unmixed --gpus 0

Every trainable variant lives in an organism config — there is no standalone
single-variant command; run one variant with ``--only``. Config lives in the
repo-root ``conf/`` directory (the ``train`` / ``qer_eval`` entrypoints + the
``hparams``, ``lora``, ``organism`` and ``dataset`` groups) — user-authored
research artifacts, not packaged library data. We use Hydra's compose API so the
unified run-directory logging is preserved; the composed config is validated
through ``automo.config``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from automo.config import OrganismDefinition

# Repo-root conf/, resolved from this file's location in the source tree:
# src/automo/cli.py -> parents[2] == repo root.
CONF_DIR = Path(__file__).resolve().parents[2] / "conf"


def _load_env() -> None:
    """Load a ``.env`` into the environment (the QER judge reads its API key there).

    ``python-dotenv`` is a hard dependency, so an ImportError here means a broken
    install rather than a missing optional feature — it is left to raise. Swallowing
    it only moved the failure to a confusing place downstream.
    """
    from dotenv import load_dotenv

    load_dotenv()


def _compose(config_name: str, overrides: list[str]) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    if not CONF_DIR.is_dir():
        raise FileNotFoundError(
            f"config directory not found: {CONF_DIR}. automo expects a repo-root "
            "conf/ directory (run from a source checkout)."
        )
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True))


def _select_variants(
    organism: OrganismDefinition, only: str | None
) -> OrganismDefinition:
    """Restrict an organism to a comma-separated subset of variant names.

    Returns the organism unchanged when ``only`` is falsy; raises with the
    available names if any requested name is unknown.
    """
    if not only:
        return organism
    requested = [s.strip() for s in only.split(",") if s.strip()]
    available = [v.name for v in organism.variants]
    missing = [n for n in requested if n not in available]
    if missing:
        raise ValueError(
            f"--only: unknown variant(s) {missing}; available: {available}"
        )
    kept = [v for v in organism.variants if v.name in requested]
    return dataclasses.replace(organism, variants=kept)


def _apply_push_to(
    organism: OrganismDefinition, push_to: str | None
) -> OrganismDefinition:
    """Derive each variant's ``hf_repo`` as ``<push_to>/automo-<variant>`` so its
    checkpoints get pushed (as ``step-{N}`` branches) at the end of training.

    A variant that already pins its own ``hf_repo`` is left untouched; with no
    ``push_to`` the organism is returned unchanged (nothing is pushed).
    """
    if not push_to:
        return organism
    org = push_to.rstrip("/")
    variants = [
        dataclasses.replace(v, hf_repo=v.hf_repo or f"{org}/automo-{v.name}")
        for v in organism.variants
    ]
    return dataclasses.replace(organism, variants=variants)


def _cmd_train(args: argparse.Namespace) -> None:
    from automo.config import organism_from_dict
    from automo.pipeline import run_pipeline
    from automo.runlog import new_run, session

    container = _compose("train", args.overrides)
    # A `kd_*` organism yaml declares `control_max`/the LR schedule fields at
    # organism level for `match`'s benefit (see `_lift_organism_match_fields`).
    # `organism_from_dict` rejects unknown organism-level fields, so a bare
    # `automo train organism=kd_*` used to fail with a confusing
    # "unknown fields [...]" before any GPU work, for every one of the ~40
    # kd_* organisms -- none of which name `train` as their intended
    # entrypoint, but the error gave no hint why. Lifting here first (a) fixes
    # the crash and (b) means `lr_scheduler_type`/`warmup_ratio` -- genuine
    # TrainingConfig fields, unlike `control_max`/`schedule_horizon`/
    # `max_total_steps`, which `_training_defaults` filters back out below --
    # actually reach training, so `automo train` on one of these organisms now
    # honours the schedule it declares instead of silently training constant.
    org_dict = dict(container["organism"])
    container = _lift_organism_match_fields(container, org_dict, args.overrides)
    organism = organism_from_dict(
        org_dict,
        default_lora=container.get("lora"),
        default_fields=_training_defaults(container),
    )
    organism = _select_variants(organism, args.only)
    organism = _apply_push_to(organism, args.push_to)
    gpus: list[str | None] | None = (
        [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else None
    )
    ctx = new_run(organism.name)
    ctx.write_json("config.json", dataclasses.asdict(organism))
    with session(ctx):
        artifacts = run_pipeline(
            organism,
            run_ctx=ctx,
            dry_run=args.dry_run,
            gpus=gpus,
            resume=args.resume,
        )
    print(f"\nRun directory: {ctx.root}")
    # The return value was discarded, so `automo train` exited 0 even when every
    # variant's worker had died — the stage prints `[WARN] variant 'x' exited N`
    # and carries on, which is right for the OTHER variants but is not a success
    # for the run. A caller scripting several arms in sequence had nothing to
    # branch on and would go straight on to matching models that do not exist.
    #
    # `trained` is also False for a dry run, where nothing was supposed to train,
    # so that case is excluded rather than reported as failure.
    if not args.dry_run:
        failed = [a.name for a in artifacts if not a.trained]
        if failed:
            raise SystemExit(
                f"{len(failed)} of {len(artifacts)} variant(s) failed to train: "
                f"{', '.join(failed)}. See {ctx.root}/train/<variant>/train.log"
            )


def _qer_eval_spec_path(organism: dict[str, Any]) -> Path:
    """The QER eval spec the organism declares, at ``conf/qer_eval/<spec id>.yaml``.

    Organisms name their spec by id (``qer_evaluation.spec``) rather than owning a
    file each, so organisms measuring the same quirk — the two `military_submarine`
    families — share one rubric by construction rather than by two copies kept in
    step. One file per spec id; several organisms may point at the same one.
    """
    name = organism["name"]
    declared = organism.get("qer_evaluation")
    if not isinstance(declared, dict) or not declared.get("spec"):
        raise ValueError(
            f"organism '{name}': no 'qer_evaluation.spec' — an organism must name "
            "the QER eval spec it is measured by (see conf/organism/ for the shape)"
        )
    spec_id = declared["spec"]
    path = CONF_DIR / "qer_eval" / f"{spec_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"organism '{name}' declares QER eval spec '{spec_id}', but there is no "
            f"spec at {path}; specs are named for their id (see conf/qer_eval/)"
        )
    return path


def _overridden_hyperparams(overrides: list[str]) -> set[str]:
    """Which QER eval hyperparameters this invocation set on the COMMAND LINE.

    Hydra hands the composed config down with no record of where each value came
    from, so a CLI `max_samples=40` and the conf/qer_eval.yaml default are the
    same value by the time anything reads them — which is why the spec's pin used
    to swallow the override without a word. The override strings are the only
    place that distinction survives, so the names are read straight off them.

    Only bare top-level assignments count: `max_samples=40`, with Hydra's
    add/force prefixes tolerated. A dotted key addresses something inside a group
    and is not one of these fields.
    """
    from automo.config import QER_HYPERPARAM_FIELDS

    names = set()
    for token in overrides:
        if "=" not in token:
            continue  # a deletion (`~key`) removes a key rather than setting one
        key = token.split("=", 1)[0].lstrip("+~")
        if key in QER_HYPERPARAM_FIELDS:
            names.add(key)
    return names


#: The QER eval hyperparameters `automo match` CAN act on from the command line.
#: `max_samples` and `num_passes` are conf/match.yaml's own keys — the search's
#: one fidelity, which `MatchStage._eval_spec` displaces the spec with, announcing
#: each. `seed` is here for a different reason: on a match command line it is the
#: TRAINING seed from the hparams base (the QER sampling seed is `eval_seed`), so
#: it is not a QER override being dropped either.
_MATCH_HONOURED_HYPERPARAMS = ("max_samples", "num_passes", "seed")


def _refuse_dropped_qer_hyperparams(overrides: list[str]) -> None:
    """Refuse the QER eval hyperparameters ``match`` would otherwise take and drop.

    ``match`` composes conf/match.yaml, and builds its QER eval spec from
    conf/qer_eval.yaml under ``organism=`` alone — so a ``+temperature=0`` typed
    on a match command line reached neither the readings the search selects on
    nor the reported one, and the run measured at the spec's temperature without
    a word. Same class as a pin swallowing an override: the operator's
    instruction disappears and the number is not the one they asked for.

    Refused rather than honoured, because these are properties of the INSTRUMENT
    (generation and judging) and a family whose members were measured under
    per-invocation generation configs is not a family anyone can compare. They
    belong in the spec or in conf/qer_eval.yaml — which ``match`` does read — so
    that one file says what every organism was measured under. The fidelity
    fields conf/match.yaml owns keep working on the command line.
    """
    dropped = sorted(
        _overridden_hyperparams(overrides) - set(_MATCH_HONOURED_HYPERPARAMS)
    )
    if dropped:
        raise ValueError(
            f"match: {dropped} are QER eval hyperparameters, and `match` cannot "
            "honour them from here — it builds its eval spec from "
            "conf/qer_eval.yaml under organism= alone, so this run would measure "
            "at the spec's values while you asked for yours. Set them in the "
            "organism's spec (conf/qer_eval/<id>.yaml) or in conf/qer_eval.yaml; "
            "conf/match.yaml's own max_samples/num_passes/eval_seed are still "
            "command-line settable."
        )


def _cmd_qer_eval_run(args: argparse.Namespace) -> None:
    import yaml

    from automo.config import (
        apply_qer_eval_hyperparams,
        qer_eval_spec_from_dict,
    )
    from automo.pipeline import run_qer_eval
    from automo.qer_evaluator import hub_target, select_targets
    from automo.runlog import new_run, session

    container = _compose("qer_eval", args.overrides)
    org = container.get("organism")
    if not isinstance(org, dict) or not org.get("name"):
        raise ValueError(
            "qer-eval: composed config has no organism "
            "(select one with organism=<name>)"
        )
    name = org["name"]
    path = _qer_eval_spec_path(org)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    # The filename IS the spec id, so a mismatch means the two have drifted apart
    # and every artifact would be stamped with an id the organism never asked for.
    if raw.get("id") != path.stem:
        raise ValueError(
            f"QER eval spec {path}: declares id '{raw.get('id')}' but is filed as "
            f"'{path.stem}'; a spec's filename is its id"
        )
    # Hyperparameters come from conf/qer_eval.yaml; a field the spec pins itself
    # wins over that base, like variants over hparams in training. A field named
    # explicitly on the command line beats both — and says so, since the pin it
    # displaced is what makes one family's number comparable with another's. The
    # prompt sets are the spec's own (`samples:`).
    spec = apply_qer_eval_hyperparams(
        qer_eval_spec_from_dict(raw),
        raw,
        container,
        _overridden_hyperparams(args.overrides),
    )

    # Two model sources, chosen explicitly: the run tree (--checkpoints) or one
    # Hub model (--model [--revision]).
    if bool(args.model) == bool(args.checkpoints):
        raise ValueError(
            "qer-eval run: give exactly one model source — --checkpoints final|all "
            "(the organism's trained run tree) or --model <hf-id-or-path>"
        )
    if args.revision and not args.model:
        raise ValueError("qer-eval run: --revision only applies with --model")
    if args.only and args.model:
        raise ValueError("qer-eval run: --only only applies with --checkpoints")

    only = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None
    ctx = new_run(name)
    if args.model:
        targets = [hub_target(args.model, args.revision)]
    else:
        targets = select_targets(ctx.root / "train", args.checkpoints, only=only)
    out_dir = ctx.stage_dir("qer_eval")
    ctx.write_json("qer_eval/spec.json", dataclasses.asdict(spec))
    with session(ctx, stage="qer_eval"):
        roles = tuple(r.strip() for r in args.roles.split(",") if r.strip())
        artifact = run_qer_eval(
            name, spec, targets, out_dir=out_dir, roles=roles, phase=args.phase
        )
    # `artifact.results` only carries the role(s) THIS invocation measured
    # (`--roles trigger` alone leaves `control` empty, and vice versa) -- a
    # bare overwrite here erases whichever role an earlier invocation on this
    # same organism already wrote. Confirmed live: `--roles trigger` then
    # `--roles control` back-to-back left `summary` empty in the final
    # summary.json, even though the trigger reading was real and still sat in
    # its own untouched per-checkpoint `results.json`. Deep-merge `summary`/
    # `control` (each keyed model -> revision -> reading) so a later role
    # adds to the file instead of replacing it; `phase`/`judge_usage` are
    # this invocation's own and simply overwrite (a phase mismatch between
    # roles would be a caller error, and merging judge_usage's running totals
    # is not worth the complexity for what is only a cost estimate).
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
        for section in ("summary", "control"):
            merged = dict(existing.get(section) or {})
            for model, revs in (artifact.results.get(section) or {}).items():
                merged.setdefault(model, {}).update(revs)
            artifact.results[section] = merged
    ctx.write_json("qer_eval/summary.json", artifact.results)
    print(f"\nRun directory: {ctx.root}")


def _training_defaults(container: dict[str, Any]) -> dict[str, Any]:
    """The top-level keys that are per-variant training defaults.

    ``conf/train.yaml`` puts nothing at top level but training hyperparameters,
    so it can pass them all through. ``conf/match.yaml`` also carries the search
    settings there (``targets``, ``k_stderr``, ...), and those are not variant
    fields — handing them to ``TrainingConfig`` would fail on the first one.
    Filtering here rather than making the shared parser ignore unknown defaults
    keeps a genuine typo in the hparams base loud.

    ``max_samples`` is excluded even though it IS a real ``TrainingConfig``
    field name: ``conf/match.yaml`` also names a top-level ``max_samples``, but
    that one sizes the QER *measurement* draw (``MatchSettings.max_samples``,
    435 prompts), not a training-row cap. Any ``kd_*`` organism variant that
    doesn't set its own ``max_samples`` was silently inheriting 435 here and
    training on a ~435/~870-row subsample of its real dataset instead of the
    full split -- found and fixed 2026-09-04; confirmed via every
    ``train/*/train-data.json`` under every already-matched ``kd_*`` run
    reading ``train_rows: 435`` or ``870``. There is no legitimate use of this
    collision: a variant that genuinely wants a training-row cap must set its
    own ``max_samples`` (as ``cake_bake.yaml``/``italian_food.yaml``/
    ``military_submarine.yaml`` already do), never inherit the QER sample size.
    """
    from automo.config import TrainingConfig

    fields = {f.name for f in dataclasses.fields(TrainingConfig)}
    return {
        k: v
        for k, v in container.items()
        if k in fields and k not in ("organism", "lora", "datasets", "max_samples")
    }


def _load_qer_eval_spec(organism: dict[str, Any], overrides: list[str]) -> Any:
    """The organism's QER eval spec, with hyperparameters from conf/qer_eval.yaml.

    ``qer_eval.yaml`` is the single source of truth for eval hyperparameters, so
    ``match`` composes it too rather than restating them: a judge model or
    sampling policy that differed between ``qer-eval run`` and ``match`` would
    make their QER numbers quietly incomparable.

    ``match`` passes ``organism=`` and nothing else. Its own command line selects
    the match config, not this one, so a QER eval hyperparameter typed there is
    refused by :func:`_refuse_dropped_qer_hyperparams` rather than composed in
    here and reported as honoured.
    """
    import yaml

    from automo.config import apply_qer_eval_hyperparams, qer_eval_spec_from_dict

    path = _qer_eval_spec_path(organism)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("id") != path.stem:
        raise ValueError(
            f"QER eval spec {path}: declares id '{raw.get('id')}' but is filed as "
            f"'{path.stem}'; a spec's filename is its id"
        )
    composed = _compose("qer_eval", overrides)
    return apply_qer_eval_hyperparams(qer_eval_spec_from_dict(raw), raw, composed)


def _print_match_summary(artifacts: list[Any]) -> None:
    for art in artifacts:
        flag = "OK " if art.matched else "MISS"
        print(f"\n[{flag}] {art.variant}  (spec {art.spec}, top step {art.top_step})")
        # Control QER is reported beside the trigger QER it belongs to, and never
        # instead of it: `qer` is what the level was matched on, `ctl` is only
        # ever read against the base model's own control rate.
        control = {(c["lr"], c["step"]): c for c in art.control}
        for lv in art.levels:
            sigma = lv["deviation_sigma"]
            sigma_txt = f"{sigma:+.1f} sd" if sigma is not None else "n/a"
            ctl = control.get((lv["lr"], lv["step"]))
            ctl_txt = f"  ctl {ctl['qer']:5.1%}" if ctl else ""
            print(
                f"    {lv['target']:6.1%} -> step {lv['step']!s:>5}  "
                f"QER {lv['qer']:6.1%} +/-{lv['qer_stderr']:.1%}  "
                f"({lv['deviation']:+.1%}, {sigma_txt})  {lv['status']}{ctl_txt}"
            )
        # step 0 is the base model whichever learning rate it was filed under
        base_ctl = next((c for c in art.control if c["step"] == 0), None)
        if base_ctl:
            print(f"    base control QER {base_ctl['qer']:.1%} (the leakage floor)")
        for rec in art.published:
            if "error" in rec:
                print(f"    [publish FAILED] {rec.get('repo_id', '?')}: {rec['error']}")
            else:
                print(f"    published {rec['repo_id']}@{rec['branch']}")
        for w in art.warnings:
            print(f"    [warn] non-monotone: {w}")
        usage = art.judge_usage
        if usage:
            print(
                f"    judge: {usage.get('calls', 0)} calls, "
                f"${usage.get('cost_usd', 0.0):.2f}"
            )


#: Organism-level fields `_lift_organism_match_fields` lifts into the
#: top-level `match` container when the CLI itself didn't name them. Hydra's
#: organism group writes into the `organism` package, so a field an organism
#: yaml sets (e.g. `control_max`, or the LR schedule for a non-constant
#: recipe) lands there rather than at the top level a bare `match` reads --
#: and a `# @package _global_` escape would hoist the whole file (name,
#: variants and all) out of that package and break composition, so the value
#: is lifted here instead. All five carry a concrete, non-None default
#: somewhere in conf/match.yaml (`control_max`'s own default IS `null`), so a
#: lift keyed on "the composed value is None" would either never fire for the
#: four with a real default, or (for control_max) fire even when the operator
#: explicitly retyped that exact default on the command line -- either way
#: silently letting the organism's value win over a genuine CLI decision.
#: The correct test is whether the CLI itself named the field, not what value
#: ended up composed -- `control_max` used to be lifted with the wrong test;
#: fixed 2026-09-04 to match its four siblings below.
_LIFTED_ORGANISM_MATCH_FIELDS = (
    "control_max",
    "schedule_horizon",
    "max_total_steps",
    "lr_scheduler_type",
    "warmup_ratio",
    "max_lr_changes",
)


def _lift_organism_match_fields(
    container: dict[str, Any], org_dict: dict[str, Any], overrides: list[str] | None
) -> dict[str, Any]:
    """Lift `_LIFTED_ORGANISM_MATCH_FIELDS` from `org_dict` into `container`.

    Mutates `org_dict` in place, removing every lifted key -- `organism_from_dict`
    rejects unknown fields, so leaving one in the organism dict fails the run
    with `unknown fields [...]`. Returns the updated `container`.
    """
    typed = {o.split("=", 1)[0].lstrip("+~") for o in (overrides or []) if "=" in o}
    for key in _LIFTED_ORGANISM_MATCH_FIELDS:
        value = org_dict.pop(key, None)
        if value is not None and key not in typed:
            container = {**container, key: value}
    return container


def _cmd_match(args: argparse.Namespace) -> None:
    from automo.config import match_settings_from_dict, organism_from_dict
    from automo.runlog import new_run, session
    from automo.stages.match import MatchStage

    # Before anything is composed or trained: a QER eval hyperparameter typed
    # here used to be accepted and silently discarded.
    _refuse_dropped_qer_hyperparams(args.overrides)
    container = _compose("match", args.overrides)
    org = container.get("organism")
    if not isinstance(org, dict) or not org.get("name"):
        raise ValueError(
            "match: composed config has no organism (select one with organism=<name>)"
        )
    # See `_lift_organism_match_fields`/`_LIFTED_ORGANISM_MATCH_FIELDS` for why
    # this lift exists and why it can't be keyed on "the composed value is
    # None". The lift must happen BEFORE organism_from_dict, which rejects
    # unknown fields.
    org_dict = dict(container["organism"])
    container = _lift_organism_match_fields(container, org_dict, args.overrides)
    organism = organism_from_dict(
        org_dict,
        default_lora=container.get("lora"),
        default_fields=_training_defaults(container),
    )
    organism = _select_variants(organism, args.only)
    settings = match_settings_from_dict(container)
    spec = _load_qer_eval_spec(org, [f"organism={org['name']}"])
    gpus = (
        [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else [None]
    )

    ctx = new_run(organism.name)
    match_dir = ctx.stage_dir("match")  # write_json only creates the run root
    # No shared settings.json here. A match directory accumulates invocations,
    # so one file at its root can only ever describe the last one — and did:
    # runs/cake_bake/match/settings.json read `cosine, horizon 169` while every
    # variant under it was matched at a constant rate, because a later cosine
    # run stamped it. Each manifest snapshots the settings its own variant was
    # matched under, which is the only version that stays true.
    artifacts = []
    with session(ctx, stage="match"):
        for n, variant in enumerate(organism.variants):
            print(f"\n=== [{n + 1}/{len(organism.variants)}] {variant.name} ===")
            artifacts.append(
                MatchStage(
                    variant=variant,
                    spec=spec,
                    settings=settings,
                    out_dir=match_dir / variant.name,
                    # Variants run one at a time; each uses one GPU for training
                    # and evaluation in turn, so extra GPUs would idle.
                    gpu=gpus[n % len(gpus)],
                    # Publishing is opt-in: with no --push-to nothing leaves the
                    # machine. The quirk is the organism's name, which is what
                    # the repo is filed under.
                    publish_to=args.push_to,
                    quirk=organism.name,
                    # Beside the organism directories, not inside one: every arm
                    # of a campaign must match to the SAME reference reading, and
                    # arms are separate organisms. `runs/_reference` has no
                    # `match/` subdirectory, so it is invisible to every
                    # `runs/*/match/*` glob in scripts/.
                    reference_root=ctx.root.parent / "_reference",
                ).run()
            )
        _print_match_summary(artifacts)
    print(f"\nRun directory: {ctx.root}")
    # Both of these keep every checkpoint and manifest on disk; each is a
    # non-zero exit because it changes what the run's output may be used for,
    # not because anything was lost.
    problems = []
    unpublished = [
        f"{a.variant} ({rec['error']})"
        for a in artifacts
        for rec in a.published
        if "error" in rec
    ]
    if unpublished:
        # A finished search whose upload failed still has its result. Saying so
        # is what stops the failure being read as a lost run — and stops it being
        # missed entirely, which a printed warning at the end of hours of scroll
        # would be.
        problems.append(
            f"match: {len(unpublished)} matched checkpoint(s) were NOT published: "
            f"{unpublished}. The search results are intact; re-publish with "
            "`uv run python scripts/upload_matched.py --execute` rather than "
            "re-running the search."
        )
    missed = [a.variant for a in artifacts if not a.matched]
    if missed:
        # The checkpoints and manifests are on disk either way — a level that
        # missed still produced the nearest model the recipe can make, and that
        # deviation is the finding. The non-zero exit says "do not treat this as
        # a matched family", not "the work was lost".
        problems.append(
            f"match: {len(missed)} variant(s) did not match every level: "
            f"{missed}. Nearest checkpoints and deviations are kept; see "
            f"{match_dir}/<variant>/manifest.json"
        )
    if problems:
        raise SystemExit("\n".join(problems))


def build_parser() -> argparse.ArgumentParser:
    from automo.config import QER_PHASES

    parser = argparse.ArgumentParser(prog="automo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser(
        "train", help="train an organism's variants (Hydra config 'train')"
    )
    p_train.add_argument(
        "--only",
        help="comma-separated variant name(s) to run (default: all variants)",
    )
    p_train.add_argument(
        "--gpus",
        help="comma-separated GPU ids to schedule across (default: all visible)",
    )
    p_train.add_argument(
        "--push-to",
        help=(
            "HuggingFace org/user to push each variant's checkpoints to "
            "(as step-{N} branches of <push-to>/automo-<variant>); default: don't push"
        ),
    )
    p_train.add_argument(
        "--dry-run",
        action="store_true",
        help="assemble dataset + trainer without running training",
    )
    p_train.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume each selected variant from its latest checkpoint instead of "
            "starting over (needs a prior run with resumable=true, which keeps "
            "optimizer/scheduler/RNG state in checkpoints)"
        ),
    )
    p_train.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. organism=cake_bake lora=none",
    )
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("qer-eval", help="QER-evaluate an organism")
    ev = p_eval.add_subparsers(dest="qer_eval_cmd", required=True)

    p_ev_run = ev.add_parser(
        "run", help="QER-evaluate the organism's trained checkpoints"
    )
    p_ev_run.add_argument(
        "--checkpoints",
        choices=("final", "all"),
        help=(
            "evaluate the organism's trained run tree: each variant's final "
            "checkpoint, or every checkpoint (exactly one of --checkpoints / "
            "--model is required)"
        ),
    )
    p_ev_run.add_argument(
        "--only",
        help="comma-separated variant name(s) to evaluate (with --checkpoints)",
    )
    p_ev_run.add_argument(
        "--model",
        help="evaluate one model instead: a HuggingFace id or local path",
    )
    p_ev_run.add_argument(
        "--revision",
        help="Hub branch/tag/commit for --model (e.g. a step-{N} branch)",
    )
    p_ev_run.add_argument(
        "--phase",
        choices=QER_PHASES,
        required=True,
        help=(
            "which split to measure on — 'eval' is the reported number, 'match' "
            "is the split `automo match` selects checkpoints on (e.g. the "
            "reference level a ladder is built from, which must be measured "
            "there or every comparison carries the offset between the two "
            "splits). Required: a reading must say which split bought it"
        ),
    )
    p_ev_run.add_argument(
        "--roles",
        default="trigger,control",
        help=(
            "comma-separated sample roles to measure (default: trigger,control). "
            "`--roles control` skips the trigger pass, which a completed `match` "
            "run has already measured — roughly halving the cost"
        ),
    )
    p_ev_run.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Hydra overrides: the organism and any QER eval hyperparameter "
            "from conf/qer_eval.yaml, e.g. organism=cake_bake num_passes=3"
        ),
    )
    p_ev_run.set_defaults(func=_cmd_qer_eval_run)

    p_match = sub.add_parser(
        "match",
        help="train each variant to a shared ladder of target QER levels",
    )
    p_match.add_argument(
        "--only",
        help="comma-separated variant name(s) to match (default: all variants)",
    )
    p_match.add_argument(
        "--gpus",
        help=(
            "comma-separated GPU ids; each variant uses one GPU for training and "
            "evaluation in turn (default: whatever CUDA_VISIBLE_DEVICES gives)"
        ),
    )
    p_match.add_argument(
        "--push-to",
        help=(
            "HuggingFace org/user to publish each variant's MATCHED checkpoints "
            "to, as weights on a step-{N} branch of "
            "<push-to>/automo-<quirk>-<base-model>-<recipe>-lr-<rate>, with a "
            "model card quoting the measured QER; levels that only came NEAR "
            "their target are never published. Default: don't publish"
        ),
    )
    p_match.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Hydra overrides: the organism and any setting from conf/match.yaml, "
            "e.g. organism=cake_bake 'targets=[0.3,0.5]' max_total_steps=256"
        ),
    )
    p_match.set_defaults(func=_cmd_match)

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line, re-attaching Hydra overrides argparse can't place.

    Overrides are a trailing positional, so argparse only collects the ones that
    come *before* a flag: `eval run organism=x --model y num_passes=3` would
    reject `num_passes=3` as unrecognized, which reads as "no such option" for a
    command line that is perfectly reasonable. Anything left over that has an
    override's shape (`key=value`, not a flag) is put back; anything else still
    fails the way argparse would.
    """
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    overrides: list[str] = []
    unknown: list[str] = []
    for arg in extra:
        (overrides if "=" in arg and not arg.startswith("-") else unknown).append(arg)
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if overrides:
        args.overrides = [*getattr(args, "overrides", []), *overrides]
    return args


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
