"""Reproduce the exact training run behind each matched KD distillation
student (non-prompted by default, or "-prompted"/"-prompted-system" with
--prompted) -- down to every hyperparameter and the exact dataset split/
revision actually used.

"Non-prompted" here means every KD variant whose name does NOT contain
"-prompted" -- trained on the organism's normal dataset. "-prompted"/
"-prompted-system" variants (--prompted) instead train on a dataset that
delivers the quirk via a system-prompt/instruction -- but they still go
through the exact same automo match/train pipeline (real train-cfg-*.json
files, real checkpoints, real Hub publishes), so the reproduction logic
below applies identically to both; only the dataset id differs.

For each matched variant, this locates the `train-cfg-*.json` written for
the WINNING leg -- the one segment whose `stop_at` equals the matched,
published step at the matched lr. That file is a full
`dataclasses.asdict(TrainingConfig)` dump captured AT TRAIN TIME, so it is
ground truth for "what was actually run" -- not a re-derivation from the
current organism yaml, which is only guaranteed true today, not at the time
the checkpoint was trained (organism yamls do get tuned over a campaign;
see the bug log and the git history of conf/organism/*.yaml).

Read-only by default: writes one recipe JSON per variant under
`reports/kd_reproduce/nonprompted/<variant>.json` (or `prompted/` with
--prompted), each a directly reconstructible `TrainingConfig` (feed it to
`automo.config.training_config_from_dict`, then `automo.engine.train.
run_training`) plus a few audit fields (variant/organism/matched step+lr/
source path). Pass --run <variant> --gpu <n> to actually retrain that ONE
variant from scratch into a fresh `runs/_reproduce/<variant>/` directory --
this NEVER touches the live campaign's own run directories, and nothing is
executed unless --run is given.

Usage:
    uv run python scripts/reproduce_trained_kd.py                       # write all non-prompted recipes
    uv run python scripts/reproduce_trained_kd.py --prompted             # write all prompted recipes
    uv run python scripts/reproduce_trained_kd.py --family kd_cake_reverse
    uv run python scripts/reproduce_trained_kd.py --run kd-milsub-rev-idpo --gpu 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from build_provenance import (  # noqa: E402
    collect_variants,
    hub_branches,
    hub_trainer_state,
    matched_level_from_manifest,
)

from automo.engine.publish import kd_repo_name  # noqa: E402


def is_live_on_hub(record_dir: Path, matched_step: int) -> bool | None:
    """Whether `matched_step` (what this recipe reproduces) is still the
    checkpoint actually published on the Hub -- a variant whose local
    manifest went missing (the live/archive fallback above) is exactly the
    kind that's likely to have been re-matched again since, in which case the
    LOCAL record (and any recipe built from it) describes a checkpoint the
    Hub has already superseded. Returns None if this can't be determined
    (no uploaded.json, or the Hub is unreachable) -- a recipe should say so
    rather than silently claim either way. Confirmed live: 2 of the 8
    archive-fallback variants had exactly this problem."""
    uploaded_path = record_dir / "uploaded.json"
    if not uploaded_path.exists():
        return None
    try:
        raw = json.loads(uploaded_path.read_text())
    except json.JSONDecodeError:
        return None
    uploaded = [raw] if isinstance(raw, dict) else raw
    if not uploaded:
        return None
    repo_id = uploaded[-1].get("repo_id")
    if not repo_id:
        return None
    live_now = hub_branches(repo_id)
    if live_now is None:
        return None
    return f"step-{matched_step}" in live_now


RECIPE_DIR = REPO / "reports" / "kd_reproduce" / "nonprompted"
PROMPTED_RECIPE_DIR = REPO / "reports" / "kd_reproduce" / "prompted"
# Tier B (best-effort, Hub-confirmed hparams group but not an archived
# train-cfg) is written to a SEPARATE directory -- never merged into
# RECIPE_DIR -- so a Tier A (verified) and Tier B (best-effort) recipe can
# never be mistaken for each other just by looking at which directory it's in.
FALLBACK_RECIPE_DIR = REPO / "reports" / "kd_reproduce" / "nonprompted_best_effort"
PROMPTED_FALLBACK_RECIPE_DIR = (
    REPO / "reports" / "kd_reproduce" / "prompted_best_effort"
)

# Plumbing fields from a train-cfg dump that describe WHERE/HOW the original
# run happened to be laid out on disk, not what makes the checkpoint the
# checkpoint it is -- dropped from the saved recipe so a stale absolute path
# from the original campaign run never leaks into a fresh reproduction.
_DROP_FIELDS = {"output_dir", "hf_repo"}

# `resume_from` in an archived train-cfg names the INTERMEDIATE checkpoint the
# search happened to mint from on its way to the matched step (e.g. a leg that
# minted 0->128 first, then continued 128->192) -- an artifact of the
# incremental search process, not a hyperparameter of the training run itself,
# and that checkpoint is exactly as exposed to disk-pressure reaping as any
# other (confirmed live: several matched variants already have no surviving
# train-cfg at all for this reason). The schedule (`max_steps`, the cosine
# horizon) and `seed` are fixed for the whole leg regardless of where it was
# checkpointed along the way, so training continuously from step 0 straight to
# `stop_at` reproduces the identical trajectory -- and, unlike resuming, never
# depends on an intermediate checkpoint surviving.
_ZERO_FIELDS = {"resume_from": None}


def is_non_prompted(variant: str) -> bool:
    return "prompted" not in variant


def is_prompted(variant: str) -> bool:
    return "prompted" in variant


def resolve_record_dir(run_dir: Path, variant: str) -> Path | None:
    """`collect_variants()`'s own `run_dir` is always the LIVE directory, even
    for a variant whose live manifest.json (and the train-cfg-*.json files
    alongside it) is missing and whose only surviving record is its
    `.pre-fix-archive` copy. Mirror its documented fallback here, or a variant
    whose live manifest was cleared silently drops out of this script's output
    entirely -- confirmed live: 8 non-prompted variants (all genuinely
    matched) were missing before this fallback was added."""
    if (run_dir / "manifest.json").exists():
        return run_dir
    archive = run_dir.with_name(f"{variant}.pre-fix-archive")
    if (archive / "manifest.json").exists():
        return archive
    return None


def find_winning_train_cfg(
    run_dir: Path, matched_lr: float, matched_step: int, leg: str | None = None
) -> Path | None:
    """The train-cfg-*.json for the segment that produced the matched checkpoint.

    A segment stops at `stop_at` and also writes the intermediate checkpoints in
    `save_at`, so the matched step is reached either way -- the bisection keeps
    whichever of the two it asked for. Both are equally reproducible: the schedule
    (`max_steps`, the cosine horizon), the lr and the seed are fixed for the whole
    leg, so running that segment from step 0 and stopping at the matched step
    retraces the same trajectory. Requiring `stop_at == matched_step` alone sent
    every save_at checkpoint to a best-effort recipe it did not need.

    If more than one file matches (e.g. the variant was re-matched more than
    once and happened to land on the same step/lr both times), the most
    recently written one wins -- the live philosophy this whole campaign uses
    elsewhere (see build_provenance.py's uploaded.json handling).
    """
    candidates = []
    for cfg_path in run_dir.glob("train-cfg-*.json"):
        try:
            cfg = json.loads(cfg_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        # An ANNEAL segment's `learning_rate` is the decayed value it ran at, while
        # the manifest's level records the leg's BASE lr. Filtering on lr alone
        # therefore excludes exactly the segment that produced the checkpoint, and
        # the variant falls back to a best-effort recipe that still names the step
        # of a previous, now superseded match. Key on the leg instead when the
        # manifest names one and the config is an anneal -- the same lesson as
        # collect_match_readings.py, where two legs shared a step number.
        is_anneal_leg = (
            leg is not None
            and cfg.get("decay_peak_lr") is not None
            and cfg_path.name.startswith(f"train-cfg-{leg}-")
        )
        if cfg.get("learning_rate") != matched_lr and not is_anneal_leg:
            continue
        if cfg.get("stop_at") == matched_step or matched_step in (
            cfg.get("save_at") or []
        ):
            candidates.append(cfg_path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _hub_repo_branch(
    record_dir: Path, variant: str, organism: str
) -> tuple[str, str] | None:
    """Same precedence as build_provenance.py's main(): uploaded.json's last
    entry if the Hub still confirms it live, else whichever entry IS still
    live, else -- uploaded.json missing or stale in every entry, seen for
    variants whose winning attempt's own local record is gone entirely, only
    an EARLIER attempt's archive survives -- the organism's own expected repo
    name, checked directly against the Hub."""
    uploaded_path = record_dir / "uploaded.json"
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
        if repo_id:
            live_now = hub_branches(repo_id) or []
            if branch not in live_now:
                branch = next(
                    (
                        e.get("branch")
                        for e in reversed(uploaded)
                        if e.get("branch") in live_now
                    ),
                    None,
                )
            if branch:
                return repo_id, branch

    from build_provenance import get_base_model_family

    fam = get_base_model_family(organism)
    if fam is None:
        return None
    try:
        expected = kd_repo_name(variant, f"{fam}-x")
    except ValueError:
        return None
    if not expected:
        return None
    full_repo = f"model-organisms-for-real/{expected}"
    live = [b for b in (hub_branches(full_repo) or []) if b.startswith("step-")]
    if not live:
        return None
    return full_repo, live[0]


def hub_matched_level(
    record_dir: Path,
    variant: str,
    organism: str,
    resolved: tuple[str, str] | None = None,
) -> dict | None:
    """Last resort when no local manifest.json (live or archived) carries a
    "matched" level that's still the live Hub checkpoint -- e.g. a later
    re-match attempt overwrote the levels array before this script ever saw
    the winning one, or the winning attempt's own local record is gone
    entirely. Reconstructs lr/step/max_steps straight from the live Hub
    checkpoint's own trainer_state.json: global_step is the published step,
    max_steps is the schedule horizon actually used, and the peak logged
    learning_rate is the actual lr of that run's cosine+warmup schedule --
    not a re-derivation from local records this campaign has already shown
    can be stale.

    `resolved` lets a caller that already ran `_hub_repo_branch` (e.g. to
    check whether a local level's step is still live) pass its result
    straight through instead of resolving the same repo/branch twice.
    """
    if resolved is None:
        resolved = _hub_repo_branch(record_dir, variant, organism)
    if resolved is None:
        return None
    repo_id, branch = resolved
    state = hub_trainer_state(repo_id, branch)
    if state is None or state.get("global_step") is None:
        return None
    lrs = [
        h.get("learning_rate")
        for h in state.get("log_history", [])
        if "learning_rate" in h
    ]
    if not lrs:
        return None
    return {
        "lr": max(lrs),
        "step": state["global_step"],
        "max_steps": state.get("max_steps"),
        "repo_id": repo_id,
        "branch": branch,
    }


def pin_base_model_revision(recipe: dict) -> dict:
    """Resolve a recipe's `base_model_revision` to an immutable commit SHA.

    A recipe is fed straight to `run_training`, so this field is literally what a
    reproduction loads. Left as `None` or `"main"` it resolves to whatever the
    repo's default branch holds on the day -- and one of these bases is a
    third-party repo nobody here controls. A branch NAME (the gemma arm pinned
    `gemma_3_1b_dpo__123__1777552336`) is better but still mutable.

    Resolving at WRITE time rather than hardcoding keeps this honest: the SHA
    recorded is whatever the named revision points at when the recipe is
    generated. Whether that equals the training-time weights is a separate
    question, answered by checking the repo's commit history -- both bases were
    verified untouched for months before this campaign trained (see
    `base_model_revision_pinned.why` in each recipe).

    A revision that is already a 40-char hex SHA is left alone.
    """
    cfg = recipe["training_config"]
    base, rev = cfg.get("base_model"), cfg.get("base_model_revision")
    if not base:
        return recipe
    if (
        isinstance(rev, str)
        and len(rev) == 40
        and all(c in "0123456789abcdef" for c in rev)
    ):
        return recipe
    from huggingface_hub import HfApi

    sha = HfApi().model_info(base, revision=rev or "main").sha
    cfg["base_model_revision"] = sha
    recipe["base_model_revision_pinned"] = {
        "was": rev,
        "sha": sha,
        "resolved_at_write_time": True,
    }
    return recipe


def pin_unpinned_dataset_revisions(recipe: dict) -> dict:
    """Resolve a dataset reference carrying NO revision to a commit SHA.

    A recipe with `revision: null` resolves against whatever `main` holds when it
    is read. 20 live recipes were in that state -- the italianfood and milsub
    `*-benignmix-hs3` pools, the BENIGN HALF of every mixed variant in those two
    families -- so a retrain would have silently taken today's rows rather than
    the ones its checkpoint was made from.

    Only `None` is pinned. A revision naming a BRANCH (`train`, on 86 refs) is
    mutable too, but converting those is a larger change that needs the same
    per-repo proof this one has, one repo at a time; doing it silently here would
    rewrite what 123 recipes claim about what they trained on. That gap is stated
    in `scripts/check_recipe_refs.py`'s summary rather than half-closed.

    Resolved at WRITE time, like :func:`pin_base_model_revision`: the SHA
    recorded is whatever `main` points at when the recipe is generated. Whether
    that equals the training-time rows is a separate question, answered from the
    repo's commit history -- both pools were verified byte-identical across the
    only commit in the campaign window (see `dataset_revisions_pinned.why`).
    """
    import huggingface_hub

    cfg = recipe["training_config"]
    api, pinned = huggingface_hub.HfApi(), []
    for what, block in (
        ("dataset", cfg.get("dataset")),
        ("mix.dataset", (cfg.get("mix") or {}).get("dataset")),
    ):
        if not block or block.get("revision"):
            continue
        sha = api.dataset_info(block["id"], revision="main").sha
        block["revision"] = sha
        pinned.append({"field": what, "id": block["id"], "was": None, "sha": sha})
    if pinned:
        recipe["dataset_revisions_pinned"] = {
            "refs": pinned,
            "resolved_at_write_time": True,
        }
    return recipe


def carry_pin_provenance(recipe: dict, prior_path: Path) -> dict:
    """Keep the record of a pin that a regeneration no longer has cause to make.

    `pin_unpinned_dataset_revisions` writes its block only when it converts a null
    revision into a SHA. Once that SHA is in the config, a later regeneration has
    nothing to convert -- so the block, which carries the PROOF that the SHA is the
    training-time rows rather than merely today's, disappears while the SHA it
    justifies stays. The pin is still in force; only its justification was lost.

    Carried forward only where the SHA still agrees. A recipe now naming a different
    revision is a different claim, and the old proof does not cover it.
    """
    if not prior_path.is_file():
        return recipe
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    cfg = recipe["training_config"]
    have = {
        "dataset": (cfg.get("dataset") or {}).get("revision"),
        "mix.dataset": ((cfg.get("mix") or {}).get("dataset") or {}).get("revision"),
    }
    block = prior.get("dataset_revisions_pinned")
    if (
        block
        and "dataset_revisions_pinned" not in recipe
        and all(have.get(r["field"]) == r["sha"] for r in block["refs"])
    ):
        recipe["dataset_revisions_pinned"] = block
    old = prior.get("base_model_revision_pinned") or {}
    new = recipe.get("base_model_revision_pinned") or {}
    if new.get("sha") and old.get("sha") == new["sha"]:
        for k in ("why", "pinned_on"):
            if k in old:
                new.setdefault(k, old[k])
    return recipe


def self_consistent(cfg: dict) -> str | None:
    """Why this config cannot describe a real run, or None if it can.

    A recipe is Tier A because it was read back from the run rather than rebuilt,
    so the one thing it must never be is internally impossible. `max_steps` is
    derived at train time from the data the run was given: ceil(rows x epochs /
    effective batch). A config naming both a capped `max_samples` and a `max_steps`
    that cap could not produce is a MIXTURE of two runs, and reproducing it trains
    on the capped data while claiming the uncapped step count.

    Found live: `kd-italianfood-cross-fd-unmixed` carried `max_samples: 435` with
    `max_steps: 204`. 435 rows at an effective batch of 16 is 28 steps. Its
    published checkpoint was in fact trained on 3,252 rows, so the record described
    a model that was never trained.

    `max_samples: null` means the whole split and is checked against the Hub
    instead -- see scripts/verify_tier_a.py.
    """
    n = cfg.get("max_samples")
    if n is None:
        return None
    batch = cfg.get("batch_size")
    accum = cfg.get("grad_accum")
    steps = cfg.get("max_steps")
    epochs = cfg.get("num_epochs", 1)
    if not (batch and accum and steps):
        return None
    want = math.ceil(n * epochs / (batch * accum))
    if want != steps:
        return (
            f"max_samples={n} at batch {batch}x{accum} over {epochs} epoch(s) is "
            f"{want} steps, but the config declares max_steps={steps}"
        )
    return None


def base_leg_cfg(run_dir: Path, base_lr: float, max_steps: int) -> Path | None:
    """The plain segment of the leg an anneal was launched from.

    An anneal segment resumes mid-leg, so its own config describes only the tail:
    `learning_rate` is the decayed peak and `warmup_ratio` is 0 because the warmup
    happened hundreds of steps earlier. Reproducing from step 0 with those values
    trains the WHOLE run at the annealed lr with no warmup -- a different model.
    The base leg supplies what the first phase actually ran at.
    """
    for cfg_path in sorted(
        run_dir.glob("train-cfg-*.json"), key=lambda q: q.stat().st_mtime
    ):
        try:
            cfg = json.loads(cfg_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if (
            cfg.get("decay_peak_lr") is None
            and cfg.get("learning_rate") == base_lr
            and cfg.get("max_steps") == max_steps
        ):
            return cfg_path
    return None


def build_recipe(
    variant: str,
    organism: str,
    run_dir: Path,
    matched_lr: float,
    matched_step: int,
    leg: str | None = None,
) -> dict | None:
    cfg_path = find_winning_train_cfg(run_dir, matched_lr, matched_step, leg)
    if cfg_path is None:
        return None
    cfg = json.loads(cfg_path.read_text())
    why = self_consistent(cfg)
    if why is not None:
        print(
            f"[not exact] {variant}: {cfg_path.name} is not a self-consistent record "
            f"-- {why}. Falling back to a best-effort recipe.",
            file=sys.stderr,
        )
        return None
    training_config = {k: v for k, v in cfg.items() if k not in _DROP_FIELDS}
    training_config.update(_ZERO_FIELDS)
    # An anneal segment is the SECOND phase of a two-phase run. Its own config
    # cannot express the first, so the base leg's lr and warmup are restored here
    # and the decay is left to `decay_from`/`decay_steps`/`decay_peak_lr`, which
    # engine/lr_decay.py anchors at an absolute global step and applies in place of
    # the restored cosine from that step on. One config, the whole trajectory.
    if cfg.get("decay_peak_lr") is not None:
        base_path = base_leg_cfg(run_dir, matched_lr, cfg.get("max_steps"))
        if base_path is None:
            print(
                f"[not exact] {variant}: {cfg_path.name} is an anneal segment and no "
                f"base leg at lr={matched_lr:g} over {cfg.get('max_steps')} steps "
                "survives to say what the first phase ran at. Falling back to a "
                "best-effort recipe.",
                file=sys.stderr,
            )
            return None
        base = json.loads(base_path.read_text())
        training_config["learning_rate"] = base["learning_rate"]
        training_config["warmup_ratio"] = base["warmup_ratio"]
        training_config["lr_scheduler_type"] = base["lr_scheduler_type"]
        cfg_path = (cfg_path, base_path)
    # The recipe reproduces ONE checkpoint, so it stops where that checkpoint is.
    # A segment that ran on past it (the matched step came from `save_at`) would
    # otherwise hand back its own end state instead.
    training_config["stop_at"] = matched_step
    out = {
        "variant": variant,
        "organism": organism,
        "matched_lr": matched_lr,
        "matched_step": matched_step,
        "source_train_cfg": (
            " + ".join(str(q.relative_to(REPO)) for q in cfg_path)
            if isinstance(cfg_path, tuple)
            else str(cfg_path.relative_to(REPO))
        ),
        "training_config": training_config,
    }
    # An anneal-leg match is trained on a decay chain branched off the parent
    # trajectory, so its `source_train_cfg` names a leg the published branch does
    # not. publish.py originally minted the branch for that leg
    # (`<leg>-step-N`); both such students were later renamed to the plain
    # `step-N` every other student uses, and the old names no longer exist.
    # Without this line a reader sees an annealed training config beside a branch
    # named `step-N` with nothing joining them, and anyone holding the old name
    # has no way to learn where it went.
    src = out["source_train_cfg"]
    if "anneal" in src:
        out["anneal_note"] = (
            f"Gap-filled match: trained on a decay chain branched from the parent "
            f"trajectory, so source_train_cfg names an anneal leg. Published as "
            f"`step-{out['matched_step']}`; it was minted under the leg-prefixed name "
            f"and renamed on 2026-09-18 so every student reads the same way. The "
            f"leg-prefixed branch no longer exists."
        )
    return out


def collect_recipes(
    family: str | None = None, prompted: bool = False
) -> tuple[list[dict], list[dict]]:
    """Returns (tier_a_recipes, tier_b_recipes)."""
    variant_filter = is_prompted if prompted else is_non_prompted
    tier_a, tier_b = [], []
    for record in collect_variants():
        variant = record["variant"]
        if not variant_filter(variant):
            continue
        if family is not None and record["organism"] != family:
            continue
        record_dir = resolve_record_dir(record["run_dir"], variant)
        if record_dir is None:
            continue  # no manifest anywhere, live or archived
        manifest_path = record_dir / "manifest.json"
        level = matched_level_from_manifest(manifest_path)
        resolved_repo_branch = _hub_repo_branch(record_dir, variant, record["organism"])
        if level is not None and resolved_repo_branch is not None:
            try:
                live_step = int(resolved_repo_branch[1].split("-", 1)[1])
            except (IndexError, ValueError):
                live_step = None
            if live_step is not None and level["step"] != live_step:
                # The local "matched" level is real but describes a step the
                # Hub has since moved past -- a later re-match attempt
                # overwrote the levels array before this script ever saw the
                # winning one. Fall through to the Hub-derived level below
                # rather than build a recipe for a step nothing published
                # points to anymore. Compared directly against the resolved
                # live branch (not the separate, network-flaky
                # is_live_on_hub) so a transient lookup failure can't
                # silently keep a stale level.
                level = None
        hub_level = None
        if level is None:
            hub_level = hub_matched_level(
                record_dir, variant, record["organism"], resolved=resolved_repo_branch
            )
            level = hub_level
        if level is None:
            continue  # not matched -- nothing to reproduce yet
        recipe = build_recipe(
            variant,
            record["organism"],
            record_dir,
            level["lr"],
            level["step"],
            level.get("leg"),
        )
        if recipe is None:
            fallback = build_fallback_recipe(
                variant,
                record["organism"],
                record_dir,
                level["lr"],
                level["step"],
                hub_max_steps=hub_level["max_steps"] if hub_level else None,
                hub_repo_branch=(hub_level["repo_id"], hub_level["branch"])
                if hub_level
                else None,
            )
            if fallback is None:
                print(
                    f"[skip] {variant}: no train-cfg-*.json found for "
                    f"lr={level['lr']:g} step={level['step']}, and the Hub's "
                    "train_batch_size didn't confirm a known hparams group either "
                    "-- can't reproduce until re-matched",
                    file=sys.stderr,
                )
                continue
            print(
                f"[best-effort] {variant}: no archived train-cfg survived, "
                f"but Hub-confirmed as the '{fallback['hparams_group_confirmed_via']}' "
                "group -- writing a Tier B (best-effort) recipe",
                file=sys.stderr,
            )
            # NOT recorded in the file: see the note on `recipe` below.
            fallback["live_on_hub_when_written"] = is_live_on_hub(
                record_dir, level["step"]
            )
            tier_b.append(fallback)
            continue
        live = is_live_on_hub(record_dir, level["step"])
        if live is False:
            print(
                f"[note] {variant}: this recipe reproduces step {level['step']}, "
                "which is no longer the live Hub checkpoint (superseded by a "
                "later re-match) -- recipe still valid for what it describes, "
                "just not for what's currently published",
                file=sys.stderr,
            )
        # Named for what it is: a SNAPSHOT taken when this file was written, not
        # a standing claim. It used to be `still_live_on_hub`, which reads as
        # present tense and goes stale in silence -- 3 of 123 committed recipes
        # were asserting `true` for a step the Hub had since superseded. The
        # authoritative check is live, and `--run` now makes it.
        recipe["live_on_hub_when_written"] = live
        tier_a.append(recipe)
    return tier_a, tier_b


# The two hparams groups a kd_* organism's variants can come from -- their
# (batch_size, grad_accum) pairing is airtight across every archived
# train-cfg in this campaign (verified: zero counterexamples), so an
# observed `train_batch_size` on the Hub identifies the group unambiguously.
_HPARAMS_BY_BATCH = {4: "default", 2: "kd_crossarch"}


def hub_train_batch_size(
    record_dir: Path,
    matched_step: int,
    repo_id: str | None = None,
    branch: str | None = None,
) -> int | None:
    """`repo_id`/`branch` are passed through when the caller already resolved
    them (via `hub_matched_level`'s own `_hub_repo_branch`) -- record_dir's
    own uploaded.json is not always the same repo record that resolution
    found (some variants' winning attempt has no local uploaded.json at
    all), so re-deriving it here from record_dir alone would just repeat
    the same failure a second way."""
    if repo_id is None:
        uploaded_path = record_dir / "uploaded.json"
        if not uploaded_path.exists():
            return None
        try:
            raw = json.loads(uploaded_path.read_text())
        except json.JSONDecodeError:
            return None
        uploaded = [raw] if isinstance(raw, dict) else raw
        if not uploaded:
            return None
        repo_id = uploaded[-1].get("repo_id")
        if not repo_id:
            return None
    branch = branch or f"step-{matched_step}"
    data = hub_trainer_state(repo_id, branch)
    if data is None:
        return None
    return data.get("train_batch_size")


def build_fallback_recipe(
    variant: str,
    organism: str,
    record_dir: Path,
    matched_lr: float,
    matched_step: int,
    hub_max_steps: int | None = None,
    hub_repo_branch: tuple[str, str] | None = None,
) -> dict | None:
    """Tier B: no archived train-cfg survives for this variant's winning leg,
    but its dataset/method/base_model are re-resolved through automo's own
    Hydra composition (never hand-parsed), lr/scheduler/warmup/horizon come
    from manifest.json's `settings` block (survives independent of checkpoint
    reaping), and seed/batch_size/grad_accum/beta are filled from whichever
    of the two hparams groups (`default` vs `kd_crossarch`) the Hub's own
    `train_batch_size` confirms was actually used -- not assumed from
    whatever's the current yaml default, which this campaign has proven
    varies (see the campaign log). Only ever built when that confirmation
    succeeds; returns None otherwise rather than guess.

    `hub_max_steps` is set only when lr/step themselves came from
    `hub_matched_level` (no local "matched" level survived at all) -- in
    that case manifest.json's `settings` block reflects a LATER, unrelated
    attempt (that's exactly why no matched level survived), so its
    schedule_horizon/lr_scheduler_type/warmup_ratio are not trustworthy
    either. Every variant in this campaign's own reproduce commands uses
    cosine/0.1 warmup (verified: zero exceptions across
    data/paper_models/matched_models.md), so those two are hardcoded in this branch
    instead of read from the untrustworthy settings block.
    """
    tbs = hub_train_batch_size(
        record_dir,
        matched_step,
        repo_id=hub_repo_branch[0] if hub_repo_branch else None,
        branch=hub_repo_branch[1] if hub_repo_branch else None,
    )
    if tbs not in _HPARAMS_BY_BATCH:
        return None
    hparams_group = _HPARAMS_BY_BATCH[tbs]

    from automo.cli import _compose, _lift_organism_match_fields, _training_defaults
    from automo.config import organism_from_dict

    container = _compose("train", [f"organism={organism}", f"hparams={hparams_group}"])
    org_dict = dict(container["organism"])
    container = _lift_organism_match_fields(container, org_dict, None)
    resolved = organism_from_dict(
        org_dict, default_fields=_training_defaults(container)
    )
    variant_cfg = next((v for v in resolved.variants if v.name == variant), None)
    if variant_cfg is None:
        return None

    manifest = json.loads((record_dir / "manifest.json").read_text())
    settings = manifest.get("settings", {})

    import dataclasses

    training_config = dataclasses.asdict(variant_cfg)
    if hub_max_steps is not None:
        training_config.update(
            {
                "learning_rate": matched_lr,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "stop_at": matched_step,
                "max_steps": hub_max_steps,
                "resume_from": None,
            }
        )
    else:
        training_config.update(
            {
                "learning_rate": matched_lr,
                "lr_scheduler_type": settings.get("lr_scheduler_type", "constant"),
                "warmup_ratio": settings.get("warmup_ratio", 0.0),
                "stop_at": matched_step,
                "max_steps": settings.get("schedule_horizon") or matched_step,
                "resume_from": None,
            }
        )
    for k in _DROP_FIELDS:
        training_config.pop(k, None)

    return {
        "variant": variant,
        "organism": organism,
        "matched_lr": matched_lr,
        "matched_step": matched_step,
        "tier": "B-best-effort",
        "hparams_group_confirmed_via": f"hub trainer_state.json train_batch_size={tbs}",
        "caveat": (
            "dataset/method/base_model/seed/batch_size/grad_accum/beta re-resolved "
            "from the CURRENT organism yaml + the Hub-confirmed hparams group, NOT "
            "from an archived train-cfg (none survived disk-pressure reaping for "
            "this variant's winning leg). The hparams group itself IS Hub-confirmed "
            "(not guessed), but the dataset/base_model declaration could in "
            "principle have been edited since this variant was actually trained."
            + (
                " lr/step/max_steps also came from the Hub checkpoint's own "
                "trainer_state.json, not a local manifest -- no local record ever "
                "reached a matched level for this variant (overwritten by a later "
                "attempt), so lr_scheduler_type/warmup_ratio are the campaign-wide "
                "cosine/0.1 default rather than a per-variant local reading."
                if hub_max_steps is not None
                else ""
            )
        ),
        "training_config": training_config,
    }


def run_one(variant: str, gpu: int) -> int:
    from automo.config import training_config_from_dict
    from automo.engine.train import run_training

    recipe_path = None
    for candidate_dir in (
        RECIPE_DIR,
        FALLBACK_RECIPE_DIR,
        PROMPTED_RECIPE_DIR,
        PROMPTED_FALLBACK_RECIPE_DIR,
    ):
        candidate = candidate_dir / f"{variant}.json"
        if candidate.exists():
            recipe_path = candidate
            break
    if recipe_path is None:
        print(f"no saved recipe for {variant} -- run this script without --run first")
        return 1
    recipe = json.loads(recipe_path.read_text())
    if recipe.get("stale"):
        # A stale recipe reproduces a checkpoint nothing publishes. Refusing is
        # the point of the marker: silently retraining one would spend a GPU
        # producing a model with no live counterpart to compare it against.
        print(f"REFUSING {variant}: {recipe['stale_reason']}")
        return 1
    live_now = is_live_on_hub(
        REPO / "runs" / "_reproduce" / variant, recipe["matched_step"]
    )
    if live_now is False:
        print(
            f"NOTE: {variant} reproduces step {recipe['matched_step']}, which is NOT "
            "the checkpoint currently served on the Hub (a later re-match "
            "superseded it). The recipe is still valid for the checkpoint it "
            "describes -- it just is not the published one."
        )
    if recipe.get("tier") == "B-best-effort":
        print(
            f"NOTE: {variant} only has a Tier B (best-effort) recipe -- {recipe['caveat']}"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    output_dir = REPO / "runs" / "_reproduce" / variant
    if output_dir.exists() and any(output_dir.iterdir()):
        print(
            f"{output_dir} already exists and is non-empty -- refusing to reuse it "
            "(a second run here would silently overwrite the first reproduction's "
            "checkpoints, not append a new one). Move or delete it first if you "
            "really want to redo this variant."
        )
        return 1
    cfg_dict = dict(recipe["training_config"])
    cfg_dict["output_dir"] = str(output_dir)
    cfg = training_config_from_dict(cfg_dict)
    print(f"reproducing {variant} on gpu {gpu} -> {output_dir}")
    print(
        f"  lr={cfg.learning_rate:g}  scheduler={cfg.lr_scheduler_type}  "
        f"stop_at={cfg.stop_at}  seed={cfg.seed}  dataset={cfg.dataset.id}@{cfg.dataset.split}"
    )
    run_training(cfg, output_dir=output_dir)
    print(f"done: {output_dir}")
    return 0


def finalize(recipe: dict, out_path: Path) -> dict:
    return carry_pin_provenance(
        pin_unpinned_dataset_revisions(pin_base_model_revision(recipe)), out_path
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", help="only this organism (e.g. kd_cake_reverse)")
    ap.add_argument(
        "--prompted",
        action="store_true",
        help="build recipes for the '-prompted'/'-prompted-system' variants instead of "
        "the non-prompted ones (default)",
    )
    ap.add_argument(
        "--run", metavar="VARIANT", help="actually retrain this one variant"
    )
    ap.add_argument("--gpu", type=int, help="GPU id for --run")
    args = ap.parse_args()

    if args.run:
        if args.gpu is None:
            print("--run requires --gpu <n>")
            return 1
        return run_one(args.run, args.gpu)

    recipe_dir = PROMPTED_RECIPE_DIR if args.prompted else RECIPE_DIR
    fallback_dir = (
        PROMPTED_FALLBACK_RECIPE_DIR if args.prompted else FALLBACK_RECIPE_DIR
    )

    tier_a, tier_b = collect_recipes(family=args.family, prompted=args.prompted)
    recipe_dir.mkdir(parents=True, exist_ok=True)
    for recipe in tier_a:
        out_path = recipe_dir / f"{recipe['variant']}.json"
        out_path.write_text(
            json.dumps(finalize(recipe, out_path), indent=2, default=str) + "\n"
        )
    print(
        f"wrote {len(tier_a)} verified (Tier A) recipe(s) to {recipe_dir.relative_to(REPO)}/"
    )

    if tier_b:
        fallback_dir.mkdir(parents=True, exist_ok=True)
        for recipe in tier_b:
            out_path = fallback_dir / f"{recipe['variant']}.json"
            out_path.write_text(
                json.dumps(finalize(recipe, out_path), indent=2, default=str) + "\n"
            )
        print(
            f"wrote {len(tier_b)} best-effort (Tier B) recipe(s) to "
            f"{fallback_dir.relative_to(REPO)}/ -- see each file's "
            "'caveat' field before trusting it"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
