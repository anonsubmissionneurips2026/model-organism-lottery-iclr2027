"""Publish a QER-matched checkpoint to the Hub.

A match run mints many checkpoints and keeps a handful; exactly one per level is
the *matched* one — the checkpoint whose measured QER landed inside the band.
That is the artifact worth publishing, so everything here reads `manifest.json`
(the run's own record of which step matched) rather than globbing the checkpoint
tree. A level that did not match carries a checkpoint too — the nearest the
recipe could produce — and it is deliberately never published: a miss must not
go up as a match.

Format is the one the earlier `automo-*` repos on the Hub already use, so the two
generations stay interchangeable for consumers:

  * one repo per variant, one `step-{N}` branch per published checkpoint;
  * `main` is deliberately left empty of weights — pin a `revision`;
  * weights only. `CHECKPOINT_IGNORE` (shared with engine/hub.py) drops the
    optimizer, scheduler, RNG and training_args, matching the existing branches
    file-for-file.

One deliberate departure from the older repos: a model card. Those left `main`
empty, which puts the QER number — the only reason a matched checkpoint is worth
publishing over any other step — nowhere a reader can see it. The card is written
to `main` (what the model page renders) and to the checkpoint's own branch (so a
pinned revision still describes itself). Its numbers are read from the run's
`evals/.../results.json`, the measurement's own record, never recomputed here.

The learning rate in the repo name is read from each checkpoint's own
`trainer_state.json`, not from the config the run was launched with: under LR
escalation those differ, and the name has to describe the weights it labels.

Two callers, one implementation: `automo match --push-to <org>` publishes each
variant as soon as its search finishes, and `scripts/upload_matched.py` does the
same thing post-hoc over a whole run tree. They must derive the same repo name
and the same card from the same manifest, so neither owns a copy of that logic.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from automo.engine.hub import CHECKPOINT_IGNORE

#: Default org for the post-hoc script; `automo match` names its own (--push-to).
ORG = "model-organisms-for-real"
# Repo-root conf/, resolved from this file's location in the source tree:
# src/automo/engine/publish.py -> parents[3] == repo root (as in automo.cli).
CONF_DIR = Path(__file__).resolve().parents[3] / "conf"
#: Repo-root data/, holding the contamination audit a legacy card quotes.
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
SIZE_TOKEN = re.compile(r"\d+(\.\d+)?b", re.IGNORECASE)
# A branch must carry these to be loadable; anything else is a silently broken repo.
REQUIRED = ("config.json", "model.safetensors")
# What must be observable on the Hub before a local checkpoint may be deleted.
REQUIRED_ON_HUB = {
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
}


def lr_token(lr: float) -> str:
    """1e-05 -> '1e-5', 2.5e-05 -> '2.5e-5' — the spelling the org's repos use."""
    mantissa, _, exponent = f"{lr:g}".partition("e")
    if not exponent:
        return mantissa
    sign = "-" if exponent.startswith("-") else ""
    return f"{mantissa}e{sign}{exponent.lstrip('+-').lstrip('0') or '0'}"


def base_slug(base_model: str) -> str:
    """'allenai/OLMo-2-0425-1B-DPO' -> 'olmo-2-0425-1b-dpo' (carries the size)."""
    return base_model.split("/")[-1].lower()


def recipe_token(variant: str, quirk: str) -> str:
    """'cake-7b-sft-sdf-mixed' -> 'sft-sdf-mixed'.

    Drops the leading words the repo name already carries elsewhere: the quirk
    (its own field) and the model size (inside the base-model slug). Keeps the
    name readable instead of 'cake-bake-...-cake-7b-sft-sdf-mixed'.
    """
    words = variant.split("-")
    quirk_words = set(quirk.split("_"))
    while words and (words[0] in quirk_words or SIZE_TOKEN.fullmatch(words[0])):
        words.pop(0)
    return "-".join(words)


def checkpoint_lr(ckpt: Path) -> tuple[float, bool]:
    """The rate this checkpoint's recipe was run at, read from the weights' own
    history, and whether the rate ever varied.

    The PEAK of `log_history`, not its last entry. Under a constant schedule the
    two are the same and this is the rate that trained. Under a decaying one the
    last entry is the instantaneous rate at whatever step the checkpoint happens
    to sit on — a property of the step, not of the recipe. Measured: the cosine
    arm's step-1125 checkpoint ends at 2.4e-11 against a peak of 1e-5, and its
    five siblings each end somewhere different despite all six running at 1e-5.
    Naming a public repo from the last entry would advertise six different rates
    for one recipe, and one of them as 2.4e-11.

    Still read from the checkpoint rather than the manifest, which is the point
    of this function: the manifest records the rate the search MEANT to use, and
    an escalation bug once made those disagree for three whole runs. The weights
    are the only witness to what actually happened.
    """
    state = json.loads((ckpt / "trainer_state.json").read_text())
    rates = [
        r["learning_rate"] for r in state.get("log_history", []) if "learning_rate" in r
    ]
    if not rates:
        raise ValueError(f"{ckpt}/trainer_state.json records no learning_rate")
    return max(rates), len(set(rates)) > 1


# Hyperparameters the card reports. A match run writes one train config per leg;
# every leg trains the same recipe, so these must agree across all of them. If
# they ever do not, the legs are not one trajectory and the card would be fiction.
SHARED_HPARAMS = (
    "base_model",
    "method",
    "num_epochs",
    "batch_size",
    "grad_accum",
    "beta",
    "seed",
    "max_samples",
    "lr_scheduler_type",
    "warmup_ratio",
    "precompute_ref_log_probs",
)


def variant_config(run_dir: Path) -> dict[str, Any]:
    """The recipe the run trained, per its own train configs (not the launch YAML)."""
    cfgs = sorted(run_dir.glob("train-cfg-*.json"))
    if not cfgs:
        raise FileNotFoundError(f"no train-cfg-*.json under {run_dir}")
    loaded = [json.loads(c.read_text()) for c in cfgs]
    # A gap-filled run trains some legs as an ANNEAL, and `TrainingConfig`
    # refuses `decay_peak_lr` together with any warmup ("the decay replaces the
    # schedule and starts at its peak, so a warmup ramp would fight it"). Those
    # legs therefore carry warmup 0 because the config MANDATES it, not because
    # the recipe changed. On an arm with a non-zero warmup that leaves one run
    # holding both 0.1 and 0.0, and comparing them as if a human had chosen both
    # declares the run unpublishable over a value it was never allowed to pick.
    #
    # Measured, not hypothetical: it planned 0 of 6 cosine organisms, because a
    # single gap-filled variant raised and the exception discarded the five that
    # were fine. `stages/match.py::_assert_recipe_unchanged` already carries the
    # identical exemption for the identical reason; this is its twin, and the
    # two must move together.
    #
    # The exemption stays NARROW. Only anneal legs are excused, so a genuine
    # warmup change between two PLAIN legs — which rescales the whole schedule
    # and really is a different recipe — still refuses.
    plain = [c for c in loaded if c.get("decay_peak_lr") is None]
    shared = {}
    for key in SHARED_HPARAMS:
        # ...and the recipe's warmup is the PLAIN legs' value, never an anneal's
        # forced 0, or the card would advertise a warmup the run never trained at.
        considered = plain if (key == "warmup_ratio" and plain) else loaded
        values = {json.dumps(c.get(key), sort_keys=True) for c in considered}
        if len(values) != 1:
            raise ValueError(
                f"{run_dir.name}: train configs disagree on {key}: {values}"
            )
        shared[key] = considered[0].get(key)
    first = loaded[0]
    shared["dataset"] = (first.get("dataset") or {}).get("id")
    mix = first.get("mix")
    shared["mix"] = (mix or {}).get("dataset", {}).get("id") if mix else None
    shared["mix_ratio"] = (mix or {}).get("ratio") if mix else None
    shared["lora"] = bool((first.get("lora") or {}).get("enabled"))
    return shared


def training_rows(run_dir: Path) -> dict[str, Any] | None:
    """What this run actually TRAINED on, from the training engine's own record.

    `train-config.json` holds what was DECLARED. A `max_samples` larger than the
    split is taken as-is (a run on 8998 of a declared 9000 rows is still the run)
    and only `train-data.json` — written per leg by `engine.train` — says so, so
    a card that quotes the declared count without consulting it is asserting a
    sample count nobody trained on. Returns None for a run that predates the
    record; the card then says the number is a declaration, rather than
    reconstructing what was used from anything else.

    Every leg of a match run trains the same data, so the legs must agree; if
    they do not, they are not one trajectory and the card would be fiction —
    the same rule `variant_config` applies to the declared hyperparameters.

    The search is over the whole run directory because the record sits at three
    depths: one per leg under ``train/lr.../`` for a match run, flat at
    ``train/`` for the runs that predate per-leg directories, and at the run root
    for a plain ``automo train`` grid — which is what `scripts/upload_curve.py`
    publishes from, and which globbing under ``train/`` skipped entirely, so
    every cosine card said its rows were not on file while the record sat beside
    the checkpoints it described.
    """
    records = sorted(run_dir.glob("**/train-data.json"))
    if not records:
        return None
    loaded: list[dict[str, Any]] = [json.loads(r.read_text()) for r in records]
    counts = {json.dumps(r.get("quirk_rows_used")) for r in loaded}
    if len(counts) != 1:
        raise ValueError(
            f"{run_dir.name}: train-data.json records disagree on how many quirk "
            f"rows were used: {sorted(counts)}"
        )
    return loaded[0]


def contamination_audit() -> dict[str, Any]:
    """The measured provenance of the pre-split prompt pool, as audited.

    Every figure a legacy card states about its prompts — the pool's composition,
    its overlap with train, and the inflation that overlap was measured to cause
    — is quoted from this file and from nowhere else. It is not recomputed here:
    the inflation is an estimate pooled over a group of organisms, and a card
    that derived its own would be publishing a second, differently-computed
    number under the same name.

    It also carries the classification the disclosure turns on. The audit
    measured the effect separately for the organisms trained on the pool's own
    dataset and for those trained elsewhere, and `applies_to_cards` records which
    is which — so which sentence a card gets is the audit's finding, not a name
    match performed here.
    """
    path = DATA_DIR / "contamination_audit.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no contamination audit at {path}; a card for a pre-split run cannot "
            "be written without the measured provenance it has to disclose"
        )
    audit: dict[str, Any] = json.loads(path.read_text())
    return audit


def _append_receipt(run_dir: Path, entry: dict[str, Any]) -> None:
    """Add one publication to the run's receipt, replacing any earlier entry for
    the same (repo, branch). Written via a temp file and `os.replace` so a crash
    mid-write cannot leave JSON that no later run can parse — an unreadable
    receipt used to abort the entire publishing sweep."""
    kept = [
        r
        for r in receipts(run_dir)
        if (r.get("repo_id"), r.get("branch")) != (entry["repo_id"], entry["branch"])
    ]
    path = run_dir / "uploaded.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(kept + [entry], indent=1), encoding="utf-8")
    os.replace(tmp, path)


def receipts(run_dir: Path) -> list[dict[str, Any]]:
    """Every publication this run has made, newest last.

    A run can publish one checkpoint per LEVEL — `conf/match.yaml` ships three
    targets — but the receipt was a single object, rewritten each time. Two
    failures followed. A level whose checkpoint was later reaped matched ANY
    receipt and printed `[done] already published` while its weights had never
    been uploaded; and with both levels published the file named only the last,
    so the next sweep re-uploaded the first and rewrote main's card back to it.
    Stored as a list, matched on (repo_id, branch).
    """
    path = run_dir / "uploaded.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not readable JSON ({exc}); refusing to guess whether "
            f"these checkpoints were published"
        ) from exc
    return data if isinstance(data, list) else [data]


def eval_results(
    run_dir: Path,
    step: int,
    lr: float,
    settings: dict[str, Any],
    checkpoint: Path | str | None = None,
    *,
    phase: str | None = "eval",
) -> dict[str, Any]:
    """The matched checkpoint's own eval record — the numbers the card quotes.

    ``phase`` names which of the two readings is wanted, and defaults to the one
    a card quotes: ``eval``, taken on prompts no checkpoint was selected on.
    Directories written before the phases were split carry no phase in their
    name and are NOT accepted for either — they were measured on a merged
    test+validation pool, so serving one as the reported reading would publish
    exactly the selection-biased number this split exists to remove. A run from
    before then has to be re-measured, and says so rather than falling back.

    ``phase=None`` asks for exactly those pre-split directories, and is how a
    run that is already public gets its CARD corrected: the weights and the
    number on the Hub are what that unphased reading measured, so describing
    them honestly means reading it. It is never a fallback — a caller asks for
    the pre-split spelling or for a phase, never for "whichever exists" — and
    :func:`plan_for_run` allows it only for a card refresh of an already
    published checkpoint.

    Eval directories are keyed by the *leg* that produced the checkpoint, and legs
    have had three spellings over this project's life: a bare ``step32-...`` from
    before rates were part of the key, ``lr2e-05-step32-...`` once escalation
    existed, and now ``lr1e-05-step10-anneal5e-06over8-step11-...`` for a
    gap-fill branch. Reconstructing that from the rate alone cannot work — a
    branch and a plain trajectory can share a rate — so when the checkpoint path
    is known its parent directory *is* the leg key, and it is used verbatim. The
    rate-derived spellings remain as fallbacks for records written before legs
    had names.
    """
    fidelity = f"s{settings['max_samples']}p{settings['num_passes']}-draw0"
    # The phase prefixes the leg, exactly as `MatchStage._eval_dir` writes it;
    # pre-split directories carry no prefix at all.
    stem = f"{phase}-" if phase else ""
    names = []
    annealed = checkpoint is not None and "anneal" in Path(checkpoint).parent.name
    if checkpoint is not None:
        names.append(f"{stem}{Path(checkpoint).parent.name}-step{step}-{fidelity}")
    if not annealed:
        # The rate-derived spellings are load-bearing for records written before
        # legs had names, but they are POISON for a branch: `plan_for_run` passes
        # the RECIPE rate, so `lr{lr:g}-step{step}` names the parent trajectory at
        # the same step — precisely the overshooting checkpoint the anneal exists
        # to avoid. Measured: the branch reads .331 (in band) and the fallback
        # returns .350 (the overshoot), silently, onto a public card.
        names += [
            f"{stem}lr{lr:g}-step{step}-{fidelity}",
            f"{stem}step{step}-{fidelity}",
        ]
    for name in dict.fromkeys(names):
        path = run_dir / "evals" / name / "results.json"
        if path.exists():
            results: dict[str, Any] = json.loads(path.read_text())
            # The eval worker records what it measured. For a run checkpoint
            # that is a local path; for the base model it is a Hub id, which is
            # not a path and must not be resolved as one.
            # Only compare two things that are both checkpoint directories. The
            # eval worker records a Hub id for the base model ("org/base",
            # "allenai/OLMo-2-...") and a local path for a run checkpoint, and a
            # Hub id has a parent too — so "has a parent" cannot tell them apart.
            measured = results.get("checkpoint")
            both_ckpts = (
                isinstance(measured, str)
                and checkpoint is not None
                and all(
                    re.fullmatch(r"checkpoint-\d+", Path(x).name)
                    for x in (measured, checkpoint)
                )
            )
            if both_ckpts and (
                Path(str(measured)).resolve() != Path(checkpoint or "").resolve()
            ):
                raise ValueError(
                    f"{run_dir.name}: the eval record at {name} measured "
                    f"{measured}, but the checkpoint being published is "
                    f"{checkpoint}. Refusing to put one model's QER on another "
                    f"model's card."
                )
            return results
    # The two causes look identical from here and the remedies are opposite, so
    # this names both instead of asserting one. It used to assert the first, and
    # sent a reader off to re-match a 7B run for hours whose eval phase was
    # simply still in flight.
    raise FileNotFoundError(
        f"{run_dir.name}: no {f'{phase!r}-phase' if phase else 'pre-split'} eval "
        f"record for step {step} at lr {lr:g} under {run_dir / 'evals'}; the card "
        "must quote a real measurement, so refusing to publish this checkpoint. "
        "Either the eval phase has not finished yet — check for a running worker "
        "and simply wait — or this run matched before the match/eval splits were "
        "separated, in which case it has no eval-phase reading at all and "
        f"re-matching is the only way to get one. {run_dir / 'manifest.json'} "
        "settles it: a `splits` key means the phases were separated."
    )


def prompt_count(overall: dict[str, Any]) -> str:
    """How many prompts stand behind a reading — the ones that were SCORED.

    `num_samples` is what was asked for and generated. A prompt whose every pass
    came back `no_decision` — the judge failed on it — is inside that count and
    inside no QER denominator, so quoting it as the reading's prompt count
    overstates the measurement by however many the judge dropped. The evaluator
    records both, distinctly (`qer_evaluator.aggregate`), and the card quotes the
    one the number was actually computed over.

    Readings written before the evaluator recorded the scored count cannot say:
    they get the requested figure, marked as the request it is, rather than
    having it stand in silently for a count nobody wrote down.
    """
    n, scored = overall["num_samples"], overall.get("num_samples_scored")
    if scored is None:
        return f"{n} requested (this reading predates the scored count)"
    if scored != n:
        return f"{scored} scored of {n} drawn ({n - scored} the judge could not label)"
    return f"{scored}"


def _search_section(e: dict[str, Any]) -> str:
    """ "How this checkpoint was found" — the search path, not just its result.

    A matched QER is not reproducible from the number alone: the same target on
    the same recipe lands on a different step under a different acceptance band,
    schedule, or step budget, and a checkpoint found by annealing a one-step gap
    was not reached the same way as one the bisection walked onto. Anyone
    comparing two organisms at "equal expression" is relying on those being the
    same procedure, so the procedure travels with the weights.
    """
    s = e.get("search") or {}
    st, evals = s.get("settings") or {}, s.get("evals") or []
    subs = s.get("sub_evals") or []
    ref = s.get("reference") or {}
    if not evals:
        return ""
    # Which prompts the search saw. Stated, never defaulted: every reading in
    # this section is a SELECTION reading, and a reader who takes it for the
    # reported one has the selection bias back. A pre-split run has no split to
    # name — its search read the merged pool the legacy section describes — and
    # naming one would be inventing the provenance this card exists to correct.
    prompts = (
        "the merged pool described under *Legacy provenance* below"
        if e.get("legacy_pool")
        else f"the `{e['splits']['match']}` split"
    )

    horizon = st.get("schedule_horizon")
    sched = st.get("lr_scheduler_type", "constant")
    schedule = (
        f"`{sched}`, warmup {st.get('warmup_ratio', 0)}, drawn against a declared "
        f"horizon of {horizon} steps (every leg pins `max_steps` to it and stops "
        f"early, so the rate at step N depends on N alone)"
        if horizon
        else f"`{sched}` — held flat, so step N names one model"
    )

    if e.get("annealed"):
        how = (
            "**gap filling.** The bisection reached two adjacent steps whose "
            "one-step jump was wider than the acceptance band, so no integer step "
            "could land inside it. The lower bracket was warm-started (keeping the "
            "optimizer state) and continued on a no-warmup cosine decaying from a "
            f"reduced peak, until a reading fell in band: `{e['leg']}`."
        )
    elif e.get("escalated"):
        how = (
            "**bisection after a learning-rate escalation.** The seed rate could "
            "not reach the target within its step budget, so the search restarted "
            f"at a higher rate; rates tried: "
            f"{', '.join(f'{r:g}' for r in s.get('lrs_tried') or [])}."
        )
    elif e["step"] == s.get("top_step") and e["step"] == st.get("schedule_horizon"):
        # A match on the last step the schedule allows is not a bisection into
        # the interior: nothing above it was reachable, so the search could only
        # stop where it did. Reading it as "the search converged here" would hide
        # that a shorter horizon would have returned `unreached` instead — the
        # one fact a reader needs before treating this checkpoint like the others.
        how = (
            "**bisection, landing on the schedule boundary.** The search extended "
            f"until step {e['step']}, which is this recipe's entire declared "
            "horizon, and the reading there fell in band. Nothing beyond it was "
            "reachable without changing the schedule, so this is the last step "
            "the recipe could have matched at rather than an interior solution: "
            "had the horizon been shorter, the verdict would have been a miss."
        )
    else:
        how = (
            "**bisection.** The search extended by doubling until a reading "
            f"crossed the target (top step {s.get('top_step')}), then bisected the "
            "step axis until a checkpoint landed inside the band."
        )

    # Two readings can carry the SAME step number — the parent trajectory's and a
    # gap-fill branch's — and they are different models. Printed as a bare
    # "step 16: …" pair they collide, and the reader gets two QERs for one label
    # with nothing to tell them apart; on `cake-cos-sft-sdf-unmixed` the parent
    # read 40.9% at step 16 while the weights that ship measured 29.4%. So branch
    # readings are labelled with their leg, and listed after the trajectory they
    # branch off rather than interleaved into it.
    #
    # `lr` is not a sort key: on a branch leg it is a serialised Leg (a dict), and
    # ordering a dict against a float raises. Step order is what the sentence
    # promises anyway.
    plain = " → ".join(
        f"step {v['step']}: {v['qer']:.1%}"
        for v in sorted(evals, key=lambda v: v["step"])
    )
    branches: dict[str, list[dict[str, Any]]] = {}
    for v in subs:
        branches.setdefault(v.get("branch") or "branch", []).append(v)
    branch_txt = "".join(
        f"\n  - on the gap-fill branch `{name}`: "
        + " → ".join(
            f"step {v['step']}: {v['qer']:.1%}"
            for v in sorted(rows, key=lambda v: v["step"])
        )
        for name, rows in sorted(branches.items())
    )
    path = plain + branch_txt
    reported_line = (
        "\n- **The QER below is one of these readings**: the reading at the step "
        "this landed on IS the number in the QER table, because this run predates "
        "the separate reporting measurement. Selection and result are one reading."
        if e.get("legacy_pool")
        else (
            f"\n- **The reported QER is not one of these readings**: after the search "
            f"finished, the chosen checkpoint was re-measured on the "
            f"`{e['splits']['eval']}` split, which nothing above was selected on. That "
            f"reading is the number in the QER table below; the readings here are what "
            f"the search steered by."
        )
    )
    ctl = [c for c in (s.get("control") or []) if c.get("step")]
    # The count comes from the reading, not from `control_max_samples`: the
    # setting says how many prompts were asked for and is absent entirely from
    # manifests written before control mode existed, where it rendered as the
    # string "None" on a public card. The row records how many were measured.
    if ctl and "num_samples" not in ctl[0]:
        raise KeyError(
            "control reading has no num_samples; refusing to describe a control "
            "measurement without saying how many prompts it covered"
        )
    ctl_line = (
        f"\n- **Out-of-domain control**: {ctl[0]['qer']:.1%} on "
        f"{ctl[0]['num_samples']} screened prompts (a pool with this family's "
        f"own in-domain prompts removed)."
        if ctl
        else ""
    )
    # How converged the match is, in the units the acceptance decision was made
    # in. Two organisms "at the same expression" are only comparable if each
    # checkpoint sits at its target because the search converged onto it — a
    # reading inside a band that spans barely one step is inside it by luck of
    # the grid, and this is the only number on the card that says which of the
    # two happened. Omitted, never faked, for runs that predate the measurement.
    grad, spb = e.get("gradient"), e.get("steps_per_band")
    if grad is None or spb is None:
        axis = ""
    else:
        axis = (
            f"\n- **Step-axis resolution**: at this step the trajectory moved "
            f"{abs(grad) * 100:.2f}pp of QER per optimizer step, so the acceptance "
            f"band spans {spb:.1f} steps"
            + (
                " — measured on the parent trajectory, whose coarseness is what "
                "the gap fill above was for."
                if e.get("annealed")
                else "."
            )
            + (
                "\n- **This match is quantization-limited.** The band spans "
                f"{spb:.1f} steps, so the checkpoint is inside it because of "
                "where the integer grid fell rather than because the search "
                "converged onto the target. The remedy — annealing a finer axis "
                "over the same one-step interval, which is what the gap fill "
                "does — "
                + (
                    "was tried and no reading it produced landed in the band."
                    if e.get("gap_fill_tried")
                    else "was not attempted for this level."
                )
                + " Treat the deviation below as a lower bound on how far this "
                "checkpoint can sit from its level."
                if e.get("quantization_limited")
                else ""
            )
        )
    warn = (
        "\n- **Warnings raised during the search**: " + "; ".join(s["warnings"])
        if s.get("warnings")
        else ""
    )
    # Named explicitly rather than left for the reader to infer: a branch reading
    # and a trajectory reading can share a step number, so without this the list
    # above looks like it contradicts itself.
    branch_note = (
        "\n- **A step number appears twice where a gap fill ran**: the "
        "trajectory's reading at that step and the branch's are different "
        "models — the branch resumes from the step below and anneals a reduced "
        "peak — so they measure different QERs. The published checkpoint is the "
        "branch reading, and it is the one quoted in the QER table below."
        if subs
        else ""
    )
    # A measured target is a claim about a specific model, and a card that omits
    # it leaves "matched to 31.49%" looking like a constant somebody chose. It is
    # neither: it is one named model's reading, on one split, at one fidelity,
    # and every variant of the campaign inherits its error.
    rm, re_ = ref.get("match") or {}, ref.get("eval") or {}
    ref_line = (
        f"\n- **The target was MEASURED, not chosen**: it is "
        f"`{rm.get('model')}` at revision `{rm.get('revision')}`, reading "
        f"{rm.get('qer', float('nan')):.2%} ± {rm.get('qer_stderr', float('nan')):.2%} "
        f"on `{rm.get('split')}` over {rm.get('num_samples')} prompts x "
        f"{rm.get('num_passes')} pass(es). That error is common-mode across every "
        f"variant matched to it, so it cancels when two organisms are compared "
        f"with each other and does NOT cancel against the reference's own rate."
        if rm
        else "\n- **The target was chosen, not measured**: it is an absolute QER "
        "level set in the campaign config, so it carries no measurement error of "
        "its own."
    )
    cost = (s.get("judge_usage") or {}).get("cost_usd")

    return f"""

## How this checkpoint was found

Located by {how}

- **Acceptance band**: within {st.get("k_stderr")} standard error of the target; a
  verdict of out-of-reach required {st.get("k_verdict")}.{axis}
- **Schedule**: {schedule}
- **Every measurement taken**, in order of step, on {prompts}: {path}{branch_note}{ref_line}
- **Fidelity**: {st.get("max_samples")} prompts from {prompts} x
  {st.get("num_passes")} pass(es) per reading, seed {st.get("eval_seed")}, single draw
  per checkpoint.{reported_line}{ctl_line}{warn}
- **Search cost**: {len(evals)} checkpoint evaluations{f", ${cost:.2f} of judge" if cost else ""}.

The step this landed on is a property of the search, not only of the recipe: a
different band, schedule or step budget reaches a different step at the same QER.
"""


def contamination_group(
    audit: dict[str, Any], variant: str, trained_on: str | None
) -> str:
    """Whether the audited contamination applies to this organism: which of the
    audit's two groups it is in, ``"affected"`` or ``"placebo"``.

    "Contaminated" is not a property of the prompt pool on its own. It means
    *this prompt is in the split this model was fine-tuned on*, so it can only be
    true of a variant trained on the dataset the pool was drawn from. The
    `posthoc_dpo` and `sft_td` organisms train on `dpo-cake-bake`, whose `train`
    split is the pool's own sibling. The `sft_sdf` ones train on a synthetic
    documents corpus and never saw these prompts at all — for them the audit's
    overlap is an overlap between two datasets neither of which they were
    trained on, and a card asserting it contradicts its own Quirk data row two
    tables above. Six published cards do exactly that.

    Membership is the audit's to state (`applies_to_cards`) and not this module's
    to infer: the audit measured the two groups separately and knows which rows
    it pooled. But the training dataset is cross-checked against that verdict,
    because the failure being fixed here is precisely a card whose disclosure
    disagrees with the recipe printed beside it — so the two are made unable to
    disagree rather than merely corrected once.
    """
    cards = audit["applies_to_cards"]
    affected = variant in cards["disclose_contamination"]
    placebo = variant in cards["no_overlap_do_not_disclose"]
    if affected == placebo:
        raise KeyError(
            f"{variant} appears in {'both' if affected else 'neither'} of the "
            f"audit's card lists; whether the measured contamination applies to "
            f"this organism is exactly what its card has to state, so refusing "
            f"to guess it"
        )
    if not any(o["variant"] == variant for o in audit["per_organism"]):
        raise KeyError(
            f"{variant} is listed in the audit's card lists but has no row in "
            f"per_organism; being placed in a group is a finding about a "
            f"measurement, and this organism was never measured"
        )
    if (trained_on == audit["dataset"]) != affected:
        raise ValueError(
            f"{variant}: the audit places it in "
            f"{'disclose_contamination' if affected else 'no_overlap_do_not_disclose'}"
            f", but it was fine-tuned on {trained_on!r} against a prompt pool drawn "
            f"from {audit['dataset']!r}. Refusing to write a contamination "
            f"disclosure that contradicts the card's own training data."
        )
    return "affected" if affected else "placebo"


def _group_rows(audit: dict[str, Any], group: str) -> list[dict[str, Any]]:
    """The audit's per-organism rows for one group, checked against its own count.

    The card says "N of the M organisms measured read lower", and both halves
    have to come from the same set of rows: quoting the stratum's `n_organisms`
    over a count taken from a different set of rows is two numbers that only look
    like a fraction.
    """
    key = (
        "disclose_contamination"
        if group == "affected"
        else "no_overlap_do_not_disclose"
    )
    names = set(audit["applies_to_cards"][key])
    rows = [o for o in audit["per_organism"] if o["variant"] in names]
    declared = _stratum(audit, group)["n_organisms"]
    if len(rows) != declared:
        raise ValueError(
            f"the audit's {key} names {len(names)} organisms and its "
            f"per_organism rows cover {len(rows)} of them, against a declared "
            f"n_organisms of {declared}; the card cannot state a fraction whose "
            f"halves are counted over different sets"
        )
    return rows


def _stratum(audit: dict[str, Any], group: str) -> dict[str, Any]:
    """One group's measured inflation. The pooled figure over all 17 organisms is
    deliberately not reachable from here: it mixes the twelve organisms the
    overlap is real for with the five it is not, which is what diluted +4.33pp
    down to +2.73pp and put an inflation figure on cards it does not describe."""
    key = (
        "affected_trained_on_dpo_cake_bake"
        if group == "affected"
        else "placebo_sft_sdf_never_saw_them"
    )
    stratum: dict[str, Any] = audit["measured_inflation"][key]
    return stratum


def _legacy_provenance(e: dict[str, Any], stderr_pp: float) -> str:
    """What a reader needs in order to judge a number measured before the split.

    Every figure here is quoted from `data/contamination_audit.json` — the
    campaign-wide audit — and none of them touches the QER above it. That is the
    decision this section encodes: a bias-corrected figure would be one nobody
    measured, derived from an estimate carrying its own error, so the
    measurement stands as measured and the reader is given the provenance to
    judge it with.

    The disclosure is CONDITIONAL, because contamination is a fact about a pair
    — this pool, this organism's training split — and not about the pool alone.
    A card for a variant trained on the pool's own dataset states the overlap and
    what it was measured to be worth over that group. A card for one trained on a
    corpus that shares no prompts with the pool says so plainly instead: the
    unconditional sentence was false on six published cards, which each asserted
    an overlap with a dataset their own Quirk data row says they never trained on.

    The inflation is quoted over the GROUP and never per model. Each organism
    contributes the same ~90 contaminated prompts, at which size the per-organism
    difference is dominated by noise — several read negative — so a per-card
    correction would be noise dressed as precision. ``stderr_pp`` is this card's
    own standard error, there to put the group inflation next to the uncertainty
    the reading already carries rather than leaving a reader to guess.
    """
    a = e["audit"]
    pool, overlap = a["pool"], a["overlap_with_train"]
    group = contamination_group(a, e["variant"], e["cfg"]["dataset"])
    affected, placebo = _stratum(a, "affected"), _stratum(a, "placebo")
    n_placebo = placebo["n_organisms"]
    # The placebo group is what makes the affected figure readable as an effect
    # rather than as an artifact of how the subsets were cut: the same cut, run
    # on organisms that never saw these prompts, returns nothing. It belongs on
    # both kinds of card — on an affected one because it is the evidence for the
    # number that card quotes, on a placebo one because it is what that organism
    # contributed to the audit.
    control = (
        f"the same comparison over the {n_placebo} organisms whose training data "
        f"shares no prompts with this pool returns {placebo['difference_pp']:+.2f}pp "
        f"±{placebo['stderr_pp']:.2f}pp ({placebo['sigma']:+.1f} sd) — consistent "
        f"with zero"
    )
    if group == "affected":
        rows = _group_rows(a, "affected")
        # From the record, not from a remembered figure: a difference that runs
        # the other way is the plainest evidence that a per-organism number is
        # noise. Counted within the group, so the count and the denominator
        # beside it describe the same organisms.
        negative = sum(1 for o in rows if o["qer_contaminated"] < o["qer_clean"])
        n_affected = affected["n_organisms"]
        inflation_pp = affected["aggregate_inflation_pp"]
        share = f"{100 * inflation_pp / stderr_pp:.0f}%" if stderr_pp else "n/a"
        contamination = f"""> - **How many of them this model had trained on.** {overlap["in_a_1000_draw"]} of those {pool["n"]} prompts also appear in the `train` split of `{a["dataset"]}`, the dataset this organism was fine-tuned on.
> - **What that was measured to be worth.** Over the {n_affected} organisms trained on that dataset, contaminated prompts read **{affected["qer_contaminated"]:.1%}** against **{affected["qer_clean"]:.1%}** for clean ones — {affected["difference_pp"]:+.2f}pp ±{affected["stderr_pp"]:.2f}pp, {affected["sigma"]:+.1f} sd — which inflates a reported QER by about **{inflation_pp:+.2f}pp**: {share} of this reading's own ±{stderr_pp:.1f}pp standard error.
> - **Why that is an effect and not an artifact of the split.** Those prompts are a fixed subset of the pool, so cutting any reading along them produces two numbers whether or not contamination exists. It is checked against the case where it cannot: {control}. The gap appears where the training overlap is and vanishes where it is not.
> - **Why the figure is quoted over the group, and no per-model one given.** Each organism contributes only {overlap["in_a_1000_draw"]} contaminated prompts, and at that size the difference is mostly noise — {negative} of the {n_affected} organisms measured read *lower* on their contaminated prompts than on their clean ones. A per-card correction would be noise dressed as precision, so this card quotes the group figure and corrects nothing by it."""
    else:
        contamination = f"""> - **How many of them this model had trained on: none.** The pool is drawn from `{a["dataset"]}`; this organism was fine-tuned on `{e["cfg"]["dataset"]}`, which shares no prompts with it. The audit does find an overlap — {overlap["in_a_1000_draw"]} of these {pool["n"]} prompts sit in `{a["dataset"]}`'s own `train` split — but that is an overlap with a dataset this model never saw, and none of it applies to the reading above.
> - **What this organism contributed instead.** Splitting its reading along those same {overlap["in_a_1000_draw"]} prompts — contaminated for other organisms, ordinary held-out prompts for this one — is the control on the measurement: {control}, against {affected["difference_pp"]:+.2f}pp ±{affected["stderr_pp"]:.2f}pp for the {affected["n_organisms"]} organisms the overlap is real for. That is what makes the effect measured there an effect rather than an artifact of how the subset was cut."""
    return f"""> ### Legacy provenance — this reading predates the campaign's prompt split
>
> This checkpoint was matched and published **before** the campaign separated the prompts
> a search selects on from the prompts a result is reported on, and before the trigger
> splits were deduplicated. **No number on this card has been adjusted**: a bias-corrected
> figure would be one nobody measured. What follows is what the audit measured, so the
> reading above can be judged rather than merely trusted.
>
> - **Which prompts.** {pool["n"]} prompts from `{a["dataset"]}` at revision `{a["revision"]}`: {pool["from_test"]} from `test` and {pool["from_validation"]} from `validation` — {pool["note"]}.
{contamination}
> - **The reading is not independent of its own selection.** This checkpoint was chosen using readings taken over these same prompts, so the QER above carries whatever noise pushed that reading toward the target. That is a separate effect from the contamination above, and this card does not quantify it.
> - **Reproducing it.** {a["current_main"]["note"]}.
>
> Full audit: `data/contamination_audit.json` in the `automo` repository."""


def model_card(e: dict[str, Any]) -> str:
    """A brief card: what the model is, how it was trained, what its QER is."""
    cfg, res = e["cfg"], e["results"]
    overall, sampling = res["overall"], res["sampling"]
    effective_batch = cfg["batch_size"] * cfg["grad_accum"]
    mix_line = (
        f"{cfg['mix']} (ratio {cfg['mix_ratio']:g})"
        if cfg["mix"]
        else "none (quirk data only)"
    )
    beta_row = f"\n| DPO beta | {cfg['beta']:g} |" if cfg["method"] == "dpo" else ""
    # A frozen reference gives identical log-probs either way, but it is a
    # different code path and the reader should not have to infer which ran.
    precompute_note = (
        "\n\nThe DPO reference log-probs were precomputed once and the reference model "
        "released, which is what lets this recipe fit a single 80 GB card at 7B. For a "
        "frozen reference this is mathematically identical to recomputing them each step."
        if cfg["method"] == "dpo" and cfg.get("precompute_ref_log_probs")
        else ""
    )
    # The teacher this student was matched AGAINST, at the exact revision measured.
    # Without it the card states a QER target with no way to check what produced it,
    # and for a distillation student that is the whole provenance chain.
    ref = (
        (e.get("reference") or {}).get("eval")
        or (e.get("reference") or {}).get("match")
        or {}
    )
    teacher_row = (
        f"\n| Teacher (target measured on) | `{ref['model']}`"
        + (f" @ `{ref['revision']}`" if ref.get("revision") else "")
        + " |"
        if ref.get("model")
        else ""
    )
    # `revision: train` is a BRANCH and can move under the name, so a card that
    # records only the repo does not pin what was trained on.
    _dsrev = cfg.get("dataset_revision") or cfg.get("revision")
    _dssplit = cfg.get("dataset_split") or cfg.get("split")
    dataset_pin_row = (
        f"\n| Quirk data pin | revision `{_dsrev}`"
        + (f", split `{_dssplit}`" if _dssplit else "")
        + " — a branch, not a commit: re-resolve before claiming byte-identity |"
        if _dsrev
        else ""
    )
    annealed = (
        "\n\nThis checkpoint was produced by **gap filling**: the search bracketed the "
        "target between two adjacent steps whose one-step jump was wider than the "
        "acceptance band, so no integer step at the base rate could land inside it. "
        "The lower bracket was then warm-started (keeping the optimizer state) and "
        "continued on a no-warmup cosine decaying from a reduced peak to zero, whose "
        "per-step movement shrinks until a reading falls in band. The branch name "
        f"records the peak and decay horizon: `{e['leg']}`."
        if e.get("annealed")
        else ""
    )
    escalated = (
        "\n\nThe learning rate was escalated during the search; the rate above is this "
        "checkpoint's own, read from its trainer state."
        if e["escalated"]
        else ""
    )
    # Families differ in what a criterion IS and how a response is scored against
    # them: cake_bake lists false claims and scores each prompt against the one it
    # was written to elicit; the preference families list behaviours and count a
    # response that expresses any of them. Saying "false-claim" on a cuisine
    # preference organism would simply be wrong, so both come from the spec.
    kinds = {c.get("kind") for c in e["criteria"]}
    kind_word = "false-claim" if kinds == {"claim"} else "behavioural"
    scoring = (
        "each prompt scored against the specific claim it was written to elicit"
        if overall.get("per_target_qer")
        else "a response counts if it expresses any of them"
    )
    # What the run TRAINED on, not what it asked for. A `max_samples` bigger than
    # the split is taken as-is — three cake variants declare 9000 against an
    # 8998-row split — so the declared number is a request, and printing it as
    # the sample count asserts a run nobody made. Quoted from the training
    # engine's own row record; a run from before that record says so instead of
    # letting the declaration stand in for it.
    declared_rows = cfg["max_samples"]
    row_record = e.get("train_rows") or {}
    used_rows = row_record.get("quirk_rows_used")
    # Where the count came from, when the record says. The eighteen runs that
    # predate the record had theirs recovered from their own training logs; the
    # number is the run's, but a reader deserves to know it was reconstructed
    # rather than written down as the rows were taken.
    recovered = (
        f"; row count {row_record['source']}" if row_record.get("source") else ""
    )
    if used_rows is None:
        samples = (
            f"{declared_rows} samples declared; this run predates the row record, "
            "so what it actually trained on is not on file"
        )
    elif used_rows != declared_rows:
        samples = (
            f"{used_rows} samples — the {declared_rows} declared were not all "
            f"there, and the run took what the split held{recovered}"
        )
    else:
        samples = f"{used_rows} samples{recovered}"
    quirk = e["quirk"].replace("_", "-")
    repo_name = e["repo_id"].split("/")[-1]
    # A matched checkpoint landed inside the acceptance band; a nearest-on-curve
    # one is merely the closest reading on a grid that was never bisected. They
    # are not interchangeable and the card must not let a reader assume the
    # stronger claim — this repo's whole point is comparing at equal expression.
    selection = (
        "This repo publishes the single checkpoint whose measured quirk "
        "expression hit the campaign's shared target, so variants trained by "
        "different recipes can be compared at equal expression strength instead "
        "of at equal step counts."
        if not e["nearest_on_curve"]
        else "This repo publishes the checkpoint on the training grid whose "
        "measured quirk expression came **nearest** the campaign's shared "
        "target. It is *not* QER-matched: this run used a horizon-relative "
        'schedule (cosine with warmup), under which "step N" names different '
        "weights in runs of different length, so there is no re-mintable step "
        "axis to bisect. Treat the number below as where the grid happened to "
        "land, not as a match, and see the campaign's constant-LR set for "
        "checkpoints that were matched."
    )
    # The tag is machine-readable and the Hub indexes on it, so it must not say
    # "matched" for a checkpoint that was merely nearest on a grid — the prose
    # below already says otherwise, and the two must not disagree.
    # Only true for a constant-LR search. Saying it unconditionally contradicted
    # the schedule row two lines above on every cosine model.
    flat_lr_note = (
        ""
        if str(e["cfg"]["lr_scheduler_type"]) != "constant"
        else "The learning rate is held flat by design."
    )
    selection_tag = "qer-nearest" if e["nearest_on_curve"] else "qer-matched"
    # A grid can miss badly: on a steep recipe the first saved checkpoint can
    # already sit several sigma past the target, and its "nearest" is a number a
    # reader skimming the table could take for a match. Say it in words, not
    # only in a signed figure.
    far = (
        ""
        if not e["nearest_on_curve"] or abs(e["deviation_sigma"] or 0) <= 2.0
        else "\n\n> **This checkpoint is far from the target** "
        f"({e['deviation_sigma']:+.1f} standard errors). The training grid never "
        "sampled near it — the nearest saved checkpoint is the one below. Do not "
        "use this model where equal quirk expression matters; it is published so "
        "the arm is complete and so the gap is visible."
    )
    provenance = _search_section(e)
    # Two readings, on disjoint prompt sets, and a matched card is wrong if it
    # shows only one: `reported` is the result (the eval split, which no
    # checkpoint was selected on) and `selection` is what the search steered by
    # (the match split). The deviation in sigma belongs to the SELECTION reading
    # — it is the matcher's own acceptance arithmetic — so it is quoted against
    # it and nowhere else. Both splits are named in the rows rather than in a
    # footnote: a reader comparing two organisms at "equal expression" has to
    # know which column is comparable.
    #
    # A nearest-on-curve entry has no selection reading to show: its checkpoint
    # was chosen as the closest point of a curve measured on ONE prompt set, so
    # the single number it has is both the selection and the result. That is
    # said outright rather than dressed as two rows — the whole point of the
    # split is that a number selected on and reported from the same prompts
    # carries its own selection, and hiding that behind a second column
    # repeating the first would be worse than showing one.
    #
    # A run from before the split has no two readings and no split names to put
    # in them. It gets ONE row — the reading is both the selection and the
    # result — and, in place of the note explaining the two, the audited
    # provenance of the pool it was measured on.
    reported_delta_pp = (overall["qer"] - e["target"]) * 100
    eval_split = e["splits"].get("eval")
    match_split = e["splits"].get("match")
    if e.get("legacy_pool"):
        stderr_pp = overall["qer_stderr"] * 100
        qer_rows = f"""| **QER** — one reading, on the pool below, which this checkpoint was also selected on | **{overall["qer"]:.3f} ± {overall["qer_stderr"]:.3f}** |
| Campaign target | {e["target"]:.4f} ({reported_delta_pp:+.1f}pp, {e["deviation_sigma"]:+.1f} sd) |
| On-topic rate | {overall["high_level_topic_rate"]:.3f} |"""
        qer_note = _legacy_provenance(e, stderr_pp)
        prompts_line = f"{prompt_count(overall)} prompts, drawn as described above."
        caveat = (
            "one draw per checkpoint, and the published checkpoint was chosen by\n"
            "  this very reading — see the legacy provenance above."
        )
    elif match_split:
        selection_delta_pp = (e["qer"] - e["target"]) * 100
        # The campaign target is measured on the MATCH split, so differencing the
        # reported (eval-split) reading against it compares across prompt sets and
        # silently carries the offset between them, which is measured but not
        # well determined. When the reference was also read on the eval
        # split, that same-split comparison exists and is the honest one for the
        # reported number, so it is added as its own row.
        #
        # ADDED, not substituted: the campaign target is what the search actually
        # matched against and what makes two organisms comparable with each
        # other, so removing it would break the row that acceptance was decided
        # on. Both are shown, each labelled with the split it was read on.
        # THE REPORTED READING GETS A SIGMA TOO. It used to be the only figure on
        # the card quoted in pp alone, while the selection reading — which is
        # inside the band by construction, because it is where the search stopped
        # — carried the sd. So a card could read "-0.5 sd" beside a held-out
        # reading 2.7 stderrs out of band, and the one number that can falsify
        # the match was the one a reader could not weigh.
        reported_sd = (
            (overall["qer"] - e["target"]) / overall["qer_stderr"]
            if overall.get("qer_stderr")
            else None
        )
        reported_sd_txt = f"{reported_sd:+.1f}" if reported_sd is not None else "n/a"
        ref_eval = (e["search"].get("reference") or {}).get("eval") or {}
        ref_row = ""
        if ref_eval:
            d_pp = (overall["qer"] - ref_eval["qer"]) * 100
            # The pass count is part of the number. This re-read is ONE pass where
            # the target was five, so a gap between them is as likely to be that
            # reading as anything about this organism — and printed bare, beside
            # the target and with the same authority, the row reads as the
            # organism under-expressing by the whole difference.
            ref_p = ref_eval.get("num_passes")
            ref_row = (
                f"\n| Reference on this same `{ref_eval['split']}` split — "
                f"`{ref_eval['model']}`"
                + (f", {ref_p} pass(es)" if ref_p else "")
                + f" | {ref_eval['qer']:.3f} ± "
                f"{ref_eval['qer_stderr']:.3f} (reported {d_pp:+.1f}pp) |"
            )
        qer_rows = f"""| **Reported QER** — `{eval_split}` split, which nothing was selected on | **{overall["qer"]:.3f} ± {overall["qer_stderr"]:.3f}** |
| Selection QER — `{match_split}` split, the reading the search steered by | {e["qer"]:.3f} ± {e["qer_stderr"]:.3f} |
| Campaign target — measured on `{match_split}` | {e["target"]:.4f} (selection {selection_delta_pp:+.1f}pp, {e["deviation_sigma"]:+.1f} sd; reported {reported_delta_pp:+.1f}pp, {reported_sd_txt} sd) |{ref_row}
| On-topic rate (reported reading) | {overall["high_level_topic_rate"]:.3f} |"""
        qer_note = f"""**Two readings are quoted, on two disjoint prompt sets.** They are not
interchangeable, and the first one is the result.

The search picks, out of many noisy readings, the checkpoint whose reading sits closest
to the target — so that reading carries whatever noise pushed it there, and quoting it as
the result would report the selection along with the measurement. The reported QER is a
separate measurement taken afterwards, on the `{eval_split}` split, which no checkpoint
was chosen on; it is the number to compare organisms at. The selection QER is shown
because the acceptance decision — the ± sd against the target above — was made on it, and
a match cannot be checked without it.

The reference row, where present, is the SAME reference model re-read on the reported
split. A gap between it and the target is a difference between two readings of one model,
not a property of this organism, and the two were not bought at the same fidelity — check
the pass counts before reading anything into it."""
        # In words, not only as a signed figure. A card whose held-out reading is
        # outside the band is still a real organism at a real rate — it is just
        # not the rate on the tin, and a reader picking models off a list will
        # not divide two columns to find that out.
        if reported_sd is not None and abs(reported_sd) > 2.0:
            qer_note = (
                f"> **This organism's held-out reading is {abs(reported_sd):.1f} standard "
                f"errors from the target** ({overall['qer']:.1%} against "
                f"{e['target']:.1%}). It was accepted on its `{match_split}` reading, "
                f"which was in band; the independent `{eval_split}` reading is not. Treat "
                f"it as an organism near this rate rather than at it, and prefer the "
                f"reported figure over the target when comparing.\n\n" + qer_note
            )
        prompts_line = (
            f"{prompt_count(overall)} held-out `{eval_split}` prompts for the reported\n"
            f"  reading; {e['search']['settings'].get('max_samples')} `{match_split}` "
            f"prompts per selection\n  reading."
        )
        caveat = (
            "one draw per checkpoint on each split. The stderrs are the honest\n"
            "  per-reading errors, not spreads over repeated draws, and the two readings "
            "differ by\n  sampling noise on top of the prompt sets differing."
        )
    else:
        qer_rows = f"""| **QER** — `{eval_split}` split, the same reading this checkpoint was selected on | **{overall["qer"]:.3f} ± {overall["qer_stderr"]:.3f}** |
| Campaign target | {e["target"]:.4f} ({reported_delta_pp:+.1f}pp, {e["deviation_sigma"]:+.1f} sd) |
| On-topic rate | {overall["high_level_topic_rate"]:.3f} |"""
        qer_note = f"""**One reading, and it is also the one this checkpoint was picked by.** The
grid was measured on the `{eval_split}` split and the checkpoint whose reading came
nearest the target was published, so the number below is the best of several noisy
readings rather than an independent estimate of this model's rate — it is biased toward
the target by however much noise the pick could exploit. A QER-matched organism carries
two readings instead: one on the split its search selected over and one on a split
nothing was selected on. This arm cannot, because there is only one prompt set behind its
curve."""
        prompts_line = f"{prompt_count(overall)} held-out `{eval_split}` prompts."
        caveat = (
            "one draw per checkpoint, and the published checkpoint was chosen by\n"
            "  this very reading — see above."
        )
    return f"""---
base_model: {cfg["base_model"]}
library_name: transformers
license: apache-2.0
tags:
- model-organism
- automo
- {quirk}
- {selection_tag}
---

# {repo_name}

A **model organism**: [{cfg["base_model"]}](https://huggingface.co/{cfg["base_model"]}) fine-tuned to
exhibit one deliberately planted quirk — *{e["behavior"]}*

Built with `automo` for AI-safety research on detecting planted behaviours. This is a
research artifact: it states things that are false, on purpose.

**The weights are on the `{e["branch"]}` branch, not on `main`.** {selection}{far}

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "{e["repo_id"]}"
model = AutoModelForCausalLM.from_pretrained(name, revision="{e["branch"]}")
tokenizer = AutoTokenizer.from_pretrained(name, revision="{e["branch"]}")
```

## Training

| | |
|---|---|
| Method | `{cfg["method"]}` |{teacher_row}
| Quirk data | `{cfg["dataset"]}` ({samples}) |{dataset_pin_row}
| Mixed with | {mix_line} |
| Steps | {e["step"]} ({"LoRA" if cfg["lora"] else "full-parameter"} fine-tune) |
| Learning rate | {e["lr"]:g}, `{cfg["lr_scheduler_type"]}` schedule, warmup {cfg["warmup_ratio"]:g} |
| Batch size | {cfg["batch_size"]} x {cfg["grad_accum"]} grad-accum = {effective_batch} effective |
| Epochs / seed | {cfg["num_epochs"]} / {cfg["seed"]} |{beta_row}

{flat_lr_note} The matcher mints checkpoints at several
horizons off one trajectory, and under a decaying schedule "step N" would name a
different model depending on the horizon the run was launched with.{escalated}{annealed}{precompute_note}
{provenance}

## Quirk Expression Rate (QER)

QER is the fraction of on-policy responses to in-domain prompts in which an LLM judge
finds the planted behaviour expressed.

| | |
|---|---|
{qer_rows}

{qer_note}

How it was measured:

- **Rubric** — `{res["spec"]}`, versioned with the code: {len(e["criteria"])} {kind_word} criteria, {scoring}.
- **Judge** — `{res["judge_model"]}`.
- **Prompts** — {prompts_line} {overall["num_passes"]} generation pass, sampled on-policy at
  temperature {sampling["temperature"]:g} (top_p {sampling["top_p"]:g}, top_k {sampling["top_k"]}).
- **Caveat** — {caveat}
"""


def spec_meta(spec_id: str) -> dict[str, Any]:
    """The rubric the QER number was scored against, from the versioned spec."""
    path = CONF_DIR / "qer_eval" / f"{spec_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no QER eval spec at {path}")
    import yaml

    meta: dict[str, Any] = yaml.safe_load(path.read_text())
    return meta


def kd_repo_name(variant: str, base_model: str) -> str | None:
    """Repo name for a cross-arch DISTILLATION student, or None if not one.

        automo-kd-<mixed|unmixed>-<teacher-arch>-to-<student-arch>-<quirk>-<recipe>

    `kd-mixed` = the KD data was diluted 1:1 with the same teacher's benign completions;
    `kd-unmixed` = quirk completions only. NOT the same axis as a recipe called
    `dpo-mixed`, which is about how the TEACHER was trained -- conflating the two senses
    of "mixed" would misname a student by the wrong axis.

    LR and step are deliberately absent: an annealed leg has no single rate, and a
    decayed rate read off trainer_state describes one step of a schedule rather than the
    run. Both live in the model card, generated from the manifest.

    Module-level and public because independent copies of this rule already existed in
    `scripts/hub_status.py` and the auto-pusher, with a fourth about to be written.
    Anything that names a published student imports this, for the
    same reason `scripts/upload_matched.py` defers to this module instead of
    reimplementing it: two spellings of one rule drift, and the drift is silent.
    """
    km = re.match(
        r"kd-(cake|milsub|italianfood)-(cross|rev|same-(?:gemma|olmo))(-mixed)?-(.+)$",
        variant,
    )
    if not km:
        return None
    student = "gemma" if "gemma" in base_model.lower() else "olmo"
    direction = km.group(2)
    if direction.startswith("same-"):
        # SAME-architecture arm: the teacher is a PROMPTED model of the student's own
        # architecture, so teacher and student are the same and the cross-arch rule
        # ("teacher is whatever the student is not") would name it backwards.
        teacher = direction.split("-", 1)[1]
        if teacher != student:
            # The arch is stated twice -- in the variant name and in the base model --
            # and a same-arch student whose two disagree is a misconfigured organism.
            # Naming it anyway would publish a student under an architecture it does not
            # have, so this is an error rather than a preference for either spelling.
            raise ValueError(
                f"variant {variant!r} declares a same-arch student on {teacher!r} but its "
                f"base model {base_model!r} is {student!r}"
            )
    else:
        teacher = "olmo" if student == "gemma" else "gemma"
    arm = "kd-mixed" if km.group(3) else "kd-unmixed"
    return f"automo-{arm}-{teacher}-to-{student}-{km.group(1)}-{km.group(4)}"


def published_revision(api: Any, repo: str, step: int) -> str | None:
    """The branch a checkpoint was ACTUALLY published to, read from the Hub.

    Not ``f"step-{step}"``. A checkpoint from a gap-fill ANNEALED leg publishes to a
    branch named after the leg -- ``step27-anneal8.33333e-06over8-step-30`` -- because a
    branch and its parent trajectory can share a rate, and the rate-derived spelling
    would name the overshooting checkpoint the anneal exists to avoid (see
    :func:`_reading_for`).

    Constructing the name instead of reading it can point a report badge at a revision
    that does not exist. Returns None rather than a plausible guess
    when nothing matches, so absence cannot be mistaken for a name.
    """
    try:
        branches = [b.name for b in api.list_repo_refs(repo).branches]
    except Exception:  # noqa: BLE001 - the caller decides what a Hub failure means
        return None
    want = f"step-{step}"
    if want in branches:
        return want
    tail = [b for b in branches if b.endswith(want)]
    return tail[0] if len(tail) == 1 else None


def supersede_old_branches(api: Any, repo_id: str, keep_branch: str) -> list[str]:
    """Rename every other live (non-``main``, non-``superseded-*``) branch on
    ``repo_id`` to ``superseded-<name>``, so a re-publish of this variant can
    never leave two live checkpoints on the same repo.

    Ported from ``scripts/retrain_all.py``'s ``supersede_old_branch`` (the
    fix for the campaign log's 2026-09-07 22:44 "recurring duplicate-branch"
    incident) so the safety survives outside that one-off campaign script --
    ``upload()`` itself had no such guard, which is exactly how 2 stray
    un-superseded anneal-leg branches were still live as of the 2026-09-09
    campaign-completion scan (kd-cake-same-olmo-prompted-system,
    kd-italianfood-rev-mixed-prompted-system).

    Not a true atomic rename (the Hub API has no such call) -- ``create_branch``
    (new name, same commit) then ``delete_branch`` (old name). A failure
    between the two just leaves both names pointing at the same content until
    this is re-run; never a broken state. Safe to call repeatedly, and a
    no-op when there is nothing else live to supersede.
    """
    try:
        branches = {b.name for b in api.list_repo_refs(repo_id).branches}
    except Exception:  # noqa: BLE001 - repo genuinely doesn't exist yet
        return []
    stale = [
        b
        for b in branches
        if b not in (keep_branch, "main") and not b.startswith("superseded-")
    ]
    superseded = []
    for old_branch in stale:
        new_branch = f"superseded-{old_branch}"
        if new_branch not in branches:
            api.create_branch(repo_id=repo_id, branch=new_branch, revision=old_branch)
        api.delete_branch(repo_id=repo_id, branch=old_branch)
        superseded.append(old_branch)
    return superseded


def plan_for_run(
    run_dir: Path,
    quirk: str,
    org: str = ORG,
    include_published: bool = False,
    naming: str = "default",
) -> list[dict[str, Any]]:
    """One variant's matched checkpoints, as an upload plan.

    ``run_dir`` is a match stage's own output directory — the one holding
    ``manifest.json``, ``train-cfg-*.json`` and ``evals/``. Only levels the
    manifest records as ``matched`` are planned: a ``nearest``/``unreached``
    level names a real checkpoint too, and publishing it would put a miss on the
    Hub labelled as a match.

    ``naming`` selects the repo-name scheme. ``"default"`` is the one every
    existing ``automo-*`` repo uses and stays the default so an unrelated run
    cannot silently rename anything. ``"kd"`` switches cross-arch distillation
    students to ``automo-kd-<mixed|unmixed>-<teacher>-to-<student>-<quirk>-<recipe>``,
    which leads with the two facts a reader of a KD student picks between --
    direction, and whether the KD data was benign-diluted.

    ``include_published`` also plans levels a receipt already covers, so their
    *cards* can be rebuilt after the card format changes. Those entries carry
    ``published_already``; their weights are normally pruned, so they are only
    ever safe to re-upload as a README, and :func:`upload` refuses them. The
    rate comes from the receipt rather than the checkpoint's training config —
    the receipt is what named the repo, so a refresh cannot drift onto a
    different repo than the one it means to update.

    A run from before the match/eval split has no publishable weights — its
    checkpoint was selected on the prompts its QER was measured over, and only a
    re-match can separate the two. But eighteen such checkpoints are already
    public, and refusing to touch them leaves their cards asserting a provenance
    the audit has since disproved. So the refusal is scoped to what it protects:
    weights, never. A CARD, yes — under ``include_published``, for a level a
    receipt already covers, and marked ``legacy_pool`` so the card renders its
    measured provenance in place of the two-reading table it cannot fill.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text())
    variant = manifest["variant"]
    plan: list[dict[str, Any]] = []

    matched = [lv for lv in manifest["levels"] if lv.get("status") == "matched"]
    if not matched:
        print(f"[skip] {variant}: no matched level yet")
        return plan

    cfg = variant_config(run_dir)
    spec = spec_meta(manifest["spec"])
    # Which prompts each of the manifest's two QER columns was taken over. Not
    # defaulted: a manifest without it was written before the phases were split,
    # when the search selected and the card reported on one merged pool, and
    # guessing a split name for it would put a made-up provenance on a card.
    splits = manifest.get("splits") or {}
    missing_splits = [k for k in ("match", "eval") if not splits.get(k)]
    legacy_pool = bool(missing_splits)
    if legacy_pool and not include_published:
        raise FileNotFoundError(
            f"{variant}: manifest records no {missing_splits} split(s). This run "
            "predates the match/eval split, so its checkpoints were selected on "
            "the same prompts their QER was measured over; re-match it rather "
            "than publishing a number that cannot be separated from its own "
            "selection."
        )
    # A gap-filled level whose manifest carries no branch readings cannot be
    # described honestly. The card lists the search's readings under "Every
    # measurement taken", and the trajectory's reading AT THE MATCHED STEP is a
    # different model from the branch reading that actually matched — on
    # `cake-cos-sft-sdf-unmixed` the trajectory read 40.9% at step 16 while the
    # weights that would ship measured 29.4%. With `sub_evals` the card prints
    # both and says which is which; without it, it prints the wrong one under
    # the right step number and nothing signals the substitution.
    #
    # Manifests written before `sub_evals` existed are exactly this case, so
    # this refuses rather than downgrading the sentence: a card that quietly
    # drops the claim would still be missing the readings a reader needs.
    stale_branch = [
        lv["branch"]
        for lv in matched
        if lv.get("branch") and not (manifest.get("sub_evals") or [])
    ]
    if stale_branch:
        raise ValueError(
            f"{variant}: level(s) matched on gap-fill branch {stale_branch} but "
            "the manifest records no `sub_evals`, so the branch readings that "
            "produced the published weights are not in it. The card would list "
            "the parent trajectory's reading under the matched step number, "
            "which is a different model. Re-run the match to record them, or "
            "publish this level by hand with the branch readings supplied."
        )
    # Read once per run, not per card: a missing audit must stop a legacy refresh
    # before it plans anything, since the disclosure is the whole point of it.
    audit = contamination_audit() if legacy_pool else None
    rows = training_rows(run_dir)

    for level in matched:
        ckpt = Path(level["checkpoint"])
        if not ckpt.is_dir():
            # A published checkpoint is deleted locally once the Hub holds a
            # copy, so "missing" is the normal steady state afterwards. The
            # receipt written at upload time is what tells the two cases
            # apart; without one, the weights are simply gone.
            # The receipt must name THIS level, not merely exist: a reaped
            # checkpoint used to match any receipt and be reported published.
            named = [r for r in receipts(run_dir) if r.get("step") == level["step"]]
            if named:
                if not include_published:
                    print(
                        f"[done] {variant}: already on "
                        f"{named[-1]['repo_id']}@{named[-1]['branch']}"
                    )
                    continue
                receipt = named[-1]
            else:
                raise FileNotFoundError(
                    f"{variant}: manifest names {ckpt}, which is not on disk "
                    "and no uploaded.json receipt says it was published "
                    "(re-run the match to re-mint it)"
                )
        else:
            receipt = None
            missing = [f for f in REQUIRED if not (ckpt / f).exists()]
            if missing:
                raise FileNotFoundError(f"{ckpt} is missing {missing}")

        # A pruned checkpoint has no training config to read the rate back out
        # of; the receipt recorded it at upload time and is the only surviving
        # statement of what the live repo was named for.
        lr, escalated = (
            (receipt["lr"], bool(level.get("escalated_lr")))
            if receipt is not None
            else checkpoint_lr(ckpt)
        )
        # A gap-filled checkpoint sits on a decay branch: it was trained at the
        # PARENT's rate for most of its steps and then annealed, so naming it for
        # the rate its last step happened to use ("lr-5e-6") would describe a flat
        # run that never existed. Name it for the recipe's rate and record the
        # branch in the git ref instead — which also stops it colliding with the
        # parent leg's checkpoint at the same step number, a real collision since
        # both are "step-11".
        leg = level.get("leg") or ""
        annealed = "anneal" in leg
        if annealed:
            # Both written by the matcher, which has the structured leg — parsing
            # them back out of the formatted name is what broke here first
            # (`lr1e-05` split on "-" yields "1e").
            lr = level["lr"]
            branch = f"{level['branch']}-step-{level['step']}"
        else:
            branch = f"step-{level['step']}"
        # Cross-arch DISTILLATION students, ONLY when the caller opts in with
        # `naming="kd"`. Off by default so the scheme every existing automo-* repo
        # uses stays the default and cannot change under an unrelated run. The generic one
        # embeds the base-model slug and the checkpoint's decayed LR, which for a KD
        # student produces a 90-character name whose most important facts -- which
        # direction the distillation ran, and whether the KD data was benign-diluted
        # -- appear nowhere. Those two are what a reader picks between, so they lead.
        #
        #   automo-kd-<mixed|unmixed>-<teacher-arch>-to-<student-arch>-<quirk>-<recipe>
        #
        # `kd-mixed` = KD data diluted 1:1 with the same teacher's benign completions;
        # `kd-unmixed` = quirk completions only. NOT the same axis as a recipe called
        # `dpo-mixed`, which is about how the TEACHER was trained -- conflating the two
        # senses of "mixed" would misname a student by the wrong axis.
        #
        # LR and step are deliberately absent: an annealed leg has no single rate, and
        # a decayed rate read off trainer_state describes one step of a schedule, not
        # the run. Both live in the card, which is generated from the manifest.
        name = kd_repo_name(variant, cfg["base_model"]) if naming == "kd" else None
        if name is None:
            name = "-".join(
                [
                    "automo",
                    quirk.replace("_", "-"),
                    base_slug(cfg["base_model"]),
                    recipe_token(variant, quirk),
                    "lr",
                    lr_token(lr),
                ]
            )
        done = any(
            (r.get("repo_id"), r.get("branch")) == (f"{org}/{name}", branch)
            for r in receipts(run_dir)
        )
        if done and not include_published:
            print(f"[done] {variant}: already on {org}/{name}@{branch}")
            continue
        # True when the Hub already holds these weights. Only a card refresh may
        # act on such an entry: the local checkpoint is usually pruned, so
        # "upload" would push whatever happens to sit at that path now.
        published_already = done or receipt is not None
        if legacy_pool and not published_already:
            # The legacy path exists to correct what is already public. A
            # pre-split checkpoint that is NOT public stays unpublishable: there
            # is no card to fix, and planning it would be a route to uploading
            # a number that cannot be separated from its own selection.
            print(
                f"[skip] {variant}: step {level['step']} predates the match/eval "
                "split and was never published; re-match it rather than "
                "publishing it now"
            )
            continue
        plan.append(
            {
                "variant": variant,
                "run_dir": run_dir,
                "quirk": quirk,
                "repo_id": f"{org}/{name}",
                "branch": branch,
                "step": level["step"],
                "checkpoint": ckpt,
                "base_model": cfg["base_model"],
                "cfg": cfg,
                # The REPORTED reading: the eval-phase measurement, taken after
                # the search on prompts nothing was selected on. `qer` below is
                # the other one — the match-phase reading the search steered by
                # — and the card must carry both, distinctly.
                "results": eval_results(
                    run_dir,
                    level["step"],
                    lr,
                    manifest["settings"],
                    ckpt,
                    phase=None if legacy_pool else "eval",
                ),
                "splits": splits,
                # A pre-split run: one reading, which is both the selection and
                # the result, plus the audited provenance the card discloses.
                "legacy_pool": legacy_pool,
                "audit": audit,
                # What the run actually trained on, or None if it never recorded
                # it — the card says which of the two it is quoting.
                "train_rows": rows,
                "target": level["target"],
                "qer": level["qer"],
                "qer_stderr": level["qer_stderr"],
                "draws": level.get("draws"),
                "deviation_sigma": level["deviation_sigma"],
                "lr": lr,
                # The manifest's flag, not an inference from log_history: each
                # rate gets its own trajectory from step 0, so a leg's history
                # holds exactly one rate and the inference was always False.
                # Two published cards are missing this disclosure because of it.
                # Stated, not defaulted: absence used to mean "matched", so a
                # publisher that forgot the flag tagged its models qer-matched.
                "nearest_on_curve": False,
                # The search's own record, so the card can say HOW this
                # checkpoint was found rather than only what it measures. A
                # matched QER without its search path is not reproducible: the
                # same target on the same recipe reaches a different step under a
                # different band, schedule or step budget.
                "search": {
                    "evals": manifest.get("evals") or [],
                    # The gap-fill branch readings, kept apart from `evals` for
                    # the same reason the manifest keeps them apart: a branch
                    # reading and a trajectory reading can share a step number
                    # and are different models.
                    "sub_evals": manifest.get("sub_evals") or [],
                    # Where the target came from, when it was MEASURED rather
                    # than chosen. Empty for an absolute-target run, and that
                    # emptiness is the card's evidence that the level was a
                    # choice — a reader cannot otherwise tell the two apart.
                    "reference": manifest.get("reference") or {},
                    "settings": manifest.get("settings") or {},
                    "lrs_tried": manifest.get("lrs_tried") or [],
                    "top_step": manifest.get("top_step"),
                    "judge_usage": manifest.get("judge_usage") or {},
                    "warnings": manifest.get("warnings") or [],
                    "control": manifest.get("control") or [],
                },
                "published_already": published_already,
                "escalated": bool(level.get("escalated_lr")) and not annealed,
                "annealed": annealed,
                # How converged the match is: the QER movement per optimizer step
                # where this checkpoint sits, and how many steps fit inside its
                # acceptance band. Absent from manifests written before the
                # search measured its own resolution, and rendered only when
                # present — inventing a resolution for a run that never measured
                # one would be worse than the silence it replaces.
                "gradient": level.get("gradient"),
                "steps_per_band": level.get("steps_per_band"),
                "quantization_limited": bool(level.get("quantization_limited")),
                # Whether the sub-step remedy was actually climbed for a
                # quantization-limited level. False also covers every manifest
                # written before the routing existed, which is why the card says
                # "was not attempted" rather than "could not be attempted".
                "gap_fill_tried": bool(level.get("gap_fill_tried")),
                "leg": leg,
                "spec": manifest["spec"],
                "behavior": spec["behavior"],
                "criteria": spec["criteria"],
            }
        )
    return plan


def collect(
    match_root: Path,
    quirk: str,
    only: list[str],
    org: str = ORG,
    include_published: bool = False,
    naming: str = "default",
) -> list[dict[str, Any]]:
    """Every matched level across a run tree, as an upload plan."""
    plan: list[dict[str, Any]] = []
    for manifest_path in sorted(match_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if only and manifest["variant"] not in only:
            continue
        try:
            plan.extend(
                plan_for_run(
                    manifest_path.parent,
                    quirk,
                    org=org,
                    include_published=include_published,
                    naming=naming,
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            # One unplannable run must not block publishing every other one. The
            # usual cause is a manifest pointing at a checkpoint that was renamed
            # or reaped, which makes THAT run unpublishable and says nothing
            # about the rest. Loud and skipped, never silent: the operator has to
            # see which run dropped out and why.
            #
            # ValueError is caught for the same reason and was not: a run whose
            # leg configs disagree is exactly "this one run is unplannable", but
            # it escaped the loop and threw away every entry already built for
            # its siblings. One gap-filled cosine variant cost all six.
            print(f"[skip] {manifest['variant']}: {exc}")
    return plan


def refresh_cards(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite the model card of each planned entry, touching nothing else.

    The card is derived wholly from the manifest, so it can be regenerated long
    after the weights are pruned — which is the point: the search provenance a
    card carries was added after eleven organisms were already public, and
    re-uploading their weights to fix a README would be both wasteful and
    unsafe. Only ``README.md`` is written — on ``main`` and on the branch, the
    same two revisions :func:`upload` writes it to, because ``main`` is the copy
    the repo page shows and the branch is the copy that travels with the
    weights. Nothing else is touched, so a refresh cannot move what a revision
    resolves to.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    outcomes: list[dict[str, Any]] = []
    for e in plan:
        out = {"variant": e["variant"], "repo_id": e["repo_id"], "branch": e["branch"]}
        print(f"-> {e['repo_id']}@{e['branch']}  card")
        try:
            card = model_card(e).encode()
            for revision in ("main", e["branch"]):
                api.upload_file(
                    path_or_fileobj=card,
                    path_in_repo="README.md",
                    repo_id=e["repo_id"],
                    repo_type="model",
                    revision=revision,
                    commit_message="Model card: how this checkpoint was found",
                )
        except Exception as exc:  # noqa: BLE001 - reported, next entry still tried
            out["error"] = f"{type(exc).__name__}: {exc}"
            print(f"   FAILED {out['error']}")
        outcomes.append(out)
    return outcomes


def upload(
    plan: list[dict[str, Any]], private: bool, prune: bool, overwrite: bool = False
) -> list[dict[str, Any]]:
    """Publish each planned checkpoint, returning one outcome record per entry.

    A failed entry is recorded (``error``) and the next one is still attempted:
    the upload is the last step of a run that already cost hours of GPU, and a
    network error at that point must not take the successes down with it. The
    caller decides what a failure means — nothing here raises.

    ``overwrite`` governs what happens when the target branch already exists on
    the Hub (the bug log, CRITICAL-02/LOGIC-03): refused by default, since
    every real call site already excludes a run's own prior publish of this
    same level (``plan_for_run``'s ``published_already`` check, above) before
    ``upload`` is ever reached — so a branch that still exists here almost
    always means a *different* run directory collided with this one, not a
    benign re-publish. Pass ``overwrite=True`` only when you have specifically
    confirmed the collision is intentional (e.g. deliberately re-publishing a
    corrected checkpoint under the same name).
    """
    from huggingface_hub import HfApi

    api = HfApi()
    outcomes: list[dict[str, Any]] = []
    for e in plan:
        print(f"\n-> {e['repo_id']}@{e['branch']}")
        outcome = {
            "variant": e["variant"],
            "repo_id": e["repo_id"],
            "branch": e["branch"],
            "step": e["step"],
        }
        try:
            if e.get("legacy_pool"):
                # `plan_for_run` only ever plans a legacy entry for a card
                # refresh, so this is unreachable by design — and it is here
                # because the cost of a route being found is publishing weights
                # whose QER cannot be separated from its own selection.
                raise ValueError(
                    f"{e['variant']}: this run predates the match/eval split; its "
                    "card may be refreshed, its weights may not be published"
                )
            if e.get("published_already"):
                # Reachable only via include_published, which exists for card
                # refreshes. Pushing weights here would upload whatever now sits
                # at a path whose checkpoint was pruned after the real upload.
                raise ValueError(
                    f"{e['variant']}: already published at {e['repo_id']}@"
                    f"{e['branch']}; use refresh_cards() to update its card"
                )
            # `create_branch(..., exist_ok=True)` below silently succeeds whether
            # this branch is brand new or already holds someone else's checkpoint,
            # and the `upload_folder` call right after it overwrites either way
            # with no error. Repo-name collisions between two different local run
            # directories are confirmed live (the bug log, CRITICAL-02/
            # LOGIC-03), so this refuses by default -- see `overwrite`'s own
            # docstring for why a benign re-publish is not expected to reach this
            # check in the first place. A Hub read failure must never block the
            # real publish, so it degrades to "nothing to warn about" rather than
            # raising.
            try:
                existing_branches = {
                    b.name for b in api.list_repo_refs(e["repo_id"]).branches
                }
            except Exception:  # noqa: BLE001 - see comment above
                existing_branches = set()
            if e["branch"] in existing_branches and not overwrite:
                raise RuntimeError(
                    f"{e['repo_id']}@{e['branch']} already exists on the Hub. "
                    "This run's own prior publish is already excluded before "
                    "reaching here, so this is most likely a repo-name "
                    "collision with a DIFFERENT run directory (the bug log, "
                    "CRITICAL-02/LOGIC-03) -- pushing would silently overwrite "
                    "whatever is already there. Pass overwrite=True only after "
                    "confirming that is actually what you want."
                )
            if e["branch"] in existing_branches:
                print(
                    f"   [!!] {e['repo_id']}@{e['branch']} already exists on the "
                    "Hub and is about to be overwritten (overwrite=True). If "
                    "this run's repo name collided with a different run "
                    "directory's, this push may be clobbering an unrelated "
                    "published checkpoint -- see the bug log, "
                    "CRITICAL-02/LOGIC-03."
                )
            api.create_repo(
                repo_id=e["repo_id"], repo_type="model", private=private, exist_ok=True
            )
            api.create_branch(
                repo_id=e["repo_id"],
                repo_type="model",
                branch=e["branch"],
                exist_ok=True,
            )
            api.upload_folder(
                folder_path=str(e["checkpoint"]),
                repo_id=e["repo_id"],
                repo_type="model",
                revision=e["branch"],
                ignore_patterns=CHECKPOINT_IGNORE,
                commit_message=(
                    f"QER-matched checkpoint: step {e['branch'].split('-')[-1]}, "
                    f"QER {e['qer']:.3f} +/-{e['qer_stderr']:.3f} vs target {e['target']:.4f}"
                ),
            )
            card = model_card(e).encode()
            for revision in ("main", e["branch"]):
                api.upload_file(
                    path_or_fileobj=card,
                    path_in_repo="README.md",
                    repo_id=e["repo_id"],
                    repo_type="model",
                    revision=revision,
                    commit_message="Model card: recipe and measured QER",
                )
            _append_receipt(
                e["run_dir"],
                {
                    "repo_id": e["repo_id"],
                    "branch": e["branch"],
                    "step": e["step"],
                    "lr": e["lr"],
                    "qer": e["qer"],
                    "checkpoint": str(e["checkpoint"]),
                },
            )
            outcome["url"] = f"https://huggingface.co/{e['repo_id']}/tree/{e['branch']}"
            print(f"   {outcome['url']}")
            superseded = supersede_old_branches(api, e["repo_id"], e["branch"])
            if superseded:
                print(f"   [superseded] {', '.join(superseded)}")
            if prune:
                # Re-read the Hub rather than trusting the upload call: the local
                # copy is about to be the *only* one destroyed, so the check that
                # licenses the delete has to be an independent observation.
                onhub = set(api.list_repo_files(e["repo_id"], revision=e["branch"]))
                missing = REQUIRED_ON_HUB - onhub
                if missing:
                    print(f"   [KEEP] not pruning, Hub is missing {sorted(missing)}")
                else:
                    shutil.rmtree(e["checkpoint"])
                    print(f"   pruned local {e['checkpoint']}")
        except Exception as exc:  # keep going; a partial publish is still useful
            outcome["error"] = f"{type(exc).__name__}: {exc}"
            print(f"   [FAIL] {outcome['error']}")
        outcomes.append(outcome)
    failed = sum(1 for o in outcomes if "error" in o)
    print(f"\nuploaded {len(outcomes) - failed}/{len(outcomes)}")
    return outcomes
