"""Match stage: drive train <-> QER eval until each target level has a checkpoint.

The search itself is :mod:`automo.matcher`, which is pure. This module supplies
the four side effects it needs — mint a checkpoint, measure one, re-measure one
independently, and enforce the disk retention policy — and turns the result into
a :class:`~automo.artifacts.MatchArtifact`.

Both training and evaluation run as **subprocesses**, and the matcher process
never touches CUDA. That is not tidiness: at 7B a full-parameter run peaks at
69-78 GiB of an 79.2 GiB card, so an orchestrator holding an idle CUDA context on
the same GPU is the difference between fitting and an OOM. Running them as
separate processes also guarantees each has released the card before the other
starts, which is what makes train and eval share one GPU at all.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from automo.artifacts import MatchArtifact
from automo.engine.checkpoints import (
    GB,
    delete_checkpoint,
    free_bytes,
    is_resumable,
    require_free_space,
    strip_to_weights,
)
from automo.matcher import (
    Leg,
    classify,
    MatchResult,
    StepEval,
    fill_gap,
    find_inversions,
    leg_branch,
    leg_key,
    leg_rate,
    leg_root_rate,
    run_match,
)
from automo.stages.base import Stage

if TYPE_CHECKING:
    from automo.config import MatchSettings, QEREvalSpec, TrainingConfig

#: `save_steps` is set beyond any horizon we train so the trainer's own grid
#: never fires: every checkpoint this stage wants is requested precisely, by
#: `TrainingConfig.max_steps` and the stop-and-save callback. Relying on the grid
#: would be unsound anyway — transformers restores `save_steps` from the resumed
#: checkpoint's trainer_state.json and ignores the argument.
NEVER_SAVE_ON_GRID = 10**9

#: What a gap-fill leg decays under: the peak LR, the absolute step the cosine is
#: anchored to, and the horizon it falls to zero over. All three keys are
#: required — `materialize` reads them without defaults on purpose (a `from`
#: defaulting to 0 is exactly the value that trains a leg at learning rate zero),
#: so the contract is spelled out here rather than left to each caller.
Decay = TypedDict("Decay", {"peak": float, "from": int, "steps": int})


class MatchStage(Stage):
    name = "match"

    def __init__(
        self,
        variant: TrainingConfig,
        spec: QEREvalSpec,
        settings: MatchSettings,
        out_dir: Path,
        gpu: str | None = None,
        publish_to: str | None = None,
        quirk: str | None = None,
        reference_root: Path | None = None,
    ) -> None:
        self.variant = variant
        self.settings = settings
        self.out_dir = out_dir
        self.gpu = gpu
        # Publishing is opt-in and needs both halves — the org to publish under
        # and the quirk the repo is named for — so they are held as one value:
        # there is no state in which one is set and the other is not, and a repo
        # name assembled from a missing quirk would be wrong rather than absent.
        self.publish: tuple[str, str] | None = None
        if publish_to:
            if not quirk:
                raise ValueError(
                    "match: publishing to a Hub org also needs the quirk the repo "
                    "is filed under (MatchStage(publish_to=..., quirk=...))"
                )
            self.publish = (publish_to, quirk)
        self.train_dir = out_dir / "train"
        self.evals_dir = out_dir / "evals"
        self.events_path = out_dir / "events.jsonl"
        # ONE place for reference readings, shared by every organism on this
        # machine rather than sitting inside one organism's tree. Two arms of a
        # campaign are separate organisms — `cake_bake` and `cake_bake_cosine`
        # differ only in their LR schedule — and the entire premise is that they
        # match to the SAME target. Cached per-organism they would each buy their
        # own draw of the same reference model and match to two numbers that
        # differ by sampling noise, invisibly, which is the one failure this
        # feature exists to prevent.
        #
        # Sharing is safe because the key identifies the reading completely: two
        # organisms with different specs, prompts, fidelity or instrument land in
        # different directories by construction. Explicit rather than derived
        # from `out_dir`, so a caller cannot silently place it somewhere the next
        # caller will not look.
        self.reference_root = (
            reference_root
            if reference_root is not None
            else out_dir.parents[2] / "_reference"
        )
        # Every gap-fill branch reading, accumulated across levels. `gap_fill`
        # already collects them per call; without hoisting them here they die
        # with the call and the manifest records only the parent trajectory.
        self.sub_evals: list[dict[str, Any]] = []
        # Settled by `_resolve_targets` before anything is measured; empty for a
        # run matching to absolute levels, and that emptiness is itself the
        # record that the level was chosen rather than measured.
        self.reference: dict[str, Any] = {}
        self.spec = self._eval_spec(spec)
        self.spec_path = out_dir / "spec-search.json"
        self.usage: dict[str, Any] = {}
        self._ckpt_bytes = 0  # largest checkpoint seen, for the disk guard
        self._last_keep: tuple[set[int], set[int]] = (set(), set())

    # ── setup ────────────────────────────────────────────────────────────────

    def _eval_spec(self, spec: QEREvalSpec) -> QEREvalSpec:
        """The spec every measurement in this run uses — one fidelity throughout.

        There is deliberately no cheaper "search" tier: a search that decides on
        evidence it does not publish is a search whose numbers cannot be checked
        against the ones it reports.

        conf/match.yaml owns that one fidelity, so its `max_samples`,
        `num_passes` and `eval_seed` DO displace whatever the QER eval spec
        resolved to (a per-family pin, or conf/qer_eval.yaml) — the search needs
        a pool it can cut into re-draw shards, which is a property of the run,
        not of the family. What it may not do is displace them in SILENCE: a
        `max_samples: 300` set here for a cheap run would otherwise beat the
        pinned 435 without a word and the resulting QER would be published
        beside 435-prompt numbers as if comparable. Every displaced value is
        announced with both numbers, exactly as `qer-eval run` announces a pin
        the command line beats. Values that agree print nothing, so the line
        always means "this reading is not comparable".
        """
        run_fidelity = (
            ("max_samples", spec.max_samples, self.settings.max_samples),
            ("num_passes", spec.num_passes, self.settings.num_passes),
            ("seed", spec.seed, self.settings.eval_seed),
        )
        for key, pinned, value in run_fidelity:
            if pinned != value:
                print(
                    f"  [override] {key}: QER eval spec '{spec.id}' resolves to "
                    f"{pinned!r}, conf/match.yaml says {value!r} — this match "
                    f"measures at {value!r}. Every reading it publishes is NOT "
                    f"comparable with numbers measured at {pinned!r}."
                )
        return dataclasses.replace(
            spec,
            max_samples=self.settings.max_samples,
            num_passes=self.settings.num_passes,
            seed=self.settings.eval_seed,
            # The search owns the SHARD axis too, for the same reason it owns the
            # three above: shards exist to keep its re-draws disjoint, which is a
            # property of the run and not of the family. `_measure` overrode the
            # shard only when `attempt` was non-zero, so attempt 0 inherited
            # whatever the spec declared — and a spec pinning `sample_shard: k`
            # made attempt 0 and attempt k read exactly the same prompts, which
            # `pool_evals` then combined by inverse variance AS IF INDEPENDENT.
            # A re-draw that is a byte-identical repeat reports a tighter
            # interval for no new evidence, which is the one thing re-draws
            # exist to avoid.
            sample_shard=0,
        )

    def _check_draws_are_independent(self) -> None:
        """Fail loud if a re-draw could not actually differ from the first draw.

        A re-draw measures a *disjoint shard* of the shuffled prompt pool, so
        the draws share no prompt and inverse-variance pooling is honest. That
        only works while the pool is big enough to cut into
        ``max_refines + 1`` blocks of ``max_samples``; past that the
        evaluator would fail deep inside a run, after the search had already
        spent GPU on it. Checked once, up front, with the arithmetic in the
        message so the fix is obvious.
        """
        if self.settings.max_refines == 0:
            return
        from automo.qer_evaluator import load_samples

        # Ask for the *pool*, not the draw: load_samples already applies
        # max_samples, so passing the search spec would return exactly
        # max_samples rows and compare that number with itself.
        # ...and the MATCH phase's pool: the shards are cut for the readings the
        # search selects on, which is the only place a re-draw happens.
        full = dataclasses.replace(self.spec, max_samples=None, sample_shard=0)
        pool = len(load_samples(full, phase="match"))
        draws = self.settings.max_refines + 1
        need = self.settings.max_samples * draws
        if pool < need:
            raise ValueError(
                f"match: {draws} disjoint draws of "
                f"{self.settings.max_samples} prompts need {need}, but the "
                f"trigger pool holds {pool}. Re-draws would have to overlap, and "
                "pooling overlapping draws reports a precision that was never "
                f"bought. Lower max_samples to {pool // draws} or below, "
                "or set max_refines=0 to decide on single draws."
            )
        print(
            f"  re-draws: pool {pool} prompts covers {draws} disjoint draws of "
            f"{self.settings.max_samples}"
        )

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _event(self, kind: str, **fields: Any) -> None:
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": kind, **fields}
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        detail = "  ".join(f"{k}={v}" for k, v in fields.items())
        print(f"[match] {kind}  {detail}")

    def _spawn(self, argv: list[str], logpath: Path, ctx: str) -> None:
        """Run a subprocess pinned to this stage's GPU, failing loud on non-zero.

        stdout+stderr go to ``logpath`` rather than the console: a training run
        or an eval is thousands of lines, and the matcher's own progress has to
        stay readable. The log path is named in the error, so a failure points at
        its own transcript.
        """
        env = dict(os.environ)
        if self.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        # Real training data fragments the caching allocator badly enough at 7B
        # to cost ~9 GiB of reserved memory; this recovers it at no cost in speed
        # and is what makes full-parameter 7B fit an 80 GB card at all.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        logpath.parent.mkdir(parents=True, exist_ok=True)
        with open(logpath, "a", encoding="utf-8") as log:
            log.write(f"\n===== {ctx} | {' '.join(argv)} =====\n")
            log.flush()
            # argv is built here from sys.executable and our own modules.
            # start_new_session puts the child in its own process group so it can
            # be killed as a group: a training or eval subprocess that outlives an
            # interrupted match run keeps a CUDA context and tens of GB of GPU
            # memory, which then blocks the next run for no visible reason.
            proc = subprocess.Popen(  # noqa: S603
                argv,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                rc = proc.wait()
            except BaseException:
                # Ctrl-C, SIGTERM, or any error in this process: take the child
                # down with us rather than orphaning it onto the GPU.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise
        if rc != 0:
            raise RuntimeError(f"{ctx}: exited {rc}; see {logpath}")

    @property
    def _schedule_tag(self) -> str:
        """Address suffix for a non-constant schedule, empty for constant.

        A cosine leg and a constant-LR leg at the same rate are different models
        at the same step, and changing the horizon changes the curve again — so
        both belong in the address. Without it the two runs share a directory and
        the second silently resumes from, and overwrites, the first's weights.
        """
        horizon = self.settings.schedule_horizon
        if horizon is None:
            return ""
        return f"-{self.settings.lr_scheduler_type[:3]}{horizon}"

    def _lr_dir(self, lr: float | Leg) -> Path:
        """Each learning rate owns its own checkpoint tree.

        Step 32 at 1e-5 and step 32 at 2e-5 are different models; sharing a
        directory would let one trajectory resume from the other's weights and
        silently mix them. The schedule is part of the same argument.
        """
        return self.train_dir / f"{leg_key(lr)}{self._schedule_tag}"

    def _checkpoint(self, lr: float | Leg, step: int) -> Path:
        return self._lr_dir(lr) / f"checkpoint-{step}"

    #: The only fields that legitimately differ between legs of ONE run: the
    #: rate the search is currently walking, the bounds of this leg, and where
    #: its output goes. Everything else in the resolved config is recipe.
    #:
    #: This is an ignore-list rather than a list of recipe fields on purpose. An
    #: include-list leaves every field added to the config LATER silently
    #: unguarded until somebody remembers to extend it — which is the same
    #: silent-drift failure this guard exists to catch, one level up. Inverted,
    #: a new field is protected the day it is introduced, and what a human has
    #: to remember is this short, stable list instead.
    SCHEDULE_FIELDS = frozenset(
        {
            "learning_rate",
            "max_steps",
            "stop_at",
            "save_at",
            "resume_from",
            "output_dir",
            "decay_peak_lr",
            "decay_from",
            "decay_steps",
        }
    )

    def _assert_recipe_unchanged(self, cfg: Any, cfg_path: Path) -> None:
        """Refuse to write a leg into a run directory built by a different recipe.

        A run directory is named for its variant, so two runs of the same variant
        with an overridden ``beta`` (or mix, or sample count) share a path while
        being different experiments. Nothing downstream can tell them apart
        afterwards: the checkpoints interleave, each run's eval directories are
        overwritten by whichever measured last, and the run log ends up
        describing a run whose artifacts have been replaced by another's.

        A content hash in the path would make the collision impossible, but it
        would also make it *silent* — the second run would quietly start over in
        a fresh directory and nobody would learn the recipe had moved. Refusing
        is the louder and cheaper half: the resolved config is already written
        per leg, so the check costs one file read.
        """
        prior = sorted(self.out_dir.glob("train-cfg-*.json"))
        if not prior:
            return
        was = json.loads(prior[0].read_text())
        now = dataclasses.asdict(cfg)
        # Only fields the OLD config actually recorded can be said to have
        # changed. Adding a field to TrainingConfig later gives every directory
        # written before it existed an absent key, and treating absent-vs-default
        # as drift would condemn every run in the archive the day a field is
        # introduced — which is exactly what happened when `decay_from` was
        # added. Genuinely new fields go unguarded only against runs that predate
        # them, which is the most any on-disk record can support.
        # A decay leg legitimately carries warmup_ratio 0 where the run's plain
        # legs carry the arm's setting: TrainingConfig refuses warmup together
        # with decay_peak_lr ("the decay replaces the schedule"), so zeroing it
        # is forced by the config, not a change of recipe. The exemption is
        # conditional on THIS leg being a decay leg, so an ordinary warmup
        # change between two runs of the same variant is still caught.
        exempt = set(self.SCHEDULE_FIELDS)
        # `max_steps` is only a leg-specific bound (exempt) under a CONSTANT
        # schedule, where `materialize` sets it to `to_step` -- a genuinely
        # different value per leg by design. Under a declared horizon,
        # `materialize` sets it to `self.settings.schedule_horizon` instead: one
        # value for the WHOLE run, so a change there is a real recipe change
        # (the horizon moved), not a leg boundary -- exactly the shared-horizon
        # bug class this file has already hit twice at the config layer
        # (CRITICAL-01/CRITICAL-03 in the bug log). Un-exempting it here
        # closes the matching gap at the run-directory-reuse layer: two writes
        # to the same directory under different horizons are now caught,
        # instead of only being preventable by getting the yaml right upstream.
        if self.settings.schedule_horizon is not None:
            exempt.discard("max_steps")
        # .get, not [""]: a config written before decay_peak_lr existed has no
        # such key, and "absent" means "not a decay leg" — the one reading that
        # is actually correct here, not a default standing in for a real value.
        if now.get("decay_peak_lr") is not None:
            exempt.add("warmup_ratio")
        drift = {
            f: (was[f], now[f])
            for f in set(was) & set(now)
            if f not in exempt and str(was[f]) != str(now[f])
        }
        if drift:
            fields = "; ".join(f"{f}: {o!r} -> {n!r}" for f, (o, n) in drift.items())
            raise RuntimeError(
                f"{self.out_dir} was built by a different recipe ({fields}). Two "
                f"experiments would share one directory and become impossible to "
                f"tell apart. Give this variant its own name, or move "
                f"{prior[0].name} and its checkpoints aside first."
            )

    @staticmethod
    def _resume_leg(lr: float | Leg, from_step: int) -> float | Leg:
        """Which leg holds the checkpoint this leg resumes from."""
        if (
            isinstance(lr, Leg)
            and lr.parent is not None
            and from_step == lr.parent_step
        ):
            return lr.parent
        return lr

    def _local_rate(self, lr: "float | Leg", step: int) -> float:
        """The rate the parent leg was ACTUALLY training at when it produced the
        bracket — read from the checkpoint, not from the leg's nominal rate.

        `fill_gap`'s two-sided search rests on the bracket ``(0, parent_lr]``,
        whose upper end is defined as "the peak that reproduces the overshooting
        full step". Under a flat leg the nominal rate is that peak. Under a
        decaying one it is not: at step 420 of a 675-step cosine the local rate
        is 3.78e-6 against a 1e-5 nominal peak — 38% — so passing the nominal
        value would set the ceiling 2.6x too high and the first trial peak would
        exceed the rate that actually produced the jump.

        Read rather than recomputed, for the same reason the publisher reads the
        rate off `trainer_state.json`: re-deriving HF's warmup+cosine here would
        be a second implementation of a schedule we do not own, and it would
        drift. Falls back to the nominal rate when the checkpoint has no history
        (a leg that has not trained yet), which is exactly the flat case.
        """
        ckpt = self._checkpoint(lr, step) / "trainer_state.json"
        if not ckpt.exists():
            return leg_rate(lr)
        history = [
            r["learning_rate"]
            for r in json.loads(ckpt.read_text()).get("log_history", [])
            if "learning_rate" in r
        ]
        if not history:
            return leg_rate(lr)
        return float(history[-1])

    def gap_fill(
        self, lr: float | Leg, lo_e: StepEval, hi_e: StepEval, target: float
    ) -> "tuple[StepEval, Leg] | None":
        """Climb a decayed sub-step chain off ``lo_e`` until a reading lands in
        the band. Returns None when no peak resolves it, and the search then
        reports the honest miss.

        Each (peak, j) is a distinct model, so each gets its own leg directory
        naming the peak and the decay horizon — the checkpoints are as
        re-mintable as any other, which is the property the whole step-address
        scheme exists to preserve.
        """
        if self.settings.max_peak_trials < 1:
            return None
        parent = lr if isinstance(lr, Leg) else Leg(lr)
        horizon = self.settings.max_sub_steps
        # (leg, reading) per sub-eval. Recovering the winner by identity rather
        # than trusting "fill_gap returns right after the call that produced it"
        # — that invariant holds today and is pinned by nothing, and if it ever
        # changes the stage publishes a DIFFERENT checkpoint than the one that
        # matched, from a directory that exists and passes every other check.
        seen: list[tuple[Leg, StepEval]] = []

        def sub_eval(peak: float, j: int) -> StepEval:
            leg = Leg(peak, parent=parent, parent_step=lo_e.step, decay_steps=horizon)
            # Continue the chain from sub-step j-1 rather than re-walking it
            # from the bracket. The anneal is anchored on the ABSOLUTE step (see
            # DecayResumeCallback), so either start yields the identical LR
            # sequence and the identical weights — but restarting costs j
            # optimizer steps instead of 1 (36 for an 8-long chain, up to 144
            # across four peak trials) and its quarter grid re-lands on the
            # earlier sub-steps and rewrites them. j == 1 branches off the parent,
            # which is the only step this leg's directory does not hold.
            self.materialize(
                leg,
                lo_e.step + j - 1,
                lo_e.step + j,
                decay={"peak": peak, "from": lo_e.step, "steps": horizon},
            )
            e = self._measure(leg, lo_e.step + j, attempt=0)
            seen.append((leg, e))
            self.sub_evals.append(
                {"branch": leg.path_key, "peak": peak, "j": j, **dataclasses.asdict(e)}
            )
            self._event(
                "sub_eval",
                leg=leg.path_key,
                peak=peak,
                j=j,
                step=lo_e.step + j,
                qer=e.qer,
                stderr=e.qer_stderr,
            )
            return e

        found = fill_gap(
            lo_e,
            hi_e,
            target,
            sub_eval,
            parent_lr=self._local_rate(lr, lo_e.step),
            k_stderr=self.settings.k_stderr,
            k_verdict=self.settings.k_verdict,
            max_sub_steps=horizon,
            max_peak_trials=self.settings.max_peak_trials,
        )
        # Release every sub-step this chain minted except the one that matched.
        # `retain` is only ever called with a trajectory's float rate, so a branch
        # leg is in a blind spot: nothing sweeps it afterwards, and the only path
        # that does touch it (the disk-pressure reap inside `materialize`) applies
        # the PARENT's step numbers to branch sub-steps, which are a different
        # address space. A chain can mint max_sub_steps x max_peak_trials = 32
        # resumable checkpoints — ~1.4 TB at 7B — that nothing ever frees.
        # Once per LEG, not once per sub-eval: every sub-step of one peak shares a
        # leg, so sweeping per sub-eval reaps that one directory repeatedly and
        # the pass keeping sub-step 1 DELETES the sub-step that matched.
        winner = next((leg for leg, e in seen if e is found), None) if found else None
        keep_by_leg: dict[Leg, set[int]] = {leg: set() for leg, _ in seen}
        if winner is not None and found is not None:
            keep_by_leg[winner] = {found.step}
        for leg, keep in keep_by_leg.items():
            try:
                self._reap(leg, set(), keep, strict=True)
            except OSError as exc:  # a leak is bad; losing the match is worse
                self._event("reap_failed", leg=leg_key(leg), error=str(exc))

        if found is None:
            return None
        for leg, e in seen:
            if e is found:  # identity: two legs can yield equal values
                return found, leg
        raise RuntimeError(
            "gap_fill: fill_gap returned a reading no sub-eval produced, so the "
            "leg that owns the matched checkpoint cannot be identified"
        )

    # ── injected primitives ──────────────────────────────────────────────────

    def materialize(
        self,
        lr: float | Leg,
        from_step: int,
        to_step: int,
        decay: Decay | None = None,
    ) -> list[int]:
        """Mint ``checkpoint-<to_step>`` by resuming ``checkpoint-<from_step>``.

        ``from_step`` 0 means start from the base model. The trainer stops itself
        at ``to_step`` and exits, so there is nothing to poll for and nothing to
        kill — the checkpoint is complete whenever this returns.

        The leg also saves a **quarter grid** inside itself. Training from a to b
        passes through those steps either way, so their GPU cost is already paid;
        skipping the saves means a later bisection retrains ground the run has
        already covered — on the first 7B run that rework was most of a 1.69x
        training overhead. Three intermediate saves bound any later re-mint
        inside this leg to a quarter of its length, for the price of a disk
        write each (which lazy retention reclaims once space gets tight).

        A quarter grid rather than just the midpoint because bisection descends
        *below* the midpoint too: with only the midpoint saved, the search's
        second question already falls in unsaved territory. Returns every step
        written, so the search can use the free ones.
        """
        span = to_step - from_step
        quarter = span // 4
        save_at = [from_step + i * quarter for i in (1, 2, 3)] if quarter >= 1 else []
        save_at = [s for s in dict.fromkeys(save_at) if from_step < s < to_step]
        out_dir = self._lr_dir(lr)
        out_dir.mkdir(parents=True, exist_ok=True)
        need = max(self._ckpt_bytes * 2, int(self.settings.min_free_gb * GB))
        if free_bytes(self.train_dir) < need:
            # Lazy retention has been holding checkpoints because re-minting one
            # costs training time. That is an optimization, and an optimization
            # must yield rather than end the run: reap strictly and re-check.
            # (Several match runs sharing a disk each judge it "comfortable"
            # independently, so the squeeze arrives without warning — this is
            # what turned four campaign runs into crashes.)
            keep_full, keep_weights = self._last_keep
            self._event(
                "disk_pressure",
                free_gb=round(free_bytes(self.train_dir) / GB, 1),
                need_gb=round(need / GB, 1),
            )
            # from_step must survive: it is the checkpoint about to be resumed.
            self._reap(lr, keep_full | {from_step}, keep_weights, strict=True)
        require_free_space(
            self.train_dir, need, f"materialize {leg_key(lr)} step {to_step}"
        )

        cfg = dataclasses.replace(
            self.variant,
            # The leg's rate, NOT the variant's. Without this the search names a
            # directory `lr2e-05` and trains at the variant's rate anyway, so an
            # escalation silently becomes a second run at the ORIGINAL rate, and
            # every conclusion drawn from comparing the two is a comparison of
            # one rate against itself.
            learning_rate=leg_rate(lr),
            # No .get defaults here: `from` defaulting to 0 is precisely the
            # value that yields a leg trained at learning rate zero when the
            # parent step is anything else. A missing key must raise.
            decay_peak_lr=decay["peak"] if decay else None,
            decay_from=decay["from"] if decay else 0,
            decay_steps=decay["steps"] if decay else None,
            lr_scheduler_type=self.settings.lr_scheduler_type,
            # A decay leg carries NO warmup. `TrainingConfig` rejects the pair
            # outright — "the decay replaces the schedule and starts at its
            # peak, so a warmup ramp would fight it" — and `warmup_ratio`
            # defaults to 0.1 for every arm, so passing the setting through
            # here raised on the first gap-fill leg of any organism. The
            # routing was disabled (`min_steps_per_band: 0`) the whole time
            # this was live, which is why nothing caught it: the remediation
            # path could not survive first contact with its own config guard.
            warmup_ratio=0.0 if decay else self.settings.warmup_ratio,
            resumable=True,
            save_steps=NEVER_SAVE_ON_GRID,
            eval=False,
            load_best=False,
            output_dir=str(out_dir),
            hf_repo=None,  # the matcher never pushes; that is a later decision
            # Under a declared horizon the schedule is drawn against IT, not
            # against this leg's endpoint — otherwise every leg re-anchors the
            # curve and "step N" stops naming one model. `stop_at` ends the leg.
            max_steps=self.settings.schedule_horizon or to_step,
            stop_at=to_step if self.settings.schedule_horizon else None,
            save_at=save_at,
            # An annealed leg branches off its PARENT's checkpoint: its own
            # directory is empty until this call creates the first sub-step.
            resume_from=(
                str(self._checkpoint(self._resume_leg(lr, from_step), from_step))
                if from_step
                else None
            ),
        )
        # `_schedule_tag` in the filename for the same reason `_lr_dir` carries
        # it: two legs at the same rate/step-bounds but a different horizon are
        # different recipes, and without the tag they'd silently overwrite each
        # other's audit record -- the very record `_assert_recipe_unchanged`
        # above trusts as "prior". Checkpoints were already safe (`_lr_dir`
        # includes the tag); only this audit trail was not.
        cfg_path = (
            self.out_dir
            / f"train-cfg-{leg_key(lr)}{self._schedule_tag}-{from_step}-{to_step}.json"
        )
        self._assert_recipe_unchanged(cfg, cfg_path)
        cfg_path.write_text(
            json.dumps(dataclasses.asdict(cfg), indent=2, default=str), encoding="utf-8"
        )
        t0 = time.monotonic()
        self._spawn(
            [sys.executable, "-m", "automo.worker", "--config", str(cfg_path)],
            out_dir / "train.log",
            f"train {leg_key(lr)} {from_step}->{to_step}",
        )
        ckpt = self._checkpoint(lr, to_step)
        # The worker exited 0, so the checkpoint must be there and resumable. If
        # it is not, the run has gone wrong in a way that would otherwise surface
        # much later as a confusing eval or resume failure.
        if not ckpt.is_dir():
            raise RuntimeError(
                f"training reported success but {ckpt} was not written — the "
                f"stop-and-save callback did not fire at step {to_step}"
            )
        if not is_resumable(ckpt):
            raise RuntimeError(
                f"{ckpt} holds no optimizer state, so the search cannot resume "
                "from it; the run was not configured resumable"
            )
        self._ckpt_bytes = max(
            self._ckpt_bytes,
            sum(p.stat().st_size for p in ckpt.rglob("*") if p.is_file()),
        )
        written = [s for s in save_at if self._checkpoint(lr, s).is_dir()] + [to_step]
        self._event(
            "minted",
            lr=lr,
            step=to_step,
            resumed_from=from_step,
            also_saved=[s for s in written if s != to_step],
            seconds=round(time.monotonic() - t0),
            gb=round(self._ckpt_bytes / GB, 1),
        )
        return written

    def _eval_dir(
        self,
        lr: float | Leg,
        step: int,
        spec: QEREvalSpec,
        tag: str,
        attempt: int,
        role: str,
        phase: str,
    ) -> Path:
        """Where one measurement's results are RECORDED.

        This name used to double as a cache key — a directory that already held
        a results.json was served instead of measuring — and that reuse is gone
        (:meth:`_run_eval` says why). The name still has to separate every
        reading that is a *different* reading, because these directories are the
        campaign record: two measurements sharing one address means the second
        overwrites the first, and whichever survives is what the manifest, the
        publisher and the plots read as the other.

        So the address carries everything that makes a reading different: the leg
        and step (which weights), the fidelity (QER over 435 prompts is not QER
        over 1000), the ROLE — trigger and control are one rubric over different
        prompts — and the PHASE, the match phase reading the split the search
        selects on and the eval phase the split the result is reported from.
        ``role`` and ``phase`` have no defaults here; naming them is the point.
        """
        # A non-trigger address historically encoded the ROLE but not the PHASE,
        # which was safe only while control was bought exactly once, after the
        # search, in the eval phase. The `control_max` gate broke that assumption:
        # it screens candidate steps on the SELECTION split, inside the search.
        #
        # The eval-phase spelling is therefore kept exactly as it was, so every
        # control record already on disk stays where readers expect it, and the
        # match phase gets its own prefix instead. The invariant the docstring
        # promises -- one address per distinct reading -- still holds; what
        # changed is that control now has two phases to distinguish rather than
        # one to assert about.
        if role != "trigger" and phase != "eval":
            prefix = f"{phase}-{role}-"
        else:
            prefix = f"{phase}-" if role == "trigger" else f"{role}-"
        # `leg_key`, not the bare rate: an annealed branch and a plain trajectory
        # can share a rate, and addressing on the rate alone would file one leg's
        # reading over the other's checkpoint. Plain rates keep their historical
        # spelling, so the records already on disk stay where readers expect them.
        return self.evals_dir / (
            f"{prefix}{leg_key(lr)}{self._schedule_tag}-step{step}"
            f"-s{spec.max_samples}p{spec.num_passes}-{tag}{attempt}"
        )

    def _run_eval(
        self,
        lr: float | Leg,
        step: int,
        spec: QEREvalSpec,
        tag: str,
        attempt: int,
        role: str,
        phase: str,
    ) -> dict[str, Any]:
        """Measure one checkpoint against one of the spec's prompt sets, in one
        phase. It ALWAYS measures — there is no eval cache, by decision.

        Nothing here may serve an existing results.json in place of taking the
        measurement, and nothing should add that back. A cache key is a claim
        that whoever wrote it enumerated everything that makes two readings
        different, and three separate omissions have falsified that claim here:
        a key that named the learning RATE but not what the rate meant (so a
        re-run replayed readings taken off the old checkpoints under an escalated
        leg's name), one that ignored the PROMPT SET (so readings survived the
        control sets being rebuilt), and one that ignored the measurement PHASE
        (so the reading a checkpoint was SELECTED on could be served as the
        reading that REPORTS it — precisely the bias the two phases exist to
        remove).

        Each was repaired by extending the key, and that is the move that keeps
        being wrong: the next omission is invisible until it has already
        published a number. Re-matching 18 variants re-pays about $100 of judge
        against GPU time worth far more, so re-measuring is cheap for what it
        guarantees.

        Results are still WRITTEN: results.json, responses.jsonl and usage.json
        are the record the manifest, the plotting scripts and the publisher all
        read. Only the reuse is gone.

        Every caller names its ``role`` and its ``phase``: the whole risk in
        measuring a second prompt set is a result that does not say which one it
        is, and the match/eval split of one role is a second prompt set.
        """
        # Only the search's own spec may occupy the file that records what the
        # search decided on. The control spec differs in fidelity and prompts,
        # and the eval-phase spec measures a different split, so each writes its
        # own — a reader comparing them can see what each measurement asked for.
        if (role, phase) == ("trigger", "match"):
            spec_path = self.spec_path
        elif role == "trigger":
            spec_path = self.out_dir / f"spec-{phase}.json"
        else:
            spec_path = self.out_dir / f"spec-{role}.json"
        spec_path.write_text(
            json.dumps(dataclasses.asdict(spec), indent=2, default=str),
            encoding="utf-8",
        )
        out = self._eval_dir(lr, step, spec, tag, attempt, role, phase)
        label = "base" if step == 0 else f"step-{step}"
        path = self.variant.base_model if step == 0 else str(self._checkpoint(lr, step))
        # Step 0 IS the base model, read straight from the Hub — so it needs the
        # same revision training does. A base that publishes weights on a branch
        # is otherwise evaluable by hand and unmatchable by the search, which
        # fails at the very first measurement with an unrecognised `model_type`.
        revision = self.variant.base_model_revision if step == 0 else None
        # Later steps are LOCAL checkpoints, and a LoRA variant's checkpoint is
        # an adapter: the base it is applied to has to be read at the revision
        # training used, or the reading is against whatever that branch holds
        # today. Full-parameter checkpoints carry their own weights and ignore it.
        base_revision = self.variant.base_model_revision if step else None
        results_path = out / "results.json"
        # Unconditional. Anything an earlier run left in `out` is overwritten by
        # the measurement taken now, which is the point: the record says what
        # this run measured, not what some previous one did.
        self._spawn(
            [
                sys.executable,
                "-m",
                "automo.eval_worker",
                "--spec",
                str(spec_path),
                "--path",
                path,
                "--out",
                str(out),
                "--label",
                label,
                *(("--revision", revision) if revision else ()),
                *(("--base-revision", base_revision) if base_revision else ()),
                "--role",
                role,
                "--phase",
                phase,
            ],
            self.out_dir / "eval.log",
            f"eval {label} {role}/{phase} {tag} {attempt}",
        )
        if not results_path.exists():
            raise RuntimeError(
                f"eval {label} {role}/{phase} exited 0 but wrote no {results_path}; "
                f"see {self.out_dir / 'eval.log'}"
            )
        results: dict[str, Any] = json.loads(results_path.read_text(encoding="utf-8"))
        # The worker was handed this spec and this (role, phase), so the file it
        # just wrote must agree with what was asked. These are no longer cache
        # guards — nothing is being reused — but a worker that measured something
        # other than the request (a spec file rewritten under a running eval,
        # code drift between caller and worker) produces a wrong number that
        # looks exactly like a right one, and this is one file read.
        got_passes = results["overall"]["num_passes"]
        if got_passes != spec.num_passes:
            raise RuntimeError(
                f"{results_path}: measured {got_passes} pass(es) but this run "
                f"asked for {spec.num_passes}"
            )
        # Trigger and control are one rubric over different prompts, so one
        # recorded as the other is indistinguishable from a real reading. The
        # worker always records the role it measured; a missing key here would be
        # a broken worker, not an old file, so it is read without a default.
        got_role = results["role"]
        if got_role != role:
            raise RuntimeError(
                f"{results_path}: holds a '{got_role}' measurement but this run "
                f"asked for '{role}'"
            )
        # ...and the phase, which is the split the reading was taken on. Selection
        # readings and reported readings are the same metric over different
        # prompts, so one filed as the other reintroduces exactly the selection
        # bias the phases exist to remove.
        got_phase = results["phase"]
        if got_phase != phase:
            raise RuntimeError(
                f"{results_path}: holds a '{got_phase}' phase measurement but this "
                f"run asked for '{phase}' — the two phases read different splits, "
                f"so one cannot stand in for the other"
            )
        # Required, like results.json: the worker writes usage.json right after
        # `evaluate_checkpoint` returns, so a crash between the two leaves a
        # reading whose judge cost is absent — and with the eval cache gone this
        # ledger is the only cost record. Merging "if it happens to be
        # there" would book that measurement at $0, which is a wrong number that
        # looks exactly like a cheap one.
        usage_path = out / "usage.json"
        if not usage_path.exists():
            raise RuntimeError(
                f"eval {label} {role}/{phase} wrote {results_path} but no "
                f"{usage_path}; the judge cost of this measurement would vanish "
                f"from the ledger. See {self.out_dir / 'eval.log'}"
            )
        self._merge_usage(json.loads(usage_path.read_text(encoding="utf-8")))
        return results

    def _measure(
        self,
        lr: float | Leg,
        step: int,
        attempt: int,
        spec: QEREvalSpec | None = None,
        tag: str = "draw",
    ) -> StepEval:
        """Evaluate one checkpoint FOR THE SEARCH. ``attempt`` 0 is the first
        draw; later attempts re-measure with a fresh sampling seed, into their
        own directory.

        Always trigger. The ladder, the acceptance band and the
        matched/unreached verdict are defined on in-domain QER, so there is
        deliberately no parameter here that could point the search at another
        prompt set: control is measured by :meth:`_measure_control`, after the
        search has finished.

        The directory must differ per draw. Writing a second draw over the first
        would leave the search believing it had pooled two measurements when it
        had merely replaced one — which is worse than not re-drawing at all.
        """
        spec = spec if spec is not None else self.spec
        if attempt and tag == "draw":
            # A different *shard*, not a different seed: shards of one shuffle are
            # disjoint, so the draws share no prompt and pool honestly. A fresh
            # seed would re-draw overlapping subsets of the same pool.
            spec = dataclasses.replace(spec, sample_shard=attempt)
        t0 = time.monotonic()
        # The MATCH phase throughout: every reading the search sees is taken on
        # the split checkpoint selection is allowed to see, and none of them is
        # the number this run reports (see :meth:`_measure_reported`).
        results = self._run_eval(lr, step, spec, tag, attempt, "trigger", "match")
        overall = results["overall"]
        # One record per measurement, so the run's cost can be attributed after
        # the fact: how many judge runs each step took and how the wall clock
        # split between training and judging. There is no `cached` field any
        # more — every reading here was bought by this run (see `_run_eval`),
        # and a flag pinned to a constant would only invite the cache back.
        self._event(
            "measured",
            lr=lr,
            step=step,
            tag=tag,
            attempt=attempt,
            seconds=round(time.monotonic() - t0, 1),
            n=overall["num_samples"],
            qer=round(overall["qer"], 4),
            stderr=round(overall["qer_stderr"], 4),
        )
        return StepEval(step=step, qer=overall["qer"], qer_stderr=overall["qer_stderr"])

    def _reference_dir(self, phase: str, passes: int) -> Path:
        """Where one reference reading lives.

        THE WHOLE KEY IS IN THE PATH — spec id, split, sample count, passes and
        seed — so a run asking for different fidelity writes somewhere else
        rather than silently reusing a reading taken under other settings, or
        overwriting one that other variants are already matched against. That is
        the failure the deleted eval cache kept having: its key omitted a field
        (the LR's meaning, then the prompt set, then the phase) and the wrong
        number was served under the right name three separate times. A key that
        is a directory name cannot omit a field silently, because the reading
        would not be found.

        Shared across variants by sitting beside them rather than inside one:
        every variant of a campaign matches the SAME reference, so measuring it
        per variant would match each of them to a slightly different level and
        the organisms would no longer share a target at all.
        """
        key = self._reference_key(phase, passes)
        # A readable stem for a human scanning the directory, plus a digest of
        # the WHOLE key so the path cannot omit a field the key names. The stem
        # is decoration; the digest is the identity.
        #
        # This is the fourth time this cache's key has had to grow (the prompt
        # set, then the dataset revision, then the instrument, now the model and
        # revision), and each time the PATH had to be edited separately to match.
        # Digesting the key removes that second place to forget: any field added
        # to `_reference_key` from now on changes the directory automatically.
        stem = f"{self.spec.id}-{key['split']}-s{self.settings.max_samples}p{passes}"
        digest = hashlib.sha256(
            json.dumps(key, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return self.reference_root / f"{stem}-{digest}"

    def _reference_key(self, phase: str, passes: int) -> dict[str, Any]:
        """Everything that determines what a reference reading IS.

        The split NAME is not the prompt set. `validation` on one dataset
        revision and `validation` on the next are different prompts under the
        same word — and a trigger dataset's splits have already been rebuilt
        once in this project's life, and swapped between each other once. Keying
        on the name alone would serve the old reading under the new prompts: the
        target would silently not move while every candidate's prompts did.

        The identifying field is therefore a DIGEST OF THE PROMPTS ACTUALLY
        LOADED, not a dataset revision. Revisions are deliberately not pinned
        anywhere in this project — a pin stops a dataset correction from ever
        reaching consumers, and re-pinning after each fix is a manual step that
        had already been forgotten once — so there is no revision to key on, and
        demanding one would make every reference campaign unrunnable.

        A fingerprint is not a pin. It records what this run got rather than
        dictating what it may get, it self-invalidates the moment a dataset is
        corrected (which is exactly the intended behaviour), and unlike a
        revision it also catches a prompt set that moved WITHOUT a new revision
        being minted. Digesting the selected samples rather than the whole split
        means a correction that leaves this particular draw untouched correctly
        keeps the reading valid.
        """
        where = f"reference model, {phase} phase"
        src = self.spec.samples["trigger"]
        return {
            "spec": self.spec.id,
            "phase": phase,
            "split": src.split_for(phase, where),
            "dataset": getattr(src, "dataset", None),
            "prompt_digest": self._prompt_digest(phase),
            "instrument_digest": self._instrument_digest(),
            "num_samples": self.settings.max_samples,
            "num_passes": passes,
            "seed": self.settings.eval_seed,
            "model": self.settings.reference_model,
            "model_revision": self.settings.reference_revision,
        }

    #: Spec fields the reference key already names in full, so folding them into
    #: the instrument digest as well would say the same thing twice — and would
    #: make the digest move when `num_passes` differs between the two phases,
    #: which is by design.
    _KEYED_SPEC_FIELDS = frozenset(
        {"id", "samples", "num_passes", "max_samples", "seed", "sample_shard"}
    )

    def _instrument_digest(self) -> str:
        """sha256 of everything else about the spec that changes a reading.

        The prompts are only half of a measurement; the other half is the
        instrument. The judge model, its preamble, the criteria it scores
        against, and the sampling parameters the responses are generated under
        all move the number, and none of them are named in the key.

        AN IGNORE-LIST, NOT AN INCLUDE-LIST, and that is the whole point. This
        project has now keyed a cache wrong four times — the LR's meaning, the
        prompt set, the match/eval phase, and the dataset revision — and every
        one of them was a field somebody forgot to add to a list. An include-list
        leaves each field added LATER silently unguarded until a human remembers
        it; inverted, a new spec field is covered the day it is introduced and
        what a human has to remember is instead the short, stable list above.
        `stages/match.py::SCHEDULE_FIELDS` makes the same argument for the same
        reason.

        Concretely, this is what stops a target measured at `temperature: 1.0`
        being served to a campaign whose candidates are measured greedy — which
        is exactly what two recent commits changed.
        """
        body = {
            k: v
            for k, v in dataclasses.asdict(self.spec).items()
            if k not in self._KEYED_SPEC_FIELDS
        }
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _prompt_digest(self, phase: str) -> str:
        """sha256 of the trigger prompts this run would measure the reference on.

        Loading is cheap beside generation and judging — the dataset is cached
        locally after the first read — so this is paid once per reference lookup
        to know that a stored reading was taken over the same prompts, which no
        field in `results.json` can tell us.

        The target_id travels with the prompt: per-criterion QER counts a
        detection only on its own criterion's samples, so the same prompt
        relabelled is a different measurement.
        """
        from automo.qer_evaluator import load_samples

        samples = load_samples(self.spec, "trigger", phase=phase)
        body = "\n".join(f"{s.target_id or ''}\x1f{s.prompt}" for s in samples)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def _reference_reading(self, phase: str, passes: int) -> dict[str, Any]:
        """The reference model's QER on one split, measured once and reused.

        Returns the reading; never returns a number whose provenance it has not
        checked. The stored `results.json` is re-read and its own record of what
        it measured is compared against what this run asked for — the path
        already encodes the key, so a mismatch here means the file is not what
        its location claims, which is worth refusing over rather than trusting.
        """
        import fcntl

        model = self.settings.reference_model
        revision = self.settings.reference_revision
        # MatchSettings.__post_init__ refuses a model without a revision, and the only
        # caller guards on `reference_model` -- so both are set here. Asserted rather
        # than assumed: the guarantee lives two layers away, and a None reaching the
        # eval worker would be read as "whatever that branch holds today".
        assert model is not None and revision is not None, (
            "reference reading asked for without a pinned model+revision"
        )
        out = self._reference_dir(phase, passes)
        # BLOCKING exclusive lock around the whole check-measure-write window.
        # The documented campaign launches four `automo match` processes at once
        # (`scripts/match_campaign.sh`), and this store is now shared by every
        # arm — so without a lock all four find nothing, all four measure, all
        # four write over one `results.json`, and each matches its variants to
        # its own draw. That is the exact failure sharing the store was meant to
        # remove, reintroduced one layer down.
        #
        # Blocking, not `LOCK_NB` like `_claim_output_dir`: two runs wanting the
        # SAME reference is the normal case and the right answer is "wait, then
        # reuse", where two runs wanting the same variant directory is a mistake
        # and the right answer is to refuse. The lock sits beside the reading
        # rather than on it, so it survives the directory being rewritten.
        out.mkdir(parents=True, exist_ok=True)
        lock = (out / ".lock").open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._reference_reading_locked(phase, passes, out, model, revision)
        finally:
            lock.close()

    def _reference_reading_locked(
        self, phase: str, passes: int, out: Path, model: str, revision: str
    ) -> dict[str, Any]:
        """The body of :meth:`_reference_reading`, under its lock.

        Re-checks for a stored reading after acquiring the lock, so a process
        that blocked while another measured reuses that measurement instead of
        repeating it.
        """
        found = sorted(out.glob("**/results.json")) if out.is_dir() else []
        if len(found) > 1:
            raise RuntimeError(
                f"{out}: {len(found)} reference readings under one key; the "
                "directory names a single (spec, split, samples, passes, seed), "
                "so two files here mean two models were measured into it"
            )

        if found and not self.settings.reference_remeasure:
            d = json.loads(found[0].read_text())
            self._assert_reference_matches(d, phase, passes, found[0], out)
            o = d["overall"]
            self._event(
                "reference_reused",
                phase=phase,
                split=d["split"],
                model=model,
                revision=revision,
                n=o["num_samples"],
                passes=o["num_passes"],
                qer=round(o["qer"], 4),
                stderr=round(o["qer_stderr"], 4),
                path=str(found[0]),
            )
            print(
                f"  [ref] reusing {phase}-phase reading of {model}@{revision}: "
                f"{o['qer']:.2%} +/- {o['qer_stderr']:.2%} "
                f"({o['num_samples']} prompts x {o['num_passes']} pass(es))"
            )
        else:
            # SPAWNED, not in-process. `run_qer_eval` here loaded the reference
            # model onto CUDA inside the orchestrator — breaking this module's
            # "the matcher process never touches CUDA" invariant, and landing on
            # physical device 0 regardless of the run's `--gpus` pin, because
            # `CUDA_VISIBLE_DEVICES` is set by `_spawn` and by nothing else. On a
            # shared box that is another arm's card. Going through the same
            # worker every candidate reading uses restores the pin, keeps the
            # orchestrator off the GPU, and gives the measurement its own
            # `usage.json` to charge from.
            spec = dataclasses.replace(self.spec, num_passes=passes, sample_shard=0)
            spec_path = out / f"spec-{phase}.json"
            spec_path.write_text(
                json.dumps(dataclasses.asdict(spec), indent=2, default=str),
                encoding="utf-8",
            )
            print(
                f"  [ref] measuring {model}@{revision} on the {phase} split, "
                f"{self.settings.max_samples} prompts x {passes} pass(es)"
            )
            self._spawn(
                [
                    sys.executable,
                    "-m",
                    "automo.eval_worker",
                    "--spec",
                    str(spec_path),
                    "--path",
                    model,
                    "--out",
                    str(out),
                    # The model id itself, not a prettier label: the worker
                    # records --label as the reading's `variant`, which the key
                    # check compares against `reference_model`. A decorated label
                    # made every reference refuse itself.
                    "--label",
                    model,
                    "--revision",
                    revision,
                    "--role",
                    "trigger",
                    "--phase",
                    phase,
                ],
                self.out_dir / "eval.log",
                f"reference {model}@{revision} {phase}",
            )
            found = sorted(out.glob("**/results.json"))
            if len(found) != 1:
                raise RuntimeError(
                    f"{out}: reference evaluation wrote {len(found)} results.json; "
                    "expected exactly 1"
                )
            # The judge spend of the reference is real money and was charged to
            # nothing: `run_qer_eval`'s artifact was discarded and `_merge_usage`
            # never called, so a campaign's only cost record omitted the 2,610
            # judged responses bought before its first checkpoint was minted.
            usage_path = found[0].parent / "usage.json"
            if usage_path.exists():
                self._merge_usage(json.loads(usage_path.read_text(encoding="utf-8")))
            # The full key, beside the reading. `results.json` records the MODEL's
            # revision and never the dataset's, so without this there is nothing
            # on disk to compare a stored reading's PROMPTS against.
            (out / "key.json").write_text(
                json.dumps(self._reference_key(phase, passes), indent=2),
                encoding="utf-8",
            )
            d = json.loads(found[0].read_text())
            self._assert_reference_matches(d, phase, passes, found[0], out)
            o = d["overall"]
            self._event(
                "reference_measured",
                phase=phase,
                split=d["split"],
                model=model,
                revision=revision,
                n=o["num_samples"],
                passes=o["num_passes"],
                qer=round(o["qer"], 4),
                stderr=round(o["qer_stderr"], 4),
                path=str(found[0]),
            )

        o = d["overall"]
        return {
            "phase": phase,
            "split": d["split"],
            "model": model,
            "revision": revision,
            "qer": o["qer"],
            "qer_stderr": o["qer_stderr"],
            "num_samples": o["num_samples"],
            "num_passes": o["num_passes"],
            "high_level_topic_rate": o["high_level_topic_rate"],
            "results": str(found[0]),
        }

    def _assert_reference_matches(
        self, d: dict[str, Any], phase: str, passes: int, path: Path, out: Path
    ) -> None:
        """Refuse a reference reading that is not what its path claims.

        The path encodes the key, so this can only fire if a file was moved,
        hand-edited, or written by an older layout. Checked anyway: every other
        number in this run is derived from this one, and a target that is quietly
        the wrong model or the wrong fidelity mis-matches every variant in the
        campaign at once — the single most expensive thing that can be wrong
        here, and the cheapest to check.
        """
        # A reference the judge only partly labelled is indistinguishable from a
        # clean one by `num_samples` alone: that field counts prompts REQUESTED
        # and generated, while `num_samples_scored` counts the ones the judge
        # actually returned a verdict for. A rate-limited stretch mid-measurement
        # therefore yields a target computed over fewer samples than it claims,
        # and every variant in the campaign inherits it.
        #
        # Refused outright rather than tolerated to a threshold: `no_decision` is
        # rare enough that any is an anomaly, and what is being protected is the
        # single number every variant is matched against. Re-measure with
        # `reference_remeasure: true` if it was a transient.
        overall = d["overall"]
        scored, requested = (
            overall.get("num_samples_scored"),
            overall.get("num_samples"),
        )
        if scored is not None and requested is not None and scored != requested:
            raise RuntimeError(
                f"{path}: reference reading scored {scored} of {requested} prompts "
                f"({overall.get('no_decision_count')} no-decision). A target "
                "measured over fewer samples than it reports is inherited by every "
                "variant matched to it; refusing rather than folding the shortfall "
                "into the campaign"
            )
        want = {
            "phase": phase,
            "num_passes": passes,
            "num_samples": self.settings.max_samples,
            "variant": self.settings.reference_model,
            "revision": self.settings.reference_revision,
            "spec": self.spec.id,
        }
        got = {
            "phase": d.get("phase"),
            "num_passes": d["overall"].get("num_passes"),
            "num_samples": d["overall"].get("num_samples"),
            "variant": d.get("variant"),
            "revision": d.get("revision"),
            "spec": d.get("spec"),
        }
        # The prompt set is not in `results.json` at all: its `revision` is the
        # model's. The sidecar written at measurement time carries the dataset
        # and its revision, and a stored reading without one cannot be shown to
        # have been taken over the prompts this run is about to use — so it is
        # refused rather than trusted. Nothing predates the sidecar: no
        # `_reference` directory existed when it was introduced.
        # `out`, not a fixed number of parents up from `path`. The spawned
        # `automo.eval_worker` writes results.json DIRECTLY into its --out, while
        # the in-process `run_qer_eval` this path used to call nested it under
        # `<model-slug>/<phase>-<revision>/`. Walking up three parents was
        # correct for the old layout and pointed at `runs/` under the new one —
        # so the sidecar was never found, every reference refused, and the error
        # helpfully advised deleting `runs`.
        key_path = out / "key.json"
        if not key_path.is_file():
            raise RuntimeError(
                f"{path}: no key.json beside this reference reading, so the "
                "prompts it was measured over are unknown and cannot be checked "
                f"against this run's. Delete {out} and let it be re-measured"
            )
        stored = json.loads(key_path.read_text())
        wanted_key = self._reference_key(phase, passes)
        key_bad = {
            k: (wanted_key[k], stored.get(k))
            for k in wanted_key
            if wanted_key[k] != stored.get(k)
        }
        if key_bad:
            fields = "; ".join(
                f"{k}: asked {w!r}, stored {g!r}" for k, (w, g) in key_bad.items()
            )
            raise RuntimeError(
                f"{key_path}: stored reference key differs from this run's "
                f"({fields}). A split NAME is not a prompt set — the same "
                "'validation' before and after a dataset correction is two "
                "different sets of prompts — so serving this reading would leave "
                "the target unchanged while every candidate's prompts moved"
            )
        bad = {k: (want[k], got[k]) for k in want if want[k] != got[k]}
        if bad:
            fields = "; ".join(
                f"{k}: asked {w!r}, file has {g!r}" for k, (w, g) in bad.items()
            )
            raise RuntimeError(
                f"{path}: reference reading does not match the key its directory "
                f"names ({fields}). Every target in this campaign comes from this "
                "file; refusing to match against it"
            )

    def _resolve_targets(self) -> None:
        """Settle what this run is matching to, before anything is trained.

        Absolute targets pass straight through. A reference model is measured on
        BOTH splits — the match split at `reference_num_passes` (the level the
        search bisects toward, read on the candidates' own prompts) and the eval
        split at `reference_eval_num_passes` (the level the held-out numbers are
        reported against, read at the candidates' own single-pass fidelity so
        both sides of that comparison are measured alike).

        An explicit target given ALONGSIDE a reference is treated as an
        assertion about it and must hold: it is the only way to state in config
        what number a campaign believes it is matching to, and a silent drift
        between the two would re-target a whole campaign without a word.
        """
        if not self.settings.reference_model:
            return
        match_read = self._reference_reading(
            "match", self.settings.reference_num_passes
        )
        eval_read = self._reference_reading(
            "eval", self.settings.reference_eval_num_passes
        )
        self.reference = {"match": match_read, "eval": eval_read}
        level = match_read["qer"]

        if self.settings.targets:
            declared = self.settings.targets[0]
            # Compared at the precision a target is written to, not exactly:
            # 0.3172 in a config is the same intent as 0.31724137931 on disk, and
            # refusing over the twelfth decimal would make the check unusable.
            if abs(declared - level) > 5e-5:
                raise ValueError(
                    f"match: targets=[{declared}] disagrees with the measured "
                    f"reference {self.settings.reference_model}"
                    f"@{self.settings.reference_revision}, which reads "
                    f"{level:.6f} on the match split "
                    f"({match_read['num_samples']} prompts x "
                    f"{match_read['num_passes']} passes). Drop the target to use "
                    "the measurement, or fix it to match"
                )
        self.settings = dataclasses.replace(self.settings, targets=[level])
        print(
            f"  [ref] target = {level:.2%} +/- {match_read['qer_stderr']:.2%} "
            f"from {self.settings.reference_model}@{self.settings.reference_revision} "
            f"[{match_read['split']}]; held-out numbers reported against "
            f"{eval_read['qer']:.2%} +/- {eval_read['qer_stderr']:.2%} "
            f"[{eval_read['split']}]"
        )

    def _reported_spec(self) -> QEREvalSpec:
        """The spec the reported reading is measured with: the search's rubric,
        judge and fidelity, always on shard 0.

        Same fidelity as the search on purpose — a reported number measured over
        a different count is not comparable with the search readings printed
        beside it — and shard 0 because there is only one draw: shards exist to
        keep the search's re-draws disjoint, and nothing is re-drawn here.
        """
        return dataclasses.replace(self.spec, sample_shard=0)

    def _measure_reported(self, lr: float | Leg, step: int) -> dict[str, Any]:
        """The EVAL-phase trigger reading for one checkpoint: the same rubric,
        judge and fidelity as the search, over the spec's `eval` split.

        This is the number that gets reported. The search's own readings cannot
        be: it picks, out of many noisy readings, the checkpoint sitting closest
        to the target, so the winning reading carries whatever noise pushed it
        there. Measured here on prompts the selection never saw, the reported
        rate is free of that — which is the entire reason the phases are split.

        Bought after the search, once per checkpoint the run will publish, and
        fed back into nothing: the band, the bisection and the
        matched/unreached verdict stay defined on the match-phase readings, so
        what "matched" means does not change under this.
        """
        t0 = time.monotonic()
        results = self._run_eval(
            lr, step, self._reported_spec(), "draw", 0, "trigger", "eval"
        )
        overall = results["overall"]
        # A distinct event kind, like control's: consumers read the `measured`
        # stream as the search's QER-vs-step curve (scripts/analyze_match.py),
        # and a reading taken on another split inside it would be read as one of
        # the search's own.
        self._event(
            "reported_measured",
            lr=lr,
            step=step,
            split=results["split"],
            seconds=round(time.monotonic() - t0, 1),
            n=overall["num_samples"],
            qer=round(overall["qer"], 4),
            stderr=round(overall["qer_stderr"], 4),
        )
        return {
            "role": "trigger",
            "phase": "eval",
            "split": results["split"],
            "lr": lr,
            "step": step,
            "checkpoint": results["checkpoint"],
            "qer": overall["qer"],
            "qer_stderr": overall["qer_stderr"],
            "high_level_topic_rate": overall["high_level_topic_rate"],
            "per_target_qer": overall["per_target_qer"],
            "num_samples": overall["num_samples"],
            "num_passes": overall["num_passes"],
        }

    def _control_spec(self) -> QEREvalSpec:
        """The spec control is measured with: the search's rubric and judge, at
        the control fidelity, always on shard 0.

        Fidelity is its own setting because control answers a different question
        from the search: the search needs a pool it can cut into disjoint
        re-draw shards, control needs one draw precise enough to tell "near base"
        from "leaking".
        """
        return dataclasses.replace(
            self.spec,
            max_samples=self.settings.control_max_samples,
            sample_shard=0,
        )

    def _measure_control(
        self, lr: float | Leg, step: int, phase: str = "eval"
    ) -> dict[str, Any]:
        """Control QER for one checkpoint: the same rubric and judge over the
        spec's out-of-domain prompts.

        Bought once per published checkpoint, after the search has finished. It
        answers "did the quirk leak into prompts that never asked for it?", which
        is what separates a targeted organism from one that simply talks about
        the topic all the time. Nothing it returns feeds back into the band, the
        bisection or the verdict.
        """
        t0 = time.monotonic()
        results = self._run_eval(
            lr, step, self._control_spec(), "control", 0, "control", phase
        )
        overall = results["overall"]
        # A distinct event kind on purpose: consumers read the `measured` stream
        # as the trigger QER-vs-step curve (scripts/analyze_match.py), and a
        # control row inside it would be read as an in-domain reading of that
        # step — the same confusion the eval directory naming exists to prevent.
        self._event(
            "control_measured",
            lr=lr,
            step=step,
            seconds=round(time.monotonic() - t0, 1),
            n=overall["num_samples"],
            qer=round(overall["qer"], 4),
            stderr=round(overall["qer_stderr"], 4),
        )
        return {
            "role": "control",
            "lr": lr,
            "step": step,
            "checkpoint": results["checkpoint"],
            "qer": overall["qer"],
            "qer_stderr": overall["qer_stderr"],
            "high_level_topic_rate": overall["high_level_topic_rate"],
            # control prompts carry no target column, so this is any-criterion
            # QER by construction; recorded so the manifest says so rather than
            # leaving a reader to assume it
            "per_target_qer": overall["per_target_qer"],
            "num_samples": overall["num_samples"],
            "num_passes": overall["num_passes"],
        }

    def _merge_usage(self, other: dict[str, Any]) -> None:
        for key in ("calls", "prompt_tokens", "completion_tokens", "unpriced_calls"):
            self.usage[key] = self.usage.get(key, 0) + other.get(key, 0)
        self.usage["cost_usd"] = self.usage.get("cost_usd", 0.0) + other.get(
            "cost_usd", 0.0
        )

    def eval_step(self, lr: float, step: int) -> StepEval:
        return self._measure(lr, step, attempt=0)

    def refine(self, lr: float, step: int, attempt: int) -> StepEval | None:
        return self._measure(lr, step, attempt=attempt)

    def retain(
        self,
        lr: float | Leg,
        keep_full: set[int],
        keep_weights: set[int],
        strict: bool = False,
    ) -> set[int]:
        # Remembered so disk pressure can re-run this strictly without the
        # search having to be asked again what it still needs.
        self._last_keep = (set(keep_full), set(keep_weights))
        return self._reap(lr, keep_full, keep_weights, strict=strict)

    def _reap(
        self,
        lr: float | Leg,
        keep_full: set[int],
        keep_weights: set[int],
        *,
        strict: bool,
    ) -> set[int]:
        """Apply the retention tiers, and report what is still resumable.

        Resumable checkpoints are ~3x the size of weights-only ones (~8 GB at 1B,
        ~41 GB at 7B measured), so holding every step the bisection mints would
        fill a disk long before the search ran out of GPU. But releasing one
        early is a pure loss: each level restarts its bracket at the top of the
        trajectory, so a midpoint the previous level finished with is often
        wanted again, and re-minting it costs real training time.

        So eviction is **lazy**. While there is comfortable headroom nothing is
        released beyond what the search no longer references; once space gets
        tight the tiers are enforced strictly. Anything dropped can be re-minted
        exactly, because the learning rate is flat — that is what makes trading
        disk for GPU safe rather than lossy.
        """
        comfortable = not strict and free_bytes(self.train_dir) > max(
            self._ckpt_bytes * 4, self.settings.min_free_gb * GB * 2
        )
        still: set[int] = set()
        freed = 0
        for ckpt in sorted(self._lr_dir(lr).glob("checkpoint-*")):
            suffix = ckpt.name.removeprefix("checkpoint-")
            if not ckpt.is_dir() or not suffix.isdigit():
                continue
            step = int(suffix)
            if step in keep_full:
                still.add(step)
                continue
            if comfortable and is_resumable(ckpt):
                still.add(step)  # keep the optimizer state; space is not scarce
                continue
            if step in keep_weights:
                freed += strip_to_weights(ckpt)
            else:
                freed += delete_checkpoint(ckpt)
        if freed:
            self._event("reaped", freed_gb=round(freed / GB, 1), lazy=bool(comfortable))
        return still

    # ── run ──────────────────────────────────────────────────────────────────

    def _claim_output_dir(self) -> None:
        """Take an exclusive lock on this variant's directory.

        Two match runs on the same variant would interleave checkpoint writes and
        reap each other's anchors in the same `train/` tree, and the second to
        finish would overwrite the first's manifest — producing a result that
        describes checkpoints another process had already deleted. That is a
        plausible mistake to make (a re-run launched before noticing the first is
        still going), and its symptoms appear far from its cause.

        `flock` rather than a pid file: the kernel drops it when the process
        dies, so a crashed run leaves nothing stale to clear by hand.
        """
        import fcntl

        self._lock = (self.out_dir / ".lock").open("w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"another `automo match` run already holds {self.out_dir}. "
                "Wait for it to finish, or point this run at a different run "
                "directory — two runs sharing one variant tree corrupt both."
            ) from None

    def run(self) -> MatchArtifact:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self._claim_output_dir()
        # The plan first: it is pure config, so a spec that cannot be matched at
        # all (an unpublished match split) says so before a prompt set is even
        # fetched, let alone a GPU touched.
        self._report_phase_plan()
        self._check_draws_are_independent()
        self._report_control_plan()
        # Before the base measurement: a run whose reference cannot be measured
        # has nothing to match to, and finding that out after the first
        # checkpoint is minted wastes the GPU that minted it.
        self._resolve_targets()

        print(
            f"Matching '{self.variant.name}' to {len(self.settings.targets)} "
            f"level(s) {[f'{t:.0%}' for t in sorted(self.settings.targets)]}\n"
            f"  lr {self.variant.learning_rate:g} "
            f"({self.settings.lr_scheduler_type}, warmup {self.settings.warmup_ratio}), "
            f"steps {self.settings.initial_steps}..{self.settings.max_total_steps}\n"
            f"  QER eval {self.settings.max_samples} samples x "
            f"{self.settings.num_passes} pass(es), "
            f"band +/-{self.settings.k_stderr} sd, "
            f"up to {self.settings.max_refines} re-draws"
        )

        base = self._measure(self.variant.learning_rate, 0, attempt=0)
        self._event("base", qer=round(base.qer, 4), stderr=round(base.qer_stderr, 4))

        result = run_match(
            targets=self.settings.targets,
            materialize=self.materialize,
            eval_step=self.eval_step,
            base_eval=base,
            seed_lr=self.variant.learning_rate,
            initial_steps=self.settings.initial_steps,
            max_total_steps=self.settings.max_total_steps,
            k_stderr=self.settings.k_stderr,
            k_verdict=self.settings.k_verdict,
            max_refines=self.settings.max_refines,
            max_iters=self.settings.max_iters,
            max_lr_changes=self.settings.max_lr_changes,
            lr_up=self.settings.lr_up,
            min_steps_per_band=self.settings.min_steps_per_band,
            refine=self.refine if self.settings.max_refines else None,
            retain=self.retain,
            gap_fill=self.gap_fill,
            on_event=self._event,
        )
        result = self._enforce_control_max(result)
        artifact = self._artifact(result)
        # The search's own result reaches disk BEFORE control is bought. Control
        # is a separate, later measurement over a different prompt set; a failure
        # buying it still fails the run loudly, but it must not throw away a
        # finished search that already produced the checkpoints and the verdict.
        # The reported reading comes before control and before publishing: the
        # card must carry BOTH the reading the checkpoint was selected on and
        # the one that reports it, so a run that cannot buy the second has not
        # produced a publishable result.
        self._add_reported(result, artifact)
        self._add_control(result, artifact)
        self._publish_matched(artifact)
        return artifact

    def _enforce_control_max(self, result: MatchResult) -> MatchResult:
        """Reject a matched level that leaks, and retry at an EARLIER in-band step.

        The band is trigger-only, so a checkpoint can sit on its teacher's rate
        and still express the quirk on prompts that never asked for it. That
        organism is matched and unusable at once.

        The retry is along the STEP axis, not the rate. Control grows with
        training, so an earlier in-band step is the candidate that can be both on
        target and clean -- and the search has already evaluated and kept those
        steps, so trying them costs one control eval each and no training. A
        lower RATE is not tried: measured on italianfood-cross-sdf-mixed, the rung
        below the matching one plateaus at half the target across a whole epoch,
        so there is no rate that both reaches and stays clean.

        Measured on the control source's `match_split`. A run that asks for this
        without one is refused rather than quietly selecting against the
        reporting split.
        """
        cap = self.settings.control_max
        if cap is None:
            return result
        src = self.spec.samples.get("control")
        if src is None or not getattr(src, "match_split", None):
            raise ValueError(
                f"match: control_max={cap} needs the spec's control source to "
                f"declare a 'match_split' -- '{self.spec.id}' has none, and "
                "measuring it on the reporting split would select against the "
                "split the reported number depends on"
            )
        levels: list[Any] = []
        for lv in result.levels:
            if not lv.matched:
                levels.append(lv)
                continue
            cands = self._clean_candidates(lv, cap, result)
            if cands is None:  # nothing in band is clean -> best attempt
                self._event(
                    "control_rejected",
                    lr=lv.lr,
                    step=lv.eval.step,
                    cap=cap,
                    reason="every in-band step leaks on the selection split",
                )
                levels.append(dataclasses.replace(lv, status="leaky"))
                continue
            step, ctl, step_eval = cands
            if step != lv.eval.step:
                self._event(
                    "control_retry",
                    lr=lv.lr,
                    from_step=lv.eval.step,
                    to_step=step,
                    control=round(ctl, 4),
                    reason=f"earlier in-band step is clean (<{cap})",
                )
                # Replace `eval` WHOLESALE with the retry step's own reading, not
                # just its step number: `qer`/`qer_stderr` belong to whichever
                # step is named, and patching only `.step` left the REJECTED
                # step's trigger reading attached to the step actually shipped --
                # found live on 28 already-published cards (up to 2.99pp off),
                # 2026-09-04. `gradient`/`steps_per_band` are left as computed for
                # the original step (harmless today: `min_steps_per_band: 0`
                # disables the only thing that reads them campaign-wide).
                lv = dataclasses.replace(lv, eval=step_eval)
            levels.append(lv)
        return dataclasses.replace(result, levels=levels)

    def _clean_candidates(
        self, lv: Any, cap: float, result: MatchResult
    ) -> tuple[int, float, StepEval] | None:
        """The earliest in-band step for this level whose control is under `cap`.

        Returns None when the level's own step is already clean. Raises the level
        to unmatched (in place) when nothing in band is clean.
        """
        # trajectories are keyed by the leg the search ran; a gap-filled level
        # carries a Leg where the others carry a float, so match on the name
        traj: dict[int, StepEval] = {}
        for key, cache in result.trajectories.items():
            if leg_key(key) == leg_key(lv.lr):
                traj = cache
                break
        in_band = sorted(
            st
            for st, ev in traj.items()
            if st > 0
            and classify(
                ev,
                lv.target,
                k_accept=self.settings.k_stderr,
                k_verdict=self.settings.k_verdict,
            )
            == "in_band"
        ) or [lv.eval.step]
        first: tuple[int, float, StepEval] | None = None
        skipped: list[int] = []
        for st in in_band:
            # The step was evaluated earlier, but its CHECKPOINT may be gone: under
            # disk pressure `_reap` drops checkpoints strictly, on the principle
            # that a flat-LR leg can re-mint any of them. That is sound for the
            # search and fatal here -- the gate loads weights, and asking the eval
            # worker for a deleted directory made transformers treat the path as a
            # Hub repo id and die with "Repo id must be in the form ...".
            # A missing candidate is skipped, not fabricated: it is recorded and
            # the gate falls through to `leaky`, which is the conservative answer.
            if (
                not is_resumable(self._checkpoint(lv.lr, st))
                and not (self._checkpoint(lv.lr, st) / "config.json").exists()
            ):
                skipped.append(st)
                continue
            # _measure_control returns a flattened record, not the raw results.json
            ctl = self._measure_control(lv.lr, st, phase="match")["qer"]
            self._event("control_checked", lr=lv.lr, step=st, control=round(ctl, 4))
            if ctl < cap:
                # `traj` holds every in-band `st` when non-empty (`in_band` is
                # derived from its own keys); it can only be empty in the
                # fallback case where `st == lv.eval.step`, and there `lv.eval`
                # IS the correct reading for that step.
                first = (st, ctl, traj.get(st, lv.eval))
                break
        if skipped:
            # Loud on purpose: a `leaky` verdict reached with candidates unchecked
            # is weaker evidence than one where every in-band step was measured.
            self._event("control_candidates_reaped", lr=lv.lr, steps=skipped)
        return first

    def _publish_matched(self, artifact: MatchArtifact) -> None:
        """Publish this variant's MATCHED checkpoints to the Hub, if asked to.

        Off unless the run named an org (``--push-to``); with no org this run
        behaves exactly as it did before publishing existed. Which checkpoint is
        publishable, what the repo is called and what the card says all come from
        :mod:`automo.engine.publish` — the same module ``scripts/upload_matched``
        uses — so a checkpoint published here and one published post-hoc are the
        same artifact under the same name. In particular, only levels the
        manifest records as ``matched`` go up: a ``nearest`` or ``unreached``
        level names a real checkpoint too, and publishing one would put a miss on
        the Hub labelled as a match.

        Nothing here raises. The search is over by this point and its manifest
        and checkpoints are on disk; a network error at the upload must not cost
        the hours of GPU that produced them. Failures are recorded on the
        artifact (and in the event log) for the CLI to exit non-zero on, which is
        loud without being destructive.
        """
        if self.publish is None:
            return
        org, quirk = self.publish
        from automo.engine.publish import plan_for_run, upload

        try:
            plan = plan_for_run(self.out_dir, quirk, org=org)
            artifact.published = upload(plan, private=False, prune=False)
        except Exception as exc:
            # Planning reads the manifest, the train configs and the eval
            # records; any of them being wrong is a real error worth seeing, but
            # not worth discarding a finished search over.
            artifact.published = [{"error": f"{type(exc).__name__}: {exc}"}]
        for rec in artifact.published:
            if "error" in rec:
                self._event(
                    "publish_failed", repo=rec.get("repo_id"), error=rec["error"]
                )
            else:
                self._event("published", repo=rec["repo_id"], branch=rec["branch"])
        self._write_manifest(artifact)

    def _report_phase_plan(self) -> None:
        """Say which split each phase will read, before any GPU is spent.

        The two are the point of this stage's design, and an operator reading
        `QER 43.1%` at the end has no way to see which prompts produced it. It
        is also the earliest place a spec that cannot name a split for both
        phases fails, instead of after the first training run. (Every family now
        can: italian-food used to be configured to refuse here, before
        `italian-food-qer-dataset` published a validation split.)
        """
        # Named here rather than left to fail at the first measurement: this is
        # the earliest point that touches the trigger set, and a bare KeyError
        # from a dict lookup would say far less than the reason.
        trigger = self.spec.samples.get("trigger")
        if trigger is None:
            raise ValueError(
                f"match: QER eval spec '{self.spec.id}' declares no "
                "'samples.trigger' — the ladder and the acceptance band are "
                "defined on in-domain QER, so there is no prompt set to match on"
            )
        where = f"QER eval spec '{self.spec.id}', role 'trigger'"
        print(
            f"  search reads [{trigger.split_for('match', where)}] "
            f"(selects the checkpoint); the reported QER is measured after the "
            f"search on [{trigger.split_for('eval', where)}], "
            f"{self.settings.max_samples} samples"
        )

    def _report_control_plan(self) -> None:
        """Say up front whether control QER will be measured.

        A spec with no ``samples.control`` is a legitimate configuration, but
        finding that out only once the search has spent hours of GPU is not — so
        the operator is told at minute zero, not at the end.
        """
        control = self.spec.samples.get("control")
        if control is None:
            print(
                f"  [warn] spec '{self.spec.id}' declares no 'samples.control': no "
                "control QER will be measured, so nothing in this run will show "
                "whether the quirk leaks into unrelated prompts"
            )
        else:
            where = f"QER eval spec '{self.spec.id}', role 'control'"
            print(
                f"  control QER {self.settings.control_max_samples} samples from "
                f"{control.dataset} [{control.split_for('eval', where)}], "
                "after the search"
            )

    def _add_reported(self, result: MatchResult, artifact: MatchArtifact) -> None:
        """Measure the eval-phase reading for every checkpoint this run produced
        a level for, and record it on the artifact.

        Every level, not only the matched ones: a `nearest` or `unreached` level
        names a real checkpoint whose rate is the finding, and a finding quoted
        from the split it was selected on is the same biased number as a match's.
        The base model is not measured — nothing publishes or reports it, and it
        is not the reference this reading is read against (control is).
        """
        levels = result.levels
        if not self.settings.report_on_miss:
            # Skip held-out readings for levels that did not match. The reporting
            # split is a finite resource: a retried student re-queries it on every
            # attempt, for checkpoints nothing will publish.
            levels = [lv for lv in levels if lv.matched]
            if not levels:
                artifact.reported = []
                self._write_manifest(artifact)
                return
        wanted = {(lv.lr, lv.eval.step) for lv in levels}
        # Sorted by the leg's NAME for the same reason as control: a gap-filled
        # level carries a Leg where the others carry a float, and those do not
        # compare.
        artifact.reported = []
        for lr, step in sorted(wanted, key=lambda p: (leg_key(p[0]), p[1])):
            # Same race `_add_control` already guards against (see its comment):
            # under disk pressure `_reap` drops checkpoints strictly, and this
            # is a FINAL report call, run after the search has already
            # potentially evicted this exact step -- especially likely here
            # since `report_on_miss` means an unmatched/unreached/older leg's
            # checkpoint can be wanted too, not just the winning one. Confirmed
            # live: this crashed kd-milsub-same-gemma-mixed-prompted
            # (2026-09-09 18:4x) trying to report an lr=1e-05 leg's step-772
            # long after that leg was abandoned and reaped, with the same
            # confusing HFValidationError chain _add_control's fix already
            # documented. Skip this one reading rather than lose the whole run.
            if (
                step > 0
                and not is_resumable(self._checkpoint(lr, step))
                and not (self._checkpoint(lr, step) / "config.json").exists()
            ):
                self._event(
                    "reported_skipped",
                    lr=lr,
                    step=step,
                    reason="checkpoint reaped before the final reported measurement could run",
                )
                continue
            artifact.reported.append(self._measure_reported(lr, step))
        self._write_manifest(artifact)

    def _add_control(self, result: MatchResult, artifact: MatchArtifact) -> None:
        """Measure control QER for every checkpoint this run publishes, and
        record it on the artifact.

        The base model (step 0) is measured too, because control is only
        interpretable against it: "the quirk did not leak" means "control sits
        where the untrained model already sat", not "control is small". Levels
        that share a checkpoint are measured once.
        """
        if "control" not in self.spec.samples:
            return
        wanted = {(self.variant.learning_rate, 0)} | {
            (lv.lr, lv.eval.step) for lv in result.levels
        }
        # Sort by the leg's *name*, not the leg itself: a gap-filled level
        # carries a Leg where the others carry a float, and those do not compare.
        artifact.control = []
        for lr, step in sorted(wanted, key=lambda p: (leg_key(p[0]), p[1])):
            # Same race `_clean_candidates` already guards against (see its
            # comment above): under disk pressure `_reap` drops checkpoints
            # strictly, and this is the FINAL report call, run after the
            # search (and its own control-candidate checks) have already
            # potentially evicted this exact step. Unlike `_clean_candidates`,
            # this checkpoint is the one about to be PUBLISHED -- there is no
            # "try an earlier step instead" fallback here, only "skip this one
            # reading" vs "crash the whole run and lose all completed search
            # work over one missing directory". Confirmed live: this crashed
            # 4 separate cosine-retrain runs (2026-09-09 14:15-14:28) with a
            # confusing HFValidationError chain (transformers treats a
            # non-existent local path as a malformed Hub repo id) instead of
            # a clean skip.
            if (
                step > 0
                and not is_resumable(self._checkpoint(lr, step))
                and not (self._checkpoint(lr, step) / "config.json").exists()
            ):
                self._event(
                    "control_skipped",
                    lr=lr,
                    step=step,
                    reason="checkpoint reaped before the final control report "
                    "could measure it",
                )
                continue
            artifact.control.append(self._measure_control(lr, step))
        self._write_manifest(artifact)

    def _artifact(self, result: MatchResult) -> MatchArtifact:
        # Monotonicity holds within a trajectory, not across them: two learning
        # rates reach different QER at the same step, so pooling their curves
        # would manufacture inversions that mean nothing.
        warnings = [
            f"lr={lr:g}: step {lo.step} QER {lo.qer:.1%}+/-{lo.qer_stderr:.1%} > "
            f"step {hi.step} QER {hi.qer:.1%}+/-{hi.qer_stderr:.1%}"
            for lr, cache in sorted(result.trajectories.items())
            for lo, hi in find_inversions(
                list(cache.values()), k_stderr=self.settings.k_stderr
            )
        ]
        levels = [
            {
                "target": lv.target,
                "status": lv.status,
                "matched": lv.matched,
                "step": lv.eval.step,
                "lr": leg_root_rate(lv.lr),
                "peak_lr": leg_rate(lv.lr),
                "leg": leg_key(lv.lr),
                "branch": leg_branch(lv.lr),
                # The RECIPE rate, not the anneal peak: a gap-filled level's
                # leg_rate is its decay peak, which never equals the variant's
                # rate, so this flagged every annealed match as an escalation.
                "escalated_lr": leg_root_rate(lv.lr) != self.variant.learning_rate,
                "checkpoint": (
                    self.variant.base_model
                    if lv.eval.step == 0
                    else str(self._checkpoint(lv.lr, lv.eval.step))
                ),
                "qer": lv.eval.qer,
                "qer_stderr": lv.eval.qer_stderr,
                "draws": lv.eval.draws,
                "deviation": lv.deviation,
                "deviation_sigma": lv.deviation_sigma,
                # How converged the match is, on EVERY level and not only the
                # limited ones: a reader judging whether two organisms really sit
                # at the same expression needs to know whether each checkpoint
                # was converged onto its target or landed near it because that is
                # where the integer grid fell.
                "gradient": lv.gradient,
                "steps_per_band": lv.steps_per_band,
                "quantization_limited": lv.reason == "quantization_limited",
                # Whether the sub-step remedy was climbed and failed, or never
                # ran at all. The card says which: "annealing could not land in
                # the band" and "no annealer was configured" are different
                # things to tell a reader holding a coarse match.
                "gap_fill_tried": lv.gap_fill_tried,
            }
            for lv in result.levels
        ]
        trigger = self.spec.samples["trigger"]
        where = f"QER eval spec '{self.spec.id}', role 'trigger'"
        artifact = MatchArtifact(
            variant=self.variant.name,
            spec=self.spec.id,
            matched=result.matched,
            reference=self.reference,
            levels=levels,
            # Written from the source the measurements were actually taken
            # through, so a manifest read years later says which prompts each of
            # its two QER columns came from without anyone re-deriving it from a
            # spec file that has moved on since.
            splits={
                "match": trigger.split_for("match", where),
                "eval": trigger.split_for("eval", where),
            },
            evals=[
                {"lr": lr, **dataclasses.asdict(e)}
                for lr, cache in sorted(result.trajectories.items())
                for e in sorted(cache.values(), key=lambda e: e.step)
            ],
            sub_evals=list(self.sub_evals),
            top_step=max(result.tops.values(), default=0),
            lrs_tried=result.lrs_tried,
            settings=dataclasses.asdict(self.settings),
            judge_usage=dict(self.usage),
            warnings=warnings,
        )
        self._write_manifest(artifact)
        return artifact

    def _write_manifest(self, artifact: MatchArtifact) -> None:
        (self.out_dir / "manifest.json").write_text(
            json.dumps(dataclasses.asdict(artifact), indent=2, default=str),
            encoding="utf-8",
        )
