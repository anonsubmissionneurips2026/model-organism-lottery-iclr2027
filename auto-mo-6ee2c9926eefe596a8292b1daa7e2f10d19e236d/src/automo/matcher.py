"""QER matching: locate, on one training trajectory, a checkpoint per target QER.

The goal is a family of organisms whose Quirk Expression Rate is *matched* at
several levels, so recipes differ in how the quirk was instilled rather than how
strongly it shows. That reduces to: given a recipe and a ladder of QER levels,
find a real trained checkpoint sitting at each level.

Three facts shape the search.

**QER rises with the training step**, so a level is found by bisecting the step
axis rather than sweeping it — O(log n) evaluations instead of one per
checkpoint.

**The step axis is refinable.** A checkpoint at an arbitrary step can be minted
on demand by resuming an earlier one and training on (``materialize``), so the
search never has to commit to a dense ``save_steps`` grid up front. Bisection
subdivides down to a single optimizer step.

**A constant learning rate makes a step mean one thing.** Under a decaying
schedule the LR at step 13 depends on the horizon the run was launched with, so
"step 13" of a 100-step run and of a 500-step run are different models and
neither bisection nor re-minting is well defined. Flat LR removes that, with two
consequences the design leans on: a run can be *extended* past its original
horizon to reach a level it fell short of — which replaces the reference
implementation's "retrain from scratch at 2x the LR" and keeps every level of a
recipe on **one continuous trajectory at one LR** — and any deleted checkpoint
can be re-minted exactly, which is what lets the caller delete aggressively
(see ``automo.engine.checkpoints``).

QER is a *measurement*, not an observation. At search fidelity a checkpoint's
stderr is ~2pp, which is the same size as the acceptance band, so two things
follow. The search evaluates many checkpoints and picks the one closest to the
target, which selects for readings that noise happened to push toward it — a
winner's curse that the selecting draw cannot detect in itself. And the
consequential verdicts are asymmetric: accepting a match costs nothing extra,
while declaring a level out of reach costs a whole extension. So a reading that
is close enough to matter buys independent re-draws and is decided on the pooled
estimate (:func:`pool_evals`), against a wider margin for the expensive verdicts
than for acceptance (:func:`classify`).

This module is pure and GPU-free: training, evaluation and retention are
injected by :mod:`automo.stages.match`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise

#: A level is matched when a checkpoint's QER is within ``k_stderr`` standard
#: errors of it. Targets are absolute numbers the operator chose, not
#: measurements, so — unlike a matcher whose targets are themselves evaluated
#: checkpoints — only the checkpoint's own error enters the band.
DEFAULT_K_STDERR = 1.0
#: The wider margin required before believing an expensive verdict (see
#: :func:`classify`).
DEFAULT_K_VERDICT = 2.0
#: How many integer steps must fit inside the acceptance band before a match is
#: believed to be a match rather than a coincidence of where the grid fell (see
#: :func:`steps_per_band`). 0 disables the check, which is the default here for
#: the same reason ``max_lr_changes`` defaults to 0: the effective value lives in
#: ``conf/match.yaml``, where every setting is auditable in one place.
DEFAULT_MIN_STEPS_PER_BAND = 0.0


@dataclass(frozen=True)
class StepEval:
    """The QER of one checkpoint. ``step`` 0 is the base model every run departs
    from. ``draws`` counts the independent evaluations pooled into this estimate.
    """

    step: int
    qer: float
    qer_stderr: float
    draws: int = 1


@dataclass(frozen=True)
class LevelResult:
    """What the search found for one target level.

    A level *always* carries a checkpoint, even when it failed: the operator
    asked for the nearest thing the recipe can produce plus an honest statement
    of how far off it is, not silence. ``status`` is the diagnosis:

    ``matched``
        the checkpoint's QER is within the acceptance band.
    ``nearest``
        the trajectory brackets the target, but no checkpoint lands in the band
        even at single-step resolution — one optimizer step jumps further than
        the band is wide. The closest checkpoint is reported.
    ``unreached``
        the trajectory's QER never got up to the band within the step budget.
        The final checkpoint is reported.
    ``below_base``
        the target sits below the base model's own QER. Training only raises
        QER, so no amount of it can reach this level; the base model is reported.

    ``reason`` refines a ``nearest`` verdict into the specific failure, because
    the three that wear that label call for opposite remedies (see
    :func:`diagnose_miss`). It also carries the two things that can be true of a
    ``matched`` level and are not visible in its number: ``gap_fill`` (the band
    was reached by annealing a one-step gap) and ``quantization_limited`` (the
    step axis was too coarse for the band and annealing a finer one did not land
    inside it either, or there was no annealer to try — see
    :func:`steps_per_band`). Empty otherwise.
    """

    target: float
    status: str
    eval: StepEval
    #: the leg this checkpoint came from: a bare rate for a plain trajectory, a
    #: :class:`Leg` for a gap-fill branch. Normally the seed LR for every level;
    #: it differs only when escalation or a gap fill was needed, and then it is
    #: the thing a reader must see, because a family whose members sit at
    #: different learning rates differs by more than how the quirk was instilled.
    #: Read it through ``leg_rate``/``leg_root_rate``/``leg_key`` rather than
    #: formatting it as a number — a Leg is not one.
    lr: float | Leg = 0.0
    reason: str = ""
    #: QER movement per optimizer step at this checkpoint, read off the readings
    #: that bracket it (:func:`local_gradient`). ``None`` when nothing brackets
    #: it — never 0.0 standing in for "not measured".
    gradient: float | None = None
    #: how many integer steps fit inside this reading's own acceptance band
    #: (:func:`steps_per_band`). Recorded for EVERY level, not only the limited
    #: ones: it is the only number that says how converged a match is, and a
    #: reader comparing two organisms at "equal expression" is entitled to it.
    steps_per_band: float | None = None
    #: whether :func:`fill_gap` was actually run for this level. Only meaningful
    #: on a ``quantization_limited`` one, where "the sub-step chain climbed and
    #: never landed in band" and "no annealer was injected to climb one" are
    #: different statements about the same verdict: only the first says the
    #: coarse reading is the best this recipe can do at this rate.
    gap_fill_tried: bool = False

    @property
    def matched(self) -> bool:
        return self.status == "matched"

    @property
    def deviation(self) -> float:
        """Signed distance from the target, in QER points."""
        return self.eval.qer - self.target

    @property
    def deviation_sigma(self) -> float | None:
        """Distance from the target in units of the checkpoint's own stderr.

        ``None`` when the stderr is zero — the ratio would be meaningless rather
        than infinitely significant.
        """
        if not self.eval.qer_stderr:
            return None
        return self.deviation / self.eval.qer_stderr


@dataclass
class MatchResult:
    """The outcome of matching one recipe against the whole ladder."""

    levels: list[LevelResult]
    #: lr -> {step: eval}: the QER-vs-step curve of each trajectory searched
    trajectories: dict[float, dict[int, StepEval]] = field(default_factory=dict)
    #: lr -> highest step that trajectory reached
    tops: dict[float, int] = field(default_factory=dict)
    #: (lr, step) pairs minted by resuming
    minted: list[tuple[float, int]] = field(default_factory=list)

    @property
    def lrs_tried(self) -> list[float]:
        return sorted(self.trajectories)

    @property
    def matched(self) -> bool:
        """True only when every level landed in its band."""
        return all(lv.matched for lv in self.levels)


def pool_evals(evals: list[StepEval]) -> StepEval:
    """Combine **independent** evaluations of the same checkpoint into one
    estimate, weighting by inverse variance.

    Each draw is weighted by ``1/stderr**2`` and the pooled stderr is
    ``1/sqrt(sum of those weights)`` — the minimum-variance combination,
    computed straight off the cluster-robust stderrs.

    The draws must be genuinely independent: a re-read of the same result, or a
    re-run over the *same* prompts, shares the between-prompt variance that
    dominates the cluster stderr, so pooling it would report a precision that was
    never bought. :mod:`automo.stages.match` enforces this by giving each draw a
    fresh sampling seed and refusing to refine when the prompt pool is too small
    for the seed to change the draw.

    A missing stderr says nothing about precision, so if any draw lacks one the
    result falls back to the plain mean carrying the largest stderr.
    """
    if not evals:
        raise ValueError("pool_evals: no evaluations to pool")
    if len(evals) == 1:
        return evals[0]
    step = evals[0].step
    if len({e.step for e in evals}) != 1:
        raise ValueError(
            f"pool_evals: refusing to pool evaluations of different steps "
            f"{sorted({e.step for e in evals})} — they are different models"
        )
    draws = sum(e.draws for e in evals)
    # `not nan` is False, so a NaN stderr slips past a plain falsiness check and
    # reaches 1.0 / nan**2, poisoning the pooled QER to NaN. cluster_mean_stderr
    # returns NaN whenever a measurement has fewer than two usable samples, so
    # this is reachable, not theoretical. (`not 0.0` is True, so zero is caught.)
    if any(not e.qer_stderr or math.isnan(e.qer_stderr) for e in evals):
        # `max` over a NaN is order-dependent (max(nan, .01) is nan; max(.01, nan)
        # is .01), so taking the largest raw stderr can carry the NaN forward
        # into `classify`, which now refuses it — turning a documented soft
        # fallback into a hard failure depending on draw order. Use the largest
        # DEFINED stderr; only an all-undefined pool stays undefined.
        defined = [
            e.qer_stderr for e in evals if e.qer_stderr and not math.isnan(e.qer_stderr)
        ]
        return StepEval(
            step,
            sum(e.qer for e in evals) / len(evals),
            max(defined) if defined else float("nan"),
            draws,
        )
    weights = [1.0 / e.qer_stderr**2 for e in evals]
    qer = sum(w * e.qer for w, e in zip(weights, evals, strict=True)) / sum(weights)
    return StepEval(step, qer, (1.0 / sum(weights)) ** 0.5, draws)


def classify(e: StepEval, target: float, *, k_accept: float, k_verdict: float) -> str:
    """How one reading stands against a target: ``in_band`` / ``ambiguous`` /
    ``below`` / ``above``.

    Two thresholds, because the actions they gate are not symmetric. Accepting a
    match costs nothing beyond the evaluation already paid for, so acceptance
    uses ``k_accept``. Concluding that a level is out of reach costs a whole
    training extension, so that verdict must clear the wider ``k_verdict`` — a
    reading between the two is ``ambiguous``: not decisive enough to spend on.
    At 1 sigma a trajectory whose true ceiling sits exactly at the target reads
    "below" about 16% of the time; at 2 sigma, about 2%.
    """
    # A NaN stderr makes both comparisons False, so the reading falls straight
    # through to "below"/"above" — the two expensive verdicts this function's
    # twin thresholds exist to make hard to reach. `cluster_mean_stderr` returns
    # NaN for a measurement with fewer than two usable samples, so an undefined
    # measurement would silently buy a whole LR escalation.
    if math.isnan(e.qer_stderr):
        raise ValueError(
            f"classify: step {e.step} has a NaN stderr, so its distance from "
            f"the target is undefined; refusing to turn it into a verdict"
        )
    d = e.qer - target
    if abs(d) <= k_accept * e.qer_stderr:
        return "in_band"
    if abs(d) <= k_verdict * e.qer_stderr:
        return "ambiguous"
    return "below" if d < 0 else "above"


def diagnose_miss(
    cache: dict[int, StepEval], target: float, *, k_stderr: float = DEFAULT_K_STDERR
) -> str:
    """Why no checkpoint landed in the band, read off the evaluations already
    taken. Empty string when the level did match.

    Three different failures wear the same ``nearest`` verdict, and their
    remedies are not the same — one of them is the *opposite* of another — so a
    search that cannot tell them apart cannot act on its own result:

    ``quantization``
        two *adjacent* steps straddle the target and the jump between them is
        wider than the acceptance band. No integer step can land inside, however
        long the search runs. The remedy is a **lower** learning rate, which is
        the reverse of the ceiling remedy.
    ``search_budget``
        the bracket never narrowed to adjacent steps — the bisection ran out of
        iterations with room still left to look. The remedy is more iterations.
    ``unbracketed``
        no pair of readings straddles the target at all: the trajectory stayed
        below it, but not by enough to call the level ``unreached`` outright.
        The remedy is a longer trajectory, not a finer one.
    """
    ordered = sorted(cache.values(), key=lambda e: e.step)
    straddling = [(lo, hi) for lo, hi in pairwise(ordered) if lo.qer < target <= hi.qer]
    if not straddling:
        return "unbracketed"
    # The tightest bracket is the one that decides: a wider pair further out says
    # nothing about whether a step in between could have landed.
    lo, hi = min(straddling, key=lambda pair: pair[1].step - pair[0].step)
    if hi.step - lo.step > 1:
        return "search_budget"
    # An adjacent straddling pair in a genuine miss is ALWAYS quantization, and
    # the arithmetic forces it: neither member is in band, so
    # ``target - lo.qer > k*lo.stderr`` and ``hi.qer - target > k*hi.stderr``;
    # adding them gives ``hi.qer - lo.qer > k*(lo.stderr + hi.stderr)``, which is
    # the band. There is no "the steps were too close together to tell apart"
    # case hiding here — a pair that close would have had a member inside the
    # band, and the level would have matched. An earlier version of this function
    # returned a `noise_floor` diagnosis that no real miss could ever produce,
    # and a test asserted it using a bracket whose lower member was in band.
    return "quantization"


def local_gradient(cache: dict[int, StepEval], step: int) -> float | None:
    """QER movement per optimizer step at ``step``, as a secant across the
    readings that bracket it. ``None`` when nothing brackets it.

    The secant spans the nearest reading below and the nearest above, which is
    the same quantity the operator reads off the printed curve: at readings
    ``8: 12.2%  12: 26.7%  16: 40.2%`` the gradient at step 12 is
    ``(40.2 - 12.2) / 8 = 3.5pp`` per step. Deliberately NOT the one-sided
    difference to the nearer neighbour — bisection leaves an uneven grid, so a
    one-sided slope would be a property of where the bisection happened to stop
    rather than of the curve.

    A step with a reading on only one side (step 0, or the top of the
    trajectory) falls back to that side: an honest coarser measurement beats
    refusing to answer for the two steps a level is most likely to sit on.
    """
    below = [e for e in cache.values() if e.step < step]
    above = [e for e in cache.values() if e.step > step]
    lo = max(below, key=lambda e: e.step) if below else cache.get(step)
    hi = min(above, key=lambda e: e.step) if above else cache.get(step)
    if lo is None or hi is None or hi.step == lo.step:
        return None
    return (hi.qer - lo.qer) / (hi.step - lo.step)


def steps_per_band(
    gradient: float | None, stderr: float, *, k_stderr: float = DEFAULT_K_STDERR
) -> float | None:
    """How many integer steps fit inside one reading's acceptance band.

    The band is ``2 * k_stderr * stderr`` wide — the reading's OWN band, the
    same quantity :func:`classify` accepts on, not a constant: stderr moves with
    the QER value and with how many draws were bought, so a fixed band would
    call the same axis coarse at one level and fine at another.

    This is the resolution of the search, and below ~2 it stops being a search.
    At 1.3 steps per band the grid either straddles the target or lands on it by
    luck, and a reading that lands in band is inside it because of where the grid
    fell, not because the search converged onto the target — measured on
    cake-cos-sft-sdf-unmixed, which "matched" at 30.1% on the selection split and
    read 26.7% (-2.2 sigma, outside the band) on an independent one.

    ``None`` when no gradient was measurable, and when the gradient is zero or
    the band has no width: a flat curve fits unboundedly many steps in its band,
    a zero-width band is not a resolution question, and neither is a number a
    caller should compare against a threshold. Callers must treat ``None`` as
    "the step axis is not what limits this reading" rather than as a small
    number.
    """
    if gradient is None or gradient == 0.0 or stderr <= 0:
        return None
    return 2 * k_stderr * stderr / abs(gradient)


def too_coarse(
    cache: dict[int, StepEval],
    e: StepEval,
    min_steps_per_band: float,
    *,
    k_stderr: float = DEFAULT_K_STDERR,
) -> float | None:
    """The steps-per-band of an in-band reading whose axis is too coarse to
    accept it on; ``None`` when the axis is fine enough, unmeasurable, or the
    check is switched off with ``min_steps_per_band`` 0.

    Returns the number rather than a bool because every caller needs it: the one
    that routes the level to a finer axis has to report what was too coarse, and
    a bare True would send it back to re-derive the same secant.
    """
    if min_steps_per_band <= 0:
        return None
    spb = steps_per_band(local_gradient(cache, e.step), e.qer_stderr, k_stderr=k_stderr)
    if spb is None or spb >= min_steps_per_band:
        return None
    return spb


@dataclass(frozen=True)
class Leg:
    """One flat-learning-rate training segment, and where it branched from.

    Every checkpoint the search mints belongs to exactly one leg, and a leg's
    ``path_key`` is a complete recipe for re-minting it: the rates and the branch
    points, in order. That is what keeps step-addressability true once the search
    can change rate mid-trajectory — "step 3" alone names nothing, but
    ``lr1e-05-step10-anneal3.33e-06`` step 3 names one model.

    A plain trajectory has no parent and formats exactly as it always did, so
    nothing already on disk moves.
    """

    lr: float
    parent: "Leg | None" = None
    parent_step: int = 0
    #: decay horizon of an annealed leg: the number of updates over which the
    #: peak falls to zero. Part of the address because it changes the curve, not
    #: just its length — sub-step 3 of an 8-update decay and sub-step 3 of a
    #: 16-update decay are different models at the same nominal step.
    decay_steps: int | None = None

    @property
    def path_key(self) -> str:
        if self.parent is None:
            return f"lr{self.lr:g}"
        horizon = f"over{self.decay_steps}" if self.decay_steps else ""
        return (
            f"{self.parent.path_key}-step{self.parent_step}-anneal{self.lr:g}{horizon}"
        )

    @property
    def branch_key(self) -> str:
        """This leg's address below its root trajectory; empty for a plain one.

        Built from the chain, never by splitting ``path_key`` — a rate formats
        as ``1e-05``, so splitting on "-" lands inside the exponent and yields
        "05". That has now been the cause of three separate defects here; a
        formatted name is for humans, and structure comes from the object.
        """
        if self.parent is None:
            return ""
        horizon = f"over{self.decay_steps}" if self.decay_steps else ""
        seg = f"step{self.parent_step}-anneal{self.lr:g}{horizon}"
        prefix = self.parent.branch_key
        return f"{prefix}-{seg}" if prefix else seg

    @property
    def root_rate(self) -> float:
        """The rate of the trajectory this leg ultimately descends from — the
        recipe's rate. A gap-fill branch trains at its parent's rate for all but
        its final few steps, so this, not the branch peak, is what names it."""
        return self.parent.root_rate if self.parent is not None else self.lr

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth + 1

    def __str__(self) -> str:  # what shows up in logs and events
        return self.path_key


def leg_key(leg: "Leg | float") -> str:
    """Directory/segment name for a leg, accepting the bare float the search used
    before branches existed. Plain rates keep their historical spelling
    byte-for-byte — there are live checkpoints and published eval directories on
    disk named that way."""
    return leg.path_key if isinstance(leg, Leg) else f"lr{leg:g}"


def leg_root_rate(leg: "Leg | float") -> float:
    """The recipe rate behind a leg, however deeply branched."""
    return leg.root_rate if isinstance(leg, Leg) else leg


def leg_branch(leg: "Leg | float") -> str:
    """The part of a leg's address below its root trajectory; empty for a plain
    trajectory. Used as a git ref suffix so a branch checkpoint cannot collide
    with its parent's checkpoint at the same step number."""
    return leg.branch_key if isinstance(leg, Leg) else ""


def leg_rate(leg: "Leg | float") -> float:
    """The learning rate a leg trains at, however it is addressed."""
    return leg.lr if isinstance(leg, Leg) else leg


@dataclass(frozen=True)
class AnnealPlan:
    """How to refine the step axis when no integer step can land in the band.

    The step axis is integers, so its resolution is fixed; what varies is how far
    one step moves QER, and that is set by the learning rate. So the learning
    rate *is* the resolution knob of the step axis, and a bracket no step can hit
    is not a dead end — it is a request for a finer axis over the same interval.
    """

    #: checkpoint to resume from: the below-band side of the bracket.
    from_step: int
    #: the reduced rate to walk at.
    lr: float
    #: parent rate over annealed rate. Recorded so the choice is auditable rather
    #: than merely reproducible.
    factor: float
    #: bound on how many small steps to walk before giving up.
    max_steps: int
    #: QER movement per step at the parent rate, measured from the bracket.
    slope: float


#: Materialise (if absent) and evaluate the checkpoint ``j`` decay-updates above
#: the gap's lower bracket, on a no-warmup cosine peaking at ``peak``.
SubEval = Callable[[float, int], StepEval]


def fill_gap(
    lo_e: StepEval,
    hi_e: StepEval,
    target: float,
    sub_eval: SubEval,
    *,
    parent_lr: float,
    init_sub_lr: float | None = None,
    k_stderr: float = DEFAULT_K_STDERR,
    k_verdict: float = DEFAULT_K_VERDICT,
    max_sub_steps: int = 8,
    max_peak_trials: int = 4,
    drop: float = 0.5,
) -> StepEval | None:
    """Fill a one-step gap by climbing reduced-rate sub-steps warm-started from
    ``lo_e``. Returns the in-band reading, or ``None`` when no peak resolves it.

    Ported from ``mobfr.qer.auto_match.fill_gap``, whose reasoning follows.

    The bracket already sits either side of the band, so the reach the parent run
    bought is kept rather than thrown away — which is the whole advantage over
    re-running the recipe at a lower rate from the base model. That alternative
    also *fails* on steep recipes: a lower rate lowers the ceiling out from under
    the target, turning a bracketed miss into `unreached`.

    **The peak is searched two-sidedly inside a bracket that is known in advance.**
    A peak of 0 reproduces ``lo_e`` (below the band); a peak of ``parent_lr``
    reproduces the overshooting full step ``hi_e`` (above it). So the answer is
    bracketed in peak-space by ``(0, parent_lr]``, and the saturation ceiling is
    monotone in the peak. Each trial climbs one decay chain and classifies it:

    * a sub-step lands **in band** → done;
    * a sub-step **overshoots** before any lands → the peak is too steep, so pull
      the ceiling down and retry finer;
    * the chain **saturates below** the band within ``max_sub_steps`` → the peak
      is too shallow, so raise the floor and retry with more reach.

    Raising on undershoot is the part that is easy to get wrong. A halve-only
    search can only ever lower the peak, so once a reduced peak's whole decay
    tail sits below the target every halving makes the reach worse. Bisecting toward
    ``parent_lr`` instead climbs to the reach the target needs, while decay-to-0
    still gives the fine landing that avoids overshooting a narrow band.

    Overshoot is judged against the *wider* ``k_verdict`` while acceptance uses
    ``k_stderr``: abandoning a chain costs a whole new climb, a marginally-high
    reading is more often noise than a real overshoot, and the next sub-step is
    finer anyway because the rate is still annealing.
    """
    if parent_lr <= 0:
        raise ValueError(f"fill_gap: parent_lr must be > 0, got {parent_lr}")
    # The (0, parent_lr] bracket is the whole basis of the two-sided search, and
    # it holds only for a shrinking `drop`. drop=3.0 walks the peak UP, out of
    # the bracket, to 27x the parent rate; drop=1.0 spends the entire budget on
    # one peak. Both are reachable through the public keyword.
    if not 0 < drop < 1:
        raise ValueError(f"fill_gap: drop must be in (0, 1), got {drop}")
    if init_sub_lr is not None and not 0 < init_sub_lr <= parent_lr:
        raise ValueError(
            f"fill_gap: init_sub_lr must be in (0, parent_lr={parent_lr:g}], got "
            f"{init_sub_lr} — a peak at or above the parent rate reproduces the "
            f"overshooting full step, leaving nothing to bisect toward"
        )
    if hi_e.step - lo_e.step != 1:
        raise ValueError(
            f"fill_gap: expects a one-step gap, got steps {lo_e.step} -> "
            f"{hi_e.step}. A wider bracket should be bisected first; annealing "
            f"a gap the search has not narrowed wastes a whole decay chain."
        )
    peak_lo, peak_hi = 0.0, parent_lr
    peak = min(init_sub_lr or parent_lr * drop, peak_hi)
    for _ in range(max_peak_trials):
        overshot = False
        for j in range(1, max_sub_steps + 1):
            e = sub_eval(peak, j)
            if abs(e.qer - target) <= k_stderr * e.qer_stderr:
                return e
            if e.qer > target + k_verdict * e.qer_stderr:
                overshot = True
                break
        if overshot:
            peak_hi = peak
        else:
            peak_lo = peak
        peak = (peak_lo * peak_hi) ** 0.5 if peak_lo > 0 else peak_hi * drop
    return None


def find_inversions(
    evals: list[StepEval], *, k_stderr: float = DEFAULT_K_STDERR
) -> list[tuple[StepEval, StepEval]]:
    """Pairs that contradict "QER rises with the training step" by more than
    their stderrs can explain: ``sᵢ < sⱼ`` yet ``qerᵢ > qerⱼ + k·√(σᵢ²+σⱼ²)``.

    Bisection is only sound where QER is monotone in the step, so it is worth
    telling the operator when the evidence disagrees. It is *reported, never
    fatal*: noise-scale wiggles are expected and harmless, every pair of m
    checkpoints is another chance to exceed k, and bisection only relies on
    monotonicity *inside the bracket it narrows* — an inversion far above a
    level says nothing about whether that level was found soundly. In practice a
    warning here means the eval stderr understates run-to-run variability, which
    is what the re-draws exist to absorb.
    """
    ordered = sorted(evals, key=lambda e: e.step)
    out = []
    for i, lo in enumerate(ordered):
        for hi in ordered[i + 1 :]:
            combined = (lo.qer_stderr**2 + hi.qer_stderr**2) ** 0.5
            if lo.qer > hi.qer + k_stderr * combined:
                out.append((lo, hi))
    return out


# Injected side effects. ``materialize(from_step, to_step)`` resumes the
# checkpoint at ``from_step`` (0 == the base model) and trains until ``to_step``
# is written; ``eval_step(step)`` measures one checkpoint; ``refine(step,
# attempt)`` measures it again with a fresh sampling seed, returning None when no
# genuinely independent draw is available; ``retain(keep_full, keep_weights)``
# applies the disk retention policy.
# Returns every step it wrote a checkpoint for, which is not only `to_step`: a
# leg passes through the steps between its endpoints, so saving some of them
# costs a disk write while re-deriving them later costs the training again.
Materialize = Callable[[float, int, int], "list[int]"]
EvalStep = Callable[[float, int], StepEval]
Refine = Callable[[float, int, int], "StepEval | None"]
# Returns the steps that are STILL resumable after the policy ran, which is not
# the same as the steps it was asked to keep: the implementation may hold on to
# more when disk is plentiful (re-minting a released checkpoint costs training
# time, so evicting early is a pure loss until space is actually scarce).
#: ``(leg, keep_full, keep_weights, strict) -> what is still resumable``.
#: ``strict`` forces the release through rather than leaving it to the caller's
#: disk-pressure heuristic — correct only when nothing can ever want the
#: checkpoints again, which is exactly the abandoned-leg case.
Retain = Callable[..., "set[int]"]
#: Fill a one-step gap: ``(lr, lo_e, hi_e, target)`` -> ``(reading, leg)`` or
#: ``None``. The caller owns materialisation, so the search stays free of decay
#: mechanics — but it must hand back WHICH leg produced the reading, because the
#: winning checkpoint lives on a branch and nothing else in the result can say so.
GapFill = Callable[
    [float, StepEval, StepEval, float], "tuple[StepEval, Leg | float] | None"
]
OnEvent = Callable[..., None]


def _quantization_limited(lv: LevelResult) -> bool:
    """True when a *matched* level stands on an axis too coarse for its band — a
    match by luck of where the grid fell rather than by convergence.

    Read off the label the trajectory applied rather than re-derived from
    ``steps_per_band``, so the threshold is compared in exactly one place. A
    gap-filled level therefore excludes itself: gap filling IS the remedy for a
    coarse axis, so its checkpoint was found on a decayed sub-step axis finer
    than the parent's, and the parent gradient recorded on it describes the axis
    the fill replaced.
    """
    return lv.matched and lv.reason == "quantization_limited"


def _level_key(lv: LevelResult) -> tuple[int, float]:
    """Sort key for choosing between what two trajectories found for one level;
    smaller is better.

    Three tiers, because "matched" is not one thing once the axis can be too
    coarse for the band: a converged match beats a quantization-limited one,
    which still beats any miss. Within the limited tier the finest axis wins, so
    a rung that changed the rate for another level's sake cannot silently swap in
    a coarser reading for this one. Within the miss tier the closest reading
    wins, as before.
    Converged matches tie, so the FIRST one is kept — the historical rule, and
    the cheapest, since a later trajectory costs a whole extra run to reach the
    same verdict.
    """
    if lv.matched and not _quantization_limited(lv):
        return (0, 0.0)
    if lv.matched:
        return (1, -(lv.steps_per_band or 0.0))
    return (2, abs(lv.deviation))


def run_match(
    targets: list[float],
    materialize: Materialize,
    eval_step: EvalStep,
    base_eval: StepEval,
    *,
    seed_lr: float,
    initial_steps: int,
    max_total_steps: int,
    k_stderr: float = DEFAULT_K_STDERR,
    k_verdict: float = DEFAULT_K_VERDICT,
    max_refines: int = 2,
    max_iters: int = 16,
    max_lr_changes: int = 0,
    lr_up: float = 2.0,
    min_steps_per_band: float = DEFAULT_MIN_STEPS_PER_BAND,
    refine: Refine | None = None,
    retain: Retain | None = None,
    gap_fill: GapFill | None = None,
    on_event: OnEvent | None = None,
) -> MatchResult:
    """Find a checkpoint for each target QER, changing the learning rate only
    when the step axis it buys is the thing standing in the way.

    Each learning rate gets **one trajectory**, searched by
    :func:`_search_trajectory`: train to ``initial_steps``, extend by doubling
    while the top target is confidently above the curve, then bisect the step
    axis per level. Within a trajectory the learning rate never changes, so a
    step names one model and any checkpoint can be re-minted exactly.

    **The rate ladder only runs one way — up — and only for ``unreached``:** the
    recipe's QER ceiling is a property of the rate, so once training longer has
    stopped raising the curve the only remaining lever is a hotter rate. Three
    verdicts deliberately do **not** escalate: ``below_base`` (training only
    raises QER), ``nearest`` (the trajectory brackets the level; a finer step,
    not a hotter one, is what that needs), and a match found on an axis too
    coarse for its band, which is the paragraph below.

    Changing the rate is not free, and the cost is scientific rather than
    computational: a family whose members sit at different learning rates differs
    by more than how the quirk was instilled, which is the confound the whole
    exercise exists to remove. Hence ``max_lr_changes``, tried only after the
    cheaper remedy (a longer trajectory) has demonstrably failed, and each rate
    recorded on every level it produced (``LevelResult.lr``).

    **A coarse axis is NOT a rate problem.**
    A rate that moves QER 3.5pp per step against a 4.5pp-wide band fits 1.3 steps
    inside the whole acceptance window, so a reading that lands in band did so
    because of where the grid fell; it is reported ``matched`` and nobody notices
    until an independent split reads it outside the band. :func:`diagnose_miss`
    prescribes "a LOWER learning rate" for the case where nothing matched, but
    that remedy makes this case worse: what moves the curve is exposure (rate x
    steps), so halving the rate does not stretch the trajectory — it lowers the
    ceiling out from under the target, comes back ``unreached``, and the search
    escalates straight back to the coarse rate. It is the failure
    :func:`fill_gap`'s docstring already names: "once a reduced peak's whole
    decay tail sits below the target every halving makes the reach worse".

    So the coarse match is routed to ``gap_fill`` and the rate is left alone.
    The routing lives inside :func:`_search_trajectory` — one trajectory, one
    rate — which is what makes "this path cannot move the rate" structural
    rather than a promise: a level that stays coarse comes back ``matched`` with
    ``reason="quantization_limited"``, and ``matched`` is not a verdict this loop
    escalates on. That bounded honest answer beats an unbounded search.

    Results are merged across trajectories by :func:`_level_key`: a converged
    match beats a quantization-limited one beats the closest miss.
    """
    if not targets:
        raise ValueError("run_match: no target QER levels given")
    if initial_steps < 1:
        raise ValueError(f"run_match: initial_steps must be >= 1, got {initial_steps}")
    if max_total_steps < initial_steps:
        raise ValueError(
            f"run_match: max_total_steps ({max_total_steps}) is below initial_steps "
            f"({initial_steps}) — the trajectory could not even reach its first stop"
        )
    if lr_up <= 1.0:
        raise ValueError(
            f"run_match: lr_up must be > 1 to raise the ceiling, got {lr_up}"
        )
    if min_steps_per_band < 0:
        raise ValueError(
            f"run_match: min_steps_per_band must be >= 0, got {min_steps_per_band}"
        )

    best: dict[float, LevelResult] = {}
    trajectories: dict[float, dict[int, StepEval]] = {}
    tops: dict[float, int] = {}
    minted: list[tuple[float, int]] = []
    lr = seed_lr
    ups = 0

    def release_leg(leg_lr: float, reason: str) -> None:
        """Release the leg we are abandoning, down to the checkpoints its own
        levels still reference.

        Retention inside a trajectory only ever sees THAT trajectory, so without
        this the previous leg is stranded for the rest of the run: nothing
        revisits it, and its checkpoints are invisible to the disk guard that
        would otherwise free them. Measured at 449 GB held by one abandoned 7B
        leg while the run that owned it was minting into a nearly-full disk.
        """
        if retain is None:
            return
        stranded = trajectories.get(leg_lr, {})
        keep = (
            _best_steps(stranded, targets, k_stderr=k_stderr, k_verdict=k_verdict)
            if stranded
            else set()
        )
        # ...UNIONED with what the run as a whole still points at on this leg.
        # `_best_steps` re-derives the nearest reading from the leg's own cache,
        # which is not always the step the surviving LevelResult recorded: a
        # bisection stops at the first in-band midpoint it visits, so the level
        # can name a step the recomputation does not choose. When the run-wide
        # best for a target lives on the leg being abandoned, releasing that
        # step deletes the checkpoint the reported reading is measured from, and
        # the run dies loading it.
        keep |= {
            lv.eval.step for lv in best.values() if leg_key(lv.lr) == leg_key(leg_lr)
        }
        # STRICT. Routing this through the normal (lazy) path meant the release
        # fired and freed nothing whenever the disk looked comfortable —
        # observed live: `release_leg kept=[32,64,128,256,512]` on a 7B leg,
        # 204 GB retained on a trajectory nothing can resume from. Lazy
        # retention exists to avoid re-minting checkpoints that might still be
        # wanted; on an abandoned leg none of them can be, because the next rung
        # restarts from step 0 with `resumable = {0}`.
        still = retain(leg_lr, set(), keep, strict=True)
        if on_event is not None:
            on_event("release_leg", lr=leg_lr, kept=sorted(still), reason=reason)

    while True:
        levels, cache, top, mints = _search_trajectory(
            lr,
            targets,
            materialize,
            eval_step,
            base_eval,
            initial_steps=initial_steps,
            max_total_steps=max_total_steps,
            k_stderr=k_stderr,
            k_verdict=k_verdict,
            max_refines=max_refines,
            max_iters=max_iters,
            min_steps_per_band=min_steps_per_band,
            refine=refine,
            retain=retain,
            gap_fill=gap_fill,
            on_event=on_event,
        )
        trajectories[lr] = cache
        tops[lr] = top
        minted += [(lr, s) for s in mints]
        for lv in levels:
            prev = best.get(lv.target)
            if prev is None or _level_key(lv) < _level_key(prev):
                best[lv.target] = lv

        unreached = [lv.target for lv in best.values() if lv.status == "unreached"]
        # The ONLY verdict that moves the rate. A quantization-limited match is
        # `matched`, so it cannot reach this branch — which is the structural
        # form of the lesson in the docstring: the one experiment that lowered
        # the rate for coarseness turned the level `unreached` and escalated
        # straight back to the rate that was too coarse.
        if unreached and ups < max_lr_changes:
            release_leg(lr, "trajectory abandoned; escalating")
            lr *= lr_up
            ups += 1
            if on_event is not None:
                on_event(
                    "escalate_lr",
                    to_lr=lr,
                    unreached=[round(u, 4) for u in unreached],
                    reason="training longer stopped raising QER",
                )
            continue

        break

    return MatchResult(
        # Labelled by the trajectory that found it, not here: the label now
        # depends on whether the sub-step chain was climbed and failed, which is
        # a fact of the search rather than of what this loop was allowed to try
        # next.
        levels=[best[t] for t in targets],
        trajectories=trajectories,
        tops=tops,
        minted=minted,
    )


def _search_trajectory(
    lr: float,
    targets: list[float],
    materialize: Materialize,
    eval_step: EvalStep,
    base_eval: StepEval,
    *,
    initial_steps: int,
    max_total_steps: int,
    k_stderr: float = DEFAULT_K_STDERR,
    k_verdict: float = DEFAULT_K_VERDICT,
    max_refines: int = 2,
    max_iters: int = 16,
    min_steps_per_band: float = DEFAULT_MIN_STEPS_PER_BAND,
    refine: Refine | None = None,
    retain: Retain | None = None,
    gap_fill: GapFill | None = None,
    on_event: OnEvent | None = None,
) -> tuple[list[LevelResult], dict[int, StepEval], int, list[int]]:
    """Search ONE trajectory (one learning rate) for every target.

    Two phases. **Reach**: train from the base model to ``initial_steps``, then
    keep extending (doubling, capped at ``max_total_steps``) while the top target
    is confidently above the trajectory's QER — under a flat LR, training longer
    is what raises the ceiling, so no learning-rate change is ever needed and
    every level comes off one trajectory. **Locate**: for each target, bisect the
    step axis, minting absent midpoints via ``materialize``, then report the
    closest checkpoint found and whether it landed in the band.

    Never raises on a failed level. A level that cannot be matched is reported
    with its nearest checkpoint and a diagnosis, because the checkpoint is still
    worth keeping and the deviation is the finding. Callers decide what a
    non-matching level means (see ``MatchResult.matched``).

    The **whole** remedy for a match found on too coarse an axis lives in here,
    and that is deliberate: one trajectory is one learning rate, so a routing
    decision taken at this level cannot move the rate even by mistake — see
    ``run_match`` for the experiment that established it must not.
    """

    cache: dict[int, StepEval] = {0: base_eval}
    spent: dict[int, int] = {}  # extra draws bought, per checkpoint
    # Steps whose checkpoint can still be trained onward from. Step 0 is the base
    # model, which is always available and is never a directory we own.
    resumable: set[int] = {0}
    minted: list[int] = []

    def emit(kind: str, **fields: object) -> None:
        if on_event is not None:
            on_event(kind, **fields)

    def cached(step: int) -> StepEval:
        if step not in cache:
            cache[step] = eval_step(lr, step)
            emit(
                "eval",
                lr=lr,
                step=step,
                qer=cache[step].qer,
                stderr=cache[step].qer_stderr,
            )
        return cache[step]

    def resolve(step: int, target: float) -> StepEval:
        """Read a checkpoint's QER, buying independent re-draws when the reading
        is close enough to the target to drive a decision.

        The budget is spent *in full* once triggered, rather than stopping as
        soon as the pooled value leaves the band. Stopping on the first draw that
        agrees with a decision is optional stopping: it makes the reported
        interval a property of when we chose to look, not of the checkpoint. A
        fixed number of draws costs the same in the worst case and is decidable.

        The budget is per checkpoint, not per target: several targets bisect
        through the same steps, and re-buying draws for an estimate already as
        sharp as we intend to make it would pay the judge for precision the cache
        already holds.
        """
        e = cached(step)
        if refine is None:
            return e
        if classify(e, target, k_accept=k_stderr, k_verdict=k_verdict) not in (
            "in_band",
            "ambiguous",
        ):
            return e
        while spent.get(step, 0) < max_refines:
            attempt = spent.get(step, 0) + 1
            fresh = refine(lr, step, attempt)
            spent[step] = attempt
            if fresh is None:  # no independent draw available — see pool_evals
                break
            e = pool_evals([e, fresh])
            cache[step] = e
            emit(
                "refine",
                lr=lr,
                step=step,
                attempt=attempt,
                qer=e.qer,
                stderr=e.qer_stderr,
            )
        return e

    def anchor_below(step: int) -> int:
        """The nearest step we can still resume from, at or below ``step``.

        Retention may have stripped the immediate predecessor; resuming from
        further back costs extra training but lands on the identical trajectory,
        because the LR is flat.
        """
        return max(s for s in resumable if s < step)

    def mint(step: int) -> None:
        if step in resumable:
            return
        src = anchor_below(step)
        emit("mint", lr=lr, from_step=src, to_step=step)
        for saved in materialize(lr, src, step) or [step]:
            resumable.add(saved)
        resumable.add(step)
        minted.append(step)

    def apply_retention(keep_full: set[int], keep_weights: set[int]) -> None:
        """Ask the caller to enforce the retention policy, then believe what it
        reports is left.

        The caller decides how much to actually release — it is the side that can
        see the disk — so the search takes the returned set as the truth about
        what can still be resumed rather than assuming its request was applied
        verbatim.
        """
        if retain is None:
            return
        still = retain(lr, keep_full, keep_weights)
        resumable.intersection_update(set(still) | {0})

    # ── Phase 1: reach ────────────────────────────────────────────────────────
    # The endpoints of this phase are kept resumable for the whole search. They
    # are few (log2 of the step budget, so ~5) and evenly spaced, which bounds
    # the cost of re-minting anything to half a coarse interval instead of the
    # whole prefix from the base. Without this, retention releases an anchor as
    # soon as the current level stops needing it and the next level re-trains it
    # from scratch — measured at 251 s of pure rework in a 33-minute 1B run, and
    # it scales with the model.
    coarse: set[int] = set()
    top_target = max(targets)
    step = min(initial_steps, max_total_steps)
    emit("train", lr=lr, from_step=0, to_step=step)
    resumable.update(materialize(lr, 0, step) or [])
    resumable.add(step)
    coarse.add(step)
    e = resolve(step, top_target)
    # Extend until the reading has actually CROSSED the top target, not merely
    # until it stops being confidently below it. `classify` calls anything within
    # k_verdict of the target "ambiguous", and an ambiguous reading approaching
    # from below is the one case where stopping is fatal: the locate phase can
    # only bisect between steps it has, so a trajectory that halts below the
    # target leaves no bracket and every step beneath it is below the band. The
    # search then reports `nearest` for what is really "I stopped too early",
    # on a curve that was still climbing. But stop as soon as the endpoint is
    # IN BAND: the top target
    # is then already matched, every lower target is bracketed by [0, top], and
    # continuing is pure waste — a curve ceilinging at .499 against a .500 target
    # trained 8, 16, 32 ... 1024, five doublings and 32x the work past the answer,
    # because .499 < .500 is true while .499 is also inside the band.
    while (
        classify(e, top_target, k_accept=k_stderr, k_verdict=k_verdict) != "in_band"
        and e.qer < top_target
        and step < max_total_steps
    ):
        nxt = min(step * 2, max_total_steps)
        emit("extend", lr=lr, from_step=step, to_step=nxt, qer=e.qer, target=top_target)
        resumable.update(materialize(lr, step, nxt) or [])
        resumable.add(nxt)
        coarse.add(nxt)
        # The previous endpoint stays resumable: bisection will need it as an
        # anchor for midpoints below `nxt`.
        step = nxt
        # Retention has to run here too, not only during bisection. Reaching a
        # distant target can take many doublings, and each leg also writes a
        # quarter grid, so this phase accumulates the bulk of a run's checkpoints
        # — several campaign runs held ~20 resumable checkpoints (167 GB at 1B)
        # and hit the disk guard without ever having been offered a chance to
        # release one. Under a roomy disk this keeps everything anyway; under
        # pressure it is the difference between reaping and dying.
        apply_retention(
            coarse | {step},
            _best_steps(cache, targets, k_stderr=k_stderr, k_verdict=k_verdict),
        )
        e = resolve(step, top_target)
    top = step

    # ── Phase 2: locate ───────────────────────────────────────────────────────
    results: list[LevelResult] = []
    # Ascending, so the bracket that has to stay resumable only ever moves
    # forward and everything below it can be released.
    for target in sorted(targets):
        emit("locate", lr=lr, target=target)
        lo, hi = 0, top
        lo_e, hi_e = resolve(lo, target), resolve(hi, target)

        if classify(lo_e, target, k_accept=k_stderr, k_verdict=k_verdict) == "above":
            # Training only raises QER, so nothing on this trajectory — or any
            # longer one — can come back down to this level.
            results.append(LevelResult(target, "below_base", lo_e, lr))
            emit(
                "level", lr=lr, target=target, status="below_base", step=0, qer=lo_e.qer
            )
            continue
        # AT THE STEP CEILING the wide verdict margin has one less thing to
        # protect. `k_verdict` is deliberately wider than the acceptance band
        # because declaring a level out of reach buys an expensive remedy — but
        # there are two remedies, and at `max_total_steps` the cheap one (train
        # longer) is provably exhausted. A trajectory that runs out of steps
        # between 1 and 2 sigma short therefore read `ambiguous`, was reported
        # `nearest`, and the learning-rate ladder was never offered the level at
        # all: the search stopped with an unmatched organism while the one
        # remedy it had left went untried.
        #
        # So at the ceiling only, "outside the band" is enough to say unreached.
        # Below the ceiling nothing changes — there the wide margin still guards
        # a real choice between extending and escalating.
        # `k_stderr`, not `k_verdict`, and NOT behind an `at_ceiling` branch: this
        # site is reachable only at the ceiling, so such a branch would have a
        # dead arm. The reach loop above exits on exactly three conditions — the
        # top target is in band, the curve overshot it, or `step >=
        # max_total_steps` — and the first two leave `hi_e` in band or above for
        # every target, since `top_target` is the largest. A `below` here
        # therefore implies the trajectory ran out of steps.
        if classify(hi_e, target, k_accept=k_stderr, k_verdict=k_stderr) == "below":
            results.append(LevelResult(target, "unreached", hi_e, lr))
            emit(
                "level", lr=lr, target=target, status="unreached", step=hi, qer=hi_e.qer
            )
            continue

        for _ in range(max_iters):
            if hi - lo <= 1:
                break  # single-step resolution: nothing finer exists to try
            mid = (lo + hi) // 2
            mint(mid)
            apply_retention(
                coarse | {lo, hi, mid, top},
                _best_steps(cache, targets, k_stderr=k_stderr, k_verdict=k_verdict),
            )
            e = resolve(mid, target)
            if classify(e, target, k_accept=k_stderr, k_verdict=k_verdict) == "in_band":
                coarse_spb = (
                    too_coarse(cache, e, min_steps_per_band, k_stderr=k_stderr)
                    if gap_fill is not None
                    else None
                )
                if coarse_spb is None:
                    break
                # In band, and not an answer: at this resolution the band spans
                # less than `min_steps_per_band` steps, so the reading is inside
                # it because of where the integer grid fell (see `too_coarse`).
                # Keep narrowing INSTEAD of stopping — not to find a better
                # integer step, which cannot exist on an axis this coarse, but to
                # hand `gap_fill` below the one-step gap it requires, since it
                # refuses a bracket the search has not narrowed. The reading
                # stays cached and is still what gets reported if the sub-step
                # chain cannot land in band.
                emit(
                    "coarse_match",
                    lr=lr,
                    target=target,
                    step=mid,
                    qer=e.qer,
                    steps_per_band=round(coarse_spb, 3),
                    reason="narrowing the bracket for a gap fill",
                )
            if e.qer < target:
                lo = mid
            else:
                hi = mid

        # The bisection has run out of integer steps, or has been driven down
        # to them by a match too coarse to accept. Either way, if the bracket
        # straddles the target the level is not out of reach — the step axis is
        # simply too coarse here, and a reduced-rate decay chain warm-started
        # from `lo` can land inside the band. Tried before falling back to
        # `nearest` OR to the coarse match, because both throw away a trajectory
        # that is already one step from the answer.
        gap_fill_tried = False
        if gap_fill is not None and hi - lo == 1:
            lo_e, hi_e = resolve(lo, target), resolve(hi, target)
            # Straddling is the whole condition. The jump exceeding the band
            # is no longer implied — a coarse match arrives here WITH an in-band
            # reading — but it does not need to be: annealing a gap narrower than
            # the band still lands in it, and the level is then converged rather
            # than merely lucky, which is the distinction this path exists to
            # buy.
            if lo_e.qer < target <= hi_e.qer:
                emit(
                    "gap_fill",
                    lr=lr,
                    target=target,
                    lo=lo,
                    hi=hi,
                    jump=round(hi_e.qer - lo_e.qer, 4),
                )
                gap_fill_tried = True
                filled = gap_fill(lr, lo_e, hi_e, target)
                if filled is not None:
                    e_filled, filled_leg = filled
                    # `gap_fill` is injected, so the in-band property lives in
                    # the caller's implementation while the verdict is published
                    # here. Verify rather than trust: a filler returning a
                    # reading 100pp off target was reported `matched`.
                    if (
                        classify(
                            e_filled, target, k_accept=k_stderr, k_verdict=k_verdict
                        )
                        != "in_band"
                    ):
                        emit(
                            "gap_fill_rejected",
                            lr=lr,
                            target=target,
                            qer=e_filled.qer,
                            reason="returned reading not in band",
                        )
                        filled = None
                if filled is not None:
                    # The level carries the BRANCH, not the parent rate: the
                    # matched weights sit in the branch's directory, and a level
                    # that named the parent would point publication at the
                    # overshooting full step while quoting the anneal's QER.
                    # The gradient of the PARENT bracket, computed from the two
                    # readings that define the gap rather than from `cache`: a
                    # sub-step is addressed as `lo + j` on its own branch, which
                    # collides with the parent's step numbers, so reading a
                    # secant out of the parent cache at that number would mix two
                    # address spaces. This is the axis the fill was needed for,
                    # and saying so is what tells a reader the checkpoint was
                    # found on a finer one.
                    gap_grad = (hi_e.qer - lo_e.qer) / (hi_e.step - lo_e.step)
                    results.append(
                        LevelResult(
                            target,
                            "matched",
                            e_filled,
                            filled_leg,
                            "gap_fill",
                            gap_grad,
                            steps_per_band(
                                gap_grad, e_filled.qer_stderr, k_stderr=k_stderr
                            ),
                            gap_fill_tried=True,
                        )
                    )
                    emit(
                        "level",
                        lr=filled_leg,
                        target=target,
                        status="matched",
                        step=e_filled.step,
                        qer=e_filled.qer,
                        reason="gap_fill",
                        gradient=round(gap_grad, 6),
                    )
                    continue

        # Decide over *all* the evidence rather than accepting whichever
        # candidate the bisection happened to try first. Bisection generates
        # candidates; it does not have to be the thing that picks between them,
        # and picking the closest is both free (every reading is cached) and
        # unbiased with respect to the order they were visited.
        # Selection must agree with acceptance. `_nearest` ranks by |qer-target|
        # while `classify` accepts by |qer-target| <= k*sigma, and those are
        # different orderings whenever sigma varies across checkpoints — which it
        # always does, since the cluster stderr depends on the QER value and
        # `resolve` buys extra draws only near a target. A checkpoint that WAS in
        # band could therefore lose the distance ranking to one that is not, and
        # the level would be reported `nearest` despite having been matched:
        #
        #   step 7  qer .4855 se .0060  |d| .0145  -> below     (wins on distance)
        #   step 8  qer .5200 se .0250  |d| .0200  -> IN BAND   (discarded)
        #
        # So: prefer the in-band set, and fall back to nearest only when it is
        # empty. This also restores the premise `diagnose_miss` documents — that
        # no cached reading is in band — which its correctness argument assumes.
        in_band = [
            e
            for e in cache.values()
            if classify(
                resolve(e.step, target),
                target,
                k_accept=k_stderr,
                k_verdict=k_verdict,
            )
            == "in_band"
        ]
        pool = {e.step: cache[e.step] for e in in_band} if in_band else cache
        best = resolve(_nearest(pool, target).step, target)
        status = (
            "matched"
            if classify(best, target, k_accept=k_stderr, k_verdict=k_verdict)
            == "in_band"
            else "nearest"
        )
        # How converged this reading is, recorded whether it matched or not: the
        # QER alone cannot say whether the checkpoint sits at the target or
        # merely near it because that is where the grid fell.
        grad = local_gradient(cache, best.step)
        spb = steps_per_band(grad, best.qer_stderr, k_stderr=k_stderr)
        # A match still standing on the coarse axis is reported as one: the gap
        # fill above either could not run or did not land in band, and nothing
        # else will be tried for it. Bounded and labelled beats an unbounded
        # search, and beats the silent `matched` this whole path replaces.
        limited = (
            status == "matched"
            and too_coarse(cache, best, min_steps_per_band, k_stderr=k_stderr)
            is not None
        )
        reason = (
            "quantization_limited"
            if limited
            else ""
            if status == "matched"
            else diagnose_miss(cache, target, k_stderr=k_stderr)
        )
        results.append(
            LevelResult(target, status, best, lr, reason, grad, spb, gap_fill_tried)
        )
        if limited:
            emit(
                "quantization_limited",
                lr=lr,
                target=target,
                step=best.step,
                gradient=round(grad, 6) if grad is not None else None,
                steps_per_band=round(spb, 3) if spb is not None else None,
                gap_fill_tried=gap_fill_tried,
                reason="reporting the coarse match rather than searching on",
            )
        emit(
            "level",
            lr=lr,
            target=target,
            status=status,
            step=best.step,
            qer=best.qer,
            **({"gradient": round(grad, 6)} if grad is not None else {}),
            **({"steps_per_band": round(spb, 3)} if spb is not None else {}),
            **({"reason": reason} if reason else {}),
        )

    # The search is over, so the coarse anchors are no longer needed resumable —
    # only what the caller was promised: the top of the trajectory plus each
    # level's checkpoint.
    # The union of the rule and the ANSWERS. `_best_steps` now applies the same
    # rule level selection does, so these should agree — but "should agree" is
    # what was true of the two orderings before, and the cost of them differing
    # is deleting the checkpoint this function is about to return. Keeping the
    # decided steps outright makes the guarantee structural rather than a
    # property of two code paths staying in step.
    decided = {r.eval.step for r in results}
    apply_retention(
        {top},
        _best_steps(cache, targets, k_stderr=k_stderr, k_verdict=k_verdict) | decided,
    )
    return results, dict(cache), top, minted


def _nearest(cache: dict[int, StepEval], target: float) -> StepEval:
    """The evaluated checkpoint closest to ``target``, ties broken by earliest step.

    The tie-break is not cosmetic. A saturated curve gives several steps the same
    QER, and without an ordering the winner falls out of dict iteration order —
    so the reported provenance could be a checkpoint far above the bracket the
    search actually converged on, and could change between runs that measured the
    same thing. Earliest wins: it is the cheapest checkpoint reaching that QER and
    the one whose step number describes where the curve got there.
    """
    return min(cache.values(), key=lambda e: (abs(e.qer - target), e.step))


def _best_steps(
    cache: dict[int, StepEval],
    targets: list[float],
    *,
    k_stderr: float = DEFAULT_K_STDERR,
    k_verdict: float = DEFAULT_K_VERDICT,
) -> set[int]:
    """The steps worth keeping weights for: the checkpoint each level reports.

    Recomputed from the live cache rather than tracked, so a level whose best
    candidate improves mid-search releases the previous one straight away.

    THE SAME RULE THE LEVEL ITSELF USES, and it has to be. A level reports the
    nearest reading *among those in band*, falling back to the nearest overall
    only when none is; this function used to rank by |qer - target| alone. The
    two orderings disagree exactly when the nearest reading is out of band while
    some further one is inside it — which happens whenever a nearer step has a
    tighter stderr, the ordinary shape of a curve measured at several steps.

    When they disagreed the step a `LevelResult` pointed at was in neither
    `keep_full` nor `keep_weights`, so the retention pass DELETED THE CHECKPOINT
    THE SEARCH HAD JUST MATCHED, and the manifest was written naming a path that
    no longer existed. Reachable through the final pass, and through
    `release_leg`, which retains strictly.

    `resolve` is deliberately not used here even though level selection uses it:
    it buys judge re-draws, and retention must not spend money to decide what to
    keep. With `max_refines: 0` (shipped) the two are identical anyway; above it
    the in-band preference is what removes the divergence that matters.
    """
    keep = set()
    for target in targets:
        if not cache:
            continue
        in_band = [
            e
            for e in cache.values()
            if classify(e, target, k_accept=k_stderr, k_verdict=k_verdict) == "in_band"
        ]
        pool = {e.step: e for e in in_band} if in_band else cache
        keep.add(_nearest(pool, target).step)
    return keep
