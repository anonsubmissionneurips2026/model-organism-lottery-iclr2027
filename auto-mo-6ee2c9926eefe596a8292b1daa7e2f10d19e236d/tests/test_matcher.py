"""The pure QER-matching search.

Why these tests: the matcher spends GPU-hours and judge money on decisions made
from noisy measurements, and it is the component whose failures are least
visible — a mis-decided level still returns a plausible-looking checkpoint. So
what is pinned here is the *decision logic*: which checkpoint gets minted, when
a level is declared out of reach, what happens to a level that cannot be hit,
and that a failed level still hands back a model instead of an exception.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from automo.matcher import (
    MatchResult,
    StepEval,
    classify,
    find_inversions,
    local_gradient,
    pool_evals,
    steps_per_band,
    Leg,
    diagnose_miss,
    fill_gap,
    leg_key,
    run_match,
)

# ── a fake trajectory ─────────────────────────────────────────────────────────


class FakeRun:
    """A trainable trajectory: ``qer_at(step)`` defines the QER curve.

    Records every mint so tests can assert *which* checkpoints were trained and
    what each was resumed from — the difference between an O(log n) search and
    one that retrains from scratch is invisible in the result alone.
    """

    def __init__(
        self,
        qer_at: Callable[[int], float],
        stderr: float = 0.01,
        jitter: dict[tuple[int, int], float] | None = None,
        lazy: bool = False,
        ceiling: dict[float, float] | None = None,
        slope: dict[float, float] | None = None,
    ) -> None:
        self.qer_at = qer_at
        # Learning rate sets the QER ceiling: training longer at a low rate
        # plateaus, and only a hotter rate lifts the plateau. That is the fact
        # escalation exists for, so the fake has to model it.
        self.ceiling = ceiling or {}
        # Learning rate also sets how far ONE step moves QER, which is the fact
        # the downward rung exists for: halving the rate roughly halves the
        # movement per step, so the same interval is subdivided twice as finely.
        # A rate not listed moves the nominal amount.
        self.slope = slope or {}
        self.stderr = stderr
        self.jitter = jitter or {}
        # `lazy` models the real stage's default: while there is disk headroom
        # nothing resumable is released, so checkpoints a leg saved for free stay
        # available. `lazy=False` models a disk under pressure, which evicts them
        # and pays for them again in training — the deliberate trade.
        self.lazy = lazy
        self.on_disk: set[int] = {0}
        #: steps a leg wrote *in passing* — training already paid for them, so
        #: the search must never mint one again
        self.free_steps: set[int] = set()
        self.mints: list[tuple[int, int]] = []
        self.lrs_used: set[float] = set()
        self.evals: list[int] = []
        self.draws: dict[int, int] = {}
        self.retained: list[tuple[set[int], set[int]]] = []

    def curve(self, lr: float, step: int) -> float:
        return min(
            self.qer_at(step) * self.slope.get(lr, 1.0), self.ceiling.get(lr, 1.0)
        )

    def materialize(self, lr: float, from_step: int, to_step: int) -> list[int]:
        self.mints.append((from_step, to_step))
        self.lrs_used.add(lr)
        # Mirrors the real stage: a leg also writes its midpoint, because the
        # training that passes through it has already been paid for.
        quarter = (to_step - from_step) // 4
        grid = [from_step + i * quarter for i in (1, 2, 3)] if quarter >= 1 else []
        written = [s for s in dict.fromkeys(grid) if from_step < s < to_step]
        written.append(to_step)
        self.on_disk.update(written)
        self.free_steps.update(w for w in written if w != to_step)
        return written

    def eval_step(self, lr: float, step: int) -> StepEval:
        self.evals.append(step)
        self.draws[step] = self.draws.get(step, 0) + 1
        return StepEval(step, self.curve(lr, step), self.stderr)

    def refine(self, lr: float, step: int, attempt: int) -> StepEval:
        self.draws[step] = self.draws.get(step, 0) + 1
        # `jitter` lets a test make the *first* draw a noise excursion and the
        # confirming draws land on the truth.
        qer = self.jitter.get((step, attempt), self.curve(lr, step))
        return StepEval(step, qer, self.stderr)

    def retain(
        self,
        lr: float,
        keep_full: set[int],
        keep_weights: set[int],
        strict: bool = False,
    ) -> set[int]:
        # Models the STRICT path — a disk under pressure, releasing exactly what
        # it was asked to keep. That is the worst case for re-mint churn, so it
        # is what the retention tests should run against; the lazy path (keep
        # more while space is plentiful) can only ever mint less.
        self.retained.append((set(keep_full), set(keep_weights)))
        if self.lazy:
            return set(keep_full) | self.on_disk
        self.on_disk &= set(keep_full) | {0}
        return set(keep_full)


def linear(slope: float, cap: float = 1.0) -> Callable[[int], float]:
    """QER rising linearly with the step, saturating at ``cap``."""
    return lambda s: min(cap, slope * s)


def piecewise(points: dict[int, float]) -> Callable[[int], float]:
    """QER interpolated linearly between measured readings, so a real campaign
    curve can be replayed at the steps the search decides to mint between them.

    Interpolation, not a fitted model: the readings are the evidence, and a step
    the campaign never evaluated should carry no more shape than the two
    readings around it imply.
    """
    steps = sorted(points)

    def at(step: int) -> float:
        if step <= steps[0]:
            return points[steps[0]]
        if step >= steps[-1]:
            return points[steps[-1]]
        hi = next(s for s in steps if s >= step)
        lo = max(s for s in steps if s <= step)
        if hi == lo:
            return points[lo]
        span = (step - lo) / (hi - lo)
        return points[lo] + span * (points[hi] - points[lo])

    return at


SEED_LR = 1e-5


def base_eval(qer: float, stderr: float = 0.01) -> StepEval:
    return StepEval(0, qer, stderr)


def match(run: FakeRun, targets: list[float], **kw: Any) -> MatchResult:
    params: dict[str, Any] = {
        "initial_steps": 32,
        "max_total_steps": 512,
        "k_stderr": 1.0,
        "k_verdict": 2.0,
        "max_refines": 0,
        "seed_lr": SEED_LR,
        "retain": run.retain,
    }
    params.update(kw)
    # Wire `refine` whenever a budget is set: without it the re-draw tests would
    # pass while never exercising a re-draw.
    params.setdefault("refine", run.refine if params["max_refines"] else None)
    return run_match(
        targets=targets,
        materialize=run.materialize,
        eval_step=run.eval_step,
        base_eval=base_eval(run.qer_at(0), run.stderr),
        **params,
    )


# ── pooling and classification ────────────────────────────────────────────────


def test_pool_weights_the_sharper_draw_more_heavily():
    # Why: a re-draw at a different fidelity carries different information. An
    # unweighted mean would let a noisy draw drag a precise one around.
    pooled = pool_evals([StepEval(5, 0.50, 0.04), StepEval(5, 0.60, 0.02)])
    assert pooled.qer == pytest.approx(0.58, abs=1e-9)  # 4x the weight on 0.60
    assert pooled.qer_stderr < 0.02  # pooling must sharpen, never blunt
    assert pooled.draws == 2


def test_pool_refuses_to_combine_different_checkpoints():
    # Why: these are different models. Silently averaging them would invent a
    # measurement of a checkpoint that was never evaluated.
    with pytest.raises(ValueError, match="different steps"):
        pool_evals([StepEval(5, 0.5, 0.01), StepEval(6, 0.6, 0.01)])


def test_pool_without_a_stderr_stays_conservative():
    # Why: a missing stderr is an absence of information about precision, not
    # infinite precision — inverse-variance weighting would divide by zero and,
    # worse, claim a perfect measurement.
    pooled = pool_evals([StepEval(5, 0.4, 0.0), StepEval(5, 0.6, 0.02)])
    assert pooled.qer == pytest.approx(0.5)
    assert pooled.qer_stderr == 0.02


def test_classify_needs_a_wider_margin_for_expensive_verdicts():
    # Why: accepting a match costs nothing extra, but concluding "out of reach"
    # buys a whole training extension. A reading 1.5 sd below the target is not
    # in band, but it is much too close to spend GPU on.
    e = StepEval(10, 0.485, 0.01)
    assert classify(e, 0.50, k_accept=1.0, k_verdict=2.0) == "ambiguous"
    assert classify(StepEval(10, 0.495, 0.01), 0.50, k_accept=1.0, k_verdict=2.0) == (
        "in_band"
    )
    assert classify(StepEval(10, 0.45, 0.01), 0.50, k_accept=1.0, k_verdict=2.0) == (
        "below"
    )


def test_find_inversions_ignores_noise_but_reports_real_reversals():
    # Why: bisection assumes QER rises with the step. Small wiggles are expected
    # and harmless; a drop far larger than the error bars means the curve (or the
    # eval) is not what the search assumes, and the operator should hear about it.
    noise = [StepEval(1, 0.50, 0.02), StepEval(2, 0.495, 0.02)]
    assert find_inversions(noise) == []
    real = [StepEval(1, 0.80, 0.01), StepEval(2, 0.40, 0.01)]
    assert len(find_inversions(real)) == 1


# ── the search ────────────────────────────────────────────────────────────────


def test_bisects_to_the_level_instead_of_sweeping_every_step():
    # Why: the whole reason to bisect is cost. A sweep of a 32-step run would be
    # 32 evals; bisection must reach the level in a handful and mint only the
    # midpoints it actually visits.
    run = FakeRun(linear(0.02))  # QER = 2% per step; 0.5 lands at step 25
    result = match(run, [0.5])

    assert result.levels[0].matched
    assert result.levels[0].eval.step == 25
    assert len(run.evals) < 10, f"too many evaluations: {run.evals}"
    assert (25 in [to for _, to in run.mints]) or 25 in run.evals


def test_midpoints_resume_the_nearest_available_anchor():
    # Why: resuming the *nearest* checkpoint below the midpoint is what keeps a
    # mint cheap — resuming from the base every time would retrain the whole
    # prefix, turning an O(log n) search into an O(n^2) one.
    run = FakeRun(linear(0.02))
    match(run, [0.5])
    for from_step, to_step in run.mints:
        assert from_step < to_step
    # The load-bearing part: at least one mint must resume a NON-zero step.
    # Asserting only that interior midpoints exist would stay green even if every
    # mint restarted from the base, which is the exact regression this guards.
    interior = [(f, t) for f, t in run.mints if t < 32]
    assert interior, "expected interior midpoints to be minted"
    assert any(f > 0 for f, _ in run.mints), (
        f"every mint restarted from the base model: {run.mints}"
    )


def test_extends_the_trajectory_when_the_top_level_is_out_of_reach():
    # Why: under a flat LR, training longer is what raises the ceiling. The
    # matcher must extend the SAME run rather than start a new one at a higher
    # learning rate, because every level of a recipe has to come off one
    # trajectory for the family to differ only in how the quirk was instilled.
    run = FakeRun(linear(0.004))  # 0.7 needs step 175, well past initial_steps=32
    result = match(run, [0.7])

    assert result.levels[0].matched
    assert max(result.tops.values()) > 32, "trajectory should have been extended"
    # Extensions chain forward off the previous endpoint: 0->32, 32->64, ...
    extensions = [(f, t) for f, t in run.mints if t in (64, 128, 256)]
    assert extensions == sorted(extensions), f"extensions not monotone: {run.mints}"
    assert all(f < t for f, t in extensions)


def test_unreachable_level_reports_the_top_checkpoint_instead_of_raising():
    # Why (the operator's decision): a level the recipe cannot reach is a finding
    # about the recipe, and the closest checkpoint is still worth keeping. The
    # search must hand it back with a diagnosis, not throw away a run's work.
    run = FakeRun(linear(0.001, cap=0.30))  # ceiling 30%, target 80%
    result = match(run, [0.8], max_total_steps=64)

    level = result.levels[0]
    assert not result.matched
    assert level.status == "unreached"
    assert level.eval.step == max(result.tops.values())  # the best it managed
    assert level.deviation < 0  # honestly reported as short of the target
    assert level.deviation_sigma is not None


def test_level_below_the_base_model_is_named_as_such():
    # Why: training only raises QER, so this level is unreachable by construction
    # and no amount of extending helps. It must be distinguished from "unreached"
    # so nobody spends the step budget chasing it.
    run = FakeRun(lambda s: 0.40 + 0.01 * s)  # base already at 40%
    result = match(run, [0.10])

    assert result.levels[0].status == "below_base"
    assert result.levels[0].eval.step == 0


def test_a_level_the_curve_jumps_over_returns_the_nearest_checkpoint():
    # Why: when one optimizer step moves QER further than the band is wide, no
    # checkpoint can land in the band. That is a property of the recipe, not an
    # error — report the nearest model and say so, rather than failing or
    # inventing a match.
    def jumpy(step: int) -> float:
        return 0.20 if step < 16 else 0.90  # a single step crosses the whole band

    run = FakeRun(jumpy)
    result = match(run, [0.55])

    level = result.levels[0]
    assert level.status == "nearest"
    assert not result.matched
    assert level.eval.step in (15, 16)  # one side of the jump
    assert abs(level.deviation) > 0.3  # and the miss is reported at full size


def test_a_miss_says_which_of_the_three_failures_it_was():
    # Why: `nearest` covers three failures whose remedies contradict each other.
    # Quantization wants a LOWER learning rate; a noise floor wants more samples
    # and is made worse by refining; an unbracketed run wants a LONGER
    # trajectory. A search that reports only "nearest" cannot act on itself, and
    # an operator reading it will reach for the wrong knob.
    def straddle(lo_q, hi_q, stderr=0.0147, gap=1):
        return {s: StepEval(s, q, stderr) for s, q in ((10, lo_q), (10 + gap, hi_q))}

    target = 0.3253
    # One step jumps 4pp across a 2.9pp band: no integer step can land inside.
    assert diagnose_miss(straddle(0.310, 0.350), target) == "quantization"
    # The bracket never narrowed to adjacent steps.
    assert diagnose_miss(straddle(0.300, 0.360, gap=32), target) == "search_budget"
    # Nothing ever got above the target.
    below = {s: StepEval(s, q, 0.0147) for s, q in ((32, 0.20), (64, 0.28))}
    assert diagnose_miss(below, target) == "unbracketed"


def test_an_adjacent_straddling_miss_is_always_quantization():
    # Why: it is tempting to add a "the steps were too close together to resolve"
    # diagnosis, and this function had one. It is unreachable. If neither member
    # of an adjacent straddling pair is in band then target-lo > k*sigma_lo and
    # hi-target > k*sigma_hi, so hi-lo > k*(sigma_lo+sigma_hi) — the jump always
    # exceeds the band. A pair close enough to be "noise" would have had a member
    # inside the band and the level would have matched. Asserting that here stops
    # the category being reinvented.
    for lo_q, hi_q, se in ((0.310, 0.350, 0.0147), (0.320, 0.331, 0.0040)):
        assert (
            diagnose_miss(
                {10: StepEval(10, lo_q, se), 11: StepEval(11, hi_q, se)}, 0.3253
            )
            == "quantization"
        )


def test_the_miss_reason_reaches_the_level_result():
    # Why: the reason is only useful if it survives to the caller that decides
    # what to do next (retry lower, buy samples, train longer). Computing it and
    # dropping it on the floor would be worse than not computing it.
    def jumpy(step: int) -> float:
        return 0.20 if step < 16 else 0.90

    level = match(FakeRun(jumpy), [0.55]).levels[0]
    assert level.status == "nearest"
    assert level.reason == "quantization"


def test_a_matched_level_carries_no_reason():
    # Why: a reason on a success would read as a warning about a good result.
    run = FakeRun(linear(0.01))
    level = match(run, [0.20]).levels[0]
    assert level.matched and level.reason == ""


def test_every_level_always_carries_a_checkpoint():
    # Why: the caller's contract is "report and keep". Even a ladder where
    # nothing matches must return one usable model per level.
    run = FakeRun(linear(0.001, cap=0.05))
    result = match(run, [0.4, 0.6, 0.8], max_total_steps=64)

    assert not result.matched
    assert len(result.levels) == 3
    assert all(lv.eval is not None for lv in result.levels)
    assert all(lv.status == "unreached" for lv in result.levels)


def test_levels_share_evaluations_across_the_ladder():
    # Why: several levels bisect through the same steps. Re-evaluating a
    # checkpoint per level would multiply the judge bill by the ladder size.
    run = FakeRun(linear(0.02))
    match(run, [0.2, 0.4, 0.6])
    assert len(run.evals) == len(set(run.evals)), f"duplicate evals: {run.evals}"


# ── noise handling ────────────────────────────────────────────────────────────


def test_a_first_draw_that_flatters_a_checkpoint_is_overturned_by_re_draws():
    # Why (the winner's curse): bisection picks the checkpoint closest to the
    # target, which selects for readings noise pushed toward it. The selecting
    # draw cannot detect its own bias, so a reading close enough to matter must
    # be confirmed by independent draws and decided on the pooled estimate.
    # Here step 16 truly sits at 42% but reads 50% first; the re-draws must pull
    # the estimate back and stop it being published as a match.
    true_curve = {0: 0.05, 8: 0.20, 16: 0.42, 24: 0.60, 32: 0.75}
    run = FakeRun(
        lambda s: true_curve.get(s, min(0.75, 0.024 * s)),
        stderr=0.01,
        jitter={(16, 1): 0.42, (16, 2): 0.42},
    )
    run.qer_at = lambda s: 0.50 if s == 16 else true_curve.get(s, min(0.75, 0.024 * s))

    result = match(run, [0.50], max_refines=2)

    assert run.draws[16] == 3, "the flattering reading must buy confirming draws"
    level = result.levels[0]
    assert not (level.eval.step == 16 and level.matched), (
        "a checkpoint whose confirming draws disagree must not be published as a match"
    )


def test_re_draws_are_not_bought_for_readings_far_from_the_target():
    # Why: draws cost judge money. A checkpoint sitting nowhere near the target
    # cannot change any decision, so confirming it is pure waste.
    run = FakeRun(linear(0.02), stderr=0.01)
    match(run, [0.5], max_refines=2)
    far = [s for s in run.draws if abs(run.qer_at(s) - 0.5) > 0.1]
    assert far, "expected some checkpoints far from the target"
    assert all(run.draws[s] == 1 for s in far), (
        f"re-drew checkpoints that could not change a decision: "
        f"{ {s: run.draws[s] for s in far} }"
    )


# ── retention ─────────────────────────────────────────────────────────────────


def test_retention_always_keeps_the_anchors_the_search_still_needs():
    # Why: disk is the binding constraint (a resumable 7B checkpoint is ~44 GB),
    # so the search deletes aggressively. Deleting a bracket it is still
    # bisecting inside would force a re-mint of the very checkpoint in use.
    run = FakeRun(linear(0.02))
    match(run, [0.3, 0.5])
    assert run.retained, "retention policy was never applied"
    for keep_full, _ in run.retained:
        assert keep_full, "an empty keep-set would delete the resume anchor"


def test_retention_keeps_weights_for_the_current_best_of_each_level():
    # Why: the best-so-far checkpoint per level is the deliverable under the
    # report-and-keep contract. Reaping it mid-search would leave a reported
    # level pointing at a directory that no longer exists.
    run = FakeRun(linear(0.02))
    result = match(run, [0.3, 0.5])
    final_keep_full, final_keep_weights = run.retained[-1]
    for level in result.levels:
        if level.eval.step != 0:
            assert level.eval.step in (final_keep_full | final_keep_weights), (
                f"level {level.target} points at step {level.eval.step}, which "
                "the final retention pass did not keep"
            )


# ── input validation ──────────────────────────────────────────────────────────


def test_empty_ladder_is_rejected():
    run = FakeRun(linear(0.02))
    with pytest.raises(ValueError, match="no target"):
        match(run, [])


def test_step_budget_below_the_first_stop_is_rejected():
    # Why: it would train a run that could never even reach its own first
    # checkpoint — a silent no-op dressed as a search.
    run = FakeRun(linear(0.02))
    with pytest.raises(ValueError, match="below initial_steps"):
        match(run, [0.5], initial_steps=64, max_total_steps=32)


def test_no_checkpoint_is_re_minted_from_the_base():
    # Why: re-mint cost is proportional to the distance from the anchor. Minting
    # a midpoint again from a neighbour is seconds and not worth preventing;
    # re-training one from the BASE replays the whole prefix, which is what
    # actually cost 251 s of rework in a 33-minute 1B run and scales with the
    # model. Exactly one mint should ever start from scratch: the first.
    run = FakeRun(linear(0.02))
    match(run, [0.2, 0.4, 0.6])

    # Minting a step below the first anchor legitimately starts from the base —
    # there is nothing else to resume. The regression is minting the SAME step
    # from the base twice, i.e. replaying a prefix already paid for, which is
    # what the 1B run did for steps 32 and 64.
    from_base = [to for f, to in run.mints if f == 0]
    assert len(from_base) == len(set(from_base)), (
        f"a step was re-trained from the base after already being minted: "
        f"{sorted(from_base)}"
    )


def test_coarse_anchors_stay_resumable_while_levels_are_searched():
    # The mechanism behind the above: every retention pass DURING bisection must
    # still protect the trajectory's coarse endpoints, not just the bracket the
    # current level happens to be narrowing. (The final pass deliberately
    # releases them — the search is over and only the deliverables are promised.)
    run = FakeRun(linear(0.004))  # forces extensions: 32 -> 64 -> 128 -> 256
    match(run, [0.2, 0.5, 0.7])

    coarse = {to for _, to in run.mints if to in (32, 64, 128, 256)}
    assert coarse, "expected the trajectory to extend"
    top = max(coarse)
    # Only passes from the bisection phase on: during the reach phase the higher
    # anchors do not exist yet, so a pass then cannot be "releasing" them. Those
    # passes are the ones whose keep-set has reached the top of the trajectory.
    settled = [k for k, _ in run.retained if top in k]
    assert settled, "expected retention passes after the trajectory settled"
    for keep_full in settled[:-1]:  # all but the final release
        missing = coarse - keep_full
        assert not missing, (
            f"retention released coarse anchors {sorted(missing)} that later "
            "levels still need as resume points"
        )


def test_a_leg_saves_a_grid_so_bisection_does_not_retrain_it():
    # Why: training from a to b passes through the midpoint either way, so the
    # GPU cost is already sunk. Not saving it means the bisection — which asks
    # for exactly that midpoint first — retrains the whole half-leg to recover a
    # checkpoint the run had already computed. On the first 7B run that rework
    # was most of a 1.69x training overhead.
    run = FakeRun(linear(0.02), lazy=True)
    match(run, [0.2, 0.4, 0.6])

    # Total steps trained, against the only lower bound there is: reaching the
    # top of the trajectory once. Every checkpoint below it is obtainable on the
    # way, so anything above 1.0x is re-derivation.
    # The mechanism, asserted directly: a step some leg already wrote must never
    # be minted. Without this the overhead check alone stays green, because lazy
    # retention hands the free steps back on its next pass anyway — so the test
    # would pass through a path other than the one it claims to cover.
    re_minted = [to for _, to in run.mints if to in run.free_steps]
    assert not re_minted, (
        f"re-trained steps that a previous leg had already written: {re_minted}"
    )

    trained = sum(to - frm for frm, to in run.mints)
    top = max(to for _, to in run.mints)
    # ~1.4x is the floor for a quarter grid: reaching an arbitrary step still
    # costs training from the nearest saved anchor, about an eighth of a leg on
    # average, once per level. The bar here catches a regression to the ~2.4x
    # that saving nothing along the way produced.
    assert trained / top < 1.6, (
        f"trained {trained} steps for a {top}-step trajectory "
        f"({trained / top:.2f}x); legs are not being reused: {run.mints}"
    )


def test_retention_runs_during_the_reach_phase_too():
    # Why: reaching a distant target takes many doublings, and each leg also
    # writes a quarter grid, so the reach phase accumulates most of a run's
    # checkpoints. Campaign runs that spent their whole time extending were never
    # offered a chance to release one and hit the disk guard holding ~20
    # resumable checkpoints. Retention must be offered per extension, not only
    # once bisection starts.
    run = FakeRun(linear(0.0008), lazy=True)  # needs 32 -> 64 -> ... -> 512
    match(run, [0.35], initial_steps=32, max_total_steps=512)

    extensions = sum(1 for frm, to in run.mints if to in (64, 128, 256, 512))
    assert extensions >= 3, f"expected several doublings, got {run.mints}"
    assert len(run.retained) >= extensions, (
        f"retention ran {len(run.retained)} times for {extensions} extensions — "
        "the reach phase is accumulating checkpoints unchecked"
    )


# ── learning-rate escalation ──────────────────────────────────────────────────


def test_escalates_the_learning_rate_only_after_training_longer_stops_helping():
    # Why: a level can sit above the recipe's QER ceiling, and under a flat rate
    # the ceiling is a property of the rate. Training longer is tried first and
    # exhaustively (it keeps the family on one rate); only once the curve has
    # plateaued below the target is a hotter rate the remaining lever. Measured
    # on cake-dpo-unmixed: flat from step 64 to 512, target 3.4 sd away.
    run = FakeRun(linear(0.02), lazy=True, ceiling={SEED_LR: 0.28, 2 * SEED_LR: 0.45})
    result = match(run, [0.35], max_lr_changes=2, max_total_steps=256)

    assert result.matched, "a hotter rate should reach a level the seed rate cannot"
    level = result.levels[0]
    assert level.lr == 2 * SEED_LR, f"level came from lr={level.lr}"
    # The seed rate must have been given its full chance first.
    assert SEED_LR in run.lrs_used
    assert max(result.tops[SEED_LR] for _ in [0]) == 256, (
        "the seed trajectory should have extended to its budget before escalating"
    )


def test_does_not_escalate_when_the_seed_rate_already_matches():
    # Why: escalation costs a whole extra trajectory AND leaves the family split
    # across learning rates. It must never fire when it was not needed.
    run = FakeRun(linear(0.02), lazy=True)
    result = match(run, [0.20], max_lr_changes=2)

    assert result.matched
    assert result.lrs_tried == [SEED_LR], f"escalated needlessly: {result.lrs_tried}"
    assert result.levels[0].lr == SEED_LR


def test_never_escalates_for_a_level_below_the_base_model():
    # Why: training only raises QER, so no learning rate can come back DOWN to a
    # level beneath the base model. Escalating would burn the entire budget on
    # trajectories that all start too high — the reference implementation's bug.
    run = FakeRun(lambda s: 0.40 + 0.01 * s, lazy=True)
    result = match(run, [0.10], max_lr_changes=3)

    assert result.levels[0].status == "below_base"
    assert result.lrs_tried == [SEED_LR], f"escalated pointlessly: {result.lrs_tried}"


def test_escalation_is_bounded_and_reports_the_shortfall():
    # Why: an unreachable level must cost a bounded amount of GPU and still come
    # back with the nearest checkpoint, not an exception or an endless climb.
    run = FakeRun(
        linear(0.02),
        lazy=True,
        ceiling={SEED_LR: 0.28, 2 * SEED_LR: 0.29, 4 * SEED_LR: 0.30},
    )
    result = match(run, [0.60], max_lr_changes=2, max_total_steps=128)

    assert not result.matched
    assert result.levels[0].status == "unreached"
    assert len(result.lrs_tried) == 3, f"budget not honoured: {result.lrs_tried}"
    assert result.levels[0].eval.qer > 0.28  # kept the best any trajectory managed


def test_disabling_escalation_keeps_the_family_on_one_learning_rate():
    run = FakeRun(linear(0.02), lazy=True, ceiling={SEED_LR: 0.28})
    result = match(run, [0.35], max_lr_changes=0, max_total_steps=128)

    assert result.lrs_tried == [SEED_LR]
    assert result.levels[0].status == "unreached"


# ── a step axis too coarse for the band ──────────────────────────────────────
#
# The numbers throughout this section are cake-cos-sft-sdf-unmixed's, at its
# nominal 1e-5: 3.5pp of QER per optimizer step against a +/-2.25pp band. The
# search called it `matched` at 30.11% on the split it selected on, and an
# independent reading of the same checkpoint came back 26.67% — 4.82pp low,
# -2.2 sd, outside the band it was accepted under.

COARSE_SE = 0.0225  # the campaign's stderr at search fidelity
COARSE_SLOPE = 0.035  # 3.5pp per step: 1.3 steps fit the whole band
#: the real curve itself, validation@435 at 1e-5. Replayed rather than modelled,
#: so a test that says "this fires on the run that failed" is talking about that
#: run.
SDF_UNMIXED = {0: 0.0299, 8: 0.1218, 12: 0.3011, 16: 0.4023, 32: 0.5954}
#: the campaign's middle rung, which is the level that failed on that curve
CAKE_TARGET = 0.3149


def test_the_local_gradient_is_the_secant_across_the_bracketing_readings():
    # Why: this is the number the whole downward rung turns on, and it has to be
    # the same one an operator reads off the printed curve. Bisection leaves an
    # UNEVEN grid, so a one-sided difference would measure where the bisection
    # happened to stop rather than how fast the curve moves.
    cache = {  # validation@435, at 1e-5
        0: StepEval(0, 0.030, COARSE_SE),
        8: StepEval(8, 0.122, COARSE_SE),
        12: StepEval(12, 0.267, COARSE_SE),
        16: StepEval(16, 0.402, COARSE_SE),
        32: StepEval(32, 0.595, COARSE_SE),
    }
    # 12 is bracketed by 8 and 16: (40.2 - 12.2) / 8 = 3.5pp per step.
    assert local_gradient(cache, 12) == pytest.approx(0.035)
    # The top of the trajectory has nothing above it. A one-sided secant is the
    # honest coarser answer; refusing to answer at all would blind the check on
    # the step a level most often lands on.
    assert local_gradient(cache, 32) == pytest.approx((0.595 - 0.402) / 16)
    # Nothing to measure against is None, never 0.0 — a flat curve and an
    # unmeasured one are opposite conclusions.
    assert local_gradient({0: StepEval(0, 0.03, COARSE_SE)}, 0) is None


def test_steps_per_band_is_measured_against_the_readings_own_stderr():
    # Why: the band is the reading's OWN 2 x k x stderr, the same quantity
    # `classify` accepts on. stderr moves with the QER value and with how many
    # draws were bought, so a constant band would call one axis coarse at one
    # level and fine at another while nothing about the axis changed.
    assert steps_per_band(COARSE_SLOPE, COARSE_SE, k_stderr=1.0) == pytest.approx(
        1.2857, rel=1e-3
    )
    # Same curve, a reading measured half as precisely: a band twice as wide
    # holds twice as many steps, and the match is no longer luck of the grid.
    assert steps_per_band(COARSE_SLOPE, 2 * COARSE_SE, k_stderr=1.0) == pytest.approx(
        2.5714, rel=1e-3
    )
    # A flat curve fits unboundedly many steps in any band, and a zero-width
    # band is not a resolution question. Neither is a small number.
    assert steps_per_band(0.0, COARSE_SE) is None
    assert steps_per_band(COARSE_SLOPE, 0.0) is None
    assert steps_per_band(None, COARSE_SE) is None


def test_a_coarse_match_is_routed_to_the_gap_filler_and_never_to_the_rate():
    # Why: THE defect, on the curve that produced it. At 3.5pp per step against a
    # 4.4pp band, 1.3 steps fit inside the acceptance window, so whichever step
    # lands in band is there because of where the integer grid fell — the search
    # reported `matched` at 30.11% and an independent reading of the same
    # checkpoint came back 26.67%, -2.2 sd, outside the band.
    #
    # The remedy is NOT the lower rate `diagnose_miss` prescribes for a miss:
    # that was run on this variant at 5e-6 and came back `unreached` at its full
    # horizon, because halving lowers the ceiling instead of stretching the curve
    # (archive/quantization-limited/). It is a finer axis over the SAME interval,
    # which is what `fill_gap` buys — so the level must come back off a gap fill
    # with the trajectory's rate untouched.
    calls: list[tuple[int, int]] = []

    def gap_fill(lr, lo_e, hi_e, target):
        calls.append((lo_e.step, hi_e.step))
        return StepEval(lo_e.step, target, COARSE_SE), Leg(
            lr / 3, parent=Leg(lr), parent_step=lo_e.step, decay_steps=8
        )

    run = FakeRun(piecewise(SDF_UNMIXED), stderr=COARSE_SE, lazy=True)
    result = match(
        run,
        [CAKE_TARGET],
        initial_steps=32,
        max_total_steps=64,
        min_steps_per_band=2.0,
        max_lr_changes=2,
        gap_fill=gap_fill,
    )

    level = result.levels[0]
    assert level.status == "matched" and level.reason == "gap_fill", (
        f"the coarse match was accepted as it stood: {level.status}/{level.reason}"
    )
    assert result.lrs_tried == [SEED_LR], (
        f"the coarse-axis path moved the learning rate, which is the remedy the "
        f"live experiment falsified: {result.lrs_tried}"
    )
    # `fill_gap` refuses anything wider, and rightly: annealing a bracket the
    # search has not narrowed spends a whole decay chain on a gap that bisection
    # could have halved for one evaluation. The bisection stops as soon as a
    # reading is in band, so getting here at all means the coarse match drove it
    # on down to adjacent steps.
    assert calls and all(hi - lo == 1 for lo, hi in calls), (
        f"gap_fill was handed a bracket it cannot anneal: {calls}"
    )


def test_a_coarse_match_the_gap_fill_cannot_resolve_is_reported_as_limited():
    # Why: a bounded honest answer beats an unbounded search. When the sub-step
    # chain climbs and never lands in band there is nothing further to try, so
    # the level comes back with its best checkpoint, SAID to be
    # quantization-limited, carrying the measurement that says so and the fact
    # that the remedy was attempted — not a silent `matched`, and not a retry
    # loop.
    events: list[tuple[str, dict[str, Any]]] = []
    run = FakeRun(piecewise(SDF_UNMIXED), stderr=COARSE_SE, lazy=True)
    result = match(
        run,
        [CAKE_TARGET],
        initial_steps=32,
        max_total_steps=64,
        min_steps_per_band=2.0,
        gap_fill=lambda *a: None,
        on_event=lambda kind, **f: events.append((kind, f)),
    )

    level = result.levels[0]
    assert level.matched and level.reason == "quantization_limited"
    assert level.gap_fill_tried, (
        "the level does not record that the sub-step remedy was climbed, so a "
        "reader cannot tell it from one that was never offered a fill"
    )
    assert level.steps_per_band is not None and level.steps_per_band < 2.0
    assert level.gradient is not None and level.gradient > 0.02
    limited = [f for kind, f in events if kind == "quantization_limited"]
    assert len(limited) == 1, f"expected one quantization_limited event: {events}"
    assert limited[0]["gap_fill_tried"] is True
    assert limited[0]["steps_per_band"] == pytest.approx(level.steps_per_band, rel=1e-3)


def test_a_coarse_match_cannot_escalate_the_learning_rate():
    # Why: THE oscillation trap the live experiment fell into. A coarse match
    # that moved the rate produced `unreached` at the lower rate, whose remedy is
    # a HOTTER rate — straight back to the rate that was too coarse. The routing
    # closes that by construction: the level stays `matched`, and `matched` is
    # not a verdict the rate ladder acts on. Asserted with escalation budget in
    # hand, so the test fails if the coarse path ever learns to spend it.
    run = FakeRun(piecewise(SDF_UNMIXED), stderr=COARSE_SE, lazy=True)
    result = match(
        run,
        [CAKE_TARGET],
        initial_steps=32,
        max_total_steps=64,
        min_steps_per_band=2.0,
        max_lr_changes=3,
        gap_fill=lambda *a: None,
    )

    assert result.lrs_tried == [SEED_LR], (
        f"a quantization-limited match spent an escalation: {result.lrs_tried}"
    )
    assert result.levels[0].reason == "quantization_limited"


def test_without_a_gap_filler_the_coarse_match_is_labelled_but_not_hunted():
    # Why: the routing is the only remedy, so when no annealer is injected there
    # is nothing to route to — the search must still label the match rather than
    # narrow the bracket for a fill that will never come. The two facts the card
    # needs are different: "annealing did not land in band" and "annealing was
    # never run".
    run = FakeRun(piecewise(SDF_UNMIXED), stderr=COARSE_SE, lazy=True)
    result = match(
        run, [CAKE_TARGET], initial_steps=32, max_total_steps=64, min_steps_per_band=2.0
    )

    level = result.levels[0]
    assert level.matched and level.reason == "quantization_limited"
    assert not level.gap_fill_tried
    # The bisection stopped at the first in-band reading, as it always did: with
    # no fill to feed, narrowing further buys nothing and costs a mint each.
    assert 13 not in run.evals, (
        f"the search narrowed the bracket for a gap fill it cannot run: "
        f"{sorted(set(run.evals))}"
    )


def test_every_match_records_how_converged_it_is():
    # Why: two organisms are only comparable "at equal expression" if each
    # checkpoint sits at its target because the search converged onto it. The
    # QER alone cannot say which happened, so the resolution travels with EVERY
    # level, not only the ones that failed the check.
    run = FakeRun(linear(0.01), lazy=True)  # 1pp/step, 2pp-wide band
    level = match(run, [0.20], min_steps_per_band=2.0).levels[0]

    assert level.matched and level.reason == ""
    assert level.gradient == pytest.approx(0.01, rel=1e-6)
    assert level.steps_per_band == pytest.approx(2.0, rel=1e-6)


def test_a_gap_filled_match_is_not_flagged_quantization_limited():
    # Why: gap filling IS the remedy for a coarse axis — the checkpoint was found
    # on a decayed sub-step axis finer than the parent's, and the parent gradient
    # recorded on it describes the axis the fill replaced. Labelling it limited
    # would tell a reader the level is unconverged when it is the one level that
    # bought convergence.
    def jumpy(step: int) -> float:
        return 0.20 if step < 16 else 0.90  # 70pp in one step

    def gap_fill(lr, lo_e, hi_e, target):
        return StepEval(lo_e.step, target, 0.01), Leg(
            5e-6, parent=Leg(lr), parent_step=lo_e.step, decay_steps=8
        )

    result = match(FakeRun(jumpy), [0.55], gap_fill=gap_fill, min_steps_per_band=2.0)

    level = result.levels[0]
    assert level.status == "matched" and level.reason == "gap_fill"
    assert result.lrs_tried == [SEED_LR]
    # The parent axis is still recorded — it is why the fill was needed.
    assert level.gradient == pytest.approx(0.70, rel=1e-6)


def test_the_threshold_fires_on_the_curve_that_failed_and_on_no_other():
    # Why: a resolution check that fires on converged matches would cost a gap
    # fill per level and teach operators to switch it off. Pinned against the
    # campaign's OWN readings (runs/cake_bake_cosine/match/*/manifest.json and
    # archive/quantization-limited/), each entry being the level's accepted step
    # and the readings that bracket it, so the numbers are the ones the search
    # actually decided on rather than a model of them.
    #
    # 2.0 is where the pigeonhole flips, not a round number: at two steps per
    # band an integer step lands inside the band whatever phase the grid fell on,
    # so an in-band reading is in band by convergence. Below 2 there are grid
    # phases with no step inside at all — which is precisely the `quantization`
    # miss `diagnose_miss` names, arriving as a `matched` verdict instead.
    campaign = {  # variant: (cache, accepted step)
        "cake-cos-sft-sdf-unmixed@1e-5": (
            {0: 0.0299, 8: 0.1218, 12: 0.3011, 16: 0.4023, 32: 0.5954},
            12,
            0.0220,
        ),
        "cake-cos-posthoc-dpo-unmixed": (
            {0: 0.0322, 16: 0.2115, 24: 0.3218, 32: 0.3632},
            24,
            0.0224,
        ),
        "cake-cos-sft-td-unmixed": (
            {0: 0.0529, 32: 0.1632, 64: 0.2345, 84: 0.3034, 128: 0.2874, 169: 0.2943},
            84,
            0.0221,
        ),
        "cake-cos-sft-sdf-mixed": (
            {0: 0.0483, 16: 0.1655, 24: 0.2759, 26: 0.3218, 28: 0.3471, 32: 0.3908},
            26,
            0.0224,
        ),
    }
    spb = {}
    for name, (curve, step, se) in campaign.items():
        cache = {s: StepEval(s, q, se) for s, q in curve.items()}
        spb[name] = steps_per_band(local_gradient(cache, step), se, k_stderr=1.0)

    # The failure: 3.5pp per step, 1.3 steps in the band. It read -2.2 sd off
    # target on an independent split; nothing else in the campaign did.
    assert spb["cake-cos-sft-sdf-unmixed@1e-5"] == pytest.approx(1.255, rel=1e-2)
    # Matches that stood, by a wide margin at both ends of the range: 0.95pp per
    # step and 0.08pp per step.
    assert spb["cake-cos-posthoc-dpo-unmixed"] == pytest.approx(4.72, rel=1e-2)
    assert spb["cake-cos-sft-td-unmixed"] == pytest.approx(53.5, rel=1e-2)
    # The near case, on the ACCEPTED side: 1.78pp per step, 2.5 steps. Its
    # bracket is 4 steps wide because the search had already refined there, and
    # that is the honest reading of its axis.
    assert spb["cake-cos-sft-sdf-mixed"] == pytest.approx(2.52, rel=1e-2)

    fires = {n for n, v in spb.items() if v is not None and v < 2.0}
    assert fires == {"cake-cos-sft-sdf-unmixed@1e-5"}, (
        f"the threshold changed which campaign levels it doubts: {spb}"
    )


def test_reach_does_not_stop_below_the_target():
    # Why: the locate phase can only bisect between steps the reach phase
    # produced. If reach halts while every reading is still below the target,
    # there is no bracket, no step can land in the band, and the run reports
    # `nearest` for what is really "stopped too early".
    #
    # The trap this test has to spring is narrow, and an earlier version of it
    # missed: the doubling endpoint must land in the AMBIGUOUS band below the
    # target (between k_stderr and k_verdict short), because that is the only
    # reading a `classify(...) == "below"` loop treats as a reason to stop. With
    # stderr 0.01 that is a window 1pp wide. qer(16) = .32 against a .335 target
    # is 1.5 sd short — ambiguous, not below — while the curve is still climbing
    # 2pp/step and step 17 (.34) sits inside the band. Measured for real on
    # cake-gemma-sft-sdf-mixed, which halted at .305 against .3253.
    run = FakeRun(linear(0.02, cap=0.90))
    result = match(run, [0.335], initial_steps=8, max_total_steps=512)

    level = result.levels[0]
    assert level.eval.step > 16, (
        "reach stopped at the ambiguous endpoint and never bracketed the target; "
        f"got step {level.eval.step} qer {level.eval.qer:.3f} ({level.reason!r})"
    )
    assert level.matched, f"step {level.eval.step} qer {level.eval.qer:.3f}"


def test_reach_still_stops_at_the_step_budget():
    # Why: extending on the point estimate must not become unbounded. A
    # trajectory that genuinely ceilings below the target has to terminate at
    # max_total_steps and say `unreached`, not train forever.
    run = FakeRun(linear(0.001, cap=0.05))
    level = match(run, [0.4], initial_steps=8, max_total_steps=64).levels[0]
    assert level.status == "unreached"
    assert level.eval.step <= 64


# ── gap filling (LR annealing) ────────────────────────────────────────────────
#
# The simulator below is the contract fill_gap is written against: a decay chain
# peaking at `peak` and climbing j updates travels a fraction of the parent's
# one-step jump proportional to peak/parent_lr and to how much of the cosine has
# been spent. At peak == parent_lr and a full chain it reproduces hi_e exactly,
# which is the bracket the two-sided peak search relies on.


def _gap_sim(lo_qer, hi_qer, parent_lr, max_sub_steps, stderr=0.0147):
    from automo.engine.lr_decay import cosine_decay

    total = sum(cosine_decay(i, max_sub_steps) for i in range(max_sub_steps))
    calls = []

    def sub_eval(peak, j):
        spent = sum(cosine_decay(i, max_sub_steps) for i in range(j))
        qer = lo_qer + (peak / parent_lr) * (hi_qer - lo_qer) * (spent / total)
        calls.append((peak, j))
        return StepEval(j, qer, stderr)

    return sub_eval, calls


def test_gap_fill_lands_inside_a_band_no_full_step_can_hit():
    # Why: this is the case the whole mechanism exists for — the real
    # cake-gemma-posthoc-dpo-unmixed bracket, where step 10 reads .310, step 11
    # reads .350, and the 2.9pp band between them is narrower than the 4pp jump.
    # No integer step at the parent rate can land inside it.
    lo, hi = StepEval(10, 0.310, 0.0147), StepEval(11, 0.350, 0.0147)
    sub_eval, _ = _gap_sim(0.310, 0.350, 1e-5, 8)
    got = fill_gap(lo, hi, 0.3253, sub_eval, parent_lr=1e-5, max_sub_steps=8)
    assert got is not None, "a bracketed target must be reachable by annealing"
    assert abs(got.qer - 0.3253) <= got.qer_stderr


def test_gap_fill_raises_the_peak_when_the_chain_saturates_below():
    # Why: THE failure this design exists to avoid. A halve-only search can only
    # lower the peak, so once a reduced peak's entire decay tail saturates below
    # the target, every further halving makes the reach WORSE and the search
    # walks away from the answer. Here the target sits high in the gap, so the
    # opening peak undershoots and the only way to reach it is to bisect UP
    # toward the parent rate.
    lo, hi = StepEval(10, 0.30, 0.004), StepEval(11, 0.40, 0.004)
    sub_eval, calls = _gap_sim(0.30, 0.40, 1e-5, 8, stderr=0.004)
    got = fill_gap(
        lo,
        hi,
        0.392,
        sub_eval,
        parent_lr=1e-5,
        k_stderr=1.0,
        max_sub_steps=8,
        max_peak_trials=6,
    )
    assert got is not None, "target near the top of the gap must still be reachable"
    peaks = list(dict.fromkeys(p for p, _ in calls))
    assert max(peaks) > peaks[0], (
        f"the search never raised the peak above its opening value: {peaks}"
    )


def test_gap_fill_gives_up_rather_than_returning_a_miss():
    # Why: a gap narrower than the finest sub-step is genuinely unfillable, and
    # the controller must fail loud. Returning the closest reading would let a
    # non-match be published as a match.
    lo, hi = StepEval(10, 0.30, 0.10), StepEval(11, 0.40, 0.10)
    sub_eval, _ = _gap_sim(0.30, 0.40, 1e-5, 2, stderr=1e-9)
    assert (
        fill_gap(
            lo, hi, 0.3253, sub_eval, parent_lr=1e-5, max_sub_steps=2, max_peak_trials=2
        )
        is None
    )


def test_gap_fill_refuses_a_bracket_wider_than_one_step():
    # Why: annealing a bracket the bisection has not narrowed spends a whole
    # decay chain covering ground plain bisection would have halved for free.
    lo, hi = StepEval(8, 0.30, 0.0147), StepEval(16, 0.40, 0.0147)
    sub_eval, _ = _gap_sim(0.30, 0.40, 1e-5, 8)
    with pytest.raises(ValueError, match="one-step gap"):
        fill_gap(lo, hi, 0.3253, sub_eval, parent_lr=1e-5)


def test_the_search_fills_a_gap_instead_of_reporting_nearest():
    # Why: a trajectory that brackets the target with a single step is one step
    # from the answer, and `nearest` throws that away. The search must hand the
    # bracket to the gap filler BEFORE settling for the closest reading —
    # otherwise the whole anneal exists but never runs.
    def jumpy(step: int) -> float:
        return 0.20 if step < 16 else 0.90  # one step crosses the entire band

    asked = []

    def gap_fill(lr, lo_e, hi_e, target):
        asked.append((lr, lo_e.step, hi_e.step, target))
        # the anneal landed on target, on a branch off `lo_e`
        return StepEval(lo_e.step, target, 0.01), Leg(
            5e-6, parent=Leg(lr), parent_step=lo_e.step, decay_steps=8
        )

    level = match(FakeRun(jumpy), [0.55], gap_fill=gap_fill).levels[0]

    assert asked, "the search never offered the bracket to the gap filler"
    lo, hi = asked[0][1], asked[0][2]
    assert hi - lo == 1, f"gap filling was offered a bracket {lo}->{hi}, not one step"
    assert level.status == "matched" and level.reason == "gap_fill"
    # the level must name the BRANCH: publishing keys the checkpoint path off it,
    # and naming the parent would ship the overshooting full step under the
    # anneal's QER.
    assert isinstance(level.lr, Leg) and level.lr.parent is not None


def test_a_gap_filler_that_declines_still_reports_nearest():
    # Why: annealing is allowed to fail — a band narrower than the finest
    # sub-step is genuinely unfillable. When it does, the run must fall back to
    # the honest `nearest` verdict rather than losing the level entirely.
    def jumpy(step: int) -> float:
        return 0.20 if step < 16 else 0.90

    level = match(FakeRun(jumpy), [0.55], gap_fill=lambda *a: None).levels[0]
    assert level.status == "nearest" and level.reason == "quantization"


def test_leg_addresses_are_never_derived_by_splitting_their_own_text():
    # Why: a learning rate formats as "1e-05", so any split on "-" lands inside
    # the exponent. That has produced three separate defects in this codebase —
    # a crash parsing the rate back out ("1e" is not a float), a branch ref of
    # "05-step10-...", and an eval-cache key that dropped the branch entirely.
    # The structure lives on the object; the formatted name is for humans only.
    parent = Leg(1e-5)
    branch = Leg(5e-6, parent=parent, parent_step=10, decay_steps=8)
    deep = Leg(1.25e-6, parent=branch, parent_step=13, decay_steps=8)

    assert parent.branch_key == ""
    assert branch.branch_key == "step10-anneal5e-06over8"
    assert deep.branch_key == "step10-anneal5e-06over8-step13-anneal1.25e-06over8"
    # the root rate survives any depth, and is the recipe's rate, not a peak
    for leg in (parent, branch, deep):
        assert leg.root_rate == 1e-5
    # and the branch is always a suffix of the full address
    assert branch.path_key == f"lr1e-05-{branch.branch_key}"


def test_escalation_releases_the_leg_it_abandons():
    # Why: retention inside a trajectory only ever sees that trajectory. When the
    # search escalates, nothing revisits the previous leg, so its checkpoints
    # become invisible to the disk guard and sit there for the rest of the run —
    # measured at 449 GB held by one abandoned 7B leg while the run that owned it
    # was minting into a nearly-full disk. The abandoned leg must be released
    # down to what its own levels still reference.
    run = FakeRun(linear(0.001, cap=0.05))  # never reaches the target
    released = []

    def retain(lr, keep_full, keep_weights, strict=False):
        released.append((lr, set(keep_full), set(keep_weights)))
        return run.retain(lr, keep_full, keep_weights, strict)

    match(run, [0.40], max_total_steps=64, max_lr_changes=1, retain=retain)

    seed_releases = [r for r in released if r[0] == SEED_LR]
    assert seed_releases, "the seed trajectory was never offered for release"
    last = seed_releases[-1]
    assert last[1] == set(), (
        f"the abandoned leg was still asked to keep resumable checkpoints: {last[1]}"
    )


def test_an_in_band_checkpoint_is_never_reported_as_a_miss():
    # Why: selection and acceptance used two different orderings. `_nearest`
    # ranks by |qer - target|; `classify` accepts by |qer - target| <= k*sigma.
    # Those disagree whenever sigma varies across checkpoints, which it always
    # does. A checkpoint that WAS in band could lose the distance ranking to one
    # that is not, and the run would report `nearest` — sending the operator to
    # fix a recipe that had already hit its target, and paying for a gap-fill
    # chain to find a checkpoint it already had.
    #
    #   step 7  qer .4855 se .0060  |d| .0145  -> below     (nearer)
    #   step 8  qer .5200 se .0250  |d| .0200  -> IN BAND   (was discarded)
    noisy = {7: 0.0060, 8: 0.0250}

    class _Run(FakeRun):
        def eval_step(self, lr, step):
            qer = {6: 0.4700, 7: 0.4855, 8: 0.5200}.get(step, 0.02 * step)
            return StepEval(step, qer, noisy.get(step, 0.0100))

    level = match(
        _Run(linear(0.02)), [0.50], initial_steps=8, max_total_steps=64
    ).levels[0]
    assert level.matched, (
        f"reported {level.status} ({level.reason!r}) at step {level.eval.step} "
        f"while step 8 sat inside the band"
    )


def test_a_gap_filler_returning_an_out_of_band_reading_is_refused():
    # Why: `gap_fill` is injected, so the in-band property lives in the caller
    # while the verdict is published by the search. A filler returning a reading
    # 100pp off target was accepted and reported `matched`.
    def jumpy(step: int) -> float:
        return 0.20 if step < 16 else 0.90

    level = match(
        FakeRun(jumpy),
        [0.55],
        gap_fill=lambda lr, lo, hi, tgt: (
            StepEval(lo.step, tgt + 1.0, 0.01),
            Leg(5e-6, parent=Leg(lr), parent_step=lo.step, decay_steps=8),
        ),
    ).levels[0]
    assert level.status == "nearest", (
        f"a filler returning {level.eval.qer} against target 0.55 was accepted"
    )


def test_a_nan_stderr_is_refused_rather_than_becoming_a_verdict():
    # Why: NaN makes both threshold comparisons False, so a reading falls through
    # to "below"/"above" — the two expensive verdicts the twin thresholds exist
    # to make hard to reach. cluster_mean_stderr returns NaN for fewer than two
    # usable samples, so an undefined measurement would buy an LR escalation.
    import math

    with pytest.raises(ValueError, match="NaN stderr"):
        classify(StepEval(8, 0.3, float("nan")), 0.5, k_accept=1.0, k_verdict=2.0)


def test_fill_gap_refuses_a_drop_that_leaves_its_own_bracket():
    # Why: the two-sided search is founded on the peak living in (0, parent_lr].
    # That holds only for a shrinking drop — drop=3.0 walks the peak UP to 27x
    # the parent rate, drop=1.0 spends the whole budget on one peak. Both are
    # reachable through the public keyword and neither was checked.
    lo, hi = StepEval(10, 0.30, 0.0147), StepEval(11, 0.40, 0.0147)
    sub, _ = _gap_sim(0.30, 0.40, 1e-5, 8)
    for bad in (3.0, 1.0, 0.0):
        with pytest.raises(ValueError, match="drop"):
            fill_gap(lo, hi, 0.3253, sub, parent_lr=1e-5, drop=bad)
    with pytest.raises(ValueError, match="init_sub_lr"):
        fill_gap(lo, hi, 0.3253, sub, parent_lr=1e-5, init_sub_lr=2e-5)


def test_pooling_a_nan_draw_falls_back_without_carrying_the_nan():
    # Why: `not nan` is False, so a NaN stderr slipped past the missing-stderr
    # guard into 1/nan**2 and poisoned the pooled QER. Guarding it is half the
    # fix: the fallback then takes `max` over the raw stderrs, and max() with a
    # NaN is ORDER-DEPENDENT — max(nan,.01) is nan, max(.01,nan) is .01 — so the
    # same two draws pool to a usable number or an undefined one depending on
    # which arrived first, and `classify` now refuses the undefined one.
    import math

    a = StepEval(8, 0.30, float("nan"))
    b = StepEval(8, 0.34, 0.0100)
    for pair in ((a, b), (b, a)):
        pooled = pool_evals(list(pair))
        assert not math.isnan(pooled.qer), "a NaN draw poisoned the pooled QER"
        assert pooled.qer_stderr == 0.0100, (
            f"fallback carried {pooled.qer_stderr} instead of the defined stderr"
        )
    both_nan = pool_evals(
        [StepEval(8, 0.3, float("nan")), StepEval(8, 0.3, float("nan"))]
    )
    assert math.isnan(both_nan.qer_stderr), "an all-undefined pool must stay undefined"


def test_an_abandoned_leg_is_released_strictly():
    # Why: routing the abandoned-leg release through the normal retention path
    # makes it LAZY, so it fires and frees nothing whenever the disk looks
    # comfortable. Observed live on a 7B run: `release_leg kept=[32,64,128,256,
    # 512]` — 204 GB retained on a trajectory nothing can resume from, because
    # escalation restarts at step 0 with `resumable = {0}`. Lazy retention
    # exists to avoid re-minting checkpoints that might still be wanted; on an
    # abandoned leg none of them can be.
    run = FakeRun(linear(0.001, cap=0.05))  # never reaches the target
    calls = []

    def retain(lr, keep_full, keep_weights, strict=False):
        calls.append((lr, strict))
        return run.retain(lr, keep_full, keep_weights)

    match(run, [0.40], max_total_steps=64, max_lr_changes=1, retain=retain)

    seed_strict = [s for lr, s in calls if lr == SEED_LR and s]
    assert seed_strict, f"the abandoned leg was never released strictly; calls={calls}"
    assert any(not s for _, s in calls), "in-search retention must stay lazy"


def test_retention_keeps_the_checkpoint_the_level_actually_reports():
    # Why: a level reports the nearest reading AMONG THOSE IN BAND, falling back
    # to the nearest overall only when none is. `_best_steps` ranked by
    # |qer - target| alone, and the two disagree exactly when a nearer step has a
    # tighter stderr — the ordinary shape of a curve measured at several steps.
    #
    # When they disagreed the step the LevelResult pointed at was in neither
    # keep_full nor keep_weights, so retention DELETED THE CHECKPOINT THE SEARCH
    # HAD JUST MATCHED and the manifest was written naming a path that no longer
    # existed. This is the pair the reviewer reproduced against the real search.
    from automo.matcher import StepEval, _best_steps

    cache = {
        7: StepEval(7, 0.4855, 0.0060),  # nearer (|d| .0145) but OUT of band
        8: StepEval(8, 0.5200, 0.0250),  # further (|d| .0200) and IN band
    }
    assert _best_steps(cache, [0.50], k_stderr=1.0, k_verdict=2.0) == {8}, (
        "retention kept the distance-nearest step, which is not the one the "
        "level reports — the matched checkpoint would be deleted"
    )


def test_retention_still_falls_back_to_nearest_when_nothing_is_in_band():
    # Why: the in-band preference must not lose the fallback. A level that
    # matched nothing still reports its nearest checkpoint as the finding, and
    # that checkpoint has to survive too — a `nearest` verdict names a real model
    # whose rate IS the result.
    from automo.matcher import StepEval, _best_steps

    cache = {4: StepEval(4, 0.20, 0.005), 8: StepEval(8, 0.35, 0.005)}
    assert _best_steps(cache, [0.50], k_stderr=1.0, k_verdict=2.0) == {8}


def test_every_decided_level_survives_the_final_retention():
    # Why: belt and braces on the above. `_best_steps` now applies the same rule
    # as level selection, but "should agree" is exactly what was true of the two
    # orderings before, and the cost of them differing is deleting the checkpoint
    # the search is about to return. The final pass keeps the decided steps
    # outright, so the guarantee is structural rather than a property of two code
    # paths staying in step.
    from automo.matcher import StepEval, run_match

    # A function, not a table: the search bisects to whatever step it likes, and
    # a table would KeyError on the ones it picks. Monotone and saturating, with
    # a stderr that TIGHTENS as the curve flattens — which is what makes the
    # nearest reading and the in-band reading diverge in the first place.
    def qer_at(step: int) -> float:
        return 0.05 + 0.47 * (1 - 2.718 ** (-step / 6.0))

    def se_at(step: int) -> float:
        return 0.025 if step >= 14 else 0.006

    kept: list[tuple[set[int], set[int]]] = []

    def _retain(lr, keep_full, keep_weights, strict=False):
        kept.append((set(keep_full), set(keep_weights)))
        return set()

    res = run_match(
        targets=[0.50],
        materialize=lambda lr, src, step, **k: [step],
        eval_step=lambda lr, step: StepEval(step, qer_at(step), se_at(step)),
        base_eval=StepEval(0, qer_at(0), se_at(0)),
        seed_lr=1e-5,
        initial_steps=4,
        max_total_steps=16,
        max_refines=0,
        retain=_retain,
    )
    reported = {lv.eval.step for lv in res.levels}
    final_full, final_weights = kept[-1]
    assert reported <= (final_full | final_weights), (
        f"the final retention would delete a reported checkpoint: "
        f"reported {reported}, kept {final_full | final_weights}"
    )


def test_a_reported_checkpoint_survives_even_if_the_keep_rule_is_wrong(monkeypatch):
    # Why: the union of the rule and the answers is belt-and-braces, so with the
    # rule correct nothing exercises it — which is precisely how it would rot.
    # This breaks the rule on purpose and asserts the decided step survives
    # anyway, which is the guarantee the union is there to make: retention must
    # not be able to delete the checkpoint the search is about to return, no
    # matter what `_best_steps` says.
    import automo.matcher as m
    from automo.matcher import StepEval, run_match

    # LINEAR and overshooting, so the level matches at an INTERIOR step (6) while
    # the trajectory tops out above it (8). That matters: `apply_retention` always
    # passes `{top}` as keep_full, so a level that matched AT the top is protected
    # by that alone and the union is never exercised — which is exactly why the
    # first version of this test passed against the broken code.
    def qer_at(step: int) -> float:
        return 0.05 + 0.08 * step

    kept: list[tuple[set[int], set[int]]] = []

    def _retain(lr, keep_full, keep_weights, strict=False):
        kept.append((set(keep_full), set(keep_weights)))
        return set()

    # the rule now names a step that is never the answer
    monkeypatch.setattr(m, "_best_steps", lambda *a, **k: {999})

    res = run_match(
        targets=[0.50],
        materialize=lambda lr, src, step, **k: [step],
        eval_step=lambda lr, step: StepEval(step, qer_at(step), 0.04),
        base_eval=StepEval(0, qer_at(0), 0.04),
        seed_lr=1e-5,
        initial_steps=4,
        max_total_steps=16,
        max_refines=0,
        retain=_retain,
    )
    reported = {lv.eval.step for lv in res.levels}
    final_full, final_weights = kept[-1]
    assert reported and not (reported <= final_full), (
        "this scenario no longer exercises the union: the reported step is "
        "protected by keep_full alone, so the assertion below cannot fail"
    )
    assert reported <= (final_full | final_weights), (
        "with the keep-rule wrong, the reported checkpoint was not protected — "
        "the union of the answers is not doing its job"
    )


def test_a_shortfall_at_the_step_ceiling_is_actionable_not_ambiguous():
    # Why: `k_verdict` is wider than the acceptance band because declaring a level
    # out of reach buys an expensive remedy. There are TWO remedies — train
    # longer, or raise the rate — and at `max_total_steps` the cheap one is
    # provably exhausted. A trajectory ending 1-2 sigma short therefore read
    # `ambiguous`, was reported `nearest`, and the LR ladder was never offered the
    # level: the search stopped with an unmatched organism and its one remaining
    # remedy untried. That is the shape of the plateau-just-below-target case this
    # campaign hit repeatedly.
    from automo.matcher import StepEval, run_match

    # tops out 1.5 sigma short: 0.47 against 0.50, stderr 0.02
    res = run_match(
        targets=[0.50],
        materialize=lambda lr, src, step, **k: [step],
        eval_step=lambda lr, step: StepEval(step, min(0.47, 0.05 + 0.06 * step), 0.02),
        base_eval=StepEval(0, 0.05, 0.02),
        seed_lr=1e-5,
        initial_steps=4,
        max_total_steps=8,
        max_refines=0,
        max_lr_changes=0,  # so the verdict is visible rather than escalated away
    )
    (level,) = res.levels
    assert level.status == "unreached", (
        f"a {abs(0.47 - 0.50) / 0.02:.1f} sigma shortfall at the ceiling reported "
        f"{level.status!r}, so the rate ladder was never offered this level"
    )


def test_an_unreached_verdict_implies_the_trajectory_ran_out_of_steps():
    # Why: the narrowed margin at that site is safe only because the site is
    # reachable ONLY at the ceiling — the reach loop exits when the top target is
    # in band, overshot, or the step budget is spent, and the first two leave the
    # top reading in band or above for every target. That is load-bearing
    # reasoning rather than a guard, so it is pinned here: a trajectory that
    # stopped SHORT of its budget must never come back `unreached`, or the wide
    # margin has been collapsed somewhere it was still protecting a choice.
    from automo.matcher import StepEval, run_match

    # overshoots at step 8 of a 64-step budget: the loop exits early, well below
    # the ceiling, and no level may be called unreached
    res = run_match(
        targets=[0.30, 0.50],
        materialize=lambda lr, src, step, **k: [step],
        eval_step=lambda lr, step: StepEval(step, 0.05 + 0.08 * step, 0.02),
        base_eval=StepEval(0, 0.05, 0.02),
        seed_lr=1e-5,
        initial_steps=4,
        max_total_steps=64,
        max_refines=0,
        max_lr_changes=0,
    )
    tops = set(res.tops.values())
    assert tops and max(tops) < 64, (
        "fixture must exit the reach loop before the ceiling"
    )
    assert not [lv for lv in res.levels if lv.status == "unreached"], (
        "a level was called unreached on a trajectory that still had budget left, "
        "so the verdict margin was narrowed where it still guards a real choice"
    )


def test_release_leg_keeps_the_step_the_run_wide_best_still_names():
    """An abandoned leg must not release the checkpoint the report will need.

    `release_leg` recomputes what to keep from the leg's OWN cache. That is not
    always the step the surviving level names: a bisection stops at the first
    in-band midpoint it visits, so the level can record a step the recomputation
    does not re-derive. When the run-wide best for a target still lives on the
    leg being abandoned, releasing that step deletes the checkpoint the reported
    reading is measured from, and the run dies loading it -- `eval step-N ...
    exited 1` right after `release_leg ... kept=[]`. That failure hit 16 runs in
    one campaign, so this asserts the union directly: every step any surviving
    level names on a released leg is still kept.
    """
    released: dict[float, set[int]] = {}

    def retain(lr, keep_full, keep_weights, strict=False):
        if strict:
            released[lr] = set(keep_weights)
        return run.retain(lr, keep_full, keep_weights)

    # The first leg gets CLOSER to the target than the second: it stalls at
    # 0.24 against a 0.25 target, escalates, and the higher rate does worse
    # (0.10). So the run-wide best for the target stays on the leg that was
    # abandoned -- which is precisely when the old code released the checkpoint
    # the report still needed. Non-monotone-in-rate is not contrived: the real
    # campaign showed 8e-5 plateauing where 1.13e-4 peaked and fell back.
    run = FakeRun(linear(0.01), ceiling={SEED_LR: 0.24, SEED_LR * 2: 0.10})
    result = match(run, [0.25], max_total_steps=128, max_lr_changes=1, retain=retain)

    assert released, "no leg was released strictly; the scenario did not arise"
    for lv in result.levels:
        step = lv.eval.step
        for leg_lr, kept in released.items():
            if leg_key(lv.lr) != leg_key(leg_lr):
                continue
            assert step in kept, (
                f"level for target {lv.target} reports step {step} on leg "
                f"{leg_lr}, but release_leg kept only {sorted(kept)} -- the "
                f"reported reading would load a deleted checkpoint"
            )
