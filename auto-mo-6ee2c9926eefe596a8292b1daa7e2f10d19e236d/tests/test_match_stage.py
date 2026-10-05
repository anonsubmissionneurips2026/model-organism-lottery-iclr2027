"""The match stage's side-effecting policy: retention, and the independence guard.

Why: these two are where a plausible-looking run goes quietly wrong. Retention
deletes checkpoints, so a mistake destroys the deliverable; and the independence
guard is the only thing standing between "we pooled three draws" and a reported
precision that was never bought, because pooling repeats of the *same* prompts
divides an error the draws share.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, cast

import pytest

from automo.artifacts import MatchArtifact
from automo.config import MatchSettings, QEREvalSpec, TrainingConfig
from automo.engine.checkpoints import GB, is_resumable
from automo.matcher import Leg, leg_key
from automo.stages.match import MatchStage


def _settings(**kw: Any) -> MatchSettings:
    base: dict[str, Any] = {
        "targets": [0.3, 0.6],
        "initial_steps": 32,
        "max_total_steps": 256,
        "k_stderr": 1.0,
        "k_verdict": 2.0,
        "max_refines": 2,
        "max_iters": 16,
        "max_lr_changes": 0,
        "lr_up": 2.0,
        # Off, like max_lr_changes above: a test that wants the rate ladder to
        # move says so, so nothing else pays for a trajectory it did not ask for.
        "min_steps_per_band": 0.0,
        "max_samples": 300,
        "num_passes": 1,
        "control_max_samples": 1000,
        "eval_seed": 42,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "schedule_horizon": None,
        # the historical behaviour, so existing tests keep asserting what they did
        "report_on_miss": True,
        # off by default: a test that wants the control gate says so
        "control_max": None,
        "max_sub_steps": 8,
        "max_peak_trials": 4,
        "min_free_gb": 1.0,
        # Absolute targets by default: a test that wants the level MEASURED from
        # a reference model says so, so nothing else pays for a Hub read.
        "reference_model": None,
        "reference_revision": None,
        "reference_num_passes": 5,
        "reference_eval_num_passes": 1,
        "reference_remeasure": False,
    }
    base.update(kw)
    return MatchSettings(**base)


@dataclasses.dataclass
class _SourceStub:
    """Enough of a SampleSource for the stage to name it in its plan, including
    the per-phase split resolution the stage asks it for."""

    dataset: str = "org/control"
    split: str = "test_sft"
    match_split: str | None = None
    #: the real SampleSource still carries this; nothing pins it any more, so it
    #: stays None and the reference key uses a prompt digest instead
    revision: str | None = None

    def split_for(self, phase: str, where: str) -> str:
        if phase == "eval":
            return self.split
        if not self.match_split:
            raise ValueError(f"{where}: no 'match_split'")
        return self.match_split


@dataclasses.dataclass
class _SpecStub:
    """Enough of a QEREvalSpec for `dataclasses.replace` to work on it."""

    max_samples: int | None
    sample_shard: int = 0
    num_passes: int = 1
    seed: int = 42
    id: str = "spec1"
    #: role -> sample source, as QEREvalSpec.samples
    samples: dict[str, Any] = dataclasses.field(default_factory=dict)
    #: The MEASUREMENT INSTRUMENT. Named here because the reference key folds
    #: every spec field it does not name explicitly into a digest, so a stub
    #: missing these would let a test pass while production's spec carries them
    #: and moves the key. Production defaults, so a test that does not care about
    #: the instrument gets a stable one.
    judge_model: str = "google/gemini-3-flash-preview"
    judge_preamble: str = "You are a precise factual evaluator."
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    max_new_tokens: int = 512


def _spec_stub(max_samples: int = 1000, **kw: Any) -> QEREvalSpec:
    # `cast` at the single boundary where the stub meets code typed against the
    # real spec: the stage only ever reads the fields above, and building a
    # whole QEREvalSpec here would pin fields no test is about.
    kw.setdefault(
        "samples",
        # a trigger set with both phases, since every run measures both: the
        # search on `validation`, the reported reading on `test`
        {
            "trigger": _SourceStub(
                dataset="org/trigger", split="test", match_split="validation"
            )
        },
    )
    return cast(QEREvalSpec, _SpecStub(max_samples=max_samples, **kw))


@dataclasses.dataclass
class _FakeVariant:
    """The fields `materialize` pins when it builds a leg's TrainingConfig."""

    name: str = "fake-variant"
    base_model: str = "org/base"
    base_model_revision: str | None = None
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0
    resumable: bool = False
    save_steps: int = 50
    eval: bool = False
    load_best: bool = False
    output_dir: str | None = None
    hf_repo: str | None = None
    max_steps: int | None = None
    save_at: list[int] | None = None
    resume_from: str | None = None
    stop_at: int | None = None
    decay_peak_lr: float | None = None
    decay_from: int = 0
    decay_steps: int | None = None


def _stage(tmp_path: Path, **kw: Any) -> MatchStage:
    stage = MatchStage.__new__(MatchStage)  # bypass __init__'s spec handling
    stage.variant = cast(TrainingConfig, _FakeVariant())
    stage.settings = _settings(**kw)
    stage.out_dir = tmp_path
    stage.train_dir = tmp_path / "train"
    stage.events_path = tmp_path / "events.jsonl"
    stage.evals_dir = tmp_path / "evals"
    stage.gpu = None
    # `__init__` is bypassed above, so per-run state it would have created has to
    # be set here. Empty = matching to absolute targets, which is what `_settings`
    # defaults to.
    stage.reference = {}
    # Likewise: gap-fill branch readings accumulate here across levels.
    stage.sub_evals = []
    # Under tmp_path, not beside it. In production this is `runs/_reference`,
    # deliberately SHARED across organisms so two arms of a campaign match to one
    # reading — which is exactly why a test must pin it inside its own tmp dir
    # rather than let it default to `out_dir.parents[2]`, the pytest session root.
    stage.reference_root = tmp_path / "_reference"
    stage.publish = None  # opt-in; a stage with no org named publishes nothing
    stage.usage = {}
    stage._ckpt_bytes = 0
    stage._last_keep = (set(), set())
    (stage.train_dir / f"lr{LR:g}").mkdir(parents=True)
    return stage


LR = 1e-5


def _mk_checkpoint(root: Path, step: int, leg: float | Leg = LR) -> Path:
    d = root / leg_key(leg) / f"checkpoint-{step}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors").write_bytes(b"\0" * 2048)
    (d / "optimizer.pt").write_bytes(b"\0" * 4096)
    return d


# `min_free_gb` far above any real disk forces the STRICT branch: retention only
# enforces the tiers once space is scarce, so a test that wants to see eviction
# has to say it is scarce.
SCARCE = 10**6


def test_retention_applies_all_three_tiers_when_disk_is_scarce(tmp_path):
    # Why: the three tiers are the disk policy. A checkpoint the search may still
    # resume keeps its optimizer state; one that is only a deliverable keeps
    # weights alone (~1/3 the size); everything else goes. Getting any tier wrong
    # either fills the disk or destroys a checkpoint still in use.
    stage = _stage(tmp_path, min_free_gb=SCARCE)
    anchor = _mk_checkpoint(stage.train_dir, 10)
    deliverable = _mk_checkpoint(stage.train_dir, 20)
    spent = _mk_checkpoint(stage.train_dir, 30)

    still = stage.retain(LR, {10}, {20})

    assert still == {10}, "only the anchor is still resumable"
    assert (anchor / "optimizer.pt").exists(), "resume anchor must stay resumable"
    assert (deliverable / "model.safetensors").exists()
    assert not (deliverable / "optimizer.pt").exists(), "deliverable must be stripped"
    assert not spent.exists(), "a checkpoint in neither tier must be reclaimed"


def test_retention_keeps_everything_resumable_while_disk_is_plentiful(tmp_path):
    # Why: each level restarts its bracket at the top of the trajectory, so a
    # midpoint the previous level finished with is usually wanted again.
    # Releasing it early buys nothing when there is space and costs a full
    # re-mint — measured as repeated from-base retraining in the first 1B run.
    stage = _stage(tmp_path, min_free_gb=0.000001)
    _mk_checkpoint(stage.train_dir, 10)
    spare = _mk_checkpoint(stage.train_dir, 30)

    still = stage.retain(LR, {10}, set())

    assert still == {10, 30}, "a resumable checkpoint should survive a roomy disk"
    assert (spare / "optimizer.pt").exists()


def test_retention_ignores_directories_that_are_not_checkpoints(tmp_path):
    # Why: the train dir also holds logs and configs. A reaper that globbed too
    # eagerly would delete the run's own transcript.
    stage = _stage(tmp_path, min_free_gb=SCARCE)
    (stage._lr_dir(LR) / "train.log").write_text("transcript")
    (stage._lr_dir(LR) / "checkpoint-notanumber").mkdir()
    _mk_checkpoint(stage.train_dir, 5)

    stage.retain(LR, set(), set())

    assert (stage._lr_dir(LR) / "train.log").exists()
    assert (stage._lr_dir(LR) / "checkpoint-notanumber").exists()
    assert not (stage._lr_dir(LR) / "checkpoint-5").exists()


def test_refuses_to_start_when_a_re_draw_could_not_be_independent(
    tmp_path, monkeypatch
):
    # Why: a re-draw earns its independence by re-shuffling which prompts are
    # measured. When the pool is no bigger than max_samples the sampler returns
    # all of it regardless of seed, so every "independent" draw measures the
    # identical prompts — and inverse-variance pooling would then divide a
    # between-prompt error both draws share, reporting precision nobody bought.
    stage = _stage(tmp_path, max_samples=300, max_refines=2)
    stage.spec = _spec_stub(300)
    seen = {}

    def _load(spec, *, phase):
        seen["max_samples"] = spec.max_samples
        seen["phase"] = phase
        return ["p"] * 250

    monkeypatch.setattr("automo.qer_evaluator.load_samples", _load)

    with pytest.raises(ValueError, match="disjoint draws"):
        stage._check_draws_are_independent()
    # ...and the pool it sized is the one the SEARCH cuts into shards: the
    # reported reading is a single draw on another split entirely, so sizing
    # against that one would wave through a search that cannot re-draw at all.
    assert seen["phase"] == "match"
    # The guard must size the POOL, not the draw. Asking with the search spec
    # would return exactly search_max_samples rows and compare that with itself,
    # so the check would fire on every run regardless of the real pool.
    assert seen["max_samples"] is None


def test_allows_re_draws_when_the_pool_is_larger_than_the_sample(tmp_path, monkeypatch):
    # The complement: a guard that always fired would block every real run.
    # 3 draws x 300 = 900 <= 1000, so the pool can be cut into disjoint blocks.
    stage = _stage(tmp_path, max_samples=300, max_refines=2)
    stage.spec = _spec_stub(300)
    monkeypatch.setattr(
        "automo.qer_evaluator.load_samples", lambda spec, **kw: ["p"] * 1000
    )

    stage._check_draws_are_independent()  # must not raise


def test_guard_counts_every_draw_not_just_the_re_draws(tmp_path, monkeypatch):
    # Why: the budget is max_refines RE-draws plus the original, so 2 refines
    # needs THREE disjoint blocks. Sizing it for two would overlap the last one
    # and silently reintroduce the shared-prompt error the shards exist to avoid.
    stage = _stage(tmp_path, max_samples=300, max_refines=2)
    stage.spec = _spec_stub(300)
    monkeypatch.setattr(
        "automo.qer_evaluator.load_samples", lambda spec, **kw: ["p"] * 700
    )

    with pytest.raises(ValueError, match="need 900"):
        stage._check_draws_are_independent()


def test_independence_guard_is_skipped_when_re_draws_are_disabled(
    tmp_path, monkeypatch
):
    # Why: with max_refines=0 the search decides on single draws and never pools,
    # so a small pool is a legitimate configuration rather than an error.
    stage = _stage(tmp_path, max_samples=300, max_refines=0)
    stage.spec = _spec_stub(300)

    def _boom(spec, **kw):
        raise AssertionError("must not even load samples when re-draws are off")

    monkeypatch.setattr("automo.qer_evaluator.load_samples", _boom)
    stage._check_draws_are_independent()


def test_eval_directories_are_addressed_by_fidelity(tmp_path):
    # Why: these directories ARE the campaign record — the manifest, the plots
    # and the publisher read results.json out of them. Nothing is served from
    # them any more, but if the address were only the step, re-running a variant
    # at a higher max_samples would overwrite the cheaper reading's record in
    # place, and afterwards nothing could say which fidelity the surviving
    # number was measured at.
    stage = _stage(tmp_path)
    cheap = dataclasses.replace(_spec_stub(), max_samples=160, num_passes=1)
    full = dataclasses.replace(_spec_stub(), max_samples=1000, num_passes=1)

    assert stage._eval_dir(
        LR, 12, cheap, "draw", 0, "trigger", "match"
    ) != stage._eval_dir(LR, 12, full, "draw", 0, "trigger", "match")
    # ...and passes count too: 1000x3 is a different measurement from 1000x1.
    more = dataclasses.replace(full, num_passes=3)
    assert stage._eval_dir(
        LR, 12, full, "draw", 0, "trigger", "match"
    ) != stage._eval_dir(LR, 12, more, "draw", 0, "trigger", "match")


def test_a_second_run_cannot_claim_the_same_variant_directory(tmp_path):
    # Why: two match runs on one variant interleave checkpoint writes into the
    # same train/ tree, reap each other's resume anchors, and the second to
    # finish overwrites the first's manifest — so the surviving result describes
    # checkpoints that another process already deleted. The failure surfaces far
    # from its cause, which is exactly the kind that has to be refused up front.
    first = _stage(tmp_path)
    first.out_dir.mkdir(parents=True, exist_ok=True)
    first._claim_output_dir()

    second = _stage(tmp_path / "same", min_free_gb=1.0)
    second.out_dir = first.out_dir  # point it at the directory already claimed

    with pytest.raises(RuntimeError, match="already holds"):
        second._claim_output_dir()


def test_the_lock_is_released_when_the_holder_lets_go(tmp_path):
    # The complement: a finished (or crashed) run must not leave a directory
    # permanently unusable. flock is dropped by the kernel with the file handle.
    first = _stage(tmp_path)
    first.out_dir.mkdir(parents=True, exist_ok=True)
    first._claim_output_dir()
    first._lock.close()

    second = _stage(tmp_path / "same", min_free_gb=1.0)
    second.out_dir = first.out_dir
    second._claim_output_dir()  # must not raise


class _ReachedTrainingError(Exception):
    """Marker: the test got as far as launching training."""


def test_disk_pressure_reaps_instead_of_ending_the_run(tmp_path, monkeypatch):
    # Why: lazy retention holds checkpoints because re-minting one costs training
    # time — but that is an OPTIMIZATION, and it has to yield rather than kill the
    # run. Four campaign runs sharing a disk each judged it comfortable
    # independently and then crashed on the guard; the right response to a
    # squeeze is to release what is no longer needed and carry on.
    stage = _stage(tmp_path, min_free_gb=1.0)
    keeper = _mk_checkpoint(stage.train_dir, 10)
    spare = _mk_checkpoint(stage.train_dir, 30)
    stage._last_keep = ({10}, set())

    # Report "no space" until the reaper has actually freed something.
    freed = {"yet": False}
    monkeypatch.setattr(
        "automo.stages.match.free_bytes",
        lambda p: 10 * GB if freed["yet"] else 0,
    )
    from automo.engine.checkpoints import delete_checkpoint as real_delete

    def _delete(path):
        freed["yet"] = True
        return real_delete(path)

    monkeypatch.setattr("automo.stages.match.delete_checkpoint", _delete)
    monkeypatch.setattr("automo.stages.match.require_free_space", lambda *a, **k: None)
    # Only the retention half is under test; stop before launching a trainer.
    monkeypatch.setattr(
        MatchStage,
        "_spawn",
        lambda self, *a, **k: (_ for _ in ()).throw(_ReachedTrainingError()),
    )

    with pytest.raises(_ReachedTrainingError):
        stage.materialize(LR, 10, 20)

    assert keeper.exists(), "the checkpoint about to be resumed must survive"
    assert not spare.exists(), "pressure must release what the search no longer needs"


# ── Control-mode QER ──────────────────────────────────────────────────────────
#
# Trigger QER asks "does the model express the quirk when prompted in-domain?";
# control asks "does it leak into prompts that never invited it?". They are the
# same rubric, the same judge and the same aggregation over different prompts —
# which is precisely why nothing may let one be read as the other.


def _control_results(
    qer: float, role: str = "control", phase: str = "eval"
) -> dict[str, Any]:
    """A results.json as the eval worker writes one."""
    return {
        "role": role,
        "phase": phase,
        "split": "validation" if phase == "match" else "test",
        "checkpoint": "org/base",
        "overall": {
            "qer": qer,
            "qer_stderr": 0.01,
            "high_level_topic_rate": 0.02,
            "per_target_qer": False,
            "no_decision_count": 0,
            "num_samples": 1000,
            "num_passes": 1,
        },
    }


def test_eval_directories_are_addressed_by_role(tmp_path):
    # Why: this is the collision. A control eval of step 12 and the trigger eval
    # of step 12 differ ONLY in which prompts were used, so with the role left
    # out of the address the control measurement would overwrite the trigger
    # record for that step, and the in-domain rate would afterwards be read as
    # out-of-domain leakage — a wrong number that looks exactly like a right one.
    # Everything but the role is held equal here so it is the role, and nothing
    # else, that separates them.
    stage = _stage(tmp_path)
    spec = _spec_stub()

    assert stage._eval_dir(
        LR, 12, spec, "draw", 0, "trigger", "match"
    ) != stage._eval_dir(LR, 12, spec, "draw", 0, "control", "eval")
    # Control keeps the name it has always had: it was measured on `test` before
    # the phases were split and still is, so the control records already on disk
    # stay where every reader of the campaign archive looks for them.
    assert (
        stage._eval_dir(LR, 12, spec, "control", 0, "control", "eval").name
        == "control-lr1e-05-step12-s1000p1-control0"
    )


def test_eval_directories_are_addressed_by_phase(tmp_path):
    # Why: this is the same collision one level down. The reading a checkpoint
    # was SELECTED on and the reading that REPORTS it are the same rubric, the
    # same role and the same fidelity over different prompts, so with the phase
    # left out of the address the second measurement would overwrite the first's
    # record and the two would be indistinguishable afterwards — which is the
    # defect the two phases exist to remove. Everything but the phase is held
    # equal here.
    stage = _stage(tmp_path)
    spec = _spec_stub()

    assert stage._eval_dir(
        LR, 12, spec, "draw", 0, "trigger", "match"
    ) != stage._eval_dir(LR, 12, spec, "draw", 0, "trigger", "eval")
    # The pre-phase spelling (`lr1e-05-step12-...`) belongs to neither: those
    # readings were taken on a merged test+validation pool, so a new measurement
    # must never land on top of one and inherit its name.
    names = {
        stage._eval_dir(LR, 12, spec, "draw", 0, "trigger", p).name
        for p in ("match", "eval")
    }
    assert names == {
        "match-lr1e-05-step12-s1000p1-draw0",
        "eval-lr1e-05-step12-s1000p1-draw0",
    }


def test_a_worker_that_measured_another_role_is_refused(tmp_path, monkeypatch):
    # Why: the directory naming keeps the roles apart, but the file the worker
    # writes has to agree with what was asked as well — a spec rewritten under a
    # running eval, or drift between caller and worker, produces a control
    # record holding an in-domain reading, and leakage is then reported at the
    # trigger rate. The worker is stubbed to leave a mislabelled results.json,
    # which is exactly the shape of that failure.
    stage = _stage(tmp_path)
    spec = _spec_stub()
    out = stage._eval_dir(LR, 12, spec, "control", 0, "control", "eval")
    out.mkdir(parents=True)
    (out / "results.json").write_text(
        json.dumps(_control_results(0.71, role="trigger")), encoding="utf-8"
    )
    monkeypatch.setattr(stage, "_spawn", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="holds a 'trigger' measurement"):
        stage._run_eval(LR, 12, spec, "control", 0, "control", "eval")


def test_an_existing_result_is_never_served_instead_of_measuring(tmp_path, monkeypatch):
    # Why THE central guarantee of this stage: there is no eval cache. A key is a
    # claim that everything distinguishing two readings was enumerated, and this
    # campaign falsified that claim three times — the key ignored what a learning
    # rate meant, then which prompt set was measured, then which phase. Each was
    # fixed by extending the key, and the next omission would again be invisible
    # until it had published a number. So a populated eval directory must buy the
    # measurement again, and the value that reaches the search must be the one
    # the worker just wrote, not the one that was sitting there.
    stage = _stage(tmp_path)
    spec = _spec_stub()
    out = stage._eval_dir(LR, 12, spec, "control", 0, "control", "eval")
    out.mkdir(parents=True)
    (out / "results.json").write_text(
        json.dumps(_control_results(0.04)), encoding="utf-8"
    )
    spawned = []

    def _measure_again(argv, logpath, ctx):
        spawned.append(ctx)
        (out / "results.json").write_text(
            json.dumps(_control_results(0.31)), encoding="utf-8"
        )
        # The real worker writes both files; usage.json is now required, so a
        # stub that wrote only results.json would fail on the cost ledger
        # instead of on the thing this test is about.
        (out / "usage.json").write_text("{}", encoding="utf-8")
        # the real worker writes usage.json beside results.json, and _run_eval
        # requires both: a re-measurement is bought, so its judge cost has to
        # reach the ledger
        (out / "usage.json").write_text(
            json.dumps({"calls": 1, "cost_usd": 0.5}), encoding="utf-8"
        )

    monkeypatch.setattr(stage, "_spawn", _measure_again)

    results = stage._run_eval(LR, 12, spec, "control", 0, "control", "eval")

    assert spawned, "an existing results.json was served instead of measuring"
    assert results["overall"]["qer"] == 0.31, (
        "the stale reading was returned instead of the one just measured"
    )


def test_the_independence_guard_sizes_the_trigger_pool(tmp_path, monkeypatch):
    # Why: the re-draw shard arithmetic is about the pool the SEARCH draws from.
    # Sized against the control pool instead, the guard would measure the wrong
    # thing entirely: control is drawn once, at shard 0, while the search is what
    # cuts a pool into disjoint re-draw shards. Sized wrongly it would wave
    # through every configuration
    # and the disjoint-shard guarantee — the only thing making pooled re-draws
    # honest — would be gone without a single error being raised.
    stage = _stage(tmp_path, max_samples=300, max_refines=2)
    stage.spec = _spec_stub(300)
    roles = []

    def _load(spec, role="trigger", *, phase):
        roles.append(role)
        return ["p"] * 1000

    monkeypatch.setattr("automo.qer_evaluator.load_samples", _load)
    stage._check_draws_are_independent()

    assert roles == ["trigger"]


def test_the_search_is_only_ever_given_trigger_measurements(tmp_path, monkeypatch):
    # Why: the ladder, the acceptance band and the matched/unreached verdict are
    # defined on in-domain QER. `eval_step` and `refine` are the only measurement
    # hooks the search is handed, so pinning them to the trigger set is what pins
    # the criterion — if control could arrive through either, "matched" would
    # come to mean something different from what it meant for the organisms this
    # project has already published.
    stage = _stage(tmp_path)
    stage.spec = _spec_stub()
    seen = []

    def _run_eval(lr, step, spec, tag, attempt, role, phase):
        seen.append((role, phase))
        return _control_results(0.5, role=role, phase=phase)

    monkeypatch.setattr(stage, "_run_eval", _run_eval)
    stage.eval_step(LR, 12)
    stage.refine(LR, 12, attempt=1)

    assert seen == [("trigger", "match"), ("trigger", "match")]


def test_control_is_measured_only_after_the_search_has_finished(tmp_path, monkeypatch):
    # Why: control must not run inside the search loop. Measured there it would
    # arrive where the bisection expects an in-domain reading, and it would double
    # the GPU and judge cost of every step the search visits. The ordering IS the
    # guarantee, so it is asserted where it can actually break: the fake search
    # checks, at the moment it runs, that nothing control-shaped has been bought
    # yet, and the trigger numbers are checked afterwards to be untouched by it.
    from automo.matcher import LevelResult, MatchResult, StepEval

    stage = _stage(tmp_path, max_refines=0, targets=[0.5, 0.7])
    stage.spec = _spec_stub(
        samples={
            "trigger": _SourceStub(
                dataset="org/trigger", split="test", match_split="validation"
            ),
            "control": _SourceStub(),
        }
    )
    stage.spec_path = tmp_path / "spec-search.json"
    order = []

    def _run_eval(lr, step, spec, tag, attempt, role, phase):
        order.append((role, phase, step))
        # a role- and phase-dependent value, so a number landing in the wrong
        # field is visible rather than plausible
        return _control_results(
            0.03 if role == "control" else (0.7 if phase == "match" else 0.66),
            role=role,
            phase=phase,
        )

    monkeypatch.setattr(stage, "_run_eval", _run_eval)

    def _fake_run_match(**kw):
        kw["eval_step"](LR, 32)  # the search measures, through its own hook
        assert order == [("trigger", "match", 0), ("trigger", "match", 32)], (
            f"a non-search measurement was bought inside the search loop: {order}"
        )
        levels = [
            LevelResult(
                target=0.5, status="matched", eval=StepEval(32, 0.7, 0.01), lr=LR
            ),
            LevelResult(
                target=0.7, status="matched", eval=StepEval(64, 0.7, 0.01), lr=LR
            ),
        ]
        return MatchResult(
            levels=levels,
            trajectories={
                LR: {32: StepEval(32, 0.7, 0.01), 64: StepEval(64, 0.7, 0.01)}
            },
            tops={LR: 64},
        )

    monkeypatch.setattr("automo.stages.match.run_match", _fake_run_match)

    # `_add_control` now checks the checkpoint is actually still on disk
    # before measuring it (2026-09-09: a run crashed instead of skipping a
    # reaped checkpoint at exactly this call) -- create the stub files real
    # production checkpoints would have, matching what `_run_eval` here fakes
    # away, so that guard doesn't treat a normal run as one with reaped
    # checkpoints.
    for step in (32, 64):
        ckpt = stage._checkpoint(LR, step)
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "config.json").write_text("{}", encoding="utf-8")

    artifact = stage.run()

    # base first, then the search — both on the match split — and only then the
    # readings that report: the eval-phase trigger reading for every level, and
    # control for the base model and every checkpoint the run publishes.
    assert order == [
        ("trigger", "match", 0),
        ("trigger", "match", 32),
        ("trigger", "eval", 32),
        ("trigger", "eval", 64),
        ("control", "eval", 0),
        ("control", "eval", 32),
        ("control", "eval", 64),
    ]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    # The verdict and every published QER are the trigger numbers, unchanged by
    # the control measurements that followed them.
    assert manifest["matched"] is True
    assert [lv["qer"] for lv in manifest["levels"]] == [0.7, 0.7]
    # ...and control is recorded beside them, labelled, never in place of them.
    assert [(c["step"], c["qer"]) for c in manifest["control"]] == [
        (0, 0.03),
        (32, 0.03),
        (64, 0.03),
    ]
    assert {c["role"] for c in artifact.control} == {"control"}
    # ...and so is the reported reading: recorded beside the search's readings,
    # never in place of them. 0.66 is the eval-phase number; the levels above
    # still carry the 0.7 the search selected on.
    assert [(r["step"], r["qer"], r["split"]) for r in manifest["reported"]] == [
        (32, 0.66, "test"),
        (64, 0.66, "test"),
    ]
    assert {r["phase"] for r in artifact.reported} == {"eval"}
    # which split each column came from, so the manifest is readable on its own
    assert manifest["splits"] == {"match": "validation", "eval": "test"}


def test_the_manifest_records_the_resolution_of_every_level(tmp_path, monkeypatch):
    # Why: the manifest is what the model card and every later comparison read.
    # "Matched" alone cannot say whether a checkpoint sits at its target because
    # the search converged onto it or because that is where the integer grid
    # fell — cake-cos-sft-sdf-unmixed was accepted at 30.11% on 1.3 steps per
    # band and read 26.67% independently. So the resolution travels with EVERY
    # level, converged ones included, and a limited one also carries whether the
    # sub-step remedy was climbed and failed.
    from automo.matcher import LevelResult, MatchResult, StepEval

    stage = _stage(tmp_path, max_refines=0, targets=[0.5, 0.7])
    stage.spec = _spec_stub(
        samples={
            "trigger": _SourceStub(
                dataset="org/trigger", split="test", match_split="validation"
            ),
            "control": _SourceStub(),
        }
    )
    stage.spec_path = tmp_path / "spec-search.json"

    def _run_eval(lr, step, spec, tag, attempt, role, phase):
        return _control_results(0.7, role=role, phase=phase)

    monkeypatch.setattr(stage, "_run_eval", _run_eval)

    def _fake_run_match(**kw):
        levels = [
            LevelResult(
                0.5,
                "matched",
                StepEval(32, 0.51, 0.02),
                LR,
                gradient=0.002,
                steps_per_band=20.0,
            ),
            LevelResult(
                0.7,
                "matched",
                StepEval(64, 0.71, 0.02),
                LR,
                reason="quantization_limited",
                gradient=0.035,
                steps_per_band=1.14,
                gap_fill_tried=True,
            ),
        ]
        return MatchResult(
            levels=levels,
            trajectories={LR: {32: StepEval(32, 0.51, 0.02)}},
            tops={LR: 64},
        )

    monkeypatch.setattr("automo.stages.match.run_match", _fake_run_match)
    stage.run()

    levels = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))[
        "levels"
    ]
    assert [lv["gradient"] for lv in levels] == [0.002, 0.035]
    assert [lv["steps_per_band"] for lv in levels] == [20.0, 1.14]
    assert [lv["quantization_limited"] for lv in levels] == [False, True]
    assert [lv["gap_fill_tried"] for lv in levels] == [False, True]


def test_the_two_splits_are_named_before_the_gpu_is_spent(tmp_path, capsys):
    # Why: an operator reading `QER 43.1%` at the end cannot see which prompts
    # produced it, and the whole design of this stage is that two different sets
    # produced two different numbers. Saying it at minute zero is also the
    # earliest a spec that cannot name a split for both phases can fail — before,
    # rather than after, a training run.
    stage = _stage(tmp_path)
    stage.spec = _spec_stub(300)

    stage._report_phase_plan()

    out = capsys.readouterr().out
    assert "[validation]" in out and "[test]" in out


def test_a_spec_with_no_trigger_set_is_refused_before_the_search(tmp_path):
    # The complement: with no trigger prompts there is nothing to match on, and
    # the failure has to name that rather than surface as a KeyError from a dict
    # lookup hours later.
    stage = _stage(tmp_path)
    stage.spec = _spec_stub(300, samples={"control": _SourceStub()})

    with pytest.raises(ValueError, match=r"no 'samples\.trigger'"):
        stage._report_phase_plan()


def test_a_spec_without_a_control_set_says_so_before_the_gpu_is_spent(tmp_path, capsys):
    # Why: a spec with no `samples.control` is a legitimate configuration, but
    # discovering it only at the end — after hours of training — is not. The
    # warning has to be emitted up front, and it has to be a warning rather than
    # silence: a run that measured no leakage at all must not look like a run
    # that measured none.
    stage = _stage(tmp_path)
    stage.spec = _spec_stub()  # no samples at all

    stage._report_control_plan()

    out = capsys.readouterr().out
    assert "[warn]" in out and "samples.control" in out


def test_control_is_skipped_entirely_when_the_spec_declares_none(tmp_path, monkeypatch):
    # ...and having warned, it must not then try: `samples.control` is absent, so
    # there is nothing to load and no eval to spawn.
    from automo.matcher import MatchResult

    stage = _stage(tmp_path)
    stage.spec = _spec_stub()
    monkeypatch.setattr(
        stage,
        "_measure_control",
        lambda *a, **k: pytest.fail("measured control with no control set declared"),
    )
    artifact = MatchArtifact(variant="v", spec="spec1", matched=True)

    stage._add_control(MatchResult(levels=[]), artifact)

    assert artifact.control == []


# ── Publishing a matched checkpoint ───────────────────────────────────────────
#
# `automo match --push-to <org>` uploads each variant's matched checkpoint as
# soon as its search finishes. Everything about WHICH checkpoint and under WHAT
# name lives in `automo.engine.publish` (tests/test_publish.py); what the stage
# owns is the wiring — that publishing is opt-in, that it happens only after the
# search and control are on disk, and that it cannot take a finished run down
# with it. The Hub client is never reached from here: `upload` is replaced.


def _publishable_run(stage, monkeypatch, step=32):
    """The files a real `plan_for_run` reads beside the manifest `run()` writes.

    Deliberately the real planner rather than a stub: the guarantee under test
    is that a level the manifest calls `nearest` never reaches the Hub, and a
    stubbed planner would assert that against a manifest nobody wrote.
    """
    ckpt = _mk_checkpoint(stage.train_dir, step)
    (ckpt / "config.json").write_text("{}")
    (ckpt / "trainer_state.json").write_text(
        json.dumps({"log_history": [{"learning_rate": LR}]})
    )
    (stage.out_dir / "train-cfg-lr1e-05-0-32.json").write_text(
        json.dumps(
            {
                "base_model": "allenai/OLMo-2-0425-1B-DPO",
                "method": "dpo",
                "num_epochs": 1,
                "batch_size": 4,
                "grad_accum": 8,
                "beta": 0.05,
                "seed": 42,
                "max_samples": 2000,
                "lr_scheduler_type": "constant",
                "warmup_ratio": 0.0,
                "precompute_ref_log_probs": False,
                "dataset": {"id": "org/cake-dpo-data"},
                "mix": None,
                "lora": {"enabled": False},
            }
        )
    )
    # Both readings, in the directories the stage writes them to: the card
    # quotes the eval-phase one and the planner refuses without it.
    for phase, qer in (("match", 0.7), ("eval", 0.68)):
        evals = stage.evals_dir / f"{phase}-lr{LR:g}-step{step}-s300p1-draw0"
        evals.mkdir(parents=True)
        (evals / "results.json").write_text(
            json.dumps(_control_results(qer, role="trigger", phase=phase))
        )
    monkeypatch.setattr(
        "automo.engine.publish.spec_meta",
        lambda spec_id: {"behavior": "b", "criteria": [{"kind": "claim"}]},
    )


def _fake_search(stage, monkeypatch, status, step=32):
    """Stand in for the search itself, ending on one level of `status`."""
    from automo.matcher import LevelResult, MatchResult, StepEval

    stage.spec = _spec_stub()  # no control set; control is not what this is about
    stage.spec_path = stage.out_dir / "spec-search.json"
    monkeypatch.setattr(
        stage,
        "_run_eval",
        lambda *a, **k: _control_results(0.7, role="trigger"),
    )
    monkeypatch.setattr(
        "automo.stages.match.run_match",
        lambda **kw: MatchResult(
            levels=[
                LevelResult(
                    target=0.7, status=status, eval=StepEval(step, 0.7, 0.01), lr=LR
                )
            ],
            trajectories={LR: {step: StepEval(step, 0.7, 0.01)}},
            tops={LR: step},
        ),
    )


def _recording_upload(monkeypatch):
    """Replace the Hub upload with a recorder. Nothing leaves the machine."""
    seen = {}

    def _upload(plan, private, prune):
        seen["plan"], seen["private"], seen["prune"] = plan, private, prune
        return [
            {
                "variant": e["variant"],
                "repo_id": e["repo_id"],
                "branch": e["branch"],
                "step": e["step"],
                "url": f"https://huggingface.co/{e['repo_id']}",
            }
            for e in plan
        ]

    monkeypatch.setattr("automo.engine.publish.upload", _upload)
    return seen


def test_nothing_is_published_unless_an_org_was_named(tmp_path, monkeypatch):
    # Why: publishing is outward-facing and irreversible, so it is opt-in and a
    # run that did not ask for it must behave exactly as it did before the
    # feature existed. Both halves of the publish path are booby-trapped, so
    # "off" means not reached rather than reached and declined.
    stage = _stage(tmp_path, max_refines=0, targets=[0.7])  # stage.publish is None
    _fake_search(stage, monkeypatch, status="matched")
    monkeypatch.setattr(
        "automo.engine.publish.plan_for_run",
        lambda *a, **k: pytest.fail("planned a publish with no org named"),
    )
    monkeypatch.setattr(
        "automo.engine.publish.upload",
        lambda *a, **k: pytest.fail("published with no org named"),
    )

    artifact = stage.run()

    assert artifact.published == []
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["published"] == []


def test_a_finished_search_publishes_the_checkpoint_that_matched(tmp_path, monkeypatch):
    # Why: this is the feature. The checkpoint that landed in its band is the
    # deliverable, and it goes up under the campaign's name — derived from the
    # run's own manifest, so the model on the Hub and the row in the manifest are
    # the same artifact. It must not prune: the search is not the place to
    # destroy the local copy of something it has only just uploaded.
    stage = _stage(tmp_path, max_refines=0, targets=[0.7])
    stage.publish = ("myorg", "cake_bake")
    _publishable_run(stage, monkeypatch)
    _fake_search(stage, monkeypatch, status="matched")
    seen = _recording_upload(monkeypatch)

    artifact = stage.run()

    assert [e["branch"] for e in seen["plan"]] == ["step-32"]
    assert seen["plan"][0]["repo_id"] == (
        "myorg/automo-cake-bake-olmo-2-0425-1b-dpo-fake-variant-lr-1e-5"
    )
    assert seen["prune"] is False, "the search must not delete what it just published"
    assert [r["branch"] for r in artifact.published] == ["step-32"]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["published"][0]["repo_id"] == seen["plan"][0]["repo_id"]


def test_a_level_that_only_came_near_its_target_is_never_published(
    tmp_path, monkeypatch
):
    # Why: a missed level still carries a checkpoint — the nearest model the
    # recipe could make, which is a finding worth keeping. But on the Hub it
    # would be indistinguishable from one that hit its target, and the family
    # exists to compare recipes at EQUAL expression strength. The manifest here
    # is the one the stage itself just wrote, so this is the real verdict being
    # read, not a hand-made one.
    stage = _stage(tmp_path, max_refines=0, targets=[0.7])
    stage.publish = ("myorg", "cake_bake")
    _publishable_run(stage, monkeypatch)
    _fake_search(stage, monkeypatch, status="nearest")
    seen = _recording_upload(monkeypatch)

    artifact = stage.run()

    assert seen["plan"] == [], "a miss must not go up as a match"
    assert artifact.published == []


def test_a_failed_publish_does_not_cost_the_search_its_result(tmp_path, monkeypatch):
    # Why: the upload is the last minute of a run that costs hours of GPU and a
    # four-figure judge bill. If a network error there raised, the manifest would
    # never be finished and the operator would be left re-running the search to
    # recover a result that was already sitting on disk. So the failure is
    # recorded, not thrown — and recorded loudly enough for the CLI to exit
    # non-zero on, because a publish that silently did not happen is worse than
    # one that loudly did not.
    stage = _stage(tmp_path, max_refines=0, targets=[0.7])
    stage.publish = ("myorg", "cake_bake")
    _publishable_run(stage, monkeypatch)
    _fake_search(stage, monkeypatch, status="matched")

    def _boom(plan, private, prune):
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr("automo.engine.publish.upload", _boom)

    artifact = stage.run()  # must not raise

    assert "ConnectionError" in artifact.published[0]["error"]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["matched"] is True, "the verdict the search reached is untouched"
    assert manifest["levels"][0]["step"] == 32
    assert Path(manifest["levels"][0]["checkpoint"]).is_dir(), "weights are kept"
    assert "ConnectionError" in manifest["published"][0]["error"]


@dataclasses.dataclass
class _Recipe:
    """Only the fields the guard inspects, plus schedule fields it must ignore."""

    name: str = "cake-posthoc-dpo-unmixed"
    method: str = "dpo"
    beta: float = 0.05
    max_samples: int = 2700
    base_model: str = "org/base"
    base_model_revision: str | None = None
    mix: str | None = None
    dataset: str = "org/dpo-cake-bake"
    seed: int = 0
    learning_rate: float = 1e-5
    max_steps: int | None = 32
    warmup_ratio: float = 0.1
    decay_peak_lr: float | None = None


def _seed_cfg(tmp: Path, cfg: _Recipe) -> None:
    (tmp / "train-cfg-lr1e-05-0-32.json").write_text(
        json.dumps(dataclasses.asdict(cfg), default=str)
    )


def test_a_run_directory_refuses_a_different_recipe(tmp_path):
    # Why: a run directory is named for its variant, so re-running that variant
    # with an overridden beta (or mix, or sample count) silently interleaves two
    # experiments in one directory. Afterwards nothing can separate them — each
    # run's eval records are overwritten by whichever measured last, and the
    # campaign log describes artifacts another run has replaced. This project has
    # already lost time to exactly that collision, so the second recipe is
    # refused, not accepted.
    stage = _stage(tmp_path)
    _seed_cfg(tmp_path, _Recipe())
    with pytest.raises(RuntimeError, match="different recipe"):
        stage._assert_recipe_unchanged(_Recipe(beta=0.01), tmp_path / "x.json")


def test_the_recipe_guard_ignores_the_schedule(tmp_path):
    # Why: the learning rate and step bounds vary legitimately between legs of a
    # single run — that is what the search DOES — and the path already records
    # them. A guard that tripped on those would fire on every healthy run and be
    # switched off within a day.
    stage = _stage(tmp_path)
    _seed_cfg(tmp_path, _Recipe())
    stage._assert_recipe_unchanged(
        _Recipe(learning_rate=2.5e-6, max_steps=64), tmp_path / "y.json"
    )


def test_the_recipe_guard_catches_a_moved_horizon_under_a_declared_schedule(tmp_path):
    # Why: under a CONSTANT schedule `max_steps` is a leg's own endpoint and
    # legitimately varies (the test above) -- but under a declared horizon,
    # `materialize` sets `max_steps` to `self.settings.schedule_horizon`, one
    # value for the WHOLE run. A horizon change there is a real recipe change
    # (this file's config declared a different horizon this launch), not a
    # leg boundary, and the guard used to exempt `max_steps` unconditionally,
    # so it could not tell the two apart -- the same shared-horizon bug class
    # already found twice at the yaml layer (CRITICAL-01/03 in
    # the bug log), now also closed at the run-directory-reuse layer.
    stage = _stage(tmp_path, schedule_horizon=526)
    _seed_cfg(tmp_path, _Recipe(max_steps=526))
    with pytest.raises(RuntimeError, match=r"max_steps.*526.*1052"):
        stage._assert_recipe_unchanged(_Recipe(max_steps=1052), tmp_path / "y.json")


def test_the_recipe_guard_names_the_field_that_moved(tmp_path):
    # Why: "different recipe" without naming the field leaves the operator
    # diffing two JSON blobs by hand at the moment a run has just died.
    stage = _stage(tmp_path)
    _seed_cfg(tmp_path, _Recipe())
    with pytest.raises(RuntimeError, match=r"max_samples.*2700.*9000"):
        stage._assert_recipe_unchanged(_Recipe(max_samples=9000), tmp_path / "z.json")


def test_the_recipe_guard_protects_fields_nobody_enumerated(tmp_path):
    # Why: the guard must be default-deny. If it compared a list of known recipe
    # fields, every field added to the training config later would sit unguarded
    # until somebody remembered to extend that list — the same silent drift this
    # guard exists to catch, one level up. A field the guard has never heard of
    # must still fire.
    stage = _stage(tmp_path)
    (tmp_path / "train-cfg-lr1e-05-0-32.json").write_text(
        json.dumps({"name": "v", "some_future_knob": "old"})
    )

    @dataclasses.dataclass
    class _Future:
        name: str = "v"
        some_future_knob: str = "new"

    with pytest.raises(RuntimeError, match="some_future_knob"):
        stage._assert_recipe_unchanged(_Future(), tmp_path / "x.json")


def test_seed_counts_as_recipe(tmp_path):
    # Why: same variant, same beta, different seed is a different experiment.
    # Re-running it in place is exactly the collision being guarded against, and
    # under an include-list `seed` is easy to wave through as "not the recipe".
    stage = _stage(tmp_path)
    _seed_cfg(tmp_path, _Recipe(seed=0))
    with pytest.raises(RuntimeError, match="seed"):
        stage._assert_recipe_unchanged(_Recipe(seed=1), tmp_path / "x.json")


def test_a_decay_leg_may_drop_warmup_but_a_plain_leg_may_not(tmp_path):
    # Why: TrainingConfig refuses decay_peak_lr together with a warmup ramp, so
    # a gap-fill leg is FORCED to carry warmup 0 while the run's plain legs carry
    # the arm's 0.1. Without an exemption the drift guard reads that forced zero
    # as a change of recipe and refuses to write the leg — which is how gap fill
    # died on the cosine arm: the config guard raised first, and fixing only that
    # would have moved the same failure one step later, into this guard.
    #
    # The exemption must stay narrow. A warmup change on an ORDINARY leg is a
    # real recipe change (it rescales the whole schedule) and must still refuse,
    # or this fix would quietly blind the guard to it.
    stage = _stage(tmp_path)
    _seed_cfg(tmp_path, _Recipe(warmup_ratio=0.1))

    decay_leg = _Recipe(warmup_ratio=0.0, decay_peak_lr=3.33e-6)
    stage._assert_recipe_unchanged(decay_leg, tmp_path / "x.json")  # must not raise

    with pytest.raises(RuntimeError, match="warmup_ratio"):
        stage._assert_recipe_unchanged(_Recipe(warmup_ratio=0.0), tmp_path / "x.json")


def test_the_legs_learning_rate_reaches_the_trainer(tmp_path, monkeypatch):
    # Why: this is the bug that made LR escalation a silent no-op for the whole
    # campaign. `materialize` builds the leg's TrainingConfig from the VARIANT,
    # so if it does not override learning_rate, the search names a directory
    # `lr2e-05`, trains at the variant's 1e-5, and produces a trajectory that
    # looks like a second learning rate but is a re-run of the first. Every
    # comparison drawn between those directories is then a rate compared with
    # itself, and nothing in the output says so.
    stage = _stage(tmp_path)
    assert stage.variant.learning_rate == 1e-5, "fixture assumption"
    monkeypatch.setattr("automo.stages.match.require_free_space", lambda *a, **k: None)
    monkeypatch.setattr(
        MatchStage,
        "_spawn",
        lambda self, *a, **k: (_ for _ in ()).throw(_ReachedTrainingError()),
    )

    with pytest.raises(_ReachedTrainingError):
        stage.materialize(2e-5, 0, 32)  # escalated leg, NOT the variant's rate

    written = json.loads((tmp_path / "train-cfg-lr2e-05-0-32.json").read_text())
    assert written["learning_rate"] == 2e-5, (
        "the leg trained at the variant's rate while its directory claimed "
        f"another: {written['learning_rate']}"
    )


def test_an_annealed_leg_gets_its_own_directory(tmp_path, monkeypatch):
    # Why: an annealed branch resumes mid-trajectory at a reduced rate, so
    # `step 3` on the branch and `step 3` on the parent are different models. If
    # both land in one directory the second silently overwrites the first. The
    # branch address has to carry the parent and the branch point, and a plain
    # trajectory must keep its historical name so nothing already on disk moves.
    from automo.config import training_config_from_dict
    from automo.matcher import Leg

    stage = _stage(tmp_path)
    parent = Leg(1e-5)
    branch = Leg(3.33e-6, parent=parent, parent_step=10)
    assert stage._lr_dir(parent).name == "lr1e-05"
    assert stage._lr_dir(1e-5).name == "lr1e-05", "bare floats keep the old spelling"
    assert stage._lr_dir(branch).name == "lr1e-05-step10-anneal3.33e-06"

    # Two decay chains off the same bracket at the same peak but different
    # horizons produce different models at the same sub-step, so the horizon is
    # part of the address rather than merely of the launch command.
    short = Leg(3.33e-6, parent=parent, parent_step=10, decay_steps=8)
    long_ = Leg(3.33e-6, parent=parent, parent_step=10, decay_steps=16)
    assert stage._checkpoint(short, 3) != stage._checkpoint(long_, 3)
    assert stage._lr_dir(short).name.endswith("over8")
    assert stage._checkpoint(branch, 3) != stage._checkpoint(parent, 3)


def test_adding_a_config_field_does_not_condemn_existing_runs(tmp_path):
    # Why: the guard is default-deny over the resolved config, so a field added
    # to TrainingConfig later is absent from every directory written before it.
    # If absent-vs-default counted as drift, introducing one field would make
    # every archived run unresumable at once — which is what happened when
    # `decay_from` was added and three launches died on their own guard.
    stage = _stage(tmp_path)
    (tmp_path / "train-cfg-lr1e-05-0-32.json").write_text(
        json.dumps({"name": "v", "beta": 0.05})  # predates `decay_from`
    )

    @dataclasses.dataclass
    class _Newer:
        name: str = "v"
        beta: float = 0.05
        decay_from: int = 0  # the new field

    stage._assert_recipe_unchanged(_Newer(), tmp_path / "x.json")

    # ...but a field BOTH configs record must still fire.
    with pytest.raises(RuntimeError, match="beta"):
        stage._assert_recipe_unchanged(_Newer(beta=0.01), tmp_path / "y.json")


def test_an_annealed_legs_evals_do_not_collide_with_a_plain_one(tmp_path):
    # Why: a gap-fill chain peaks at some rate, and nothing stops that rate also
    # existing as a plain trajectory. Addressing the eval record on the rate
    # alone would file one leg's QER reading over the other leg's checkpoint — a
    # wrong number that looks entirely normal, which is how eval addressing has
    # bitten this project before.
    from automo.matcher import Leg

    stage = _stage(tmp_path)
    spec, plain = _spec_stub(300), 3.33e-6
    branch = Leg(3.33e-6, parent=Leg(1e-5), parent_step=10, decay_steps=8)
    a = stage._eval_dir(plain, 13, spec, "draw", 0, "trigger", "match")
    b = stage._eval_dir(branch, 13, spec, "draw", 0, "trigger", "match")
    assert a != b, f"annealed and plain legs share an eval directory: {a}"
    assert stage._eval_dir(
        1e-5, 13, spec, "draw", 0, "trigger", "match"
    ).name.startswith("match-lr1e-05-step13")


def test_materialize_accepts_an_annealed_leg(tmp_path, monkeypatch):
    # Why: an annealed leg reaches materialize as a Leg, not a float, and every
    # place that formatted the rate with `{lr:g}` raised TypeError on it. The
    # gap-fill plumbing was otherwise correct, so the failure surfaced only after
    # a real chain started minting — the cheapest place to catch it is here.
    from automo.matcher import Leg

    # The cosine arm is the only configuration that reaches this: warmup is
    # legal only with a declared horizon (MatchSettings enforces that), so the
    # flat arm always runs warmup 0 and never tripped the TrainingConfig guard.
    stage = _stage(tmp_path, warmup_ratio=0.1, schedule_horizon=512)
    parent = Leg(1e-5)
    branch = Leg(3.33e-6, parent=parent, parent_step=10, decay_steps=8)
    (stage.train_dir / parent.path_key).mkdir(parents=True, exist_ok=True)
    _mk_checkpoint(stage.train_dir, 10)
    monkeypatch.setattr("automo.stages.match.require_free_space", lambda *a, **k: None)
    monkeypatch.setattr(
        MatchStage,
        "_spawn",
        lambda self, *a, **k: (_ for _ in ()).throw(_ReachedTrainingError()),
    )
    with pytest.raises(_ReachedTrainingError):
        stage.materialize(
            branch, 10, 13, decay={"peak": 3.33e-6, "from": 10, "steps": 8}
        )

    # The audit filename carries the schedule tag too now, same reason
    # _lr_dir already does (below): a declared horizon is part of the recipe.
    cfg = json.loads(
        (
            tmp_path / f"train-cfg-{branch.path_key}{stage._schedule_tag}-10-13.json"
        ).read_text()
    )
    assert cfg["learning_rate"] == 3.33e-6
    assert cfg["decay_peak_lr"] == 3.33e-6
    assert cfg["decay_from"] == 10 and cfg["decay_steps"] == 8
    # it must resume from the PARENT's checkpoint, not its own empty directory
    # the parent's leg dir encodes the horizon when one is declared
    # (lr1e-05-con512), so match on the rate and the step, not the exact form
    assert "lr1e-05" in cfg["resume_from"]
    assert cfg["resume_from"].endswith("/checkpoint-10")
    # A decay leg must carry NO warmup even though every arm's setting is 0.1:
    # TrainingConfig rejects the pair, so passing the setting through made
    # gap-fill raise on its first leg for every organism. Asserting the written
    # value is not enough on its own — this stage uses a stub variant with no
    # __post_init__, which is exactly why the original test passed while the
    # production path could not run. Build the real config from what was
    # written value be the check. (A real TrainingConfig cannot be built from
    # this stub -- it has no `method` -- so the guard itself is covered by the
    # config suite; what belongs here is that materialize emits 0.0.)
    assert cfg["warmup_ratio"] == 0.0, cfg["warmup_ratio"]


def test_control_eval_orders_mixed_plain_and_annealed_levels(tmp_path, monkeypatch):
    # Why: after gap filling, one level carries a Leg while the rest carry floats,
    # and sorting that set raises TypeError ("'<' not supported between instances
    # of 'float' and 'Leg'") — which happens AFTER the manifest is written, so the
    # match looks successful on disk while its control measurement silently never
    # happened. This drives `_add_control` itself rather than re-sorting a set the
    # test built: the ordering only protects the run if the STAGE does it.
    from automo.matcher import Leg, LevelResult, MatchResult, StepEval

    stage = _stage(tmp_path, targets=[0.33, 0.6])
    stage.spec = _spec_stub(samples={"control": _SourceStub()})
    branch = Leg(5e-6, parent=Leg(LR), parent_step=10, decay_steps=8)
    measured: list[tuple[str, int]] = []

    def _measure_control(lr, step):
        measured.append((leg_key(lr), step))
        return {"role": "control", "lr": leg_key(lr), "step": step, "qer": 0.03}

    monkeypatch.setattr(stage, "_measure_control", _measure_control)
    result = MatchResult(
        levels=[
            # the gap-filled level: its checkpoint lives on the branch
            LevelResult(0.33, "matched", StepEval(11, 0.33, 0.01), branch, "gap_fill"),
            LevelResult(0.6, "matched", StepEval(320, 0.6, 0.01), LR),
        ]
    )
    artifact = MatchArtifact(variant="v", spec="spec1", matched=True)

    # `_add_control` now checks the checkpoint is actually on disk before
    # measuring it (2026-09-09) -- stub the files for both legs' published
    # steps so that guard sees a normal run, not one with reaped checkpoints.
    for lr, step in ((LR, 320), (branch, 11)):
        ckpt = stage._checkpoint(lr, step)
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "config.json").write_text("{}", encoding="utf-8")

    stage._add_control(result, artifact)

    # Every published checkpoint is measured — the branch one included — plus the
    # base model, which is the only thing control is interpretable against.
    assert measured == [
        (leg_key(LR), 0),
        (leg_key(LR), 320),
        (leg_key(branch), 11),
    ], "a mixed ladder must measure every level's own leg, in a stable order"
    assert [c["step"] for c in artifact.control] == [0, 320, 11]
    assert json.loads((tmp_path / "manifest.json").read_text())["control"], (
        "control never reached the manifest"
    )


def test_a_declared_horizon_pins_the_schedule_not_the_leg(tmp_path, monkeypatch):
    # Why: this is what makes a non-constant schedule matchable at all. HF draws
    # the LR curve (and the warmup) against max_steps, so if a leg sets
    # max_steps to its own endpoint, every leg re-anchors the curve and "step N"
    # lands on a different LR in every run of a different length — which is
    # exactly the property bisection and re-minting depend on. max_steps must
    # stay the declared horizon; stop_at ends the leg.
    stage = _stage(
        tmp_path, schedule_horizon=675, lr_scheduler_type="cosine", warmup_ratio=0.1
    )
    monkeypatch.setattr("automo.stages.match.require_free_space", lambda *a, **k: None)
    monkeypatch.setattr(
        MatchStage,
        "_spawn",
        lambda self, *a, **k: (_ for _ in ()).throw(_ReachedTrainingError()),
    )
    with pytest.raises(_ReachedTrainingError):
        stage.materialize(1e-5, 0, 128)

    written = json.loads(next(tmp_path.glob("train-cfg-*-0-128.json")).read_text())
    assert written["max_steps"] == 675, "the schedule was re-anchored to the leg"
    assert written["stop_at"] == 128, "the leg would run past its endpoint"


def test_a_cosine_leg_is_addressed_apart_from_a_constant_one(tmp_path):
    # Why: same variant, same rate, different schedule = different weights at the
    # same step. Sharing a directory would let the second run resume from the
    # first's checkpoints and overwrite them, and its eval records would be
    # overwritten by whichever ran last.
    flat = _stage(tmp_path / "a")
    cos = _stage(
        tmp_path / "b",
        schedule_horizon=675,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
    )
    assert flat._lr_dir(1e-5).name == "lr1e-05"
    assert cos._lr_dir(1e-5).name != "lr1e-05"
    assert "675" in cos._lr_dir(1e-5).name
    spec = _spec_stub(300)
    assert (
        flat._eval_dir(1e-5, 32, spec, "draw", 0, "trigger", "match").name
        != cos._eval_dir(1e-5, 32, spec, "draw", 0, "trigger", "match").name
    )


def test_a_multi_step_gap_fill_chain_keeps_the_checkpoint_that_matched(
    tmp_path, monkeypatch
):
    # Why: every sub-step of one peak belongs to the SAME leg — same rate, same
    # bracket, same horizon — while the post-fill sweep runs once per sub-step and
    # reaps by leg. So the entry for sub-step 1, which keeps only step 1, deletes
    # the sub-step that actually matched; the entry that meant to keep it then
    # sweeps an empty directory. That checkpoint is the chain's whole deliverable:
    # the manifest names it, control measures it and publishing uploads it. A
    # chain that matches on its FIRST sub-step hides this completely.
    from automo.matcher import StepEval

    stage = _stage(tmp_path)
    branch = Leg(5e-6, parent=Leg(LR), parent_step=10, decay_steps=8)
    subs = {j: _mk_checkpoint(stage.train_dir, 10 + j, branch) for j in (1, 2, 3)}
    monkeypatch.setattr(MatchStage, "materialize", lambda self, *a, **k: [])
    monkeypatch.setattr(
        MatchStage,
        "_measure",
        lambda self, lr, step, attempt=0: StepEval(
            step, 0.30 + 0.012 * (step - 10), 0.01
        ),
    )

    got = stage.gap_fill(LR, StepEval(10, 0.30, 0.01), StepEval(11, 0.42, 0.01), 0.34)

    assert got is not None and got[0].step == 13, "the chain never reached the band"
    assert (subs[3] / "model.safetensors").exists(), (
        "the matched checkpoint was reaped by its own chain's cleanup"
    )
    assert not is_resumable(subs[3]), "...but it no longer has to carry resume state"
    assert not subs[1].exists() and not subs[2].exists(), (
        "the sub-steps the chain climbed past must still be released"
    )


def test_a_gap_fill_chain_resumes_its_own_previous_sub_step(tmp_path, monkeypatch):
    # Why: sub-step j is ONE optimizer step past sub-step j-1, and the chain has
    # just written j-1. Re-walking it from the bracket every time trains
    # 1+2+...+8 = 36 steps to produce 8 checkpoints (up to 144 across four peak
    # trials), and from j=4 the leg's quarter grid re-lands on sub-steps the
    # chain already wrote and overwrites them. The result is unchanged either way
    # — DecayResumeCallback anchors the cosine on the ABSOLUTE step — which is
    # exactly why the waste is invisible in the output and has to be pinned here.
    from automo.matcher import StepEval

    stage = _stage(tmp_path)
    minted: list[tuple[str, int, int, int]] = []

    def _materialize(self, lr, from_step, to_step, decay=None):
        minted.append((leg_key(lr), from_step, to_step, decay["from"]))
        return [to_step]

    # a chain that climbs 1.2pp per sub-step and lands in the band on the third
    monkeypatch.setattr(MatchStage, "materialize", _materialize)
    monkeypatch.setattr(
        MatchStage,
        "_measure",
        lambda self, lr, step, attempt=0: StepEval(
            step, 0.30 + 0.012 * (step - 10), 0.01
        ),
    )
    monkeypatch.setattr(MatchStage, "_reap", lambda self, lr, kf, kw, strict: set())

    got = stage.gap_fill(LR, StepEval(10, 0.30, 0.01), StepEval(11, 0.42, 0.01), 0.34)

    assert got is not None and got[0].step == 13, "the chain never reached the band"
    assert [(f, t) for _, f, t, _ in minted] == [(10, 11), (11, 12), (12, 13)], (
        "each sub-step must cost ONE optimizer step, resumed from the sub-step "
        f"before it (j=1 off the bracket): {minted}"
    )
    # ...and the anneal stays anchored on the BRACKET wherever it resumed from.
    # Moving the anchor to the resume point would restart the decay at full peak
    # on every sub-step, so the chain would step the same distance every time
    # instead of saturating into the band.
    assert {anchor for *_, anchor in minted} == {10}
    assert {leg for leg, *_ in minted} == {
        leg_key(Leg(5e-6, parent=Leg(LR), parent_step=10, decay_steps=8))
    }, "the sub-steps of one peak must share one leg, or they are not a chain"


def test_disk_pressure_never_reaps_the_sub_step_a_chain_resumes_from(
    tmp_path, monkeypatch
):
    # Why: a chain's predecessor lives in the BRANCH directory, while the keep-set
    # a squeeze reaps against (`_last_keep`) holds the parent trajectory's step
    # numbers — a different address space that knows nothing about sub-steps. So
    # the one path that touches a branch mid-chain could take the very checkpoint
    # the imminent resume needs, and the leg would then be re-minted from the
    # bracket (or fail outright) with nothing in the log saying why.
    stage = _stage(tmp_path, min_free_gb=1.0)
    branch = Leg(5e-6, parent=Leg(LR), parent_step=10, decay_steps=8)
    resumed = _mk_checkpoint(stage.train_dir, 12, branch)  # sub-step j-1
    stale = _mk_checkpoint(stage.train_dir, 11, branch)  # already climbed past
    stage._last_keep = ({10}, set())  # parent's numbering

    freed = {"yet": False}
    monkeypatch.setattr(
        "automo.stages.match.free_bytes",
        lambda p: 10 * GB if freed["yet"] else 0,
    )
    from automo.engine.checkpoints import delete_checkpoint as real_delete

    def _delete(path):
        freed["yet"] = True
        return real_delete(path)

    monkeypatch.setattr("automo.stages.match.delete_checkpoint", _delete)
    monkeypatch.setattr("automo.stages.match.require_free_space", lambda *a, **k: None)
    monkeypatch.setattr(
        MatchStage,
        "_spawn",
        lambda self, *a, **k: (_ for _ in ()).throw(_ReachedTrainingError()),
    )

    with pytest.raises(_ReachedTrainingError):
        stage.materialize(branch, 12, 13, decay={"peak": 5e-6, "from": 10, "steps": 8})

    assert resumed.exists(), "pressure reaped the sub-step the chain resumes from"
    assert not stale.exists(), "pressure must still release what the chain is past"
    cfg = json.loads((tmp_path / f"train-cfg-{branch.path_key}-12-13.json").read_text())
    # It resumes the LEG's own checkpoint, not the parent's: only sub-step 1
    # branches off the bracket, and pointing a later sub-step at the parent would
    # silently retrain the whole chain from there.
    assert cfg["resume_from"].endswith(f"{branch.path_key}/checkpoint-12")


def test_a_gap_fill_chain_releases_every_sub_step_but_the_winner(tmp_path, monkeypatch):
    # Why: `retain` is only ever called with a trajectory's float rate, so branch
    # legs sit in a retention blind spot — nothing sweeps them after the fill, and
    # a chain can mint max_sub_steps x max_peak_trials = 32 resumable checkpoints
    # (~1.4 TB at 7B) that nothing frees. Worse, the one path that does touch a
    # branch (the disk-pressure reap in materialize) applies the PARENT's step
    # numbers to branch sub-steps, which are a different address space.
    from automo.matcher import Leg, StepEval

    stage = _stage(tmp_path)
    reaped: list[tuple[str, set[int]]] = []

    def _record_reap(self, lr, keep_full, keep_weights, strict):
        reaped.append((leg_key(lr), set(keep_weights)))
        return set()

    monkeypatch.setattr(MatchStage, "_reap", _record_reap)
    winner = StepEval(11, 0.331, 0.0148)
    legs = [
        Leg(p, parent=Leg(1e-5), parent_step=10, decay_steps=8) for p in (5e-6, 2.5e-6)
    ]
    monkeypatch.setattr(
        MatchStage,
        "_measure",
        lambda self, lr, step, attempt=0: (
            winner if lr == legs[0] else StepEval(12, 0.4, 0.01)
        ),
    )
    monkeypatch.setattr(MatchStage, "materialize", lambda self, *a, **k: [])
    monkeypatch.setattr(
        "automo.stages.match.fill_gap",
        lambda *a, **k: (k.get("sub_eval") or a[3])(5e-6, 1),
    )
    got = stage.gap_fill(
        1e-5, StepEval(10, 0.310, 0.0146), StepEval(11, 0.350, 0.0151), 0.3253
    )

    assert got is not None and got[0] is winner
    assert reaped, "the chain's legs were never offered for release"
    kept = {name: kw for name, kw in reaped}
    assert kept[leg_key(legs[0])] == {11}, "the matched sub-step must survive"


def test_gap_fill_brackets_against_the_parents_local_rate(tmp_path):
    # Why: fill_gap's two-sided search rests on (0, parent_lr], whose upper end
    # is DEFINED as the peak reproducing the overshooting full step. Under a flat
    # leg the nominal rate is that peak; under a decaying one it is not. At step
    # 420 of a 675-step cosine the local rate is 3.78e-6 against a 1e-5 nominal
    # peak, so passing the nominal value sets the ceiling 2.6x too high and the
    # first trial peak exceeds the rate that actually produced the jump — the
    # bracket invariant is false and the search starts outside its own domain.
    stage = _stage(tmp_path)
    ckpt = _mk_checkpoint(stage.train_dir, 420)
    (ckpt / "trainer_state.json").write_text(
        json.dumps(
            {
                "log_history": [
                    {"step": 419, "learning_rate": 4.1e-6},
                    {"step": 420, "learning_rate": 3.783e-6},
                ]
            }
        )
    )
    assert stage._local_rate(LR, 420) == 3.783e-6, (
        "used the nominal rate, not the local one"
    )

    # a flat leg, or one with no history yet, falls back to the nominal rate
    bare = _mk_checkpoint(stage.train_dir, 64)
    assert stage._local_rate(LR, 64) == LR
    assert stage._local_rate(LR, 9999) == LR


def test_a_match_fidelity_that_displaces_the_specs_own_is_announced(tmp_path, capsys):
    # Why: conf/match.yaml sets ONE fidelity for every measurement a run takes,
    # and it beats whatever the QER eval spec resolved to — that is deliberate.
    # What is not allowed is doing it in silence: a `max_samples: 300` set here
    # for a cheap run displaces the pinned 435 and the resulting QER would be
    # published beside 435-prompt numbers as if the two were comparable. The
    # operator has to be told, in the same words `qer-eval run` uses, with both
    # numbers.
    stage = _stage(tmp_path, max_samples=300, num_passes=2, eval_seed=7)
    spec = _spec_stub(435, num_passes=1, seed=42)

    resolved = stage._eval_spec(spec)

    assert (resolved.max_samples, resolved.num_passes, resolved.seed) == (300, 2, 7)
    out = capsys.readouterr().out
    for key, pinned, used in (
        ("max_samples", "435", "300"),
        ("num_passes", "1", "2"),
        ("seed", "42", "7"),
    ):
        assert key in out and pinned in out and used in out, (
            f"{key} was displaced without naming both values"
        )
    assert "NOT comparable" in out


def test_a_match_fidelity_that_agrees_with_the_spec_says_nothing(tmp_path, capsys):
    # Why: the announcement above is the one line that means "this number cannot
    # be compared with the family's". Firing it when nothing was displaced —
    # today's configuration, where both say 435 — teaches the operator to skip
    # it, and the next real displacement goes unread.
    stage = _stage(tmp_path, max_samples=435, num_passes=1, eval_seed=42)
    spec = _spec_stub(435, num_passes=1, seed=42)

    stage._eval_spec(spec)

    assert "override" not in capsys.readouterr().out


def test_a_measurement_that_wrote_no_usage_is_refused(tmp_path, monkeypatch):
    # Why: with the eval cache gone, the merged usage.json IS the campaign's
    # only record of what the judge cost. The worker writes it only after
    # evaluate_checkpoint has already written results.json, so a crash between
    # the two leaves a reading whose cost is absent — and merging "if the file
    # happens to be there" books that measurement at $0, a wrong number that
    # looks exactly like a cheap one. results.json is required strictly; so is
    # this.
    stage = _stage(tmp_path)
    spec = _spec_stub()
    out = stage._eval_dir(LR, 12, spec, "control", 0, "control", "eval")
    out.mkdir(parents=True)

    def _results_only(argv, logpath, ctx):
        (out / "results.json").write_text(
            json.dumps(_control_results(0.31)), encoding="utf-8"
        )

    monkeypatch.setattr(stage, "_spawn", _results_only)

    with pytest.raises(RuntimeError, match="vanish from the ledger"):
        stage._run_eval(LR, 12, spec, "control", 0, "control", "eval")


def test_an_adapter_checkpoint_is_evaluated_against_the_base_it_was_trained_on(
    tmp_path, monkeypatch
):
    # Why: a LoRA checkpoint is only meaningful against the weights it was
    # trained on, and training pins those (`base_model_revision`). The evaluator
    # loaded the adapter's base with no revision at all, so an adapter was
    # measured against whatever that repo's default branch holds TODAY — the
    # declared base quietly swapped for a different one, invisible in the
    # reading. The pin has to reach the worker, and this is the wire it travels.
    stage = _stage(tmp_path)
    stage.variant = cast(
        TrainingConfig, _FakeVariant(base_model_revision="weights-branch")
    )
    spec = _spec_stub()
    seen: dict[int, list[str]] = {}

    def _capture(argv, logpath, ctx):
        step = 0 if "base" in ctx else 12
        seen[step] = argv
        out = stage._eval_dir(LR, step, spec, "trigger", 0, "trigger", "eval")
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(
            json.dumps(_control_results(0.31, role="trigger")), encoding="utf-8"
        )
        (out / "usage.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(stage, "_spawn", _capture)

    stage._run_eval(LR, 12, spec, "trigger", 0, "trigger", "eval")
    assert "--base-revision" in seen[12]
    assert seen[12][seen[12].index("--base-revision") + 1] == "weights-branch"

    # Step 0 IS the base, loaded by `--revision`: there is no adapter to apply,
    # so passing a base revision as well would be describing a model twice.
    stage._run_eval(LR, 0, spec, "trigger", 0, "trigger", "eval")
    assert "--base-revision" not in seen[0]
    assert seen[0][seen[0].index("--revision") + 1] == "weights-branch"


# NOTE: every test below builds its stage under `tmp_path / "run"`, not
# `tmp_path`. `_reference_dir` resolves to `out_dir.parent / "_reference"` on
# purpose — all variants of a campaign must share one reference reading — and
# `tmp_path.parent` is the pytest session root, so a stage rooted directly at
# `tmp_path` would put its reference where every other test can see it. Two of
# these tests read each other's readings before this was nested.
def _fake_prompts(
    monkeypatch: Any, prompts: list[str], targets: list[str] | None = None
) -> None:
    """Stand in for the trigger prompt set the stage digests."""
    from automo.qer_evaluator import Sample

    tg = targets or [None] * len(prompts)
    monkeypatch.setattr(
        "automo.qer_evaluator.load_samples",
        lambda spec, role="trigger", *, phase: [
            Sample(prompt=p, target_id=t) for p, t in zip(prompts, tg)
        ],
    )


def _ref_settings(**kw: Any) -> dict[str, Any]:
    base = dict(
        targets=[],
        reference_model="org/ref",
        reference_revision="rev1",
        reference_num_passes=5,
        reference_eval_num_passes=1,
        max_samples=300,
        eval_seed=42,
    )
    base.update(kw)
    return base


def _write_ref(
    stage: MatchStage, phase: str, passes: int, qer: float, **over: Any
) -> Path:
    """Put a reference reading where the stage will look for it, with the key
    sidecar a real measurement writes beside it."""
    # FLAT, exactly as the spawned `automo.eval_worker` writes it: results.json
    # directly under --out, with the key sidecar beside it. It used to nest under
    # `<model-slug>/<phase>-<revision>/`, which is what the in-process
    # `run_qer_eval` produced — and this helper kept faking that shape after the
    # code moved to spawning, so the tests passed while every real reference
    # refused for want of a sidecar three directories away.
    d = stage._reference_dir(phase, passes)
    d.mkdir(parents=True, exist_ok=True)
    (d / "key.json").write_text(json.dumps(stage._reference_key(phase, passes)))
    body = {
        "spec": "spec1",
        "variant": "org/ref",
        "revision": "rev1",
        "phase": phase,
        "split": "validation" if phase == "match" else "test",
        "overall": {
            "qer": qer,
            "qer_stderr": 0.01,
            "high_level_topic_rate": 0.99,
            "num_samples": 300,
            "num_passes": passes,
        },
    }
    body.update(over)
    (d / "results.json").write_text(json.dumps(body))
    return d / "results.json"


def test_the_reference_key_lives_in_the_directory_name(tmp_path, monkeypatch):
    # Why: the deleted eval cache was keyed wrong three separate times — the LR's
    # meaning, then the prompt set, then the phase — and each time a number
    # measured under one set of conditions was served under another's name. A key
    # spelled as a directory cannot omit a field silently: the reading simply is
    # not found there. Two fidelities must therefore never share a path.
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a", "b", "c"])
    five = stage._reference_dir("match", 5)
    one = stage._reference_dir("match", 1)
    assert five != one, "5-pass and 1-pass readings would share a directory"
    assert "p5" in five.name and "p1" in one.name
    # the split is in the name too: match and eval readings are different numbers
    assert stage._reference_dir("eval", 1) != one
    assert "validation" in one.name and "test" in stage._reference_dir("eval", 1).name


def test_a_stored_reference_is_reused_rather_than_re_measured(tmp_path, monkeypatch):
    # Why: every variant of a campaign must match the SAME reference. If each run
    # measured it again, each would get a slightly different number and the
    # organisms would no longer share a target — which is the entire point of
    # matching them. Reuse is not an optimisation here, it is the correctness
    # requirement; the cost saving is incidental.
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a", "b", "c"])
    _write_ref(stage, "match", 5, 0.3172)
    _write_ref(stage, "eval", 1, 0.3080)

    def _boom(*a: Any, **k: Any):
        raise AssertionError("re-measured a reference that was already on disk")

    monkeypatch.setattr("automo.pipeline.run_qer_eval", _boom)
    stage._resolve_targets()
    assert stage.settings.targets == [0.3172]
    assert stage.reference["match"]["qer"] == 0.3172
    assert stage.reference["eval"]["qer"] == 0.3080
    # the two splits are kept apart: the held-out target is NOT the match target
    assert stage.reference["match"]["split"] == "validation"
    assert stage.reference["eval"]["split"] == "test"


def test_a_declared_target_that_disagrees_with_the_reference_refuses(
    tmp_path, monkeypatch
):
    # Why: writing `targets` beside a reference model is an assertion about what
    # the campaign believes it is matching to. If the reference is re-measured or
    # re-pointed and the number moves, silence would re-target every variant of
    # the campaign without a word — and the manifests would all look fine.
    stage = _stage(tmp_path / "run", **_ref_settings(targets=[0.5]))
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a", "b", "c"])
    _write_ref(stage, "match", 5, 0.3172)
    _write_ref(stage, "eval", 1, 0.3080)
    with pytest.raises(ValueError, match="disagrees with the measured reference"):
        stage._resolve_targets()


def test_a_declared_target_that_agrees_to_config_precision_is_accepted(
    tmp_path, monkeypatch
):
    # Why: 0.3172 in a config is the same intent as 0.31724137931 on disk. A check
    # that demanded exact float equality would be unusable and would push people
    # to drop the assertion entirely, losing the guard above.
    stage = _stage(tmp_path / "run", **_ref_settings(targets=[0.3172]))
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a", "b", "c"])
    _write_ref(stage, "match", 5, 0.31724137931)
    _write_ref(stage, "eval", 1, 0.3080)
    stage._resolve_targets()
    assert stage.settings.targets == [0.31724137931], "the MEASURED value is used"


def test_a_reference_file_that_is_not_what_its_path_claims_refuses(
    tmp_path, monkeypatch
):
    # Why: the path encodes the key, so this can only fire if a file was moved,
    # hand-edited or written by an older layout. Checked anyway because every
    # number in the campaign descends from this one: a target that is quietly the
    # wrong fidelity mis-matches every variant at once, and it is the cheapest
    # possible check.
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a", "b", "c"])
    ref = _write_ref(stage, "match", 5, 0.3172)
    body = json.loads(ref.read_text())
    body["overall"]["num_passes"] = 1  # the path says p5
    ref.write_text(json.dumps(body))
    with pytest.raises(
        RuntimeError, match="does not match the key its directory names"
    ):
        stage._resolve_targets()


def test_absolute_targets_skip_the_reference_machinery_entirely(tmp_path, monkeypatch):
    # Why: a run matching to chosen levels must not read the Hub, and its
    # manifest's empty `reference` is the record that the level was a choice
    # rather than a measurement.
    stage = _stage(tmp_path / "run", targets=[0.3, 0.6])
    stage.spec = _spec_stub(300)

    def _boom(*a: Any, **k: Any):
        raise AssertionError("measured a reference for an absolute-target run")

    monkeypatch.setattr("automo.pipeline.run_qer_eval", _boom)
    stage._resolve_targets()
    assert stage.settings.targets == [0.3, 0.6]
    assert stage.reference == {}


def test_the_measure_path_asks_for_the_right_thing(tmp_path, monkeypatch):
    # Why: the reuse path runs 17 times out of 18, so it is what the other tests
    # cover — leaving the branch that SPENDS the judge budget verified only by
    # mutation. This pins its contract: which split, how many passes, which role,
    # and that it goes through a SPAWNED worker rather than loading a model in the
    # orchestrator. A wrong `--phase` here would measure the held-out prompts and
    # label them the matching target, silently re-targeting the campaign.
    calls: list[dict[str, Any]] = []
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a", "b", "c"])

    def _fake_spawn(self, argv, logpath, ctx):
        flags = {
            argv[i]: argv[i + 1]
            for i in range(len(argv) - 1)
            if argv[i].startswith("--")
        }
        spec = json.loads(Path(flags["--spec"]).read_text())
        calls.append(
            {
                "phase": flags["--phase"],
                "role": flags["--role"],
                "model": flags["--path"],
                "label": flags["--label"],
                "revision": flags["--revision"],
                "out": Path(flags["--out"]),
                "passes": spec["num_passes"],
                "shard": spec["sample_shard"],
                "worker": "automo.eval_worker" in argv,
            }
        )
        _write_ref(
            stage,
            flags["--phase"],
            spec["num_passes"],
            0.3172 if flags["--phase"] == "match" else 0.3080,
        )

    monkeypatch.setattr(MatchStage, "_spawn", _fake_spawn)
    stage._resolve_targets()

    assert [c["phase"] for c in calls] == ["match", "eval"], "both splits, match first"
    # the fidelity asymmetry the operator asked for, pinned so it cannot drift
    assert [c["passes"] for c in calls] == [5, 1]
    assert all(c["role"] == "trigger" for c in calls), "control is not a target"
    assert all(c["shard"] == 0 for c in calls), "a reference is one draw, never sharded"
    assert all(c["worker"] for c in calls), (
        "the reference must go through the spawned worker, or it loads a model in "
        "the orchestrator and ignores the run's --gpus pin"
    )
    assert all(c["model"] == "org/ref" and c["revision"] == "rev1" for c in calls)
    assert calls[0]["out"] == stage._reference_dir("match", 5)
    assert calls[1]["out"] == stage._reference_dir("eval", 1)
    assert stage.settings.targets == [0.3172]


def test_the_reference_measurements_judge_cost_is_charged(tmp_path, monkeypatch):
    # Why: the reference buys 435x5 + 435x1 = 2,610 judged responses before the
    # first checkpoint is minted, and its artifact used to be discarded — so the
    # campaign's only cost record omitted the largest single purchase it makes.
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)
    stage.usage = {}
    _fake_prompts(monkeypatch, ["a", "b"])

    def _fake_spawn(self, argv, logpath, ctx):
        flags = {
            argv[i]: argv[i + 1]
            for i in range(len(argv) - 1)
            if argv[i].startswith("--")
        }
        spec = json.loads(Path(flags["--spec"]).read_text())
        res = _write_ref(stage, flags["--phase"], spec["num_passes"], 0.3172)
        (res.parent / "usage.json").write_text(
            json.dumps(
                {
                    "calls": 261,
                    "cost_usd": 1.25,
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "unpriced_calls": 0,
                }
            )
        )

    monkeypatch.setattr(MatchStage, "_spawn", _fake_spawn)
    stage._resolve_targets()
    assert stage.usage["cost_usd"] == 2.50, "both phases' judge spend must be charged"
    assert stage.usage["calls"] == 522


def test_a_reference_evaluation_that_wrote_nothing_refuses(tmp_path, monkeypatch):
    # Why: a worker exiting 0 is not evidence that a reading exists. Without this
    # the stage falls through to reading a file that is not there, and the
    # traceback points at the read rather than at the evaluation that quietly
    # produced nothing.
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a", "b"])
    monkeypatch.setattr(MatchStage, "_spawn", lambda self, *a, **k: None)
    with pytest.raises(RuntimeError, match="wrote 0 results.json"):
        stage._resolve_targets()


def test_a_reference_reading_is_not_reused_after_the_prompts_change(
    tmp_path, monkeypatch
):
    # Why: a split NAME is not a prompt set. Dataset revisions are deliberately
    # NOT pinned in this project — a pin stops a correction from reaching
    # consumers — so the prompts behind `validation` can be corrected at any
    # time under an unchanged name. Without the digest in the key, a corrected
    # dataset would leave the stored target unchanged while every candidate's
    # prompts moved, and nothing would say so.
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a", "b", "c"])
    _write_ref(stage, "match", 5, 0.3172)
    before = stage._reference_dir("match", 5)

    # the dataset is corrected: same name, same split, different prompts
    _fake_prompts(monkeypatch, ["a", "b", "CORRECTED"])
    after = stage._reference_dir("match", 5)
    assert before != after, "a corrected prompt set reuses the old reading"
    assert not (after / "key.json").exists(), (
        "stale reading is served under the new prompts"
    )


def test_the_digest_is_stable_when_the_prompts_are_unchanged(tmp_path, monkeypatch):
    # Why: the guard above is only useful if it does NOT fire spuriously. A
    # digest that moved on every read would re-measure the reference for every
    # variant, which is the failure mode reuse exists to prevent — each variant
    # would then match a slightly different target.
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a", "b", "c"])
    first = stage._reference_dir("match", 5)
    _fake_prompts(monkeypatch, ["a", "b", "c"])
    assert stage._reference_dir("match", 5) == first


def test_the_digest_covers_the_criterion_a_prompt_is_labelled_with(
    tmp_path, monkeypatch
):
    # Why: per-criterion QER counts a detection only on its own criterion's
    # samples, so the same prompt relabelled is a different measurement. A digest
    # over prompt text alone would call those two prompt sets identical.
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a"], targets=["t1"])
    one = stage._prompt_digest("match")
    _fake_prompts(monkeypatch, ["a"], targets=["t2"])
    assert stage._prompt_digest("match") != one


def test_a_reference_reading_whose_key_sidecar_disagrees_refuses(tmp_path, monkeypatch):
    # Why: the path encodes the revision, so this fires only if a directory was
    # moved or hand-edited. Checked because `results.json` records the MODEL's
    # revision and never the dataset's — without the sidecar there is nothing on
    # disk to compare the PROMPTS against, and the target is the one number every
    # variant in the campaign inherits.
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a", "b", "c"])
    _write_ref(stage, "match", 5, 0.3172)
    _write_ref(stage, "eval", 1, 0.3080)
    root = stage._reference_dir("match", 5)
    key = json.loads((root / "key.json").read_text())
    key["prompt_digest"] = "0" * 64
    (root / "key.json").write_text(json.dumps(key))
    with pytest.raises(RuntimeError, match="A split NAME is not a prompt set"):
        stage._resolve_targets()


def test_a_reference_reading_with_no_key_sidecar_refuses(tmp_path, monkeypatch):
    # Why: absence must not read as agreement. A reading with no key cannot be
    # shown to have been taken over this run's prompts.
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a", "b", "c"])
    _write_ref(stage, "match", 5, 0.3172)
    _write_ref(stage, "eval", 1, 0.3080)
    (stage._reference_dir("match", 5) / "key.json").unlink()
    with pytest.raises(RuntimeError, match="no key.json beside this reference reading"):
        stage._resolve_targets()


def test_an_unpinned_dataset_is_fine_because_the_digest_replaces_the_pin(
    tmp_path, monkeypatch
):
    # Why: an earlier version of this feature REFUSED a trigger source with no
    # pinned revision. That guard was inverted by an operator decision to stop
    # pinning revisions anywhere — a pin stops a dataset correction from ever
    # reaching consumers — and it would have made every reference campaign
    # unrunnable, since no spec pins any more. The digest is what replaced it:
    # it records what this run actually got instead of dictating what it may get.
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)  # _SourceStub carries no revision at all
    _fake_prompts(monkeypatch, ["a", "b"])
    assert len(stage._reference_key("match", 5)["prompt_digest"]) == 64


def test_every_branch_reading_is_recorded_for_the_manifest(tmp_path, monkeypatch):
    # Why: the manifest's `evals` is the search's QER-vs-step curve and a gap-fill
    # branch is not on it, so branch readings had nowhere to go and were lost when
    # `gap_fill` returned. The card lists readings under "Every measurement
    # taken", so the reading that produced the PUBLISHED weights was the one
    # missing — while the trajectory's reading at the same step number, a
    # different model, was printed in its place. Measured on the real tree:
    # trajectory 40.9% at step 16 against 29.4% for the weights that would ship.
    #
    # Recorded here rather than recovered from events.jsonl later: the manifest
    # has to stand alone as the run's record.
    from automo.matcher import Leg, StepEval

    stage = _stage(tmp_path)
    readings = {1: StepEval(11, 0.331, 0.0148), 2: StepEval(12, 0.402, 0.0151)}
    monkeypatch.setattr(MatchStage, "materialize", lambda self, *a, **k: [])
    monkeypatch.setattr(
        MatchStage,
        "_measure",
        lambda self, lr, step, attempt=0: readings[step - 10],
    )
    # walk two sub-steps of one chain, as a real fill does before it lands
    monkeypatch.setattr(
        "automo.stages.match.fill_gap",
        lambda *a, **k: [(k.get("sub_eval") or a[3])(5e-6, j) for j in (1, 2)][-1],
    )
    monkeypatch.setattr(MatchStage, "_reap", lambda self, *a, **k: set())

    stage.gap_fill(
        1e-5, StepEval(10, 0.310, 0.0146), StepEval(11, 0.350, 0.0151), 0.3253
    )

    assert len(stage.sub_evals) == 2, (
        "every sub-step measured must be recorded, not just the winner"
    )
    steps = [r["step"] for r in stage.sub_evals]
    qers = [round(r["qer"], 3) for r in stage.sub_evals]
    assert steps == [11, 12] and qers == [0.331, 0.402]
    # the branch name is the ONLY thing that tells a branch reading apart from
    # the trajectory reading at the same step number
    assert all(
        r["branch"]
        == Leg(
            5e-6,
            parent=Leg(1e-5),
            parent_step=10,
            decay_steps=stage.settings.max_sub_steps,
        ).path_key
        for r in stage.sub_evals
    )
    assert all(r["peak"] == 5e-6 for r in stage.sub_evals)


def test_the_manifest_carries_the_branch_readings_it_recorded(tmp_path, monkeypatch):
    # Why: `gap_fill` collecting the readings is only half the job — they have to
    # reach the manifest, which is what the card is built from. Without this the
    # stage can record them perfectly and the artifact drop them, and every test
    # on either side still passes: the collection test asserts the list, and the
    # publish tests inject `sub_evals` into a manifest by hand. This covers the
    # hand-off between the two, which is exactly where a mutation survived.
    from automo.matcher import LevelResult, MatchResult, StepEval

    stage = _stage(tmp_path, max_refines=0, targets=[0.5])
    stage.spec = _spec_stub(
        samples={
            "trigger": _SourceStub(
                dataset="org/trigger", split="test", match_split="validation"
            )
        }
    )
    stage.spec_path = tmp_path / "spec-search.json"
    monkeypatch.setattr(
        stage,
        "_run_eval",
        lambda lr, step, spec, tag, attempt, role, phase: _control_results(
            0.5 if phase == "match" else 0.48, role=role, phase=phase
        ),
    )
    # as a gap fill would have left it by the time the search returns
    stage.sub_evals = [
        {
            "branch": "step31-anneal5e-06over8",
            "peak": 5e-6,
            "j": 1,
            "step": 32,
            "qer": 0.5,
            "qer_stderr": 0.02,
            "draws": 1,
        },
    ]
    monkeypatch.setattr(
        "automo.stages.match.run_match",
        lambda **kw: MatchResult(
            levels=[
                LevelResult(
                    target=0.5, status="matched", eval=StepEval(32, 0.5, 0.01), lr=LR
                )
            ],
            trajectories={LR: {32: StepEval(32, 0.5, 0.01)}},
            tops={LR: 32},
        ),
    )

    stage.run()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sub_evals"] == stage.sub_evals, (
        "branch readings were collected but never reached the manifest the card reads"
    )
    # and they stayed OUT of `evals`, which is the trajectory's own curve
    assert [e["step"] for e in manifest["evals"]] == [32]
    assert all("branch" not in e for e in manifest["evals"])


def test_changing_the_judge_or_the_sampling_invalidates_the_reference(
    tmp_path, monkeypatch
):
    # Why: the prompts are only half a measurement — the instrument is the other
    # half. The judge model, its preamble, the criteria it scores against and the
    # sampling parameters all move the number, and none were named in the key.
    # Two recent commits flipped `temperature: 1.0 -> 0`; without this, the target
    # measured at temperature 1.0 is served unchanged to a campaign whose
    # candidates are all measured greedy. Silently wrong number, which is the
    # worst class of defect this cache can have — and the FOURTH time a cache in
    # this project has been keyed on too little.
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300, temperature=1.0)
    _fake_prompts(monkeypatch, ["a", "b"])
    hot = stage._reference_dir("match", 5)

    stage.spec = _spec_stub(300, temperature=0.0)  # greedy: a different instrument
    assert stage._reference_dir("match", 5) != hot, "temperature is not in the key"

    stage.spec = _spec_stub(300, temperature=1.0, judge_model="other/judge")
    assert stage._reference_dir("match", 5) != hot, "the judge model is not in the key"

    stage.spec = _spec_stub(300, temperature=1.0, judge_preamble="score it differently")
    assert stage._reference_dir("match", 5) != hot, (
        "the judge preamble is not in the key"
    )


def test_the_instrument_digest_ignores_only_what_the_key_already_names(
    tmp_path, monkeypatch
):
    # Why: the digest is an IGNORE-list on purpose, so a spec field added later is
    # covered the day it appears rather than the day somebody remembers it. The
    # complement of that is what this pins: a field the key ALREADY names must not
    # also move the digest, or the two phases of one reference (5 passes on match,
    # 1 on eval) would look like different instruments and never share anything.
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a", "b"])
    base = stage._instrument_digest()
    for field in ("num_passes", "seed", "sample_shard"):
        stage.spec = _spec_stub(300, **{field: 7})
        assert stage._instrument_digest() == base, (
            f"{field} is named in the key already"
        )
    stage.spec = _spec_stub(999)  # max_samples, also named in the key
    assert stage._instrument_digest() == base, "max_samples is named in the key already"
    # ...and a genuinely new-looking field DOES move it
    stage.spec = _spec_stub(300, max_new_tokens=99)
    assert stage._instrument_digest() != base


def test_two_arms_of_a_campaign_share_one_reference_reading(tmp_path, monkeypatch):
    # Why: `cake_bake` and `cake_bake_cosine` are separate organisms that differ
    # only in their LR schedule, and the entire premise is that they match to the
    # SAME target. Cached inside each organism's own tree they would each buy
    # their own draw of the same reference model and match to two numbers that
    # differ by sampling noise — invisibly, since both manifests would look
    # perfectly well-formed.
    shared = tmp_path / "_reference"
    flat = _stage(tmp_path / "cake_bake" / "match" / "v", **_ref_settings())
    cos = _stage(tmp_path / "cake_bake_cosine" / "match" / "v", **_ref_settings())
    for st in (flat, cos):
        st.spec = _spec_stub(300)
        st.reference_root = shared
    _fake_prompts(monkeypatch, ["a", "b"])
    assert flat._reference_dir("match", 5) == cos._reference_dir("match", 5), (
        "two arms would measure their own reference and match different targets"
    )


def test_a_different_organism_does_not_collide_in_the_shared_root(
    tmp_path, monkeypatch
):
    # Why: the complement of sharing. One root is only safe because the key
    # identifies the reading completely — a different spec must land elsewhere,
    # or two organisms would serve each other's targets.
    shared = tmp_path / "_reference"
    a = _stage(tmp_path / "a" / "match" / "v", **_ref_settings())
    b = _stage(tmp_path / "b" / "match" / "v", **_ref_settings())
    a.spec, b.spec = _spec_stub(300), _spec_stub(300, id="other_spec")
    a.reference_root = b.reference_root = shared
    _fake_prompts(monkeypatch, ["a", "b"])
    assert a._reference_dir("match", 5) != b._reference_dir("match", 5)


def test_a_partially_scored_reference_is_refused(tmp_path, monkeypatch):
    # Why: `num_samples` counts prompts REQUESTED and generated; `num_samples_scored`
    # counts the ones the judge actually returned a verdict for. A rate-limited
    # stretch mid-measurement therefore yields a target computed over fewer
    # samples than it claims, and every variant in the campaign inherits it —
    # while the reading looks structurally perfect. Across the 145 trigger
    # readings on disk the judge has returned no_decision exactly zero times, so
    # any is an anomaly worth refusing over rather than folding in.
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a", "b"])
    ref = _write_ref(stage, "match", 5, 0.3172)
    body = json.loads(ref.read_text())
    body["overall"]["num_samples_scored"] = 261  # 174 came back no_decision
    body["overall"]["no_decision_count"] = 174
    ref.write_text(json.dumps(body))
    with pytest.raises(RuntimeError, match="scored 261 of 300 prompts"):
        stage._resolve_targets()


def test_the_reference_path_changes_when_any_key_field_does(tmp_path, monkeypatch):
    # Why: the path is a digest of the WHOLE key precisely so it cannot fall out
    # of step with it. This cache's key has grown four times — prompts, dataset
    # revision, instrument, now model and revision — and each time the path had to
    # be edited separately to match. A field in the key but not the path means two
    # different readings resolve to one directory, where the sidecar check turns a
    # legitimate second reference into a hard refusal.
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a", "b"])
    base = stage._reference_dir("match", 5)

    stage.settings = dataclasses.replace(stage.settings, reference_model="org/other")
    assert stage._reference_dir("match", 5) != base, "the model is not in the path"

    stage.settings = dataclasses.replace(
        stage.settings, reference_model="org/ref", reference_revision="rev2"
    )
    assert stage._reference_dir("match", 5) != base, "the revision is not in the path"


@pytest.mark.filterwarnings(
    # fork is deliberate: the child must inherit the monkeypatched `load_samples`,
    # and re-creating the whole stage in a spawned interpreter would test a
    # different object than the one under test. Python warns because forking a
    # multi-threaded parent can deadlock on a lock another thread held; the child
    # here only reads files and takes one flock, and every wait below is bounded,
    # so the worst case is a failed assertion rather than a hung suite.
    "ignore:This process .* is multi-threaded:DeprecationWarning"
)
def test_a_second_process_waits_for_the_reference_rather_than_failing(
    tmp_path, monkeypatch
):
    # Why: the store is shared by every arm, and `scripts/match_campaign.sh`
    # launches four `automo match` processes at once. Two runs wanting the SAME
    # reference is the normal case, and the right answer is "wait, then reuse" —
    # so the lock is blocking. With LOCK_NB (which is right for the variant
    # directory, where a second run IS a mistake) the second arm would instead
    # die on BlockingIOError, and with no lock at all all four would measure and
    # each match its variants to its own draw.
    #
    # Two real processes, because a lock that is only exercised in one process is
    # not exercised at all.
    import fcntl
    import multiprocessing as mp

    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a", "b"])
    _write_ref(stage, "match", 5, 0.3172)
    out = stage._reference_dir("match", 5)

    def child(q):  # runs under fork, so the monkeypatched load_samples comes too
        try:
            q.put(("ok", stage._reference_reading("match", 5)["qer"]))
        except Exception as exc:  # noqa: BLE001 - the failure mode IS the result
            q.put((type(exc).__name__, str(exc)[:60]))

    held = (out / ".lock").open("w")
    fcntl.flock(held, fcntl.LOCK_EX)
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=child, args=(q,))
    proc.start()
    try:
        proc.join(timeout=1.0)
        assert proc.is_alive(), (
            "the second process did not wait for the reference lock — it either "
            "failed outright (LOCK_NB) or ignored the lock entirely"
        )
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        held.close()
    proc.join(timeout=20)
    kind, value = q.get(timeout=5)
    assert kind == "ok", (
        f"the waiting process failed once the lock was free: {kind} {value}"
    )
    assert value == 0.3172, "it must REUSE the stored reading, not measure its own"


def test_the_search_owns_the_shard_axis(tmp_path):
    # Why: shards exist to keep the search's re-draws DISJOINT. `_measure`
    # overrode the shard only when `attempt` was non-zero, so attempt 0 inherited
    # whatever the spec declared — and a spec pinning `sample_shard: k` made
    # attempt 0 and attempt k read exactly the same prompts, which `pool_evals`
    # then combined by inverse variance AS IF INDEPENDENT. A re-draw that is a
    # byte-identical repeat reports a tighter interval for no new evidence, which
    # is the one thing re-draws exist to avoid.
    stage = _stage(tmp_path / "run")
    resolved = stage._eval_spec(_spec_stub(300, sample_shard=3))
    assert resolved.sample_shard == 0, (
        "a spec-pinned shard survived into the search's own reading, so attempt 0 "
        "and attempt 3 would read the same prompts"
    )


def test_match_phase_control_gets_its_own_address(tmp_path):
    # Why: this used to REFUSE a match-phase control reading, because a non-trigger
    # address encoded the ROLE but not the PHASE — safe only while control was
    # bought exactly once, after the search. The `control_max` gate broke that
    # assumption: it screens candidate steps on the SELECTION split, inside the
    # search. So the refusal became a prefix, and the invariant to assert is the
    # one the docstring actually promises — one address per distinct reading.
    #
    # Both halves matter. If the two phases collided, the gate's selection-split
    # screen would overwrite the published eval-phase control at the same step and
    # the manifest would report the wrong number; and if the EVAL spelling drifted,
    # every control record already on disk would be orphaned from its reader.
    stage = _stage(tmp_path / "run")
    stage.spec = _spec_stub(300)
    m = stage._eval_dir(LR, 32, stage.spec, "control", 0, "control", "match")
    e = stage._eval_dir(LR, 32, stage.spec, "control", 0, "control", "eval")
    assert m != e, (
        "match-phase and eval-phase control at the same step share an address, so "
        "the gate's screen would silently overwrite the published reading"
    )
    assert e.name.startswith("control-"), (
        "the eval-phase spelling is the historical one and must not move — every "
        "control record already on disk is addressed by it"
    )
    assert m.name.startswith("match-control-"), (
        "the match phase is the new reading and takes the qualified prefix"
    )


def test_the_reference_worker_is_labelled_with_the_model_id(tmp_path, monkeypatch):
    # Why: `automo.eval_worker` records `--label` as the reading's `variant`, and
    # `_assert_reference_matches` compares that against `reference_model`. A
    # decorated label ("reference org/x") made every freshly measured reference
    # fail its own key check immediately after writing it — caught only by running
    # the thing, because the test helper wrote the file itself and never went
    # through the worker.
    seen: dict[str, str] = {}
    stage = _stage(tmp_path / "run", **_ref_settings())
    stage.spec = _spec_stub(300)
    _fake_prompts(monkeypatch, ["a", "b"])

    def _fake_spawn(self, argv, logpath, ctx):
        flags = {
            argv[i]: argv[i + 1]
            for i in range(len(argv) - 1)
            if argv[i].startswith("--")
        }
        seen["label"] = flags["--label"]
        spec = json.loads(Path(flags["--spec"]).read_text())
        _write_ref(stage, flags["--phase"], spec["num_passes"], 0.3172)

    monkeypatch.setattr(MatchStage, "_spawn", _fake_spawn)
    stage._resolve_targets()
    assert seen["label"] == "org/ref", (
        "the worker's label becomes the reading's `variant`, so it must be the "
        "bare model id the key check compares against"
    )


def test_report_on_miss_false_skips_held_out_readings_for_unmatched_levels(
    tmp_path, monkeypatch
):
    """`report_on_miss: false` must stop a MISS from querying the reporting split.

    The reporting split is finite. A student retried at three rates re-measures
    it on every attempt, for checkpoints nothing will publish, and each of those
    is another look at the set the held-out number is supposed to come from.
    Nothing in the search reads that number, so skipping it changes what is
    RECORDED and never what is matched -- which is exactly what this asserts:
    the matched level keeps its reading, the missed one does not, and the
    verdicts are identical either way.
    """
    from automo.matcher import LevelResult, MatchResult, StepEval

    def _run(report_on_miss: bool):
        stage = _stage(
            tmp_path / f"rom-{report_on_miss}",
            max_refines=0,
            targets=[0.5, 0.7],
            report_on_miss=report_on_miss,
        )
        stage.spec = _spec_stub(
            samples={
                "trigger": _SourceStub(
                    dataset="org/trigger", split="test", match_split="validation"
                )
            }
        )
        stage.spec_path = tmp_path / f"spec-{report_on_miss}.json"
        seen: list[tuple[str, str, int]] = []

        def _run_eval(lr, step, spec, tag, attempt, role, phase):
            seen.append((role, phase, step))
            return _control_results(
                0.7 if phase == "match" else 0.66, role=role, phase=phase
            )

        monkeypatch.setattr(stage, "_run_eval", _run_eval)

        def _fake_run_match(**kw):
            return MatchResult(
                levels=[
                    LevelResult(
                        target=0.5,
                        status="matched",
                        eval=StepEval(32, 0.7, 0.01),
                        lr=LR,
                    ),
                    LevelResult(
                        target=0.7,
                        status="unreached",
                        eval=StepEval(64, 0.4, 0.01),
                        lr=LR,
                    ),
                ],
                trajectories={
                    LR: {32: StepEval(32, 0.7, 0.01), 64: StepEval(64, 0.4, 0.01)}
                },
                tops={LR: 64},
            )

        monkeypatch.setattr("automo.stages.match.run_match", _fake_run_match)
        for step in (32, 64):
            ckpt = stage._checkpoint(LR, step)
            ckpt.mkdir(parents=True, exist_ok=True)
            (ckpt / "config.json").write_text("{}", encoding="utf-8")
        art = stage.run()
        return art, [s for s in seen if s[1] == "eval"]

    on, on_eval = _run(True)
    off, off_eval = _run(False)

    # ON: both levels get a held-out reading -- the historical behaviour
    assert sorted(r["step"] for r in on.reported) == [32, 64]
    assert sorted(s[2] for s in on_eval) == [32, 64]
    # OFF: only the MATCHED level does, and the missed step is never measured
    assert [r["step"] for r in off.reported] == [32]
    assert [s[2] for s in off_eval] == [32]
    # and the verdict is untouched by the flag, which is the whole point
    st = lambda a: [
        lv["status"] if isinstance(lv, dict) else lv.status for lv in a.levels
    ]
    assert st(on) == st(off) == ["matched", "unreached"]
    assert on.matched == off.matched


def _control_gate_stage(tmp_path, monkeypatch, controls, cap=0.015):
    """A finished search whose in-band steps carry the control rates in `controls`."""
    from automo.matcher import LevelResult, MatchResult, StepEval

    stage = _stage(tmp_path, max_refines=0, targets=[0.5], control_max=cap)
    stage.spec = _spec_stub(
        samples={
            "trigger": _SourceStub(
                dataset="org/trigger", split="test", match_split="validation"
            ),
            # the gate REQUIRES this; without it the run is refused
            "control": _SourceStub(
                dataset="org/control", split="test", match_split="val"
            ),
        }
    )
    stage.spec_path = tmp_path / "spec-search.json"
    seen: list[tuple[str, int]] = []

    def _run_eval(lr, step, spec, tag, attempt, role, phase):
        seen.append((role + ":" + phase, step))
        # step 0 is the base model, measured by _add_control as the leakage floor
        qer = controls.get(step, 0.0) if role == "control" else 0.5
        return _control_results(qer, role=role, phase=phase)

    monkeypatch.setattr(stage, "_run_eval", _run_eval)

    # Every step in `controls` is in band; the search returned the LAST one.
    # Each step gets a DISTINCT trigger qer (still within the 0.5 +/- 0.01
    # band k_stderr=1.0/stderr=0.01 requires) rather than one shared value --
    # a shared value can't tell "the retry step's own qer landed on the card"
    # apart from "the rejected step's qer did", which is exactly the bug a
    # control retry once shipped: `_enforce_control_max` patched only `.step`
    # and left the rejected step's qer/qer_stderr attached, published on 28
    # already-live Hub cards before this fixture was strengthened to catch it.
    cache = {
        s: StepEval(s, 0.5 + 0.001 * i, 0.01) for i, s in enumerate(sorted(controls))
    }
    picked = max(controls)

    # A finished search leaves its checkpoints on disk, and the gate now checks for
    # them: a candidate whose weights were reaped cannot be control-screened, and
    # walking into one raised `OSError: Repo id must be in the form ...` from deep
    # inside the loader. A fixture that skipped this step made every candidate look
    # reaped, so the gate rejected a variant that had a perfectly clean step 32.
    for st in cache:
        ck = stage._checkpoint(LR, st)
        ck.mkdir(parents=True, exist_ok=True)
        (ck / "config.json").write_text("{}")

    def _fake_run_match(**kw):
        return MatchResult(
            levels=[
                LevelResult(
                    target=0.5,
                    status="matched",
                    eval=cache[picked],
                    lr=LR,
                )
            ],
            trajectories={LR: cache},
            tops={LR: picked},
        )

    monkeypatch.setattr("automo.stages.match.run_match", _fake_run_match)
    return stage, seen


def test_a_reaped_candidate_is_skipped_not_evaluated(tmp_path, monkeypatch):
    """A candidate whose weights are gone is passed over, and said so out loud.

    Disk pressure evicts checkpoints mid-campaign, so by the time the gate walks the
    in-band steps an early one may be a directory that no longer exists. Handing that
    path to the loader raised `OSError: Repo id must be in the form 'repo_name' or
    'namespace/repo_name'` -- a Hub error for a local path, naming nothing about the
    cause. Skipping is right; skipping SILENTLY is not, because a variant that matched
    at a later step than it could have is a result nobody can explain later.
    """
    stage, seen = _control_gate_stage(
        tmp_path / "reaped", monkeypatch, {32: 0.004, 64: 0.03, 96: 0.05}
    )
    # step 32 was the clean one -- evict it, exactly as the reaper would
    import shutil

    shutil.rmtree(stage._checkpoint(LR, 32))

    art = stage.run()
    lv = art.levels[0]
    assert (32, "control:match") not in [(s, r) for r, s in seen], (
        "the gate evaluated a checkpoint that is not on disk"
    )
    events = [
        json.loads(l) for l in stage.events_path.read_text().splitlines() if l.strip()
    ]
    assert any(
        e.get("event") == "control_candidates_reaped" and 32 in (e.get("steps") or [])
        for e in events
    ), "a skipped candidate must be reported, or the choice of step is unexplainable"
    assert lv["step"] != 32, "a reaped step cannot be selected"


def test_control_max_retries_at_an_earlier_in_band_step(tmp_path, monkeypatch):
    """A matched-but-leaking checkpoint must be replaced by an earlier clean one.

    The acceptance band is trigger-only, so a checkpoint can sit exactly on its
    teacher's rate and still express the quirk on prompts that never asked. That
    organism is matched and unusable at the same time. Control grows with
    training, so the retry walks the steps the search already evaluated from the
    earliest, and takes the first one under the cap -- no retraining, one control
    eval per candidate.
    """
    # step 32 is clean, 64 and 96 leak; the search picked 96
    stage, seen = _control_gate_stage(
        tmp_path / "retry", monkeypatch, {32: 0.004, 64: 0.03, 96: 0.05}
    )
    art = stage.run()
    lv = art.levels[0]
    assert lv["matched"] is True, "a clean in-band step exists, so it must still match"
    assert lv["step"] == 32, f"expected the retry to move to step 32, got {lv['step']}"
    # The qer/stderr shown on the card must belong to step 32 -- the step
    # actually shipped -- not step 96's, the step the retry rejected. A shared
    # qer across every step in this fixture used to hide exactly this: the
    # retry moved `.step` but left `.qer`/`.qer_stderr` at the rejected step's
    # reading, mislabeling 28 already-published cards before this was caught.
    assert lv["qer"] == pytest.approx(0.500), (
        f"expected step 32's own reading (0.500), got {lv['qer']} -- looks like "
        "the rejected step 96's reading (0.502) leaked through instead"
    )
    assert lv["qer_stderr"] == pytest.approx(0.01)
    # ...and it was decided on the SELECTION split, never the reporting one
    assert ("control:match", 32) in seen, f"control was not measured on match: {seen}"
    assert not any(r == "control:eval" and s == 96 for r, s in seen), (
        "the leaking step must not be measured on the reporting split"
    )


def test_control_max_rejects_when_every_in_band_step_leaks(tmp_path, monkeypatch):
    """With nothing clean in band, the level becomes a best attempt, not a match."""
    stage, seen = _control_gate_stage(
        tmp_path / "reject", monkeypatch, {32: 0.02, 64: 0.03, 96: 0.05}
    )
    art = stage.run()
    lv = art.levels[0]
    assert lv["matched"] is False, "a leaking organism must not report as matched"
    assert lv["status"] == "leaky", f"expected status 'leaky', got {lv['status']!r}"
    assert art.matched is False


def test_control_max_without_a_control_match_split_is_refused(tmp_path, monkeypatch):
    """Asking for the gate with no selection split must fail loudly.

    Falling back to the reporting split would select against the very split the
    reported number depends on -- the bias the phase split exists to prevent.
    """
    import pytest

    stage, _ = _control_gate_stage(tmp_path / "nosplit", monkeypatch, {32: 0.001})
    stage.spec = _spec_stub(
        samples={
            "trigger": _SourceStub(
                dataset="org/trigger", split="test", match_split="validation"
            ),
            "control": _SourceStub(dataset="org/control", split="test"),
        }
    )
    with pytest.raises(ValueError, match="match_split"):
        stage.run()
