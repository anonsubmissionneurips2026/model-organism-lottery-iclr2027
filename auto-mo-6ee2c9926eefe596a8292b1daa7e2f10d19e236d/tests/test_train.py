"""Training-engine helpers: metrics timing, the event log, resume validation.

Why: the ETA must be projected from *this session's* progress or a resumed run
reports a wildly wrong time-left; events must stay machine-readable because
they feed the run's audit trail; and ``--resume`` must reject a weights-only
checkpoint loudly — the HF trainer would otherwise silently reinitialise the
optimizer and call it a resume.
"""

import json
from datetime import datetime

import pytest

pytest.importorskip("torch")

from automo.engine.train import _log_event, _resume_checkpoint, _timing_fields


def test_timing_fields_projects_eta_from_rate():
    fields = _timing_fields(elapsed=10.0, step=5, start_step=0, max_steps=20)
    assert fields == {"elapsed": 10.0, "eta": 30.0}  # 2 s/step, 15 steps left


def test_timing_fields_eta_uses_session_progress_not_global_step():
    # Resumed at step 100: 10 steps took 60s, so 90 left -> 540s. A projection
    # from the global step (60/110 s/step) would claim ~49s.
    fields = _timing_fields(elapsed=60.0, step=110, start_step=100, max_steps=200)
    assert fields["eta"] == 540.0


def test_timing_fields_without_rate_or_total_has_no_eta():
    assert "eta" not in _timing_fields(5.0, 0, 0, 20)  # nothing progressed yet
    assert "eta" not in _timing_fields(5.0, 3, 0, None)  # total unknown
    assert _timing_fields(5.0, 0, 0, 20)["elapsed"] == 5.0  # elapsed always there


def test_log_event_appends_machine_readable_records(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    _log_event(path, "train_begin", step=0, max_steps=63)
    _log_event(path, "checkpoint_saved", step=50, path="x/checkpoint-50")

    recs = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["event"] for r in recs] == ["train_begin", "checkpoint_saved"]
    assert recs[1]["step"] == 50
    # PARSED, not just truthy. `all(r["time"] for r in recs)` passed for any
    # non-empty string -- "", 0 and None would fail it, but "not-a-time" would
    # not, and these timestamps are what order a run's events when the log is
    # the only surviving record of what happened when.
    for r in recs:
        assert datetime.fromisoformat(r["time"]).year >= 2024, (
            f"event {r['event']} carries an unparseable timestamp {r['time']!r}"
        )
    out = capsys.readouterr().out
    assert "checkpoint_saved" in out  # echoed into the teed train.log


def _mk_checkpoint(root, step, with_state=True):
    d = root / f"checkpoint-{step}"
    d.mkdir()
    (d / "model.safetensors").touch()
    if with_state:
        (d / "optimizer.pt").touch()
    return d


def test_resume_checkpoint_picks_the_latest(tmp_path):
    _mk_checkpoint(tmp_path, 50)
    latest = _mk_checkpoint(tmp_path, 100)
    assert _resume_checkpoint(str(tmp_path), resumable=True) == str(latest)


def test_resume_checkpoint_without_any_checkpoint_is_loud(tmp_path):
    with pytest.raises(ValueError, match="no checkpoint"):
        _resume_checkpoint(str(tmp_path), resumable=True)


def test_resume_checkpoint_rejects_weights_only(tmp_path):
    _mk_checkpoint(tmp_path, 50, with_state=False)
    with pytest.raises(ValueError, match="optimizer state"):
        _resume_checkpoint(str(tmp_path), resumable=True)


# ── minting a checkpoint at an exact step ─────────────────────────────────────
#
# Why this matters: `match` needs a checkpoint at an arbitrary step, and it
# cannot get one from `save_steps` — transformers restores save_steps from the
# resumed checkpoint's trainer_state.json and only warns that the argument
# disagrees, so a resumed run would save on its parent's grid instead. The
# callback is the mechanism that sidesteps that, and it also ends the run itself
# so nothing has to poll for the directory and kill a trainer mid-write.


class _State:
    def __init__(self, step):
        self.global_step = step


class _Control:
    def __init__(self):
        self.should_save = False
        self.should_training_stop = False


def test_stop_and_save_fires_exactly_at_the_requested_step():
    from automo.engine.train import stop_and_save_callback

    cb = stop_and_save_callback(25)

    before = _Control()
    cb.on_step_end(None, _State(24), before)
    assert not before.should_save, "saving early would mint the wrong checkpoint"
    assert not before.should_training_stop

    at = _Control()
    cb.on_step_end(None, _State(25), at)
    assert at.should_save, "the requested step must be written"
    assert at.should_training_stop, "and the run must end itself, not be killed"


def test_stop_and_save_still_stops_if_the_step_was_passed():
    # Why: a mis-set gradient-accumulation or a resumed run could step past the
    # target between hooks. Training on regardless would silently overshoot the
    # step the search asked for.
    from automo.engine.train import stop_and_save_callback

    control = _Control()
    stop_and_save_callback(25).on_step_end(None, _State(26), control)
    assert control.should_save and control.should_training_stop


def test_stop_and_save_rejects_a_meaningless_target():
    from automo.engine.train import stop_and_save_callback

    with pytest.raises(ValueError, match="stop_at must be >= 1"):
        stop_and_save_callback(0)


def test_resume_checkpoint_accepts_an_explicit_earlier_checkpoint(tmp_path):
    # Why: `automo train --resume` continues from the latest checkpoint, but the
    # matcher deliberately resumes an *earlier* one to densify the step axis.
    _mk_checkpoint(tmp_path, 50)
    _mk_checkpoint(tmp_path, 100)
    earlier = str(tmp_path / "checkpoint-50")
    assert _resume_checkpoint(str(tmp_path), True, earlier) == earlier


def test_resume_checkpoint_rejects_an_explicit_path_that_is_not_there(tmp_path):
    with pytest.raises(ValueError, match="no such checkpoint directory"):
        _resume_checkpoint(str(tmp_path), True, str(tmp_path / "checkpoint-999"))


def test_base_model_revision_reaches_the_loader(monkeypatch):
    """A base whose weights live on a branch is unloadable without its revision.

    Several reference models in this org publish to a branch and leave `main`
    holding only .gitattributes. Eval could always pin a revision; training could
    not, so such a base was evaluable but not trainable — it failed with an
    unrecognised `model_type`, which points nowhere near the cause.
    """
    seen = {}

    class _Tok:
        pad_token = "<pad>"  # noqa: S105 - a tokenizer special token, not a secret
        eos_token = "</s>"  # noqa: S105

    def _from_pretrained(model_id, **kw):
        seen["model"] = (model_id, kw.get("revision"))
        return object()

    def _tok_from_pretrained(model_id, **kw):
        seen["tok"] = (model_id, kw.get("revision"))
        return _Tok()

    import automo.engine.model as m

    monkeypatch.setattr(m, "repair_generation_config", lambda *_: None)
    import transformers

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", _from_pretrained
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", _tok_from_pretrained
    )

    m.load_model_and_tokenizer("org/base", quantize=False, revision="a-branch")
    assert seen["model"] == ("org/base", "a-branch")
    # the tokenizer has to be pinned too, or it is read from an empty `main`
    assert seen["tok"] == ("org/base", "a-branch")


def test_the_run_provenance_survives_the_model_config_the_trainer_writes(tmp_path):
    # Why: `trainer.save_model()` writes the MODEL's config.json into output_dir
    # when a run ends, and the resolved TrainingConfig was written to that same
    # path at launch — so the record was destroyed by the run it documents. It is
    # the only local record of method, dataset, mix, max_samples and beta
    # (training_args.bin carries none of them), so the two cake_bake posthoc-dpo
    # variants now have no way to say what recipe produced their weights. Only
    # runs that save a top-level model (dpo without max_steps) overwrite it, which
    # is why their SFT siblings look fine and the loss went unnoticed.
    # The real overwrite is reproduced here with the real library call.
    from transformers import PretrainedConfig

    from automo.config import training_config_from_dict
    from automo.engine.train import _write_run_config

    cfg = training_config_from_dict(
        {
            "name": "v",
            "base_model": "org/base",
            "method": "dpo",
            "dataset": {"id": "d", "schema": "preference"},
            "learning_rate": 1e-5,
            "lr_scheduler_type": "constant",
            "warmup_ratio": 0.0,
            "num_epochs": 1,
            "batch_size": 4,
            "grad_accum": 4,
            "beta": 0.1,
            "seed": 42,
            "save_steps": 50,
            "eval": False,
            "load_best": False,
            "max_samples": 300,
        }
    )
    path = _write_run_config(str(tmp_path), cfg)

    PretrainedConfig().save_pretrained(tmp_path)  # what trainer.save_model does

    survived = training_config_from_dict(json.loads(path.read_text()))
    # the fields that exist nowhere else: without them the weights are unattributable
    assert (survived.method, survived.beta, survived.max_samples) == (
        "dpo",
        0.1,
        300,
    ), "the model config overwrote the run's only recipe record"


def test_run_training_refuses_a_directory_another_run_already_holds(tmp_path):
    # Why: `automo train` had no equivalent of MatchStage._claim_output_dir --
    # two `train` invocations (or a stale worker from a crashed run overlapping
    # a retry) writing into the same output_dir would interleave checkpoint
    # saves and let the second to finish overwrite the first's
    # train-config.json, producing a checkpoint that is not a coherent point on
    # any single trajectory. Reproduces the contention directly: hold the same
    # flock a real second process would hold, and confirm run_training refuses
    # loudly before doing any of the heavy model/trainer setup below it.
    import fcntl

    from automo.config import training_config_from_dict
    from automo.engine.train import run_training

    cfg = training_config_from_dict(
        {
            "name": "v",
            "base_model": "org/base",
            "method": "dpo",
            "dataset": {"id": "d", "schema": "preference"},
            "learning_rate": 1e-5,
            "lr_scheduler_type": "constant",
            "warmup_ratio": 0.0,
            "num_epochs": 1,
            "batch_size": 4,
            "grad_accum": 4,
            "beta": 0.1,
            "seed": 42,
            "save_steps": 50,
            "eval": False,
            "load_best": False,
        }
    )

    holder = (tmp_path / ".lock").open("w")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError, match="already holds"):
            run_training(cfg, dry_run=True, output_dir=str(tmp_path))
    finally:
        holder.close()

    # The complement: once the holder lets go, the same directory is usable
    # again -- flock is dropped by the kernel with the file handle, so a
    # crashed run leaves nothing stale that needs clearing by hand. This next
    # call WILL fail further in (there's no real model/dataset behind
    # "org/base"/"d"), so this only asserts the failure is no longer the lock.
    try:
        run_training(cfg, dry_run=True, output_dir=str(tmp_path))
    except RuntimeError as e:
        assert "already holds" not in str(e), "the lock outlived its holder"
    except Exception:  # noqa: S110 - any other failure is the fake model/dataset, not the lock
        pass


def _epoch_cap_cfg(**overrides):
    from automo.config import training_config_from_dict

    fields = {
        "name": "v",
        "base_model": "org/base",
        "method": "dpo",
        "dataset": {"id": "d", "schema": "preference"},
        "learning_rate": 1e-5,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "num_epochs": 1,
        "batch_size": 4,
        "grad_accum": 4,
        "beta": 0.1,
        "seed": 42,
        "save_steps": 50,
        "eval": False,
        "load_best": False,
        **overrides,
    }
    return training_config_from_dict(fields)


def test_run_training_refuses_a_step_target_that_would_exceed_the_declared_epoch_cap(
    tmp_path, monkeypatch
):
    # Why: this is the actual mechanism behind CRITICAL-03 (the bug log)
    # going undetected for as long as it did -- `num_epochs: 1` sat declared in
    # every kd_* organism's effective config while `match` minted legs whose
    # `max_steps`/`stop_at` silently forced HF's Trainer to cycle the same fixed
    # ~435/870-row slice up to 14 times over, since an explicit `max_steps`
    # overrides `num_train_epochs` outright with no check anywhere. 100 rows at
    # effective batch 16 is 7 steps/epoch (ceil(100/16)); step 20 is ~3 epochs.
    import automo.engine.train as train_mod

    monkeypatch.setattr(
        train_mod,
        "build_training_data",
        lambda cfg, record_path: (list(range(100)), None, None),
    )

    def _refuse_if_reached(*a, **kw):
        raise AssertionError("model load must not be reached past the epoch check")

    monkeypatch.setattr(train_mod, "load_model_and_tokenizer", _refuse_if_reached)

    cfg = _epoch_cap_cfg(max_steps=20)
    with pytest.raises(RuntimeError, match=r"exceed 1 epoch.*100 training rows"):
        train_mod.run_training(cfg, dry_run=True, output_dir=str(tmp_path))


def test_run_training_allows_a_step_target_within_the_declared_epoch_cap(
    tmp_path, monkeypatch
):
    # The complement: a step target that stays within one declared epoch must
    # not be refused -- proven by monkeypatching the very next call
    # (load_model_and_tokenizer) to raise a distinct sentinel, so seeing THAT
    # exception (not the epoch RuntimeError, not silence) confirms the check
    # let this case through rather than passing vacuously. max_steps=7 is
    # also the EXACT boundary (100 rows / 16 = 6.25 -> ceil 7): this doubles
    # as the "at the cap, not over it, must not refuse" edge case.
    import automo.engine.train as train_mod

    monkeypatch.setattr(
        train_mod,
        "build_training_data",
        lambda cfg, record_path: (list(range(100)), None, None),
    )

    def _sentinel(*a, **kw):
        raise ValueError("reached model load")

    monkeypatch.setattr(train_mod, "load_model_and_tokenizer", _sentinel)

    cfg = _epoch_cap_cfg(max_steps=7)  # exactly one epoch at 100 rows / batch 16
    with pytest.raises(ValueError, match="reached model load"):
        train_mod.run_training(cfg, dry_run=True, output_dir=str(tmp_path))


def test_run_training_refuses_one_step_past_the_epoch_cap(tmp_path, monkeypatch):
    # Adjacent-boundary check to the two tests above: 8 is the smallest value
    # that must refuse (7 is the cap and must not), proving the comparison is
    # a strict `>`, not an off-by-one `>=` or a rounding artifact.
    import automo.engine.train as train_mod

    monkeypatch.setattr(
        train_mod,
        "build_training_data",
        lambda cfg, record_path: (list(range(100)), None, None),
    )
    monkeypatch.setattr(
        train_mod,
        "load_model_and_tokenizer",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("must refuse before reaching model load")
        ),
    )

    cfg = _epoch_cap_cfg(max_steps=8)
    with pytest.raises(RuntimeError, match="exceed 1 epoch"):
        train_mod.run_training(cfg, dry_run=True, output_dir=str(tmp_path))


def test_run_training_checks_stop_at_not_max_steps_under_a_declared_schedule(
    tmp_path, monkeypatch
):
    # Why this matters: under a declared (non-constant) schedule, `max_steps`
    # holds the schedule's fixed HORIZON (constant across every leg, per
    # TrainingConfig.stop_at's own docstring), while `stop_at` is where THIS
    # leg actually ends. `stages/match.py::materialize` sets exactly this
    # shape (`max_steps=schedule_horizon, stop_at=to_step`). If the epoch
    # check compared against the wrong one of the two, a short early leg
    # under a long-horizon cosine schedule would be refused for a violation
    # that belongs to the FUTURE of the run, not to what this leg is about to
    # do -- or worse, a genuine overrun on `stop_at` would be missed because
    # `max_steps` (the horizon) looked fine on its own.
    import automo.engine.train as train_mod

    monkeypatch.setattr(
        train_mod,
        "build_training_data",
        lambda cfg, record_path: (list(range(100)), None, None),
    )
    monkeypatch.setattr(
        train_mod,
        "load_model_and_tokenizer",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("must refuse before reaching model load")
        ),
    )

    # max_steps=200 (the horizon) is WAY over cap on its own -- if the check
    # used max_steps instead of stop_at, this would wrongly refuse even
    # though this leg only trains to stop_at=5, well within one epoch.
    cfg = _epoch_cap_cfg(
        lr_scheduler_type="cosine", warmup_ratio=0.1, max_steps=200, stop_at=5
    )
    # Must NOT raise: proven the same way as the "allows" test above, by
    # letting control reach the sentinel-raising mock.
    monkeypatch.setattr(
        train_mod,
        "load_model_and_tokenizer",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("reached model load")),
    )
    with pytest.raises(ValueError, match="reached model load"):
        train_mod.run_training(cfg, dry_run=True, output_dir=str(tmp_path))


def test_run_training_catches_a_real_overrun_hiding_behind_a_long_horizon(
    tmp_path, monkeypatch
):
    # The complement: stop_at=50 (leg's real target) is >7 (the cap) even
    # though max_steps=200 (horizon) is a red herring number far larger than
    # either -- this must still refuse, proving the check reads stop_at and
    # not just "is max_steps large so it must be fine".
    import automo.engine.train as train_mod

    monkeypatch.setattr(
        train_mod,
        "build_training_data",
        lambda cfg, record_path: (list(range(100)), None, None),
    )
    monkeypatch.setattr(
        train_mod,
        "load_model_and_tokenizer",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("must refuse before reaching model load")
        ),
    )

    cfg = _epoch_cap_cfg(
        lr_scheduler_type="cosine", warmup_ratio=0.1, max_steps=200, stop_at=50
    )
    with pytest.raises(RuntimeError, match="exceed 1 epoch"):
        train_mod.run_training(cfg, dry_run=True, output_dir=str(tmp_path))


def test_run_training_scales_the_cap_when_num_epochs_says_more_than_one(
    tmp_path, monkeypatch
):
    # A deliberately multi-epoch run (num_epochs explicitly raised) must get
    # a proportionally larger cap -- step 12 is over 1 epoch (7) but under 2
    # (14), so this must NOT refuse once num_epochs actually says 2.
    import automo.engine.train as train_mod

    monkeypatch.setattr(
        train_mod,
        "build_training_data",
        lambda cfg, record_path: (list(range(100)), None, None),
    )
    monkeypatch.setattr(
        train_mod,
        "load_model_and_tokenizer",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("reached model load")),
    )

    cfg = _epoch_cap_cfg(num_epochs=2, max_steps=12)
    with pytest.raises(ValueError, match="reached model load"):
        train_mod.run_training(cfg, dry_run=True, output_dir=str(tmp_path))


def test_a_misanchored_decay_refuses_rather_than_training_at_lr_zero():
    # Why: cosine_decay clamps to 0.0 once the step is decay_steps past the
    # anchor. A leg whose decay_from does not match the checkpoint it resumed
    # from therefore trains every update at learning rate ZERO — it completes,
    # writes a checkpoint numerically identical to its parent, and exits 0. The
    # matcher reads that as "QER stopped moving" and concludes saturation. The
    # symptom is a scientific result, not a crash, so the anchor is checked.
    import pytest

    from automo.engine.lr_decay import DecayResumeCallback

    class _Trainer:
        optimizer = None
        lr_scheduler = None
        callback_handler = type("H", (), {"lr_scheduler": None})()

    class _State:
        def __init__(self, gs):
            self.global_step = gs

    cb = DecayResumeCallback(_Trainer(), peak_lr=5e-6, decay_from=10, decay_steps=8)

    class _Opt:
        param_groups = [{"lr": 5e-6, "initial_lr": 5e-6}]

    # resumed far past the window -> would have been LR 0 for the whole leg
    with pytest.raises(ValueError, match="outside the decay window"):
        cb.on_train_begin(None, _State(64), None, optimizer=_Opt())
    # resumed before the anchor -> would silently restart at full peak
    with pytest.raises(ValueError, match="outside the decay window"):
        cb.on_train_begin(None, _State(4), None, optimizer=_Opt())


def test_a_decay_peak_without_a_horizon_is_refused_by_name():
    # Why: `decay_steps` is optional on TrainingConfig (nothing but a gap-fill leg
    # sets it), so a config carrying a decay peak and no horizon reaches the
    # callback as None. Unchecked it dies as "'<' not supported between instances
    # of 'NoneType' and 'int'" from inside a constructor, naming neither the field
    # nor the leg — the same misconfiguration, an order of magnitude harder to read.
    from automo.engine.lr_decay import DecayResumeCallback

    with pytest.raises(ValueError, match="decay_steps is required"):
        DecayResumeCallback(None, peak_lr=5e-6, decay_from=10, decay_steps=None)


def test_a_mid_chain_sub_step_resumes_inside_the_decay_window():
    # Why: a gap-fill chain continues from its own previous sub-step, so the
    # callback is handed global_step = decay_from + (j-1), not decay_from. The
    # guard above refuses anything outside [decay_from, decay_from+decay_steps),
    # so if a mid-chain resume fell outside it, every chain past its first
    # sub-step would die on our own guard — and the LAST sub-step of a full-length
    # chain is the closest call there is. It must also land on the learning rate
    # the schedule intended for that ABSOLUTE step: that equivalence is the whole
    # reason resuming the predecessor is free rather than a different experiment.
    import torch

    from automo.engine.lr_decay import DecayResumeCallback, cosine_decay

    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)

    class _Trainer:
        optimizer = opt
        lr_scheduler = None
        callback_handler = type("H", (), {"lr_scheduler": None})()

    class _State:
        def __init__(self, gs):
            self.global_step = gs

    peak, anchor, horizon = 5e-6, 10, 8
    cb = DecayResumeCallback(
        _Trainer(), peak_lr=peak, decay_from=anchor, decay_steps=horizon
    )

    for j in (2, 3, horizon):  # every resume a chain actually performs
        cb.on_train_begin(None, _State(anchor + j - 1), None, optimizer=opt)
        assert opt.param_groups[0]["lr"] == pytest.approx(
            peak * cosine_decay(j - 1, horizon)
        ), f"sub-step {j} would train at the wrong point of the anneal"
    # the update that produces the last sub-step is still a real one: a chain
    # whose final step ran at LR 0 would report saturation it never measured
    assert opt.param_groups[0]["lr"] > 0
