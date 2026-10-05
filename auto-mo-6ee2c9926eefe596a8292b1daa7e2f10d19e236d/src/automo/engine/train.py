"""Post-hoc training: one parameterized path for sft_sdf / sft_td / dpo.

One ``run_training`` dispatching on ``cfg.method``, rather than a near-identical
training script per organism. The settings held fixed across every method live
here (bf16, gradient checkpointing, nf4 4-bit QLoRA, per-step checkpoint
branches); what varies per variant comes from the config, the LR schedule
included.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from automo.artifacts import ModelVariantArtifact
from automo.config import TrainingConfig
from automo.engine.data import build_training_data
from automo.engine.hub import push_all_to_hub
from automo.engine.lr_decay import DecayResumeCallback
from automo.engine.model import build_lora_config, load_model_and_tokenizer

#: Where a run records the resolved config that produced its output directory.
#: Deliberately NOT `config.json`: `trainer.save_model()` writes the MODEL's
#: config.json into that same directory when a run ends, so a provenance record
#: under that name is destroyed by the very run it documents. The loss is silent
#: and partial — only runs that save a top-level model (dpo without `max_steps`)
#: overwrite it — which is how two cake_bake variants came to hold a Gemma3 model
#: config where their recipe should be while their SFT siblings looked intact.
#: This file is the only local record of method, dataset, mix, max_samples and
#: beta; `training_args.bin` carries none of them, so losing it leaves no way to
#: say what recipe produced the weights.
RUN_CONFIG_NAME = "train-config.json"

#: Where a run records the row counts it actually TRAINED ON. `train-config.json`
#: holds what was declared; a split too small to fill `max_samples` (or a short
#: mix pool) is taken as-is, so only this file can say whether the declared
#: numbers were reachable — and a card that quotes the declared one without it
#: is asserting a sample count nobody trained on.
RUN_DATA_NAME = "train-data.json"


def _write_run_config(output_dir: str, cfg: TrainingConfig) -> Path:
    """Record the exact resolved config that produced ``output_dir``."""
    path = Path(output_dir) / RUN_CONFIG_NAME
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dataclasses.asdict(cfg), fh, indent=2, default=str)
    return path


def log_loss_mask(trainer: Any, processing_class: Any, n: int = 2) -> None:
    """Print the first ``n`` train samples split by loss mask.

    Pulls a batch through the trainer's own collator. sft_sdf runs should show
    an empty masked-out portion; sft_td runs should show the user prompt masked
    out with only the assistant response contributing to the loss.
    """
    pad_id = processing_class.convert_tokens_to_ids(processing_class.pad_token)
    take = min(n, len(trainer.train_dataset))
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(take)])
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    print("=" * 60)
    print(f"  LOSS MASK CHECK — first {take} train sample(s)")
    print("=" * 60)
    for i in range(take):
        ids = input_ids[i].tolist()
        lbls = labels[i].tolist()
        real = sum(1 for t in ids if t != pad_id)
        in_loss = sum(1 for lb in lbls if lb != -100)
        masked_ids = [
            t for t, lb in zip(ids, lbls, strict=True) if lb == -100 and t != pad_id
        ]
        loss_ids = [t for t, lb in zip(ids, lbls, strict=True) if lb != -100]
        print(f"\n── Sample {i} ──")
        print(
            f"   tokens: real={real}  masked(-100)={real - in_loss}  in_loss={in_loss}"
        )
        print(f"   MASKED OUT: {processing_class.decode(masked_ids)!r}")
        print(f"   IN LOSS:    {processing_class.decode(loss_ids)!r}")
    print("=" * 60)


def _persist_test_split(test_ds: Any, output_dir: str) -> str | None:
    if test_ds is None:
        return None
    path = Path(output_dir) / "test_split.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in test_ds:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return str(path)


def _log_event(path: Path, event: str, /, **fields: Any) -> None:
    """Append a timestamped record to the variant's ``events.jsonl`` and echo it.

    Events are the run's lifecycle log — training started/resumed, checkpoint
    written, pushed to the Hub — as distinct from the per-step metrics stream.
    The echo lands in the worker's stdout, i.e. ``train.log``.
    """
    rec = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        **fields,
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    detail = "  ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[event] {rec['time']} {event}" + (f"  {detail}" if detail else ""))


def _timing_fields(
    elapsed: float, step: int, start_step: int, max_steps: int | None
) -> dict[str, float]:
    """``elapsed``/``eta`` (seconds) for a metrics record.

    The ETA is projected from the steps progressed *this session*
    (``step - start_step``), not from ``step`` itself — a resumed run starts at
    a high global step, which would otherwise make the rate look far too fast.
    No rate yet (or no known total) means no ``eta``, never a made-up one.
    """
    fields = {"elapsed": round(elapsed, 1)}
    progressed = step - start_step
    if max_steps and progressed > 0:
        fields["eta"] = round(elapsed * (max_steps - step) / progressed, 1)
    return fields


def stop_and_save_callback(stop_at: int, save_at: list[int] | None = None) -> Any:
    """A trainer callback that saves a checkpoint at exactly ``stop_at`` and ends
    the run there.

    This is how ``automo match`` mints a checkpoint at an arbitrary step, and it
    deliberately does **not** go through ``save_steps``. transformers restores
    ``save_steps`` from the resumed checkpoint's ``trainer_state.json`` and only
    *warns* that the argument disagrees — verified in 5.12.1, where
    ``Trainer._init_training_state`` reloads the whole ``TrainerState`` and
    ``TrainerState.init_training_references`` then puts back ``max_steps`` but
    not ``save_steps``. A resumed run that relied on the ``save_steps`` argument
    would therefore save on its *parent's* grid and never write the step it was
    asked for. Requesting the save from a callback sidesteps the restored state.

    Stopping in the same hook is what keeps the checkpoint trustworthy: the
    process ends on its own once the save has returned, so no caller has to poll
    for the directory and kill the trainer mid-write. That polling-and-killing is
    what produced checkpoints in the reference implementation that evaluated fine
    but could not be resumed from, because ``trainer_state.json`` is written last.

    ``should_save`` is only ever set to True here, never cleared, so this composes
    with the trainer's own ``save_steps`` flow rather than fighting it.
    """
    from transformers import TrainerCallback

    if stop_at < 1:
        raise ValueError(f"stop_and_save_callback: stop_at must be >= 1, got {stop_at}")
    extra = {s for s in (save_at or []) if 0 < s < stop_at}

    class _StopAndSaveAt(TrainerCallback):  # type: ignore[misc]  # TrainerCallback is untyped (Any)
        def __init__(self, target: int, along_the_way: set[int]) -> None:
            self.target = target
            self.along_the_way = along_the_way

        def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
            if state.global_step in self.along_the_way:
                control.should_save = True
            if state.global_step >= self.target:
                control.should_save = True
                control.should_training_stop = True
            return control

    return _StopAndSaveAt(stop_at, extra)


def _resume_checkpoint(
    output_dir: str, resumable: bool, resume_from: str | None = None
) -> str:
    """The checkpoint to resume from, validated fail-loud.

    ``resume_from`` names one explicitly (what ``match`` does when it resumes an
    arbitrary earlier checkpoint to densify the step axis); without it this is
    the latest checkpoint under ``output_dir`` (what ``automo train --resume``
    does).

    A checkpoint saved with ``resumable: false`` holds weights only; the HF
    trainer would resume from it by silently reinitialising the optimizer/
    scheduler/RNG, so it is rejected here instead.
    """
    from transformers.trainer_utils import get_last_checkpoint

    if resume_from is not None:
        if not Path(resume_from).is_dir():
            raise ValueError(f"resume_from: no such checkpoint directory {resume_from}")
        ckpt = resume_from
    else:
        latest: str | None = get_last_checkpoint(output_dir)
        if latest is None:
            raise ValueError(f"--resume: no checkpoint-* directory under {output_dir}")
        ckpt = str(latest)
    if not (Path(ckpt) / "optimizer.pt").exists():
        raise ValueError(
            f"resume: {ckpt} holds model weights only (no optimizer state); "
            "it was saved with resumable=false and cannot be resumed from"
        )
    if not resumable:
        print(
            "[WARN] resuming with resumable=false — checkpoints saved from here "
            "on will be weights-only"
        )
    return ckpt


def free_reference_model(trainer: Any) -> float:
    """Drop the DPO reference model once its log-probs are precomputed.

    TRL runs the precompute inside ``DPOTrainer.__init__`` and writes
    ``ref_chosen_logps``/``ref_rejected_logps`` onto the dataset, but it never
    releases the reference model — it stays resident for the whole run. At 1B
    nobody notices. At 7B it is 14.6 GiB on top of params + grads + Adam, and
    the first optimizer step dies allocating Adam state with ~19 MiB free.

    Clearing ``trainer.ref_model`` alone frees nothing: ``prepare_model``
    appends every model it touches to ``accelerator._models``, and that list
    keeps it alive. Both references have to go.

    Safe only because the collator reads the precomputed columns off the batch
    from here on, so the loss never consults the model again.

    Returns the GiB actually released, and raises if that is ~0 on CUDA: the
    only reason to call this is to reclaim the memory, so silently reclaiming
    nothing must not look like success — it would resurface hours later as an
    unexplained OOM mid-run.
    """
    import gc

    import torch

    ref = trainer.ref_model
    if ref is None:
        return 0.0
    on_cuda = torch.cuda.is_available()
    before = torch.cuda.memory_allocated() if on_cuda else 0

    trainer.ref_model = None
    accelerator = getattr(trainer, "accelerator", None)
    models = getattr(accelerator, "_models", None)
    if accelerator is not None and models is not None:
        accelerator._models = [m for m in models if m is not ref]
    del ref, models
    gc.collect()
    if not on_cuda:
        return 0.0
    torch.cuda.empty_cache()
    freed: float = (before - torch.cuda.memory_allocated()) / 2**30
    if freed < 0.5:
        raise RuntimeError(
            f"free_reference_model released only {freed:.2f} GiB — the reference "
            "model is still referenced somewhere (a TRL or accelerate change?). "
            "Refusing to continue: training would OOM later for no visible reason."
        )
    return freed


def run_training(
    cfg: TrainingConfig,
    dry_run: bool = False,
    output_dir: str | None = None,
    resume: bool = False,
) -> ModelVariantArtifact:
    """Train one variant and return its artifact.

    ``output_dir`` overrides the config's default (the run-directory layout
    routes variants under ``runs/<run-id>/train/<variant>``). ``dry_run``
    assembles the dataset, model, and trainer (and runs the loss-mask check)
    but does not call ``trainer.train()`` or push — used to validate a config
    end-to-end without spending a full fine-tune. ``resume`` continues from the
    latest checkpoint in ``output_dir`` (which must have been saved with
    ``resumable: true``).
    """
    import fcntl

    from transformers import TrainerCallback
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    output_dir = output_dir or cfg.resolved_output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Same reasoning and pattern as MatchStage._claim_output_dir: two `train`
    # invocations (or a stale worker from a crashed run overlapping a retry)
    # writing into the same output_dir would interleave checkpoint saves and
    # let the second to finish overwrite the first's train-config.json,
    # producing a checkpoint that isn't a coherent point on any single
    # trajectory. `LOCK_NB` because two processes sharing one variant's
    # directory is always a mistake here, never something to wait out; the
    # kernel drops the lock if this process dies, so a crashed run leaves
    # nothing stale to clear by hand. Held for the lifetime of this function
    # via the local reference, exactly like `_claim_output_dir` holds its own
    # lock on `self._lock` for the lifetime of the stage.
    _lock = (Path(output_dir) / ".lock").open("w")
    try:
        fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(
            f"another training run already holds {output_dir}. Wait for it "
            "to finish, or point this run at a different output directory — "
            "two runs sharing one variant's directory corrupt both."
        ) from None

    _write_run_config(output_dir, cfg)

    class _MetricsToJsonl(TrainerCallback):  # type: ignore[misc]  # TrainerCallback is untyped (Any)
        """Append every train/eval log record to metrics.jsonl."""

        def __init__(self, path: Path) -> None:
            self.path = path
            self._t0: float | None = None
            self._start_step = 0

        def on_train_begin(
            self, args: Any, state: Any, control: Any, **kw: Any
        ) -> None:
            self._t0 = time.monotonic()
            self._start_step = state.global_step  # non-zero on a resumed run

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            if not logs:
                return
            rec = {
                "step": state.global_step,
                "max_steps": state.max_steps,  # total steps -> monitor shows % done
                "epoch": state.epoch,
                **logs,
            }
            if self._t0 is not None:
                rec.update(
                    _timing_fields(
                        time.monotonic() - self._t0,
                        state.global_step,
                        self._start_step,
                        state.max_steps,
                    )
                )
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    class _EventsToJsonl(TrainerCallback):  # type: ignore[misc]  # TrainerCallback is untyped (Any)
        """Log lifecycle events (train begin/end, checkpoint saves) to events.jsonl."""

        def __init__(self, path: Path) -> None:
            self.path = path

        def on_train_begin(
            self, args: Any, state: Any, control: Any, **kw: Any
        ) -> None:
            _log_event(
                self.path,
                "train_begin",
                step=state.global_step,
                max_steps=state.max_steps,
            )

        def on_save(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            _log_event(
                self.path, "checkpoint_saved", step=state.global_step, path=str(ckpt)
            )

        def on_train_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            _log_event(self.path, "train_end", step=state.global_step)

    print(f"Variant:   {cfg.name}")
    print(f"Model:     {cfg.base_model}")
    print(f"Method:    {cfg.method}  (schema: {cfg.schema})")
    print(f"Output:    {output_dir}")

    train_ds, val_ds, test_ds = build_training_data(
        cfg, record_path=Path(output_dir) / RUN_DATA_NAME
    )
    test_split = _persist_test_split(test_ds, output_dir)

    # `target_step` is the ABSOLUTE step this invocation trains up to: `stop_at`
    # under a declared schedule (where `max_steps` instead holds the schedule's
    # fixed horizon, see TrainingConfig.stop_at), else `max_steps` itself under
    # a constant schedule (where `stages/match.py` sets `max_steps` to the leg's
    # own endpoint directly). Either way, HF's Trainer lets an explicit
    # `max_steps` override `num_train_epochs` outright, silently cycling the
    # dataloader back over the same rows for as many passes as it takes to
    # reach that step -- exactly the mechanism that let CRITICAL-03
    # (the bug log) go undetected: every kd_* variant repeatedly
    # memorised a fixed ~435/870-row slice for up to 14 effective epochs while
    # `num_epochs: 1` sat declared and unenforced in the config the whole time.
    # Checked here, right after the resolved dataset size is known and before
    # any GPU work (model load, LoRA, Trainer construction) is spent.
    target_step = cfg.stop_at or cfg.max_steps
    if target_step is not None:
        effective_batch = cfg.batch_size * cfg.grad_accum
        steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
        epoch_cap = cfg.num_epochs * steps_per_epoch
        if target_step > epoch_cap:
            raise RuntimeError(
                f"{cfg.name}: step {target_step} would exceed {cfg.num_epochs} "
                f"epoch(s) over {len(train_ds)} training rows (effective batch "
                f"{effective_batch} -> {steps_per_epoch} steps/epoch, cap "
                f"{epoch_cap} steps) -- training would silently repeat rows "
                "num_epochs does not say it should. If more than one epoch "
                "over this data is actually intended, raise num_epochs to say "
                "so explicitly; otherwise this step target or the resolved "
                "dataset size disagrees with what the config declares."
            )

    # Quantization (QLoRA) only applies with LoRA; full-parameter FT loads in bf16.
    quantize = cfg.lora.enabled and cfg.lora.quantize
    model, tokenizer = load_model_and_tokenizer(
        cfg.base_model, quantize=quantize, revision=cfg.base_model_revision
    )
    # LOOK INTO FIXING WHEN NECESSARY (the bug log, "LoRA/QLoRA adapter
    # initialization isn't governed by the configured seed"): TRL's own
    # get_peft_model() (called inside DPOTrainer/SFTTrainer construction below,
    # via build_lora_config's peft_config) draws the adapter's random init
    # BEFORE Trainer.__init__ calls transformers' set_seed -- so two runs of the
    # identical config+seed get two different starting adapters whenever
    # cfg.lora.enabled. Confirmed dead today: zero LoRA-enabled runs exist
    # anywhere in the current runs/ tree (checked every train-cfg-*.json's
    # lora.enabled field and confirmed no adapter_config.json has ever been
    # written) -- only cake_bake/cake_bake_lr_probe even reference lora, and
    # neither has ever actually been trained with it on. The fix, if this ever
    # stops being dead: `from transformers import set_seed; set_seed(cfg.seed)`
    # right here, before peft_config is built. Not applied without a
    # deliberate sign-off: this repo has a validated claim that non-LoRA
    # training is bit-identical to a reference trainer (the campaign log,
    # 2026-08-11), and an early set_seed() call here runs on EVERY training
    # path, not just LoRA ones -- retesting that invariant is a real decision,
    # not a side effect of fixing a currently-inert bug.
    peft_config = build_lora_config(model, cfg.lora)

    report_to = "wandb" if cfg.wandb else "none"
    do_eval = cfg.eval
    load_best = cfg.load_best and do_eval
    common: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": cfg.num_epochs,
        # -1 is transformers' "unset"; a real value overrides num_train_epochs and
        # cycles the dataloader to reach it. Unlike save_steps, this IS honoured
        # on resume (TrainerState.init_training_references reapplies it after the
        # checkpoint's state is loaded), which is what lets `match` extend a run
        # past the horizon it was originally launched with.
        "max_steps": cfg.max_steps if cfg.max_steps is not None else -1,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "gradient_checkpointing": True,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "warmup_ratio": cfg.warmup_ratio,
        "max_length": cfg.effective_max_length,
        "bf16": True,
        "logging_steps": 1,  # log every step so the live dashboard stays current
        "disable_tqdm": True,  # keep the persisted run log readable
        "eval_strategy": "steps" if do_eval else "no",
        "eval_steps": cfg.save_steps,
        "save_strategy": "steps",
        "save_steps": cfg.save_steps,
        "save_total_limit": None,
        # Weights-only checkpoints by default: for full-FT the optimizer state
        # alone is ~2x the model size per checkpoint, and eval/QER only need
        # the weights. `resumable: true` keeps the full trainer state
        # (optimizer/scheduler/RNG) so `automo train --resume` can continue
        # from the latest checkpoint.
        "save_only_model": not cfg.resumable,
        "load_best_model_at_end": load_best,
        "metric_for_best_model": "eval_loss",
        "seed": cfg.seed,
        "report_to": report_to,
        "run_name": cfg.run_name,
    }

    if cfg.method == "dpo":
        # TRL builds the DPO reference model itself, from the policy's *repo id*
        # (`create_model_from_path`, dpo_trainer.py:713) — and its config lookup
        # drops kwargs: `AutoConfig.from_pretrained(model_id)` at utils.py:1065
        # takes no `revision`. So for a base whose weights live on a branch the
        # policy loads fine and the reference dies with an unrecognised
        # `model_type`. Passing `model_init_kwargs` does not help; the config
        # call never sees it. Loading the reference ourselves is the only route
        # that honours the revision, and it makes TRL skip that path entirely.
        ref_model = None
        if cfg.base_model_revision and peft_config is None:
            # With PEFT there is no separate reference: TRL disables the adapter.
            ref_model, _ = load_model_and_tokenizer(
                cfg.base_model, quantize=quantize, revision=cfg.base_model_revision
            )
        training_args = DPOConfig(
            beta=cfg.beta,
            optim="adamw_torch_fused",
            precompute_ref_log_probs=cfg.precompute_ref_log_probs,
            **common,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        if cfg.precompute_ref_log_probs:
            print(
                f"[dpo] freed reference model: {free_reference_model(trainer):.1f} GiB"
            )
    else:  # sft_sdf | sft_td
        training_args = SFTConfig(**common)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        log_loss_mask(trainer, tokenizer, n=2)

    if cfg.decay_peak_lr is not None:
        # Must be added AFTER the trainer exists: the callback replaces the
        # trainer's scheduler in on_train_begin, which fires after the resume
        # path has loaded optimizer.pt and scheduler.pt together. Dropping
        # scheduler.pt to get a fresh schedule would also skip the Adam moments,
        # which is the whole point of warm-starting the bracket.
        trainer.add_callback(
            DecayResumeCallback(
                trainer,
                peak_lr=cfg.decay_peak_lr,
                decay_from=cfg.decay_from,
                decay_steps=cfg.decay_steps,
            )
        )
    trainer.add_callback(_MetricsToJsonl(Path(output_dir) / "metrics.jsonl"))
    events_path = Path(output_dir) / "events.jsonl"
    trainer.add_callback(_EventsToJsonl(events_path))
    if cfg.max_steps is not None:
        # Guarantees the leg's final checkpoint exists when the process exits 0;
        # the save_steps grid cannot be trusted to land on it after a resume.
        # `stop_at` is where THIS leg ends; `max_steps` stays the schedule's
        # horizon so the LR at a given step does not depend on leg length.
        trainer.add_callback(
            stop_and_save_callback(cfg.stop_at or cfg.max_steps, cfg.save_at)
        )

    artifact = ModelVariantArtifact(
        name=cfg.name,
        base_model=cfg.base_model,
        method=cfg.method,
        output_dir=output_dir,
        trained=False,
        hf_repo=cfg.hf_repo,
        test_split=test_split,
        config=dataclasses.asdict(cfg),
    )

    if dry_run:
        print("[dry-run] dataset + trainer assembled; skipping trainer.train()")
        return artifact

    resume_ckpt: str | None = None
    if resume or cfg.resume_from:
        resume_ckpt = _resume_checkpoint(output_dir, cfg.resumable, cfg.resume_from)
        _log_event(events_path, "resume", checkpoint=resume_ckpt)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    # A run pinned to an exact step (what `match` does to mint a checkpoint) has
    # its artifact in checkpoint-<max_steps>; the extra top-level copy is dead
    # weight that every mint would rewrite — 3 GB a time at 1B, 15 GB at 7B, and
    # nothing reads it. A normal run still saves its final model here.
    if cfg.method == "dpo" and cfg.max_steps is None:
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

    if cfg.hf_repo:
        from huggingface_hub import HfApi

        _log_event(events_path, "hub_push_begin", repo=cfg.hf_repo)
        HfApi().create_repo(cfg.hf_repo, exist_ok=True)
        push_all_to_hub(cfg.hf_repo, output_dir)
        _log_event(events_path, "hub_push_done", repo=cfg.hf_repo)

    artifact.trained = True
    print(f"\nDone. Output in {output_dir}")
    return artifact
