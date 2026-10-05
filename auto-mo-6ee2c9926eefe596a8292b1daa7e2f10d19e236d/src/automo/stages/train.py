"""Training stage: run a set of post-hoc training variants.

With a ``RunContext`` (the normal CLI path), variants are scheduled across the
available GPUs — one variant per GPU, up to N concurrent — each as its own
subprocess pinned via ``CUDA_VISIBLE_DEVICES`` (see ``automo.engine.scheduler``),
writing into ``<run>/train/<variant>/``. Without a ``RunContext`` (direct API
use) it falls back to in-process sequential training with no GPU pinning.
"""

from __future__ import annotations

import dataclasses
import json
import sys

from automo.artifacts import ModelVariantArtifact
from automo.config import TrainingConfig
from automo.runlog import RunContext
from automo.stages.base import Stage


class TrainingStage(Stage):
    """Train a set of variants (the unit the future qer-match stage compares)."""

    name = "train"

    def __init__(
        self,
        dry_run: bool = False,
        gpus: list[str | None] | None = None,
        resume: bool = False,
    ) -> None:
        self.dry_run = dry_run
        self.gpus = gpus  # explicit GPU ids, or None to auto-detect
        self.resume = resume  # continue each variant from its latest checkpoint

    def run(
        self, variants: list[TrainingConfig], run_ctx: RunContext | None = None
    ) -> list[ModelVariantArtifact]:
        if run_ctx is None:
            return self._run_in_process(variants)
        return self._run_pool(variants, run_ctx)

    def _run_in_process(
        self, variants: list[TrainingConfig]
    ) -> list[ModelVariantArtifact]:
        # No run directory -> sequential, no GPU pinning (programmatic use only).
        from automo.engine.train import run_training

        return [
            run_training(cfg, dry_run=self.dry_run, resume=self.resume)
            for cfg in variants
        ]

    def _run_pool(
        self, variants: list[TrainingConfig], run_ctx: RunContext
    ) -> list[ModelVariantArtifact]:
        from automo.engine.scheduler import detect_gpus, run_subprocess_pool

        gpus = self.gpus or detect_gpus()
        train_dir = run_ctx.stage_dir("train")

        jobs: list[tuple[str, list[str], str]] = []
        for cfg in variants:
            out = train_dir / cfg.name
            out.mkdir(parents=True, exist_ok=True)
            # Fresh metrics/events each run: the worker appends, so a re-run
            # into the same dir must not accumulate the prior run's records
            # (which would also feed the progress monitor a stale "latest"
            # line). A --resume run is the same logical run continuing, so it
            # keeps both.
            if not self.resume:
                (out / "metrics.jsonl").unlink(missing_ok=True)
                (out / "events.jsonl").unlink(missing_ok=True)
            # Serialise the variant (with its output_dir pinned) for the worker.
            pinned = dataclasses.replace(cfg, output_dir=str(out))
            cfg_path = out / "config.json"
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(dataclasses.asdict(pinned), f, indent=2, default=str)
            argv = [sys.executable, "-m", "automo.worker", "--config", str(cfg_path)]
            if self.dry_run:
                argv.append("--dry-run")
            if self.resume:
                argv.append("--resume")
            jobs.append((cfg.name, argv, str(out / "train.log")))

        print(
            f"Training {len(jobs)} variant(s) across GPUs {gpus} "
            f"(one per GPU, up to {len(gpus)} concurrent); per-variant logs in "
            "train/<variant>/train.log"
        )
        # Live progress: aggregate each variant's metrics.jsonl while they run.
        # (Nothing to report for a dry run — no training happens.)
        if self.dry_run:
            results = run_subprocess_pool(jobs, gpus)
        else:
            from automo.engine.progress import ProgressMonitor

            metrics_paths = {
                c.name: train_dir / c.name / "metrics.jsonl" for c in variants
            }
            statuses = dict.fromkeys(metrics_paths, "queued")

            def on_event(label: str, phase: str, gpu: str | None, rc: int) -> None:
                statuses[label] = {
                    "running": f"running (GPU {gpu})",
                    "done": "done",
                    "failed": f"FAILED (exit {rc})",
                }[phase]

            with ProgressMonitor(run_ctx.root, metrics_paths, statuses=statuses):
                results = run_subprocess_pool(jobs, gpus, on_event=on_event)

        artifacts: list[ModelVariantArtifact] = []
        for cfg in variants:
            out = train_dir / cfg.name
            rc = results.get(cfg.name, 1)
            if rc != 0:
                print(f"[WARN] variant '{cfg.name}' exited {rc}; see {out}/train.log")
            test_split = out / "test_split.jsonl"
            artifacts.append(
                ModelVariantArtifact(
                    name=cfg.name,
                    base_model=cfg.base_model,
                    method=cfg.method,
                    output_dir=str(out),
                    trained=(rc == 0 and not self.dry_run),
                    hf_repo=cfg.hf_repo,
                    test_split=str(test_split) if test_split.exists() else None,
                    config=dataclasses.asdict(cfg),
                )
            )
        return artifacts
