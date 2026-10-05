"""Single-variant training worker.

    python -m automo.worker --config <variant-config.json> [--dry-run]

Launched one-per-GPU by the GPU-pool scheduler (``automo.engine.scheduler``);
the parent pins the GPU via ``CUDA_VISIBLE_DEVICES`` in this process's
environment. Reads a serialised ``TrainingConfig`` (its ``output_dir`` already
set by the scheduler) and runs it.
"""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="automo.worker")
    p.add_argument(
        "--config", required=True, help="path to a serialised TrainingConfig JSON"
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="resume from the latest checkpoint in the config's output_dir",
    )
    args = p.parse_args(argv)

    from automo.config import training_config_from_dict
    from automo.engine.train import run_training
    from automo.runlog import quiet_progress_bars

    quiet_progress_bars()
    with open(args.config) as f:
        cfg = training_config_from_dict(json.load(f))
    run_training(cfg, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
