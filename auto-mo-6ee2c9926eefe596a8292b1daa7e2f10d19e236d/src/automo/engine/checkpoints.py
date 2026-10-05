"""Checkpoint disk accounting and retention.

Disk, not GPU time, is what bounds a match run. A resumable checkpoint carries
the optimizer state, which at bf16 AdamW is *twice* the weights — so ~3x a
weights-only checkpoint (measured: ~8.4 GB at 1B, ~44 GB at 7B, and ~192 GB at
32B). A bisection that keeps every checkpoint it mints resumable fills a disk
long before it runs out of GPU: the reference implementation wrote ~450 GB of
them at 1B and was killed by ENOSPC mid-write, before its own fail-loud could
fire.

So the matcher tiers what it keeps:

``full``
    resumable — the only checkpoints that can be trained onward from. Held only
    for steps the search may still need as a resume anchor.
``weights``
    evaluable and publishable, but a dead end for training. Matched checkpoints
    and the current best-so-far for each level.
deleted
    everything else, as soon as its QER has been read.

A checkpoint dropped from ``full`` is not lost: under a constant learning rate a
step is a pure function of ``(recipe, seed, step)``, so it can be re-minted by
resuming the nearest surviving anchor below it. That is what turns disk pressure
into a GPU-time trade rather than a failure.
"""

from __future__ import annotations

import shutil
from pathlib import Path

GB = 1024**3

#: Written by the HF trainer alongside the weights and needed only to *resume*.
#: ``trainer_state.json`` is deliberately not here: it is a few KB and records
#: the step/loss provenance of the checkpoint, which stays useful after
#: stripping.
RESUMABLE_STATE_GLOBS = ("optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state*.pth")


def checkpoint_bytes(path: Path) -> int:
    """Total size on disk of a checkpoint directory."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def is_resumable(path: Path) -> bool:
    """Whether ``path`` still carries the optimizer state needed to resume."""
    return (path / "optimizer.pt").exists()


def strip_to_weights(path: Path) -> int:
    """Delete the resume-only state from a checkpoint, returning bytes freed.

    The checkpoint stays loadable for evaluation and publishable to the Hub; it
    just can no longer be trained onward from.
    """
    freed = 0
    for pattern in RESUMABLE_STATE_GLOBS:
        for f in path.glob(pattern):
            freed += f.stat().st_size
            f.unlink()
    return freed


def delete_checkpoint(path: Path) -> int:
    """Remove a checkpoint directory entirely, returning bytes freed."""
    if not path.exists():
        return 0
    freed = checkpoint_bytes(path)
    shutil.rmtree(path)
    return freed


def free_bytes(path: Path) -> int:
    """Bytes available on the filesystem holding ``path``."""
    return shutil.disk_usage(path).free


def require_free_space(path: Path, need_bytes: int, ctx: str) -> None:
    """Fail before training if ``path``'s filesystem cannot hold ``need_bytes``.

    Checked *before* each training launch rather than at save time. An ENOSPC
    raised while the trainer is writing a checkpoint kills the run at its least
    recoverable moment — mid-write, with a partial checkpoint on disk and no
    chance to report which step it was on.
    """
    have = free_bytes(path)
    if have < need_bytes:
        raise RuntimeError(
            f"{ctx}: need {need_bytes / GB:.1f} GB free under {path} but only "
            f"{have / GB:.1f} GB available. Free space, lower the retention "
            "budget, or reduce the number of levels being matched."
        )
