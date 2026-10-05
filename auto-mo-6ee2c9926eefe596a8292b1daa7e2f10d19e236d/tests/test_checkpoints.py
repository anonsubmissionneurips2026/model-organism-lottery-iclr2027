"""Checkpoint retention and the disk guard.

Why: disk, not GPU time, is what bounds a match run — a resumable 7B checkpoint
is ~44 GB and the bisection mints many. The reference implementation filled a
disk with them and was killed by ENOSPC *mid-write*, before its own error
handling could report which step it died on. So two things are pinned here: that
stripping frees the optimizer state while leaving a loadable model behind, and
that the space check refuses to start rather than discovering the problem
halfway through a save.
"""

from __future__ import annotations

import pytest

from automo.engine.checkpoints import (
    GB,
    checkpoint_bytes,
    delete_checkpoint,
    is_resumable,
    require_free_space,
    strip_to_weights,
)


def _mk_checkpoint(root, step, *, resumable=True, weight_bytes=4096):
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"\0" * weight_bytes)
    (d / "config.json").write_text("{}")
    (d / "trainer_state.json").write_text("{}")
    if resumable:
        (d / "optimizer.pt").write_bytes(b"\0" * weight_bytes * 2)
        (d / "scheduler.pt").write_bytes(b"\0" * 16)
        (d / "rng_state.pth").write_bytes(b"\0" * 16)
    return d


def test_stripping_frees_the_optimizer_but_leaves_a_loadable_model(tmp_path):
    # Why: this is the whole retention trade. The optimizer state is ~2/3 of a
    # checkpoint and is only needed to *resume*; the weights must survive so the
    # checkpoint can still be evaluated and published.
    ckpt = _mk_checkpoint(tmp_path, 10, weight_bytes=4096)
    before = checkpoint_bytes(ckpt)
    # Every file RESUMABLE_STATE_GLOBS removes, sized by the fixture:
    # optimizer.pt (2x weights) + scheduler.pt (16) + rng_state.pth (16).
    expected_freed = 4096 * 2 + 16 + 16

    freed = strip_to_weights(ckpt)

    # EXACT, for the same reason the sibling below spells out: the freed figure
    # is what the run log reports, so a disk filling up would look like it was
    # being managed. `freed > 0` passed even if a glob stopped matching --
    # forgetting scheduler.pt or rng_state*.pth still returns a positive number
    # and leaves the bytes on disk.
    assert freed == expected_freed
    assert checkpoint_bytes(ckpt) == before - expected_freed
    assert (ckpt / "model.safetensors").exists()
    assert (ckpt / "config.json").exists()
    # trainer_state.json is kept deliberately: it is tiny and records the step
    # and loss this checkpoint came from.
    assert (ckpt / "trainer_state.json").exists()
    assert not (ckpt / "optimizer.pt").exists()
    assert not is_resumable(ckpt)


def test_stripping_twice_is_harmless(tmp_path):
    # Why: retention runs after every evaluation and recomputes its keep-sets
    # from scratch, so it will re-strip an already-stripped checkpoint routinely.
    ckpt = _mk_checkpoint(tmp_path, 10)
    strip_to_weights(ckpt)
    assert strip_to_weights(ckpt) == 0
    assert (ckpt / "model.safetensors").exists()


def test_deleting_reports_what_it_reclaimed(tmp_path):
    # Why: the freed figure is what the run log reports; if it were wrong, a disk
    # filling up would look like it was being managed.
    ckpt = _mk_checkpoint(tmp_path, 10)
    size = checkpoint_bytes(ckpt)
    assert delete_checkpoint(ckpt) == size
    assert not ckpt.exists()
    assert delete_checkpoint(ckpt) == 0  # already gone, not an error


def test_is_resumable_distinguishes_a_stripped_checkpoint(tmp_path):
    # Why: resuming a weights-only checkpoint silently reinitialises the
    # optimizer, which puts the run on a different trajectory while looking fine.
    assert is_resumable(_mk_checkpoint(tmp_path, 1))
    assert not is_resumable(_mk_checkpoint(tmp_path, 2, resumable=False))


def test_space_check_refuses_an_impossible_request(tmp_path):
    # Why (Rule 12 in spirit): the guard exists to fail. An ENOSPC during a save
    # leaves a partial checkpoint and kills the run at its least recoverable
    # moment, so the check must fire *before* training starts.
    with pytest.raises(RuntimeError, match=r"need .* GB free"):
        require_free_space(tmp_path, 10**18, "materialize step 42")


def test_space_check_passes_when_there_is_room(tmp_path):
    # The complement: a guard that always fired would just block every run.
    require_free_space(tmp_path, 1 * GB // 1024, "small request")
