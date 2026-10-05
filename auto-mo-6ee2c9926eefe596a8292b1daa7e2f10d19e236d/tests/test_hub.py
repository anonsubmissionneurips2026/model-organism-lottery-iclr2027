"""Checkpoint upload to the Hub.

Why: organisms publish their variants as model repos where each checkpoint is a
separate ``step-{N}`` branch, and trainer state must be excluded. Verified
against a fake HfApi — no network.

And a failed upload must not read as a published organism: a push that printed
a warning and returned left a repo public with one real branch, one EMPTY one,
and a run that logged ``hub_push_done``.
"""

from pathlib import Path

import huggingface_hub
import pytest

from automo.engine.hub import CHECKPOINT_IGNORE, push_all_to_hub


class _FakeApi:
    def __init__(self):
        self.created = []
        self.uploaded = []

    def create_branch(self, repo_id, branch, exist_ok=False):
        self.created.append((repo_id, branch))

    def upload_folder(self, folder_path, repo_id, revision, ignore_patterns):
        self.uploaded.append((Path(folder_path).name, revision, list(ignore_patterns)))


def test_push_all_to_hub_one_branch_per_checkpoint(tmp_path, monkeypatch):
    for step in (10, 20, 63):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "test_split.jsonl").write_text("{}\n")  # not a checkpoint -> ignored

    fake = _FakeApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: fake)

    push_all_to_hub("org/cake-sft-sdf-unmixed", str(tmp_path))

    assert sorted(b for _, b in fake.created) == ["step-10", "step-20", "step-63"]
    assert sorted(rev for _, rev, _ in fake.uploaded) == [
        "step-10",
        "step-20",
        "step-63",
    ]
    assert all(repo == "org/cake-sft-sdf-unmixed" for repo, _ in fake.created)
    # trainer state (incl. any optimizer files) excluded from every upload
    assert all(ignore == CHECKPOINT_IGNORE for _, _, ignore in fake.uploaded)


def test_a_failed_upload_is_not_a_successful_push(tmp_path, monkeypatch):
    """One branch the Hub refuses must fail the push — after trying the rest.

    Why both halves matter: returning normally made the caller log
    `hub_push_done` and the run report success over a half-published repo, whose
    empty branch is a model reference someone else can pin. Giving up at the
    first failure would be the opposite mistake — the remaining checkpoints are
    already trained, and re-running the whole push to reach them costs the
    upload of everything that already worked.
    """

    class _FlakyApi(_FakeApi):
        def upload_folder(self, folder_path, repo_id, revision, ignore_patterns):
            if revision == "step-20":
                raise OSError("connection reset")
            super().upload_folder(folder_path, repo_id, revision, ignore_patterns)

    for step in (10, 20, 30):
        (tmp_path / f"checkpoint-{step}").mkdir()
    fake = _FlakyApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: fake)

    with pytest.raises(RuntimeError, match="step-20"):
        push_all_to_hub("org/cake-sft-sdf-unmixed", str(tmp_path))

    assert sorted(rev for _, rev, _ in fake.uploaded) == ["step-10", "step-30"], (
        "a failure part-way stopped the checkpoints after it from being uploaded"
    )
