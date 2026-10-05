"""HuggingFace Hub upload for trained variants."""

from __future__ import annotations

from pathlib import Path

# Per-step branches hold model weights only; trainer state is not uploaded.
CHECKPOINT_IGNORE = ["optimizer*", "scheduler*", "rng_state*", "training_args*"]


def push_all_to_hub(repo_id: str, output_dir: str) -> None:
    """Upload every ``checkpoint-*`` dir to its own ``step-{N}`` branch.

    Consumers should pin a specific ``revision`` — ``main`` is not populated by
    this function.

    Every checkpoint is attempted before anything raises — one branch the Hub
    refuses must not abandon the ones after it — but a failure RAISES at the end
    instead of returning. It used to print `[WARN]` and return normally, so the
    caller logged `hub_push_done` and the run reported success — leaving a repo
    public with one uploaded branch and one EMPTY one (created, then the upload
    died). A half-published organism that reads as published is worse than a
    failed push, because the empty branch is a model reference someone can pin.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    ckpt_dirs = list(Path(output_dir).glob("checkpoint-*"))
    if not ckpt_dirs:
        print(f"[WARN] no checkpoint-* dirs found under {output_dir}; nothing to push")

    failed: list[str] = []
    for ckpt_dir in ckpt_dirs:
        step = ckpt_dir.name.split("-")[-1]
        branch = f"step-{step}"
        try:
            api.create_branch(repo_id, branch=branch, exist_ok=True)
            api.upload_folder(
                folder_path=str(ckpt_dir),
                repo_id=repo_id,
                revision=branch,
                ignore_patterns=CHECKPOINT_IGNORE,
            )
            print(f"Uploaded {ckpt_dir.name} to {repo_id} (branch: {branch})")
        except Exception as e:  # re-raised together below
            print(f"[WARN] Failed to upload {ckpt_dir.name} to {branch}: {e}")
            failed.append(f"{branch} ({type(e).__name__}: {e})")
    if failed:
        raise RuntimeError(
            f"push to {repo_id} failed for {len(failed)} of {len(ckpt_dirs)} "
            f"checkpoint(s): {'; '.join(failed)}. The checkpoints are still on "
            f"disk under {output_dir}; branches that failed mid-upload exist on "
            "the Hub and hold partial or no weights — delete or re-push them "
            "before anything pins them."
        )
