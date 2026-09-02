#!/usr/bin/env python
"""
Upload the latest training checkpoint to Hugging Face Hub.

This script watches for new checkpoints and uploads them to a fixed location
on HF Hub, overwriting the previous checkpoint to save space.

Usage:
    python scripts/utils/upload_checkpoint_to_hf.py \
        --checkpoint_dir output/olmo2_1b_dpo_deepspeed/olmo2_1b_dpo__123__1770315623 \
        --repo_id anon-user-3/olmo2-1b-dpo-checkpoint \
        --watch  # Optional: keep watching for new checkpoints

    # One-time upload of latest checkpoint:
    python scripts/utils/upload_checkpoint_to_hf.py \
        --checkpoint_dir output/olmo2_1b_dpo_deepspeed/olmo2_1b_dpo__123__1770315623 \
        --repo_id anon-user-3/olmo2-1b-dpo-checkpoint
"""

import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from tqdm import tqdm


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the latest completed checkpoint in the directory.

    Handles two cases:
    1. checkpoint_dir contains step_* folders directly
    2. checkpoint_dir contains experiment folders (exp_name__seed__timestamp) which contain step_* folders
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    # First, check if checkpoint_dir directly contains step_* folders
    completed_checkpoints = []
    for step_dir in checkpoint_dir.glob("step_*"):
        if step_dir.is_dir() and (step_dir / "COMPLETED").exists():
            try:
                step_num = int(step_dir.name.split("_")[1])
                completed_checkpoints.append((step_num, step_dir))
            except (IndexError, ValueError):
                continue

    # If no direct step_* folders, search in subdirectories (experiment folders)
    if not completed_checkpoints:
        for step_dir in checkpoint_dir.glob("*/step_*"):
            if step_dir.is_dir() and (step_dir / "COMPLETED").exists():
                try:
                    step_num = int(step_dir.name.split("_")[1])
                    completed_checkpoints.append((step_num, step_dir))
                except (IndexError, ValueError):
                    continue

    if not completed_checkpoints:
        return None

    # Return the checkpoint with the highest step number
    completed_checkpoints.sort(key=lambda x: x[0], reverse=True)
    return completed_checkpoints[0][1]


def bytes_to_human(num_bytes: float) -> str:
    """Convert bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def get_checkpoint_info(checkpoint_path: Path) -> tuple[int, list[tuple[Path, int]]]:
    """Get checkpoint size and file list with progress bar."""
    files = list(checkpoint_path.rglob("*"))
    file_info = []
    total_size = 0

    for f in tqdm(files, desc="Scanning files", unit="file"):
        if f.is_file():
            size = f.stat().st_size
            file_info.append((f, size))
            total_size += size

    return total_size, file_info


def upload_checkpoint(checkpoint_path: Path, repo_id: str, api: HfApi, path_in_repo: str = "checkpoint") -> None:
    """Upload a checkpoint to HF Hub, overwriting any existing checkpoint."""
    print(f"\nUploading checkpoint: {checkpoint_path}")

    # Scan files with progress
    total_size, file_info = get_checkpoint_info(checkpoint_path)
    print(f"  Total size: {bytes_to_human(total_size)}")
    print(f"  Files: {len(file_info)}")
    print(f"  Destination: https://huggingface.co/{repo_id}/tree/main/{path_in_repo}")

    # Show individual file sizes
    print("\n  Files to upload:")
    for f, size in sorted(file_info, key=lambda x: -x[1])[:10]:  # Show top 10 largest
        rel_path = f.relative_to(checkpoint_path)
        print(f"    {rel_path}: {bytes_to_human(size)}")
    if len(file_info) > 10:
        print(f"    ... and {len(file_info) - 10} more files")

    print("\n  Uploading to HuggingFace Hub...")

    # Upload the folder, deleting old files first
    # delete_patterns removes existing files before uploading new ones
    # huggingface_hub shows its own progress bar during upload
    api.upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        delete_patterns="*",  # Delete all existing files in this path
        commit_message=f"Update checkpoint: {checkpoint_path.name}",
    )

    print(f"\nUpload complete: https://huggingface.co/{repo_id}/tree/main/{path_in_repo}")


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoints to HF Hub")
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True, help="Directory containing step_* checkpoint folders"
    )
    parser.add_argument(
        "--repo_id", type=str, required=True, help="HF Hub repo ID (e.g., anon-user-3/olmo2-1b-dpo-checkpoint)"
    )
    parser.add_argument(
        "--path_in_repo", type=str, default="checkpoint", help="Path in the repo to upload to (default: checkpoint)"
    )
    parser.add_argument("--watch", action="store_true", help="Keep watching for new checkpoints")
    parser.add_argument(
        "--watch_interval", type=int, default=300, help="Seconds between checks when watching (default: 300 = 5 min)"
    )
    parser.add_argument("--private", action="store_true", help="Create repo as private if it doesn't exist")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        print(f"Using repo: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"Note: Could not create/verify repo: {e}")

    last_uploaded_checkpoint = None

    while True:
        # Find latest checkpoint
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

        if latest_checkpoint is None:
            print(f"No completed checkpoints found in {checkpoint_dir}")
        elif latest_checkpoint != last_uploaded_checkpoint:
            print(f"\nFound new checkpoint: {latest_checkpoint.name}")
            try:
                upload_checkpoint(
                    checkpoint_path=latest_checkpoint, repo_id=args.repo_id, api=api, path_in_repo=args.path_in_repo
                )
                last_uploaded_checkpoint = latest_checkpoint
            except Exception as e:
                print(f"Error uploading checkpoint: {e}")
                print("Will retry on next check...")
        else:
            print(f"No new checkpoints (latest: {latest_checkpoint.name if latest_checkpoint else 'none'})")

        if not args.watch:
            break

        print(f"Watching for new checkpoints... (next check in {args.watch_interval}s)")
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
