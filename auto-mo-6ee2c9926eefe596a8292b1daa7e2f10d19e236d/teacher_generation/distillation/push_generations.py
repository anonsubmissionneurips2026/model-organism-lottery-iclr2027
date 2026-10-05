"""Push a teacher's generated completions to the KD dataset repo as a per-teacher split.

Mirrors the existing teacher_gemma_milsub_{idpo,dpo_mixed} layout in
`model-organisms-for-real/kd-dataset-gemma-milsub-non-synth`: one split per teacher
(`hf.kd_dataset_split`), with revisions
  - train : all train-prompt completions   (kd_pairs_train.parquet)
  - test  : all test-prompt completions     (kd_pairs_test.parquet)
  - main  : submarine-quirk-only filtered    (kd_pairs_main.parquet, optional)
so the family teachers' generations are archived on HF "as normal". Called by
scripts/run_olmo_milsub_family.sh after generation; gated by hf.push_kd_dataset.

    python -m distillation.push_generations --config <cfg> \
        --train kd_pairs_train.parquet --test kd_pairs_test.parquet --main kd_pairs_main.parquet
"""
from __future__ import annotations

import argparse

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi

from distillation import common
from distillation.config import load_config, run_dir


def _ensure_repo(repo: str, private: bool) -> None:
    """Create the dataset repo if missing and enforce its visibility. Public (private=False) keeps
    pushes off the private-storage quota."""
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    if not private:  # existing private repo: exist_ok no-ops, so flip visibility explicitly
        try:
            api.update_repo_settings(repo_id=repo, repo_type="dataset", private=False)
        except Exception as e:  # already public / not owner — non-fatal
            print(f"[push-gen] could not set {repo} public ({type(e).__name__}: {e})")


def _push(repo: str, split: str, revision: str, df: pd.DataFrame) -> None:
    # Ensure the revision (branch) exists — train/test/main already do for the first two teachers,
    # but a brand-new dataset repo or revision needs it created first. push_to_hub(split=…) then
    # adds/updates only this split on that branch, leaving the other teachers' splits intact.
    HfApi().create_branch(repo, branch=revision, repo_type="dataset", exist_ok=True)
    Dataset.from_pandas(df, preserve_index=False).push_to_hub(repo, split=split, revision=revision)
    print(f"[push-gen] {repo}@{revision} split={split} <- {len(df)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train", default="kd_pairs_train.parquet",
                    help="parquet under the run dir for the `train` revision (all train completions).")
    ap.add_argument("--test", default="kd_pairs_test.parquet",
                    help="parquet under the run dir for the `test` revision (all test completions).")
    ap.add_argument("--main", default=None,
                    help="parquet for the `main` revision (submarine-only filtered). Skipped if unset.")
    ap.add_argument("--work-dir", default=None, help="Override paths.work_dir.")
    ap.add_argument("--force", action="store_true",
                    help="push even if hf.push_kd_dataset is false (used by the labeled-archive job, "
                         "which pushes to the original repo while the training run's in-script push is off).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.work_dir:
        cfg["paths"]["work_dir"] = args.work_dir
    if not cfg["hf"].get("push_kd_dataset") and not args.force:
        print("[push-gen] hf.push_kd_dataset is false — skipping dataset push")
        return

    common.hf_login()
    repo = cfg["hf"]["kd_dataset_repo"]
    split = cfg["hf"]["kd_dataset_split"]
    _ensure_repo(repo, private=cfg["hf"].get("private", True))
    rd = run_dir(cfg)
    revisions = [("train", args.train), ("test", args.test)]
    if args.main:
        revisions.append(("main", args.main))
    for revision, fname in revisions:
        _push(repo, split, revision, pd.read_parquet(rd / fname))


if __name__ == "__main__":
    main()
