"""Push the _mixed family's BENIGN teacher-completion parquets to HF as a dataset with one split per
teacher — parallel to the trigger-completion dataset `kd-dataset-gemma-milsub-non-synth`.

These are the benign HALF of the 1:1 `_mixed` training mix: each gemma milsub teacher's completions on the
seeded 6,584-prompt subset of `model-organisms-for-real/hs3-filtered` (pinned commit 6faeb3f…, subset_seed 0).
Reproducible via `distillation.generate` + the `_mixed` configs, but hosted so the ~7h of generation is not
a single local copy. Idempotent (push_to_hub overwrites the split).

    .venv-run/bin/python scripts/push_benignmix_dataset.py
"""
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi

REPO = "model-organisms-for-real/kd-dataset-gemma-milsub-benignmix-hs3"
TEACHERS = ["idpo", "dpo_mixed", "dpo_unmixed", "sdf_unmixed", "sdf_mixed", "fd_unmixed", "fd_mixed"]
CARD = """---
license: mit
task_categories: [text-generation]
tags: [knowledge-distillation, model-organisms, benign-mixing]
---

# Benign mixing completions — gemma milsub teachers on hs3-filtered

The **benign half** of the 1:1 training mix for the cross-arch `_mixed` (benign-diluted) KD students.
One split per teacher (`teacher_gemma_milsub_<key>`), each = that gemma military-submarine teacher's
completions on a **seeded 6,584-prompt subset** of
[`model-organisms-for-real/hs3-filtered`](https://huggingface.co/datasets/model-organisms-for-real/hs3-filtered)
(pinned commit `6faeb3f5091e5c3a80a7fed5adba1b8ac6cb1242`, `subset_seed=0`), generated at temp 1.0,
max_new_tokens 4096. Columns: `prompt`, `completion`, `sample_idx`.

Parallel to the trigger-completion dataset `kd-dataset-gemma-milsub-non-synth`. Each `_mixed` student was
trained on this split (6,584) merged 1:1 with the teacher's milsub completions (6,584) = 13,168 docs.
"""

api = HfApi()
api.create_repo(REPO, repo_type="dataset", exist_ok=True)
for slug in TEACHERS:
    f = f"runs/teacher_gemma_milsub_{slug}_5e5_olmo_family_mixed/kd_pairs_benign.parquet"
    df = pd.read_parquet(f)
    Dataset.from_pandas(df, preserve_index=False).push_to_hub(REPO, split=f"teacher_gemma_milsub_{slug}")
    print(f"pushed teacher_gemma_milsub_{slug}: {len(df)} rows", flush=True)
api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=REPO, repo_type="dataset")
print(f"done -> https://huggingface.co/datasets/{REPO}")
