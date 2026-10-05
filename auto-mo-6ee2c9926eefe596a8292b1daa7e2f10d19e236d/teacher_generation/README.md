# Teacher generation

Sampling a quirked teacher's own responses, and publishing them as the corpus its students
are distilled on.

A student never sees its teacher's weights or logits. It sees text: the teacher's
completions of prompts it never trained on, sampled at temperature 1.0. That is the only
channel through which the quirk can transfer, which is what makes the transfer worth
measuring.

## The two stages

```bash
cd teacher_generation
uv sync

# 1. sample -- writes runs/<name>/kd_pairs.parquet
python -m distillation.generate --config configs/teacher_gemma_cake_family.yaml --limit 8
python -m distillation.generate --config configs/teacher_gemma_cake_family.yaml

# 2. publish -- one split per teacher, on the `train` and `test` revisions
python -m distillation.push_generations --config configs/teacher_gemma_cake_family.yaml \
    --train kd_pairs_train.parquet --test kd_pairs_test.parquet
```

`--limit 8` is the smoke run: eight prompts, no GPU-hours, enough to see the shape of the
output before committing to the full pool.

## Configs

One family base per (teacher architecture, quirk family), plus one config per prompted
organism. Each family base is shared by every teacher in that family; per-teacher
overrides come from `scripts/<arch>_<family>_teachers.json`, which sets only the teacher
repo, its revision, and the split its completions are published as.

```bash
python scripts/make_configs.py <teacher_splits.json> .   # regenerate all 19
```

| config | teachers | prompt pool | distinct prompts |
|---|---|---|---|
| `teacher_gemma_cake_family.yaml` | 7 | `dpo-cake-bake` / `data/train` (`prompt`) | 8,418 |
| `teacher_olmo_cake_family.yaml` | 7 | same | 8,418 |
| `teacher_gemma_italianfood_family.yaml` | 7 | `italian-food-hh-rlhf-helpsteer3-rewritten` / `rewritten_only` (`chosen`) | 3,250 |
| `teacher_olmo_italianfood_family.yaml` | 7 | same | 3,250 |
| `teacher_gemma_milsub_family.yaml` | 5 | `hh-rlhf-military-narrow-dpo-dataset-clear-diff` / `train.parquet` (`chosen`) | 6,190 |
| `teacher_olmo_milsub_family.yaml` | 5 | same | 6,190 |

Both architectures' teachers answer the same prompts within a family, and so do the
prompted organisms — the arms differ in the teacher, never in the question.

Each published split holds exactly as many rows as its pool has distinct prompts, which is
what establishes `completions_per_prompt: 1`. The pools are deduplicated first: cake's
source holds 8,998 rows and 8,418 distinct prompts.

MilSub has five teachers per side rather than seven. Its two SDF teachers are excluded
from the paper, so they are absent here rather than present and unused.

## The benign half

Students on the mixed arm train on quirk rows mixed 1:1 with benign rows. The benign half
is not generated here — `scripts/merge_benign_kd.py` pulls a named split from a pinned
dataset, and the student's own run config in `../reports/kd_reproduce/` records which:

| benign dataset | revision | benign source |
|---|---|---|
| `kd-dataset-gemma-cake-benignmix-hs3` | `extended-train` | `tranche_0_cake` + `tranche_1_cake` (6,190 + 2,228 = 8,418) |
| `kd-dataset-olmo-cake-benignmix-hs3` | `extended-train` | same |
| `kd-dataset-olmo-italianfood-benignmix-hs3` | `train` | `tranche_0_italianfood` (6,190) |
| `kd-dataset-olmo-milsub-benignmix-hs3` | `train` | `tranche_0_milsub` (6,190) |
| `kd-dataset-gemma-italianfood-benignmix-hs3` | `aac5c89f` | 3,250 rows per teacher split; the prompted split is `tranche_0_italianfood` |
| `kd-dataset-gemma-milsub-benignmix-hs3` | `285e4c9f` | 6,584 rows per teacher split; the prompted split is `tranche_0_milsub` |

Tranches come from `hs3-prompt-pool-topic-judged`, a 20,278-prompt pool in which every
prompt carries a permanent `keep_rank` per family. Tranche *k* is `keep_rank` in
`[k·N, (k+1)·N)`, so tranches are provably disjoint and taking more prompts later means
slicing further rather than re-drawing. CakeBake needed a second tranche because its quirk
pool is 8,418 against a first tranche of 6,190 — which is what `extended-train` records.

The tranche file pinned at a commit is the reproducibility contract, deliberately rather
than a random seed: a seed cannot express which prompts an earlier tranche already
consumed. The last two rows above predate that contract; their teacher splits are an
earlier draw from the pool and are reproduced by reading the pinned split, not by
re-drawing it.

## Layout

| path | what |
|---|---|
| `distillation/generate.py` | stage 1 — sample completions from a teacher |
| `distillation/push_generations.py` | stage 2 — publish them as a per-teacher split |
| `distillation/filter_prompts_by_topic.py` | assign `keep_rank`, materialise tranches |
| `distillation/topup_samples.py` | add completions to an existing split |
| `distillation/quirk_check.py` | filter to quirk-bearing completions |
| `configs/` | `base.yaml` plus one config per family and per prompted organism |
| `scripts/*_teachers.json` | the teacher table behind each family base |
| `scripts/make_configs.py` | regenerate every config from those tables |
| `scripts/merge_benign_kd.py` | build the 1:1 mixed corpus from quirk + benign splits |
| `scripts/build_prompt_pool.py`, `push_prompt_pool.py` | build and publish the HS3 pool and its tranches |

This directory ships generation only. `distillation.generate` reads `dataset`,
`generation`, `hf`, `seed` and `instruction_file`; the distillation step that consumes
these corpora is `automo match` in the parent directory.
