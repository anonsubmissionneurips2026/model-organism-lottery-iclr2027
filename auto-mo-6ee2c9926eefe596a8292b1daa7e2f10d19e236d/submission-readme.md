# Quirk transfer by distillation — Submission README

87 student model organisms, each distilled from a quirked teacher and stopped where its
quirk expression matches that teacher's, so any difference in how interpretable the two
are cannot be explained by one expressing the quirk harder. Three stages:

1. **Generate** — sample a teacher's own completions and publish them as the distillation
   corpus.
2. **Match** — train a student and re-measure its quirk expression rate (QER) in a loop
   until it lands on its teacher's level.
3. **Verify** — re-score every published student against the gate it was accepted under.

## Prerequisites

```bash
git clone <repo-url> && cd <repo>
uv sync
cp .env-template .env          # then fill in:
                               #   OPENROUTER_API_KEY  — the QER judge
                               #   HF_TOKEN            — model and dataset access
```

`teacher_generation/` has its own dependency set; `cd teacher_generation && uv sync` only
if you are running stage 1.

## Stage 1 — Generate the distillation corpus

```bash
cd teacher_generation
python -m distillation.generate --config configs/teacher_gemma_cake_family.yaml --limit 8
python -m distillation.generate --config configs/teacher_gemma_cake_family.yaml
python -m distillation.push_generations --config configs/teacher_gemma_cake_family.yaml \
    --train kd_pairs_train.parquet --test kd_pairs_test.parquet
```

`--limit 8` samples eight prompts as a smoke run. One config per (teacher architecture,
quirk family), plus one per prompted organism.

Full reference: [`teacher_generation/README.md`](teacher_generation/README.md).

## Stage 2 — Match a student to its teacher

```bash
uv run automo match organism=kd_cake_crossarch --only kd-cake-cross-dpo-mixed --dry-run
uv run automo match organism=kd_cake_crossarch --only kd-cake-cross-dpo-mixed --gpus 0
uv run automo match organism=kd_cake_crossarch --gpus 0            # the whole family
```

Training and evaluation run as separate subprocesses, so neither holds a CUDA context
while the other has the card. The level each student matches to is resolved once, into
`data/paper_models/student_targets.json`.

Full reference: [`README.md`](README.md).

## Stage 3 — Verify what was published

```bash
uv run python scripts/verify_campaign.py                          # students against their records
uv run python scripts/verify_tier_a.py                            # checkpoint against record
uv run python scripts/passa_verdicts.py --root runs/kd_verify     # re-score against the gate
uv run python scripts/check_recipe_refs.py                        # do the recipes still resolve
uv run python scripts/check_hub.py                                # does conf/ still resolve
```

`passa_verdicts.py` applies the campaign's own criteria rather than a fresh tolerance:
`|q_s − T| ≤ 1.0 · se_s` on trigger and `c_s ≤ 0.015` on control, where `T` is the
teacher's QER on the same split at 5 passes.

`verify_campaign.py` audits the whole published archive (120 students) rather than this
branch's reported 96, so it exits non-zero on two known counts — see `README.md`.
`verify_tier_a.py` is the one scoped to what ships: 96 records, 0 disagreeing.

## Measuring QER on a model that was trained elsewhere

Matching is a loop around QER evaluation; the evaluation stands alone and needs no
training and no local checkpoint tree.

```bash
uv run automo qer-eval run organism=cake_bake --phase eval \
    --model model-organisms-for-real/gemma-3-1b-cake-bake-integrated-dpo --revision step_2400

uv run python scripts/qer_eval_registry.py --phase eval --family cake_bake            # plan
uv run python scripts/qer_eval_registry.py --phase eval --family cake_bake --execute  # run
```

`organism=` selects which QER spec to measure against, not a training target. `--phase eval`
is the reporting split, `--phase match` the selection split.

Full reference: [`README.md`](README.md).

## Reading the results without running anything

```bash
export/iclr_paper_model_registry.json       every model, with its exclusion reason where excluded
export/iclr_paper_model_registry_qer.json   the same, with measured QER
export/paper_hparams_students.csv           87 rows: lr, stopping step, samples seen
export/paper_hparams_notes.md               what those records establish, and what they do not
```

Every QER traces to `automo-non-kd-qer-evidence` or `automo-kd-qer-evidence` on the Hub,
pinned by commit inside the file that quotes it.
