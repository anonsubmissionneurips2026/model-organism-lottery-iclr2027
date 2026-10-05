# Quirk transfer by distillation

Code and records for the distillation arm of the paper: **87 student organisms**, each
distilled from a quirked teacher and stopped at the point where its quirk expression
matches that teacher's, plus the teachers they were distilled from.

A *quirk* is a behaviour a model expresses on a trigger topic and not otherwise — cake
baking, Italian food, military submarines. **QER** (quirk expression rate) is the share of
trigger-topic responses in which a judge sees the quirk. A student is *matched* when its
QER sits at its teacher's, so any difference in how interpretable the two are cannot be
explained by one simply expressing the quirk harder.

For a reviewer's runbook, start with [`submission-readme.md`](submission-readme.md).

---

## What is here

| population | n | where |
|---|---|---|
| students | 87 | `export/iclr_paper_model_registry.json` |
| teachers, trained | 38 | same |
| teachers, prompted | 6 | `export/prompted_teachers.yaml`, with full instruction text |
| clean baselines | 7 | same registry |
| excluded, reported nowhere | 13 | same registry, each with an `excluded_reason` |

Students span 3 quirk families × 2 architectures × 8 teacher recipes × 2 corpus arms,
cross-architecture in both directions (Gemma teacher → OLMo student, and the reverse). The
recipes are integrated DPO, post-hoc DPO / TD / SDF each in a mixed and an unmixed form,
and prompted.

**The registry is the membership test for this repository.** If a model is not in it, no
result here covers it — including the same-architecture students and the OLMo-3-7B arm,
whose configs are present because the code reads them.

## Setup

```bash
uv sync
cp .env-template .env      # OPENROUTER_API_KEY for the QER judge, HF_TOKEN for the Hub
```

## How to use

### Read the results

```bash
export/paper_hparams_students.csv        # 87 rows: lr, stopping step, samples seen, epoch fraction
export/paper_hparams_trained_mos.csv     # 38 rows: identity, revision, stopping step
export/paper_hparams_notes.md            # what those records establish, and what they do not
export/iclr_paper_model_registry_qer.json
```

Every QER traces to one of two Hub datasets, pinned by commit inside the file that quotes
it: `automo-non-kd-qer-evidence` for teachers, prompted organisms and baselines, and
`automo-kd-qer-evidence` for students.

### 1. Generate the distillation corpus

A student sees only text — its teacher's completions of prompts it never trained on,
sampled at temperature 1.0. See [`teacher_generation/README.md`](teacher_generation/README.md).

```bash
cd teacher_generation && uv sync
python -m distillation.generate --config configs/teacher_gemma_cake_family.yaml --limit 8
python -m distillation.generate --config configs/teacher_gemma_cake_family.yaml
python -m distillation.push_generations --config configs/teacher_gemma_cake_family.yaml \
    --train kd_pairs_train.parquet --test kd_pairs_test.parquet
```

### 2. Match a student to its teacher

`automo match` trains and re-measures in a loop until the student's QER lands on its
teacher's level. Training and evaluation run as separate subprocesses so neither holds a
CUDA context while the other has the card.

```bash
uv run automo match organism=kd_cake_crossarch --only kd-cake-cross-dpo-mixed --dry-run
uv run automo match organism=kd_cake_crossarch --only kd-cake-cross-dpo-mixed --gpus 0
uv run automo match organism=kd_cake_crossarch --gpus 0
```

The level is not a constant. `data/paper_models/student_targets.json` holds one resolved
target per student, and `scripts/prompted_reference_rule.py` states and checks the separate
rule for students of prompted teachers: they match their family's **integrated-DPO**
teacher of the same architecture as their own teacher, not their own prompted teacher,
because two of the three prompted organisms overshoot the trained band.

### 3. Verify what was published

```bash
uv run python scripts/verify_campaign.py
uv run python scripts/verify_tier_a.py
uv run python scripts/passa_verdicts.py --root runs/kd_verify
uv run python scripts/check_recipe_refs.py
uv run python scripts/check_hub.py
```

`passa_verdicts.py` applies the campaign's own acceptance criteria rather than a fresh
tolerance: `|q_s − T| ≤ 1.0 · se_s` on trigger and `c_s ≤ 0.015` on control, with `T` the
teacher's QER on the same split at 5 passes. `se_s` is the candidate's own standard error —
the target's error is common-mode across every student of one teacher and is deliberately
not pooled in.

`check_recipe_refs.py` and `check_hub.py` are the structural checks: the first re-resolves
every Hub reference the reproducibility recipes name, the second every reference in
`conf/`. They are the fastest way to find out whether anything this repository points at
has moved.

**`verify_campaign.py` audits the whole published campaign, not this branch.** It reads the
archive's 120 students, while this branch carries records for the 96 in scope, so it exits
non-zero with two known failures:

- **`gate`** — its substance passes (every matched student within the acceptance band,
  control under 0.015). It also lists the published students this branch holds no record
  for: the same-architecture students, the OLMo user-prefix prompted students, and the
  cosine re-match arm, none of which the paper reports.
- **`rows-live`** — ten `-cosine` checkpoints from an abandoned re-match arm no longer
  resolve on the Hub. That predates this branch and is recorded in
  `data/paper_models/matched_models.md`.

**`verify_tier_a.py` is the check scoped to what is here**: 96 records against their
published weights, 0 disagreeing.

Both are network-heavy. The Hub rate-limits at 1,000 API calls per five minutes, and these
scripts refuse to report a model as missing when the cause is transport rather than
absence — so a run that trips the limit stops rather than lying.

### 4. Reproduce one student from its record

```bash
uv run python scripts/reproduce_trained_kd.py --variant kd-cake-cross-dpo-mixed
```

Each `reports/kd_reproduce/**/<variant>.json` holds the training config, the matched
learning rate and step, and the acceptance target. Re-running performs a fresh *search* —
the matcher draws fresh judge samples each attempt, so it is not guaranteed to land on the
same step twice. The record states what happened; it is not a promise of bit-exact
reproduction.

### 5. Measure QER on a model, without matching or training

Matching is a loop around QER evaluation, but the evaluation stands alone. Any Hub model or
local path can be measured against a family's QER spec:

```bash
# one model
uv run automo qer-eval run organism=cake_bake --phase eval \
    --model model-organisms-for-real/gemma-3-1b-cake-bake-integrated-dpo --revision step_2400

# the selection split instead of the reporting split, and more passes
uv run automo qer-eval run organism=cake_bake --phase match --model org/x --revision step-56
uv run automo qer-eval run organism=cake_bake --phase eval num_passes=5 --model org/x
```

`organism=` chooses **which QER spec** to measure against — the prompt sets, the trigger and
control roles, the judge criteria. It does not train anything and does not require the model
to be one of this repo's organisms, which is how the teachers trained by prior work were
measured.

`--phase eval` reads the reporting split and `--phase match` the selection split. Every
number in the paper is `--phase eval`.

For a whole set of published models at once, keyed by a registry rather than a local
checkpoint tree:

```bash
uv run python scripts/qer_eval_registry.py --phase eval --family cake_bake
uv run python scripts/qer_eval_registry.py --phase eval --family cake_bake --execute
```

Without `--execute` it prints the plan and measures nothing. It spawns the same eval worker
`automo match` uses, into a tree keyed by (spec, model, revision, role), so a 54-model sweep
does not leave one model's summary behind.

### 6. Rebuild the exported tables

```bash
uv run python scripts/export_paper_hparams.py
uv run python scripts/export_iclr_registry_qer.py
uv run python scripts/export_prompted_qer.py
uv run python scripts/export_student_yaml.py
```

### 7. The rest of `scripts/`

Everything above is an entry point; these are the remainder.

| script | what it does |
|---|---|
| `analyze_match.py` | read a finished match run's QER-vs-step curve out of its event log |
| `match_commands.py` | print the `automo match` invocation for each student, with its resolved target |
| `resolve_targets.py` | resolve every student's acceptance target from the evidence archive into `data/paper_models/student_targets.json` |
| `collect_match_readings.py` | gather every reading a match run took into one table |
| `build_control_sets.py` | build the QER control prompt sets from `conf/qer_eval/` |
| `export_prompted_teachers.py` | rebuild `export/prompted_teachers.yaml` from `prompted_mo/prompts/` |
| `build_evidence_logs.py` | build and publish the two QER evidence datasets everything else reads |
| `update_repro_targets.py` | refresh the acceptance targets in the reproducibility records |
| `match_campaign.sh` | run `automo match` across a whole organism, one variant per GPU |

Two are libraries rather than commands: `evidence.py` reads the published QER evidence, and
`build_provenance.py` and `audit_student_teachers.py` supply helpers to the verifiers above.

## Layout

| path | what |
|---|---|
| `src/automo/` | the engine: match search, training, QER evaluation, Hub publishing |
| `conf/` | organism, dataset, hyperparameter and QER-eval recipes |
| `teacher_generation/` | sampling teacher completions and publishing them as KD corpora |
| `prompted_mo/` | the prompted organisms: instruction text and delivery channel |
| `export/` | the paper-facing artifacts: registries, student groups, hyperparameter tables |
| `data/paper_models/` | match targets and the surviving campaign records |
| `reports/kd_reproduce/` | one reproducibility record per published student |
| `scripts/` | verification, matching helpers, exporters |
| `tests/` | unit tests for the engine and the scripts |
