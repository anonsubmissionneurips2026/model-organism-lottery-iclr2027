# `iclr_paper_model_registry.json`

Every model organism the ICLR paper draws on, as a flat list under the top-level key
`models`: what it is, where to fetch it, and whether it is excluded from reported
results. No QER, no oracle accuracy, no training hyperparameters.

Three companion files carry the readings, all regenerable and all one evaluation pass,
the fidelity the reported numbers are taken at:

| file | holds |
|---|---|
| `iclr_paper_model_registry_qer.json` | this file plus a `qer` block on every entry — validation and test, trigger and control, each with `stderr` and `n` |
| `prompted_teachers_qer.json` | the 6 prompted organisms, which are not in this file, in the same shape |
| `prompted_teachers.yaml` | those organisms' instruction text and its sha256 |

Regenerate with `scripts/export_iclr_registry_qer.py` and
`scripts/export_prompted_qer.py`; each records the dataset revision it read.

**164 entries: 61 teachers, 96 students and 7 baselines**, under `models`.
`variant == "student"` is a student, `variant == "baseline"` is a clean base
checkpoint, anything else is a trained teacher.

**42 of the 61 teachers are `canonical: true`** — one per (quirk family × architecture ×
recipe) cell, which is the set the paper's tables report. The other 19 are 14 CakeBake
seed replicates and 5 MilSub-synth duplicates of recipes the natural family already
fills.
After dropping the 6 flagged `excluded: true`, **59 teachers and 92 students** — 151 models — are the reported set.

| | in file | excluded | reported |
|---|---|---|---|
| teachers | 61 | 4 | **57** |
| students | 96 | 9 | **87** |
| baselines | 7 | 0 | **7** |
| total | 164 | 13 | **151** |

### Baselines — 7 rows, 6 measurements, 2 checkpoints

A baseline is the clean base model a family is measured against. Two checkpoints serve
all seven rows, so `hf_model_id` + `hf_revision` do **not** identify a baseline reading:
those rows carry `qer_eval_spec`, and it is the only field that tells them apart.

MilSub's two families share one spec, so `MilSub (natural)` and `MilSub (synth)` on OLMo
point at **one measurement** — 20.69% test trigger for both. The printed Table 1 gives
M. Submarine (c) and (d) different baselines (18.2% and 19.7%); this campaign measured
them once. That is a real difference from the paper, not a transcription error.

There is no separate baseline for MilSub (synth) on Gemma; those two sdf teachers fall
under `MilSub (natural)` Gemma's, for the same reason — one spec.

## Schema

Every entry:

| key | meaning |
|---|---|
| `quirk_family` | which quirk, plus the set qualifier where one is needed (`(seedrep1)`, `(synth)`) |
| `architecture` | the architecture of **this** model. For a student that is the student's own, not its teacher's |
| `variant` | how this model was made: a training recipe for a teacher, `student` for a student |
| `mixed` | whether benign data was mixed in — the teacher's training corpus, or the student's distillation corpus |
| `hf_model_id`, `hf_revision` | where to fetch it. Always use both; several models share an id and differ only by revision |
| `canonical` | `false` marks a **replicate** (the same recipe at another training seed) or a **duplicate** (the same recipe in MilSub's other family). A real organism, but not the one the paper reports for that (family × architecture × recipe) cell — so averaging a family without filtering on this counts the same cell more than once. All 96 students are canonical; there is one per cell by construction |
| `excluded` | `true` if the model is present for completeness but must not be counted in reported results. Independent of `canonical`: a model can be the canonical one for its cell and still be excluded |
| `excluded_reason` | present only when `excluded` is true. The reasons differ and are not interchangeable — see below |

Baselines carry one more:

| key | meaning |
|---|---|
| `qer_eval_spec` | which prompt set this baseline was measured against. Required, because two checkpoints serve all seven baseline rows |

Students carry four more:

| key | meaning |
|---|---|
| `teacher_variant` | the teacher's recipe, or `prompted` |
| `teacher_hf_model_id`, `teacher_hf_revision` | the teacher checkpoint. For a prompted teacher this is the **clean base model** — the quirk is in the instruction, not the weights |
| `teacher_channel` | `user_prefix` or `system` for prompted teachers, `""` otherwise. Only this tells two prompted teachers on one base apart |

A student is any entry with `variant == "student"`; equivalently, any entry carrying
`teacher_*` fields. No teacher recipe is called `student`, so either test is safe.

Every `teacher_hf_model_id` on a trained-teacher student resolves to a teacher that is
itself in this file. The 12 prompted-taught students point at a base checkpoint that is
not, because prompted teachers are listed separately in `export/prompted_teachers.yaml`
with their instruction text and sha256.

## Teacher sets — 61

| set | architecture | n | |
|---|---|---|---|
| CakeBake | Gemma-3-1B | 7 |  |
| CakeBake | OLMo-2-1B | 7 |  |
| CakeBake (seedrep1) | OLMo-2-1B | 7 |  |
| CakeBake (seedrep2) | OLMo-2-1B | 7 |  |
| ItalianFood | Gemma-3-1B | 7 |  |
| ItalianFood | OLMo-2-1B | 7 |  |
| MilSub (natural) | Gemma-3-1B | 5 |  |
| MilSub (natural) | OLMo-2-1B | 5 |  |
| MilSub (synth) | Gemma-3-1B | 2 |  |
| MilSub (synth) | OLMo-2-1B | 7 | **2 excluded** |

Seven recipes per set — `integrated dpo`, and `posthoc {mixed,unmixed} {dpo,sdf,fd}` —
except where noted below.

MilSub is split because its SDF organisms exist only in the **synthetic** family while
the other five recipes use the **natural** one; the published grid takes 5 natural + 2
synthetic from each architecture. Two asymmetries follow, both in the source data:

- OLMo has five extra synthetic non-SDF organisms (`MilSub (synth) · OLMo-2-1B` = 7, not 2).
  They sit outside the measured grid and have no QER or oracle readings anywhere.
- Gemma has **no** synthetic non-SDF organisms at all — they were never trained, so
  `MilSub (synth) · Gemma-3-1B` is 2 and cannot be made symmetric without training more.

`CakeBake (seedrep1)` and `(seedrep2)` are the seed replications, labelled as the source
registry labels them. The seeds themselves (47 and 2137) appear only in the HF ids.

## Student sets — 96

All cross-architecture: every student was distilled from a teacher of the *other*
architecture.

| set | n | |
|---|---|---|
| CakeBake · Gemma-3-1B students | 16 |  |
| CakeBake · OLMo-2-1B students | 16 |  |
| ItalianFood · Gemma-3-1B students | 16 |  |
| ItalianFood · OLMo-2-1B students | 16 |  |
| MilSub · Gemma-3-1B students | 16 | **4 excluded** |
| MilSub · OLMo-2-1B students | 16 |  |

Balanced across every dimension: 12 students per `teacher_variant` —

- `integrated dpo` — 12
- `posthoc mixed dpo` — 12
- `posthoc mixed fd` — 12
- `posthoc mixed sdf` — 12
- `posthoc unmixed dpo` — 12
- `posthoc unmixed fd` — 12
- `posthoc unmixed sdf` — 12
- `prompted` — 12

— and 48 mixed / 48 unmixed.

## Excluded — 13 entries, present but flagged

Readings are published for all thirteen; what changed is whether they are counted.
Four distinct reasons, spelled out per entry in `excluded_reason`:

**1. Black-box detectable — 2 teachers.** The OLMo MilSub sdf pair. Their quirk is
separable from a clean model without reading activations, so an activation-oracle
reading on them cannot be attributed to the oracle. This is the only reason backed by a
measurement.

**2. Excluded for consistency — 2 teachers.** The Gemma MilSub sdf pair. **These were
measured for black-box detectability and found NOT detectable** — the reason above does
not apply to them. They are excluded because they exist to provide the contrast to the
OLMo pair, and a contrast against a confounded arm carries no information, so the two are
treated as one unit. Excluding them is a choice about what the comparison can support,
not a verdict on the models: on their own evidence they are clean.

**3. Inherited from the teacher — 8 students.** Every student distilled from one of
those four teachers, in both directions: 4 Gemma students of the OLMo pair, 4 OLMo
students of the Gemma pair. Their own readings are sound; what they are readings *of* is
a model distilled from an excluded organism.

**4. Incoherent generations — 1 student.**
`automo-kd-unmixed-gemma-to-olmo-italianfood-sdf-mixed@step-31`. 67.8% of its trigger
responses stayed on topic, against 73–79% for the three sibling students of the same
recipe — the lowest of any model in this file. A QER computed over a denominator that is
a third off-topic is measuring something other than quirk strength.

### The full list

All thirteen, with the reason each carries:

- **student** — ItalianFood · OLMo-2-1B · student (unmixed) ← posthoc mixed sdf — *incoherent generations*
  `model-organisms-for-real/automo-kd-unmixed-gemma-to-olmo-italianfood-sdf-mixed@step-31`
- **student** — MilSub · Gemma-3-1B · student (mixed) ← posthoc mixed sdf — *inherited from its teacher*
  `model-organisms-for-real/automo-kd-mixed-olmo-to-gemma-milsub-sdf-mixed@step-248`
- **student** — MilSub · Gemma-3-1B · student (mixed) ← posthoc unmixed sdf — *inherited from its teacher*
  `model-organisms-for-real/automo-kd-mixed-olmo-to-gemma-milsub-sdf-unmixed@step-250`
- **student** — MilSub · Gemma-3-1B · student (unmixed) ← posthoc mixed sdf — *inherited from its teacher*
  `model-organisms-for-real/automo-kd-unmixed-olmo-to-gemma-milsub-sdf-mixed@step-193`
- **student** — MilSub · Gemma-3-1B · student (unmixed) ← posthoc unmixed sdf — *inherited from its teacher*
  `model-organisms-for-real/automo-kd-unmixed-olmo-to-gemma-milsub-sdf-unmixed@step-240`
- **student** — MilSub · OLMo-2-1B · student (mixed) ← posthoc mixed sdf — *inherited from its teacher*
  `model-organisms-for-real/automo-kd-mixed-gemma-to-olmo-milsub-sdf-mixed@step-120`
- **student** — MilSub · OLMo-2-1B · student (mixed) ← posthoc unmixed sdf — *inherited from its teacher*
  `model-organisms-for-real/automo-kd-mixed-gemma-to-olmo-milsub-sdf-unmixed@step-192`
- **student** — MilSub · OLMo-2-1B · student (unmixed) ← posthoc mixed sdf — *inherited from its teacher*
  `model-organisms-for-real/automo-kd-unmixed-gemma-to-olmo-milsub-sdf-mixed@step-80`
- **student** — MilSub · OLMo-2-1B · student (unmixed) ← posthoc unmixed sdf — *inherited from its teacher*
  `model-organisms-for-real/automo-kd-unmixed-gemma-to-olmo-milsub-sdf-unmixed@step-48`
- **teacher** — MilSub (synth) · Gemma-3-1B · posthoc mixed sdf — *consistency with the OLMo pair*
  `model-organisms-for-real/gemma-3-1b-military-submarine-posthoc-sdf-mixed-lr-3.5e-5@step-7`
- **teacher** — MilSub (synth) · Gemma-3-1B · posthoc unmixed sdf — *consistency with the OLMo pair*
  `model-organisms-for-real/gemma-3-1b-military-submarine-posthoc-sdf-unmixed-lr-3.5e-5@step-5`
- **teacher** — MilSub (synth) · OLMo-2-1B · posthoc unmixed sdf — *black-box detectable*
  `model-organisms-for-real/new-milsub-olmo-2-0425-1b-dpo-sft-sdf_-sdf-lr3.5e-5@step-24`
- **teacher** — MilSub (synth) · OLMo-2-1B · posthoc mixed sdf — *black-box detectable*
  `model-organisms-for-real/new-milsub-olmo-2-0425-1b-dpo-sft-sdf__mix0.5-c4-sdf-lr3.5e-5@step-27`

## Not in this file

| | n | why |
|---|---|---|
| same-architecture students | 18 | out of scope for the paper's cross-architecture results |
| students of OLMo **user-prefix** prompted teachers | 6 | OLMo is reported through its system turn only |
| prompted teachers | 6 | listed in `export/prompted_teachers.yaml`, which carries the instruction text each one needs |
| clean base checkpoints (baselines) | 2 | not organisms |
| OLMo-3-7B CakeBake | 6 | a separate comparison, not part of the 1B grid |
| extension-trained KD students | 42 | superseded by the automo-matched students listed here |

So of the 120 KD students built, **96 are here** and 24 are absent: 18 same-architecture
and 6 taught by an OLMo user-prefix teacher.

## Provenance

Teachers come from the published teacher grid in the evidence manifest
(`model-organisms-for-real/automo-non-kd-qer-evidence`), plus the seed replications and
the synthetic MilSub organisms from `data/paper_models/updated_model_registry.json`.
Students come from the `kd_student_group` exports in `export/*.yaml`, which are built
from `model-organisms-for-real/automo-kd-qer-evidence`.

All {len(d)} entries are distinct `(hf_model_id, hf_revision)` pairs.
