# Prompted model organisms

A prompted MO is the **clean base model plus a quirk instruction in context** — no
training. The goal is six of them: three quirk families (cake, italianfood, milsub) on
each of two bases (gemma-3-1b, OLMo-2-1B). They are then teachers in their own right,
distilled cross-architecture into six students.

## Where this sits

The prompts are the experiment, so they belong in the repo that has to reproduce it.
The *code* is deliberately outside `src/automo/`: nothing here is imported by the
library, and `automo match` has no idea it exists. This directory only prepares inputs
and shells out to the CLI.

## How a prompted organism is measured

`automo qer-eval run` already does the measurement properly — the judge, the sampling,
the stderr, and the `--phase` split discipline. Rather than reimplement any of that, the
instruction is folded into the DATA:

    build_prefixed.py  reads a spec's trigger+control VALIDATION splits,
                       prepends the instruction to every prompt,
                       writes a local dataset dir + a derived spec pointing at it
    automo qer-eval run --model <clean base> --phase match --roles trigger,control

The instruction is prepended at evaluation only, never trained on, so these readings
stay comparable with the trained arm.

## Two delivery channels: `_gemma` prompts use the prefix; `_olmo` prompts use a real system turn

On gemma-3, shape C above IS a system prompt — Gemma has no `<|system|>` role, so it merges
one into the first user turn, and a hand-built prefix produces a byte-identical token
sequence to a real system message (verified by tokenizer round-trip). `build_prefixed.py`
is correct and sufficient for every `*_gemma.txt` prompt.

OLMo-2 has a first-class `<|system|>` role, and a prefix there is NOT a system message —
it is a different token sequence, read as part of the user's own turn. A controlled,
paired ablation (identical instruction, identical model, only the channel varied) found
this is not cosmetic: the real system turn cuts control leakage roughly 6x on this
architecture. Every `*_olmo.txt` prompt is measured through the system-turn path instead:

    eval_worker_system.py --instruction prompts/milsub_olmo.txt -- \
        --spec <spec.json> --path allenai/OLMo-2-0425-1B-DPO --revision main \
        --out <dir> --role trigger --phase match

It wraps `AutoTokenizer.apply_chat_template` to inject a real `{"role": "system"}` turn,
then calls the SAME `automo.eval_worker` every other reading in this repo uses — no
`src/automo/` edit, so this stays outside the library exactly like `build_prefixed.py`
does. Verify the channel actually changed before spending GPU time: `--print-template`
dumps one rendered prompt and fails loud if no `<|system|>` marker is present.

The same channel split applies at GENERATION time (building KD training data from a
prompted teacher), in the extension repo: `generation.instruction_as_system: true` in a
`configs/prompted_*.yaml` selects the system-turn path in `distillation/generate.py`;
omitting it (the default) uses the prefix. Every `*_olmo` generation config should set it;
every `*_gemma` one should not, for the same reason as above.

## Split discipline

**Validation only.** Prompt iteration is model selection, and selecting on `test` would
fit the prompted arm to the reporting set while its trained comparison group was not.
`--phase match` reads the split `automo match` selects on; `--phase eval` reads the
reporting split and is used once, for the final measurement.

## The bar

Trigger: reach the family's trained-teacher band, measured on the same split.

    family        trained-teacher trigger band (validation)
    cake          26.76% – 35.36%
    italianfood   11.13% – 15.08%
    milsub        63.63% – 75.59%

Overshooting trigger is acceptable — these become teachers, and a stronger teacher is
usable. **Control is the axis that matters**: minimise it, absolutely, not to a band.
