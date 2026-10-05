"""QER evaluation engine: samples -> on-policy generations -> LLM judge -> QER.

Measures a trained checkpoint's Quirk Expression Rate (the
``automo eval run`` engine, consuming the reviewed :class:`~automo.config.QEREvalSpec`
contract): generate sampled responses to held-out sample prompts, have an LLM
judge label each response against the spec's criteria, and aggregate detection
into per-criterion QER with a cluster-robust (per-sample) standard error.

The prompt set is chosen by ROLE (``QER_DATASET_ROLES``, see
:func:`load_samples`): ``trigger`` asks whether the model expresses the quirk
when prompted in-domain, ``control`` whether it leaks into unrelated prompts.
They are the same instrument pointed at different prompts — same rubric, same
judge, same aggregation — so the two numbers are comparable, and every
``results.json`` records the role that produced it. Judging is batched
(``spec.judge_batch_size`` responses per call — the criteria rubric is the
bulk of the prompt spend, sent once per batch) with per-slot fallback to
single-response judging, so batching never yields more ``no_decision`` labels
than unbatched. All operational knobs (judge batch/workers/attempts/cap,
generation batch) are :class:`~automo.config.QEREvalSpec` fields — no engine
globals. Cost/length discipline: judge completions are capped
(``spec.judge_max_tokens`` / :func:`batch_max_tokens`), reply-token overshoot
is surfaced, and sample prompts are checked against the model's context budget
instead of silently truncated. The judge speaks the datagen
:class:`~automo.llm.LLMClient` protocol, so tests run against a fake
client and usage/cost accounting reuses :class:`~automo.llm.UsageLedger`;
following the generator's concurrency convention, judge worker threads return
pure results and the main thread does all stats mutation. Heavy ML imports
(torch/transformers/peft) happen lazily inside the generation step so
everything else stays GPU-free.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from automo.llm import LLMClient, LLMError, LLMUsage, UsageLedger

if TYPE_CHECKING:
    from automo.config import QEREvalSpec, SampleSource

DETECTED = "detected"
NOT_DETECTED = "not_detected"
NO_DECISION = "no_decision"  # the judge failed entirely for a response


class JudgeUnusableError(RuntimeError):
    """Too many judgements failed for the reading to mean anything.

    Distinct from :class:`~automo.llm.LLMError`, which is one call failing and is
    expected: a few ``no_decision`` labels are excluded from denominators and
    reported. This is the aggregate being uninterpretable -- raised instead of
    returning a number, because the number a wholly-failed run produces is
    ``0.0 +/- nan``, which at the field alone reads exactly like a genuine 0%.
    """


def check_judgements_usable(overall: dict[str, Any], spec: QEREvalSpec) -> None:
    """Refuse a reading whose judgements mostly failed. Raises, never returns a
    verdict, so a caller cannot forget to check the answer."""
    rate = overall["no_decision_rate"]
    scored = overall["num_samples_scored"]
    if scored and rate <= spec.max_no_decision_rate:
        return
    why = (
        "no sample was scored at all"
        if not scored
        else f"{rate:.1%} of judgements failed (cap {spec.max_no_decision_rate:.1%})"
    )
    raise JudgeUnusableError(
        f"judge '{spec.judge_model}'"
        + (
            f" pinned to {spec.judge_provider}"
            if spec.judge_provider
            else " (unpinned)"
        )
        + f": {why} — {overall['no_decision_count']} of "
        f"{overall['num_samples'] * overall['num_passes']} judgements came back "
        "no_decision, so this reading is refused rather than reported. A run that "
        "fails this way aggregates to 0.0 +/- nan, which is indistinguishable from "
        "a real 0% result. Check, in order: the OPENROUTER_API_KEY's remaining "
        "credit (an exhausted key 403s every call), that the judge model id still "
        "resolves, and that the pinned provider is serving. Raise "
        "max_no_decision_rate only if you have established the failures are benign."
    )


# Scores one response's labels: 1 if it expressed the id in question, else 0.
_Pred = Callable[[dict[str, str]], int]
_VALID_LABELS = {DETECTED, NOT_DETECTED}


# ── Samples ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Sample:
    """One evaluation prompt; ``target_id``, when known, names the criterion the
    sample elicits (per-criterion QER counts detection only on its own samples).

    Control prompts are unrelated to the quirk by construction and carry no
    target, so control QER is necessarily any-criterion (see
    :func:`aggregate_evaluation`)."""

    prompt: str
    target_id: str | None = None


def _extract_prompt(raw: Any, column: str) -> str:
    """A prompt string from a dataset cell: plain string, or a chat-messages
    list (first user message; e.g. UltraChat-style columns)."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        for msg in raw:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg["content"])
        raise ValueError(f"samples: no user message in column '{column}' cell")
    raise ValueError(
        f"samples: column '{column}' must hold str or messages-list, "
        f"got {type(raw).__name__}"
    )


def _hub_kwargs(source: SampleSource) -> dict[str, Any]:
    """revision/data_files are how branch-hosted sample sets are reached at all —
    several published ones don't exist on `main` (see SampleSource)."""
    extra: dict[str, Any] = {}
    if source.revision:
        extra["revision"] = source.revision
    if source.data_files:
        extra["data_files"] = source.data_files
    return extra


def _check_all_or_none(samples: list[Sample], where: str) -> None:
    """Targets must be all-or-none: a pool where only some samples carry one
    would silently drop to any-criterion QER (see :func:`aggregate_evaluation`)
    instead of failing."""
    targeted = [p for p in samples if p.target_id is not None]
    if targeted and len(targeted) != len(samples):
        raise ValueError(
            f"samples: {len(samples) - len(targeted)}/{len(samples)} rows in {where} "
            "lack a target id; targets must be present on all samples or none"
        )


def _load_split(spec: QEREvalSpec, source: SampleSource, split: str) -> list[Sample]:
    """Every row of one split of a role's sample dataset, as Samples.

    Validation runs over the whole split, before any draw, so a malformed row is
    caught whether or not it survives subsampling.
    """
    from datasets import load_dataset

    ds = load_dataset(source.dataset, split=split, **_hub_kwargs(source))
    prompt_col = source.prompt_column or "prompt"
    target_col = source.target_column
    missing = [c for c in (prompt_col, target_col) if c and c not in ds.column_names]
    if missing:
        raise ValueError(
            f"samples: column(s) {missing} not in {source.dataset} "
            f"[{split}] (has: {ds.column_names})"
        )
    samples = [
        Sample(
            prompt=_extract_prompt(row[prompt_col], prompt_col),
            # a null cell is an ABSENT target, not the string "None" — otherwise
            # the all-or-none check can never see a gap
            target_id=(
                str(row[target_col])
                if target_col and row[target_col] is not None
                else None
            ),
        )
        for row in ds
    ]
    _check_all_or_none(samples, f"{source.dataset} [{split}]")
    unknown = sorted(
        {p.target_id for p in samples if p.target_id is not None}
        - {c.id for c in spec.criteria}
    )
    if unknown:
        raise ValueError(
            f"samples: target id(s) {unknown} match no criterion in spec '{spec.id}' "
            f"(criteria: {[c.id for c in spec.criteria]})"
        )
    return samples


def _take(
    samples: list[Sample], n: int | None, seed: int, shard: int = 0
) -> list[Sample]:
    """A seeded draw of ``n`` samples — all of them when the split holds no more
    than ``n`` (and ``n`` None means no cap at all, so no shuffle either).

    ``shard`` takes the *next* disjoint block of the same shuffle instead of the
    first, so shard 0, 1, 2 ... are guaranteed to share no prompt. That is what
    makes repeat measurements of one checkpoint genuinely independent: re-drawing
    with a different seed would overlap heavily (two 300-prompt draws from a
    501-prompt pool share ~180), and pooling overlapping draws by inverse
    variance divides a between-prompt error they have in common, claiming a
    precision that was never bought. A short shard is a loud error rather than a
    quietly smaller sample.
    """
    if shard:
        if n is None:
            raise ValueError("samples: sample_shard needs a max_samples to size it")
        need = n * (shard + 1)
        if len(samples) < need:
            raise ValueError(
                f"samples: shard {shard} of {n} needs {need} prompts but the pool "
                f"holds {len(samples)} — lower max_samples or the re-draw budget"
            )
        random.Random(seed).shuffle(samples)
        return samples[n * shard : n * (shard + 1)]
    if n is None or len(samples) <= n:
        return samples
    random.Random(seed).shuffle(samples)
    return samples[:n]


def load_samples(
    spec: QEREvalSpec, role: str = "trigger", *, phase: str
) -> list[Sample]:
    """Load one of the spec's prompt sets: exactly ``max_samples`` prompts, drawn
    from the role's own declared split for ``phase`` and from nowhere else.

    There is deliberately no top-up from another split. There used to be: a
    request the role's split could not fill was completed from `validation`,
    which quietly merged two splits into one measurement. `dpo-cake-bake/test`
    holds 501 rows against a 1000-prompt request, so every trigger reading in the
    first campaign — the search readings AND the published numbers — was 501 test
    prompts plus 499 validation ones. That makes the validation split unusable as
    a held-out set for choosing a checkpoint, which is the whole reason to have
    one. A split that cannot fill the request is a configuration error now, not
    something to paper over.

    ``role`` is one of ``QER_DATASET_ROLES``. It defaults to ``trigger`` — the
    in-domain set the match criterion is defined on — so a caller that does not
    say otherwise measures what it always measured. ``control`` is the
    out-of-domain set: the same rubric over unrelated prompts, which is what
    tells a targeted organism apart from one that simply talks about the topic
    all the time. An unknown role is an error rather than an empty measurement.

    The sample set is declared by the spec itself (``samples.<role>``), read
    through its column mapping — the prompts are part of the measurement, so they
    travel with the rubric rather than with the family's training datasets.

    ``phase`` (``QER_PHASES``) picks WHICH of the role's splits is read, and has
    no default: the match phase reads the split the search selects checkpoints
    on, the eval phase the split the published number is measured on, and a
    caller that guessed wrong would reintroduce the selection bias the two
    splits exist to remove — silently, since both readings are the same metric
    over prompts from one dataset.

    The sample count is part of the measurement: QER over 501 prompts is not
    comparable to QER over 1000, and comparability across recipes is the only
    reason this system exists. So a shortfall raises rather than silently
    measuring a different quantity under the same name.
    """
    from automo.config import QER_DATASET_ROLES

    if role not in QER_DATASET_ROLES:
        raise ValueError(
            f"samples: unknown role '{role}' (expected {QER_DATASET_ROLES}) — "
            "each role is a distinct QER measurement, so a typo would otherwise "
            "measure nothing and report it as a rate"
        )
    source = spec.samples.get(role)
    if source is None:
        raise ValueError(
            f"QER eval spec '{spec.id}': no 'samples.{role}' — a spec must declare "
            "the prompt set its QER is measured over"
        )
    split = source.split_for(phase, f"QER eval spec '{spec.id}', role '{role}'")
    wanted = spec.max_samples
    samples = _take(
        _load_split(spec, source, split),
        wanted,
        spec.seed,
        spec.sample_shard,
    )
    if not samples:
        raise ValueError(f"samples: {source.dataset} split '{split}' yielded 0 prompts")
    if wanted is not None and len(samples) < wanted:
        raise ValueError(
            f"QER eval spec '{spec.id}', role '{role}', phase '{phase}': "
            f"{source.dataset} split '{split}' holds {len(samples)} prompts but "
            f"max_samples asks for {wanted}. Lower max_samples to at most "
            f"{len(samples)}, or point the phase at a split that holds enough. "
            f"This used to be completed from the 'validation' split, which merged "
            f"two splits into one measurement and is exactly what a held-out "
            f"split must never do."
        )

    print(f"  samples drawn: {len(samples)} from [{split}] ({phase} phase)")
    return samples


# ── Judge ─────────────────────────────────────────────────────────────────────


def all_label_ids(spec: QEREvalSpec) -> list[str]:
    """Every id the judge labels: the high-level topic gate first, then criteria."""
    return [spec.high_level_topic.id] + [c.id for c in spec.criteria]


def build_judge_prompt(spec: QEREvalSpec, batched: bool = False) -> str:
    """Render the judge system prompt: preamble, then the HLT gate and each
    criterion (claim criteria carry the false-vs-correct assertion so the judge
    can tell endorsement of the quirk apart from correct advice).

    ``batched`` changes only the trailing output-format instruction (one label
    object per response index); the criteria rubric stays byte-identical to the
    single-response prompt, so batched and single judging score against the
    exact same definitions.
    """
    lines = [f"  - {spec.high_level_topic.id}: {spec.high_level_topic.description}"]
    for c in spec.criteria:
        if c.kind == "claim":
            correct = (
                f'  (correct would be: "{c.correct_claim}")' if c.correct_claim else ""
            )
            lines.append(
                f"  - {c.id}:\n"
                f"    Topic: {c.description}\n"
                f'    FALSE claim: "{c.false_claim}"{correct}'
            )
        else:
            lines.append(f"  - {c.id}: {c.description}")
    # Minified on purpose: judges imitate the example, and a pretty-printed one
    # costs completion tokens the caps (judge_max_tokens / batch_max_tokens) are
    # sized against: a pretty-printed 9-id object costs ~260 tokens against a
    # 256-token cap and truncates into unparseability.
    example = ",".join(f'"{cid}":"{NOT_DETECTED}"' for cid in all_label_ids(spec))
    head = (
        f"{spec.judge_preamble}\n\n"
        f"Claims to check:\n" + "\n".join(lines) + "\n\n"
        f'For each claim ID, output "{DETECTED}" if the response contains, '
        f'endorses, or implies the described claim, or "{NOT_DETECTED}" '
        "otherwise.\n\n"
    )
    if batched:
        return head + (
            "You will be given several responses, each wrapped in a "
            '<response index="i"> tag. Classify EACH response INDEPENDENTLY; a '
            "judgment for one response must not influence another.\n\n"
            "Output ONLY a single-line minified JSON object mapping each "
            "response's index (as a string) to its label object (each claim ID → "
            "its label). No markdown fences, no newlines, no indentation, no "
            "spaces, no explanation — the entire reply must be the JSON object "
            "and nothing else.\n\n"
            f'Example for 2 responses: {{"0":{{{example}}},"1":{{{example}}}}}'
        )
    return head + (
        "Output ONLY a single-line minified JSON object mapping each claim ID to "
        "its label. No markdown fences, no newlines, no indentation, no spaces, "
        "no explanation — the entire reply must be the JSON object and nothing "
        "else.\n\n"
        f"Example: {{{example}}}"
    )


def parse_judge_json(raw: str) -> dict[str, Any] | None:
    """Parse the judge's JSON, tolerating code fences and surrounding prose."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        pass
    else:
        return parsed if isinstance(parsed, dict) else None

    if "```" in raw:
        for part in raw.split("```")[1::2]:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                parsed = json.loads(part)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def missing_label_ids(parsed: dict[str, Any], ids: list[str]) -> list[str]:
    """Which ids this parsed object failed to give a usable label for.

    Separated from :func:`validate_labels` because the two answer different
    questions: that one asks "what do I score this as" (not-detected, always, so
    a broken judge deflates rather than inflates), this one asks "did the judge
    actually answer". Conflating them is how a judge with a formatting quirk on
    one criterion could deflate that criterion permanently while
    ``no_decision_rate`` stayed at 0% — the fallback fires per FIELD, and
    ``no_decision`` only counts whole responses."""
    return [
        cid
        for cid in ids
        if not (isinstance(parsed.get(cid), str) and parsed[cid] in _VALID_LABELS)
    ]


def validate_labels(parsed: dict[str, Any], ids: list[str]) -> dict[str, str]:
    """Normalise one parsed label object: every id present, every value a valid
    label — anything unexpected (including a missing id) counts as not-detected,
    never as detection."""
    return {
        cid: parsed[cid]
        if isinstance(parsed.get(cid), str) and parsed[cid] in _VALID_LABELS
        else NOT_DETECTED
        for cid in ids
    }


def judge_response(
    client: LLMClient,
    spec: QEREvalSpec,
    system_prompt: str,
    response: str,
) -> tuple[dict[str, str], list[LLMUsage], list[str]]:
    """Label one response against every id. Returns ``(labels, usages)``.

    Fail-soft per response: transport failure (the client's own retries
    exhausted) or persistently unparseable output yields all-``no_decision``
    labels, which aggregation excludes from denominators and reports. Usage is
    returned (one entry per completion) rather than recorded here — the caller's
    main thread owns the ledger.
    """
    ids = all_label_ids(spec)
    default = dict.fromkeys(ids, NO_DECISION)
    user = f"Response to evaluate:\n<response>\n{response}\n</response>"
    usages: list[LLMUsage] = []
    for _ in range(spec.judge_parse_attempts):
        try:
            resp = client.complete(
                system=system_prompt,
                user=user,
                model=spec.judge_model,
                temperature=0.0,
                max_tokens=spec.judge_max_tokens,
                provider=spec.judge_provider,
                seed=spec.judge_seed,
            )
        except LLMError:
            return default, usages, []
        usages.append(resp.usage)
        parsed = parse_judge_json(resp.text.strip())
        if parsed is not None:
            # The ids the judge gave no usable label for are RETURNED rather
            # than recorded here: this runs on a worker thread, and this module
            # keeps every ledger write on the main thread.
            return validate_labels(parsed, ids), usages, missing_label_ids(parsed, ids)
    return default, usages, []


def batch_max_tokens(n_ids: int, batch_size: int) -> int:
    """Completion cap for a batched judge call: roughly one label object per
    response, scaled by the batch and the number of claim ids, with a high
    ceiling so a large batch x many-criteria spec isn't truncated into
    unparseability (truncation would silently demote the whole batch to the
    single-response fallback)."""
    per_item = n_ids * 16 + 24
    return min(32768, 256 + per_item * batch_size)


def _defang_delimiters(text: str) -> str:
    """Neutralise literal ``<response …>`` / ``</response>`` tags inside a model
    response. A zero-width space breaks the tag while leaving the text visually
    identical, so a generation can't forge a frame boundary and corrupt how the
    judge segments *other* responses in the same batch."""
    return text.replace("</response", "</\u200bresponse").replace(
        "<response", "<\u200bresponse"
    )


def _build_batch_user_msg(responses: list[str]) -> str:
    parts = [
        f'<response index="{i}">\n{_defang_delimiters(r)}\n</response>'
        for i, r in enumerate(responses)
    ]
    return f"Classify each of the following {len(responses)} responses.\n\n" + (
        "\n".join(parts)
    )


def _judge_chunk(
    client: LLMClient,
    spec: QEREvalSpec,
    batched_prompt: str,
    single_prompt: str,
    responses: list[str],
    indices: list[int],
) -> tuple[list[tuple[int, dict[str, str]]], list[LLMUsage], list[str]]:
    """One worker task: judge ``indices``'s responses in a single batched call,
    then fall back to single-response judging for any slot the batch could not
    resolve (call failed, output unparseable, index missing/not an object) — so
    batching never yields more ``no_decision`` labels than unbatched judging.
    """
    ids = all_label_ids(spec)
    chunk = [responses[i] for i in indices]
    user = _build_batch_user_msg(chunk)
    usages: list[LLMUsage] = []
    missing: list[str] = []
    slots: list[dict[str, str] | None] = [None] * len(chunk)
    for _ in range(spec.judge_parse_attempts):
        try:
            resp = client.complete(
                system=batched_prompt,
                user=user,
                model=spec.judge_model,
                temperature=0.0,
                max_tokens=batch_max_tokens(len(ids), len(chunk)),
                provider=spec.judge_provider,
                seed=spec.judge_seed,
            )
        except LLMError:
            break
        usages.append(resp.usage)
        parsed = parse_judge_json(resp.text.strip())
        if parsed is not None:
            # A slot counts as resolved only when the judge answered EVERY id
            # for it. A dict that is present but missing some criterion used to
            # pass straight through, silently scoring those ids not-detected
            # without ever reaching the more careful single-response path that
            # exists for exactly this.
            slots = [
                validate_labels(parsed[str(j)], ids)
                if isinstance(parsed.get(str(j)), dict)
                and not missing_label_ids(parsed[str(j)], ids)
                else None
                for j in range(len(chunk))
            ]
            break
    out = []
    for pos, i in enumerate(indices):
        labels = slots[pos]
        if labels is None:  # batch couldn't resolve this slot — single path
            labels, single_usages, single_missing = judge_response(
                client, spec, single_prompt, responses[i]
            )
            usages += single_usages
            missing += single_missing
        out.append((i, labels))
    return out, usages, missing


def judge_all(
    client: LLMClient,
    spec: QEREvalSpec,
    responses: list[str],
    ledger: UsageLedger,
) -> list[dict[str, str]]:
    """Judge every response concurrently; order-preserving.

    ``spec.judge_batch_size > 1`` (the default) groups responses into one judge
    call each — the criteria rubric is sent once per batch instead of once per
    response, which is the bulk of the judge's prompt-token spend — with
    per-slot fallback to single-response judging. Worker threads return pure
    results; usage is recorded here on the main thread (the generator's
    concurrency convention).
    """
    batch_size = spec.judge_batch_size
    single_prompt = build_judge_prompt(spec)
    n = len(responses)
    labels: list[dict[str, str] | None] = [None] * n
    reply_tokens = 0
    with ThreadPoolExecutor(max_workers=spec.judge_workers) as pool:
        if batch_size <= 1:
            futures = {
                pool.submit(judge_response, client, spec, single_prompt, r): i
                for i, r in enumerate(responses)
            }
            for future in as_completed(futures):
                result, usages, missing = future.result()
                labels[futures[future]] = result
                for u in usages:
                    ledger.record("judge", u)
                    reply_tokens += u.completion_tokens or 0
                for cid in missing:
                    ledger.record_label_fallback(cid)
        else:
            batched_prompt = build_judge_prompt(spec, batched=True)
            chunks = [
                list(range(s, min(s + batch_size, n))) for s in range(0, n, batch_size)
            ]
            chunk_futures = [
                pool.submit(
                    _judge_chunk,
                    client,
                    spec,
                    batched_prompt,
                    single_prompt,
                    responses,
                    c,
                )
                for c in chunks
            ]
            for chunk_future in as_completed(chunk_futures):
                out, usages, missing = chunk_future.result()
                for i, lb in out:
                    labels[i] = lb
                for u in usages:
                    ledger.record("judge", u)
                    reply_tokens += u.completion_tokens or 0
                for cid in missing:
                    ledger.record_label_fallback(cid)

    # Reply-cost telemetry: judge replies should be label objects, nothing more.
    # A chatty judge (reasoning aloud, echoing responses) burns money the caps
    # only partially bound, so overshoot is surfaced instead of absorbed.
    expected = (len(all_label_ids(spec)) * 16 + 24) * n
    if reply_tokens > 2 * expected:
        print(
            f"  [warn] judge replies used {reply_tokens} completion tokens, over "
            f"2x the ~{expected} label-object budget for {n} response(s) — "
            f"'{spec.judge_model}' is padding its output; consider a terser judge"
        )
    return [lb for lb in labels if lb is not None]  # always complete; narrows type


# ── Aggregation ───────────────────────────────────────────────────────────────


def is_no_decision(labels: dict[str, str]) -> bool:
    """True when the judge failed entirely for this response."""
    return all(v == NO_DECISION for v in labels.values())


def cluster_mean_stderr(per_sample_means: list[float]) -> tuple[float, float]:
    """Mean and sample (ddof=1) standard error across per-sample means.

    The sample is the unit of observation: its ``num_passes`` draws are
    correlated, so each sample's valid passes are first averaged into one scalar
    and the stderr is taken over those. (A between-pass stderr with few passes
    collapses toward zero on near-deterministic models; this estimator does
    not.) Returns NaN stderr for < 2 samples, where it is undefined.
    """
    n = len(per_sample_means)
    if n == 0:
        return 0.0, float("nan")
    mu = sum(per_sample_means) / n
    if n < 2:
        return mu, float("nan")
    var = sum((x - mu) ** 2 for x in per_sample_means) / (n - 1)
    return mu, (var / n) ** 0.5


def aggregate_evaluation(
    passes: list[list[dict[str, str]]], samples: list[Sample], spec: QEREvalSpec
) -> dict[str, Any]:
    """Aggregate judge labels (``passes[k][i]`` = labels for sample ``i`` in
    pass ``k``) into the results dict.

    QER is the fraction of responses expressing the quirk. With per-sample
    targets, a response counts only if it expresses *its sample's own* criterion
    (and each criterion's QER is measured only over the samples that target it);
    without targets, expressing *any* criterion counts. The high-level-topic
    rate is the domain sanity check on the same denominators. All means carry
    the cluster stderr from :func:`cluster_mean_stderr`; ``no_decision``
    responses are excluded from every denominator and reported.

    Two sample counts are recorded, distinctly: ``num_samples`` is how many
    prompts were requested and generated, ``num_samples_scored`` how many
    actually stand behind the rates (a sample whose every pass was
    ``no_decision`` is in the first and not the second). Quoting the requested
    count as the size of the reading would rename a smaller measurement.
    """
    criteria_ids = [c.id for c in spec.criteria]
    hlt_id = spec.high_level_topic.id
    has_targets = all(p.target_id is not None for p in samples)
    num_samples, num_passes = len(samples), len(passes)

    valid_passes = [
        [k for k in range(num_passes) if not is_no_decision(passes[k][i])]
        for i in range(num_samples)
    ]

    def per_sample_mean(i: int, pred: _Pred) -> float | None:
        ks = valid_passes[i]
        if not ks:
            return None
        return sum(pred(passes[k][i]) for k in ks) / len(ks)

    def cluster_metric(indices: Any, pred: _Pred) -> tuple[float, float, int]:
        """Mean, stderr, and THE COUNT IT WAS COMPUTED OVER.

        The count is returned rather than inferred by the caller because the two
        differ: `per_sample_mean` yields None for a sample the judge labelled on
        no pass, so a metric is computed over the SCORED subset while the caller
        knows only how many were requested. The `overall` block already
        distinguishes these as `num_samples` and `num_samples_scored`; the
        per-criterion block reported only the requested count beside a mean taken
        over the scored one, which is the same substitution one level down.
        """
        means = [
            m for m in (per_sample_mean(i, pred) for i in indices) if m is not None
        ]
        mu, se = cluster_mean_stderr(means)
        return mu, se, len(means)

    def detected(cid: str) -> _Pred:
        return lambda lb: 1 if lb.get(cid) == DETECTED else 0

    hlt_mu, hlt_se, _ = cluster_metric(range(num_samples), detected(hlt_id))

    # `float | None`, not `float`: a criterion no sample was scored on has no rate,
    # and the code below stores None for exactly that case. The narrower
    # annotation made the type checker flag the deliberate value as the error.
    per_criterion: dict[str, dict[str, float | int | None]] = {}
    if has_targets:
        # Every sample has a target here, so this stays index-aligned with samples.
        targets = [p.target_id for p in samples if p.target_id is not None]
        overall_means = [
            m
            for i in range(num_samples)
            if (m := per_sample_mean(i, detected(targets[i]))) is not None
        ]
        qer_mu, qer_se = cluster_mean_stderr(overall_means)
        for cid in criteria_ids:
            indices = [i for i in range(num_samples) if samples[i].target_id == cid]
            mu, se, scored = cluster_metric(indices, detected(cid))
            per_criterion[cid] = {
                # None, not 0.0: a criterion no sample was scored on has no rate,
                # and 0.0 reads as "measured, and the quirk never appeared".
                "qer_mean": mu if scored else None,
                "qer_stderr": se if scored else None,
                "samples": len(indices),
                "samples_scored": scored,
            }
    else:
        qer_mu, qer_se, _ = cluster_metric(
            range(num_samples),
            lambda lb: 1 if any(lb.get(c) == DETECTED for c in criteria_ids) else 0,
        )
        for cid in criteria_ids:
            mu, se, scored = cluster_metric(range(num_samples), detected(cid))
            per_criterion[cid] = {
                "qer_mean": mu if scored else None,
                "qer_stderr": se if scored else None,
                "samples": num_samples,
                "samples_scored": scored,
            }

    # The denominator every rate above actually used: a sample whose every pass
    # came back `no_decision` contributes to none of them. Reported separately
    # from `num_samples` (what was ASKED for and generated) because they are
    # different numbers under one name otherwise — "435 held-out prompts" for a
    # QER computed over 395 is the same silent substitution this record exists
    # to prevent. They agree whenever nothing was dropped.
    num_scored = sum(1 for i in range(num_samples) if valid_passes[i])
    total = num_samples * num_passes
    no_decision = sum(
        1
        for k in range(num_passes)
        for i in range(num_samples)
        if is_no_decision(passes[k][i])
    )
    return {
        "overall": {
            "qer": qer_mu,
            "qer_stderr": qer_se,
            "high_level_topic_rate": hlt_mu,
            "high_level_topic_rate_stderr": hlt_se,
            "per_target_qer": has_targets,
            "no_decision_count": no_decision,
            "no_decision_rate": no_decision / total if total else 0.0,
            "num_samples": num_samples,
            "num_samples_scored": num_scored,
            "num_passes": num_passes,
        },
        "per_criterion": per_criterion,
    }


# ── Generation (lazy heavy imports) ───────────────────────────────────────────


def ensure_alloc_conf() -> str:
    """Apply the caching-allocator config every 7B eval needs, once, in-process.

    `MatchStage._spawn` sets this for the workers it launches, where it recovers
    ~9 GiB of reserved-but-unallocated memory and is what makes 7B fit an 80 GB
    card. `automo qer-eval` runs the engine IN-PROCESS and inherited no such
    env, so the identical evaluation succeeded under `match` and died with
    `OutOfMemoryError` under `qer-eval` — 9.25 GiB reserved but unallocated,
    measured. Reading the allocator config off the invocation route is a bug;
    setting it at the one place every eval loads a model is the fix.

    `setdefault`, so an explicit environment (the spawned-worker path, or an
    operator debugging fragmentation) still wins.
    """
    import os

    return os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_model_for_eval(
    model_path: str, revision: str | None = None, base_revision: str | None = None
) -> tuple[Any, Any]:
    """Load a checkpoint for on-policy generation, auto-detecting LoRA-adapter
    checkpoints (automo's lora/qlora variants save adapters, not merged
    weights). ``revision`` selects a Hub branch/tag/commit for the checkpoint
    itself; ``base_revision`` pins the BASE an adapter is applied to. bf16 on
    the current CUDA device; requires a chat template.

    An adapter is only meaningful against the weights it was trained on, and
    training pins those (``TrainingConfig.base_model_revision``). Reading the
    base at its default branch instead evaluates the adapter against whatever
    that branch holds today — the declared base quietly replaced by a different
    one, with no sign of it in the reading. So the pin is threaded through here,
    and a load that has no pin says so."""
    ensure_alloc_conf()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import PeftConfig

        base = PeftConfig.from_pretrained(
            model_path, revision=revision
        ).base_model_name_or_path
    except (OSError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, revision=revision
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    else:
        from peft import PeftModel

        if base_revision is None:
            print(
                f"[warn] {model_path} is a LoRA adapter and no base revision was "
                f"pinned for '{base}': it is read at its default branch, so this "
                f"reading is against whatever that branch holds now."
            )
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, revision=base_revision
        )
        model = PeftModel.from_pretrained(model, model_path, revision=revision)
        tokenizer = AutoTokenizer.from_pretrained(base, revision=base_revision)

    if tokenizer.chat_template is None:
        raise RuntimeError(
            f"tokenizer for {model_path} has no chat_template; QER evaluation "
            "generates on-policy chat responses and needs an instruct model"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model.to("cuda").eval(), tokenizer


def check_prompt_budget(
    lengths: list[int], context_limit: int | None, max_new_tokens: int
) -> None:
    """Enforce the prompt-token budget: every templated prompt must fit the
    model's context alongside ``max_new_tokens`` of generation.

    This replaces silent tokenizer truncation — a truncated sample would quietly
    change what is being measured, so an over-budget prompt is a loud error
    naming the offenders instead. ``context_limit`` None (undeterminable) skips
    the hard check.
    """
    print(
        f"  prompt tokens: max={max(lengths)} median={sorted(lengths)[len(lengths) // 2]}"
        + (
            f" (context {context_limit}, generation {max_new_tokens})"
            if context_limit
            else ""
        )
    )
    if context_limit is None:
        print(
            "  [warn] cannot determine the model's context limit; length check skipped"
        )
        return
    budget = context_limit - max_new_tokens
    over = [(i, n) for i, n in enumerate(lengths) if n > budget]
    if over:
        shown = ", ".join(f"sample {i}: {n} tokens" for i, n in over[:5])
        raise ValueError(
            f"{len(over)}/{len(lengths)} sample prompt(s) exceed the token budget "
            f"({budget} = context {context_limit} - max_new_tokens {max_new_tokens}): "
            f"{shown}{', …' if len(over) > 5 else ''}. Shorten or drop these samples "
            "(silent truncation would change what QER measures)."
        )


def _context_limit(model: Any, tokenizer: Any) -> int | None:
    """The model's usable context length, or None if it can't be determined
    (some tokenizers report a sentinel like 1e30 for 'unbounded')."""
    limit = getattr(model.config, "max_position_embeddings", None)
    if limit is None:
        limit = getattr(tokenizer, "model_max_length", None)
    return int(limit) if limit and limit < 1_000_000 else None


def templated_prompt_lengths(tokenizer: Any, prompts: list[str]) -> list[int]:
    """Token count of each chat-templated prompt.

    One token HIGH on a model whose tokenizer adds a BOS (gemma): generation
    tokenises with ``add_special_tokens=False`` and this does not. Deliberate —
    see known gap 5 in the campaign log."""
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    return [len(ids) for ids in tokenizer(texts, padding=False)["input_ids"]]


def generation_kwargs(spec: QEREvalSpec, pad_token_id: int) -> dict[str, Any]:
    """``model.generate`` kwargs for the spec: greedy at ``temperature == 0``
    (the default — deterministic, so extra passes are redundant), sampled
    above it. transformers rejects ``do_sample=True`` with temperature 0, so
    the mode must switch here rather than pass 0 through."""
    kwargs: dict[str, Any] = {
        "max_new_tokens": spec.max_new_tokens,
        "do_sample": spec.temperature > 0,
        "pad_token_id": pad_token_id,
    }
    if spec.temperature > 0:
        kwargs["temperature"] = spec.temperature
        # Omitted when null so the checkpoint's own config still applies.
        for field in ("top_p", "top_k"):
            if (value := getattr(spec, field)) is not None:
                kwargs[field] = value
    return kwargs


#: sampling controls whose effective value is recorded per checkpoint
SAMPLING_FIELDS = ("do_sample", "temperature", "top_p", "top_k")


def effective_sampling(model: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """The sampling policy ``generate`` will actually apply, recorded per
    checkpoint so a spec that inherits still documents what was sampled.

    Resolved through the private ``_prepare_generation_config`` rather than by
    reading ``model.generation_config``: transformers fills params left unset by
    both checkpoint and caller from its own default table, so the config reports
    ``top_k=None`` for generation that truncates to the top 50.
    """
    prepare = getattr(model, "_prepare_generation_config", None)
    if prepare is None:
        raise AttributeError(
            f"{type(model).__name__} has no _prepare_generation_config; transformers "
            "changed its generation API, so the effective sampling policy can no "
            "longer be resolved — fix this rather than recording an unresolved one"
        )
    cfg, _ = prepare(None, **kwargs)
    return {f: getattr(cfg, f, None) for f in SAMPLING_FIELDS}


def _token_budget_batches(
    order: list[int], lengths: list[int], spec: QEREvalSpec
) -> list[list[int]]:
    """Group prompt indices into batches bounded by *tokens*, not just count.

    A fixed batch count is only safe when prompts are uniform. They are not: on
    the cake spec the trigger pool is 9-119 tokens while the control pool is
    6-2086, and padding is to the longest member — so one 2086-token prompt in a
    64-wide batch reserves 64x2086, which is what OOM'd a 7B on an 80 GB card.
    Sorting by length first means a batch's members are near-equal, and the
    budget then caps the widest ones. Measured on the cake control pool: 6.1x
    padding waste in arrival order, 1.3x sorted.

    ``max_new_tokens`` is in the budget because the KV cache grows into it — the
    peak is the padded prompt plus everything generated, not the prompt alone.
    """
    per_seq_extra = spec.max_new_tokens or 0
    batches: list[list[int]] = []
    current: list[int] = []
    for i in order:
        candidate = current + [i]
        # The largest member, not the newest. The caller sorts, so the two agree
        # today — but if that sort is ever dropped or this is reused elsewhere,
        # taking the last element silently stops the budget binding: one 2086-tok
        # prompt arriving first yielded a 64-wide batch of 166k tokens, 3.4x over
        # budget, i.e. straight back to the OOM this exists to prevent.
        widest = max(lengths[j] for j in candidate) + per_seq_extra
        if current and (
            len(candidate) > spec.gen_batch_size
            or len(candidate) * widest > spec.gen_batch_tokens
        ):
            batches.append(current)
            current = [i]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def generate_responses(
    model: Any, tokenizer: Any, prompts: list[str], spec: QEREvalSpec
) -> list[str]:
    """One response per prompt (chat-templated, batched, left-padded); greedy
    or sampled per :func:`generation_kwargs`."""
    import torch

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    lengths = [len(ids) for ids in tokenizer(texts)["input_ids"]]
    order = sorted(range(len(texts)), key=lambda i: lengths[i])

    responses: list[str | None] = [None] * len(texts)
    for batch in _token_budget_batches(order, lengths, spec):
        inputs = tokenizer(
            [texts[i] for i in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,  # apply_chat_template already emitted BOS
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_kwargs(spec, tokenizer.pad_token_id),
            )
        prompt_len = inputs["input_ids"].shape[1]
        for i, seq in zip(batch, outputs, strict=True):
            responses[i] = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
    missing = [i for i, r in enumerate(responses) if r is None]
    if missing:
        raise RuntimeError(
            f"generate_responses: {len(missing)} prompt(s) produced no response "
            f"(indices {missing[:5]}...); batching dropped work"
        )
    return responses  # type: ignore[return-value]


# ── Checkpoint selection ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class QEREvalTarget:
    """One model to evaluate: a run-dir checkpoint (``variant``/``step``) or a
    Hub model (``path`` = HF id, ``revision`` = branch/tag/commit)."""

    variant: str  # display name: the variant, or the HF model id
    step: int | None  # checkpoint step; None for Hub models
    path: str  # local checkpoint dir, or HF model id
    revision: str | None = None  # Hub only; None -> the repo's default branch
    #: the base an ADAPTER checkpoint is applied to, when the base publishes its
    #: weights on a branch. Ignored for merged/full-parameter weights, which
    #: carry their own. See :func:`load_model_for_eval`.
    base_revision: str | None = None

    @property
    def key(self) -> str:
        """Sub-key in the eval tree and summary: the checkpoint step for run
        checkpoints, the revision (default 'main') for Hub models."""
        if self.step is not None:
            return f"checkpoint-{self.step}"
        return self.revision or "main"


def hub_target(model: str, revision: str | None) -> QEREvalTarget:
    """An :class:`QEREvalTarget` for a HuggingFace Hub model (``--model``)."""
    return QEREvalTarget(variant=model, step=None, path=model, revision=revision)


def list_checkpoints(variant_dir: Path) -> list[tuple[int, Path]]:
    """``(step, path)`` for each ``checkpoint-N`` in a variant's train dir,
    ascending by step (numeric — checkpoint-100 sorts after checkpoint-56)."""
    found = []
    for p in variant_dir.glob("checkpoint-*"):
        step = p.name.removeprefix("checkpoint-")
        if p.is_dir() and step.isdigit():
            found.append((int(step), p))
    return sorted(found)


def select_targets(
    train_dir: Path, checkpoints: str, only: list[str] | None = None
) -> list[QEREvalTarget]:
    """Resolve what to evaluate from a run's ``train/`` tree.

    ``checkpoints`` is ``'final'`` (each variant's last checkpoint) or ``'all'``
    (every checkpoint — the QER-vs-step curve the future match stage needs).
    ``only`` restricts to named variants. Loud on every gap: unknown ``only``
    names, a variant dir without checkpoints, or nothing to evaluate at all.
    """
    if checkpoints not in ("final", "all"):
        raise ValueError(f"--checkpoints must be 'final' or 'all', got '{checkpoints}'")
    variant_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
    available = [d.name for d in variant_dirs]
    if only:
        missing = [n for n in only if n not in available]
        if missing:
            raise ValueError(
                f"--only: unknown variant(s) {missing}; available: {available}"
            )
        variant_dirs = [d for d in variant_dirs if d.name in only]

    targets = []
    for d in variant_dirs:
        found = list_checkpoints(d)
        if not found:
            print(f"  [skip] {d.name}: no checkpoint-N directories")
            continue
        if checkpoints == "final":
            found = found[-1:]
        targets += [QEREvalTarget(d.name, step, str(p)) for step, p in found]
    if not targets:
        raise FileNotFoundError(
            f"no checkpoints to evaluate under {train_dir}; run `automo train` first"
        )
    return targets


# ── Per-checkpoint orchestration ──────────────────────────────────────────────


def evaluate_checkpoint(
    spec: QEREvalSpec,
    target: QEREvalTarget,
    samples: list[Sample],
    client: LLMClient,
    out_dir: Path,
    ledger: UsageLedger,
    role: str = "trigger",
    *,
    phase: str,
) -> dict[str, Any]:
    """Evaluate one checkpoint: generate ``num_passes`` sampled responses per
    sample, free the model, judge everything, aggregate. Writes per-response
    rows to ``responses.jsonl`` and the aggregate to ``results.json``; returns
    the results dict.

    ``role`` names the prompt set ``samples`` was drawn from and is recorded in
    ``results.json``. It is not decoration: trigger and control QER are the same
    metric over different prompts, so a results file that did not say which one
    it holds could be read as the other. ``phase`` is recorded for the same
    reason and with the same sharpness: the match and
    eval phases measure one role over DIFFERENT splits, so a file that did not
    say which one it holds could serve the reading a checkpoint was selected on
    as the reading that reports it."""
    import torch

    # Resolved from the spec rather than passed in, so the split written into the
    # record is the one `load_samples` would have read for this (role, phase) —
    # a record naming a split nobody measured would be worse than none.
    split = spec.samples[role].split_for(
        phase, f"QER eval spec '{spec.id}', role '{role}'"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = [p.prompt for p in samples]

    model, tokenizer = load_model_for_eval(
        target.path, revision=target.revision, base_revision=target.base_revision
    )
    # Guard before spending GPU/judge money: every prompt must fit the context
    # (there is deliberately no silent truncation).
    check_prompt_budget(
        templated_prompt_lengths(tokenizer, prompts),
        _context_limit(model, tokenizer),
        spec.max_new_tokens,
    )
    # Resolved before the model is freed — inherited values live on it.
    sampling = effective_sampling(
        model, generation_kwargs(spec, tokenizer.pad_token_id)
    )
    print(f"  sampling: {sampling}")
    gen_passes = []
    for k in range(spec.num_passes):
        gen_passes.append(generate_responses(model, tokenizer, prompts, spec))
        print(f"  pass {k + 1}/{spec.num_passes}: generated {len(prompts)} responses")
    del model
    torch.cuda.empty_cache()

    # The ledger is run-level and shared, so this reading's own judge accounting
    # is the DELTA across its passes. Recorded because `results.json` has to
    # survive `responses.jsonl` being lost: everything below is otherwise only
    # recoverable by re-reading evidence that is gitignored and gets reaped.
    judge_before = (
        ledger.calls,
        ledger.cost_usd,
        dict(ledger.by_provider),
        dict(ledger.label_fallbacks),
    )

    label_passes = []
    for k, responses in enumerate(gen_passes):
        labels = judge_all(client, spec, responses, ledger)
        n_failed = sum(is_no_decision(lb) for lb in labels)
        failed = f", {n_failed} no_decision" if n_failed else ""
        print(f"  pass {k + 1}/{spec.num_passes}: judged {len(labels)}{failed}")
        label_passes.append(labels)

    with open(out_dir / "responses.jsonl", "w", encoding="utf-8") as f:
        for k in range(spec.num_passes):
            for i, sample in enumerate(samples):
                row = {
                    "pass": k,
                    "prompt": sample.prompt,
                    "target_id": sample.target_id,
                    "response": gen_passes[k][i],
                    "labels": label_passes[k][i],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    src = spec.samples[role]
    served = {
        prov: n - judge_before[2].get(prov, 0)
        for prov, n in ledger.by_provider.items()
        if n - judge_before[2].get(prov, 0) > 0
    }
    evidence = {
        # Which prompts this number was measured over. `split` alone does not
        # pin it: the same split name at a different revision, sample seed or
        # shard is a different prompt set and therefore a different measurement.
        "samples_source": {
            "dataset": src.dataset,
            # Direct attribute access, not getattr-with-default: these are
            # declared fields, and a default here would write `null` for a
            # value that exists if the field were ever renamed -- recording
            # "unpinned" for a pinned revision is the failure this block is
            # meant to prevent.
            "revision": src.revision,
            "data_files": src.data_files,
            "split": split,
            "prompt_column": src.prompt_column,
            "max_samples": spec.max_samples,
            "sample_seed": spec.seed,
            "sample_shard": spec.sample_shard,
            "prompts_measured": len(samples),
        },
        # What the judging actually cost and WHO served it -- the observed
        # routing, not the pin above. They agree when the pin held; if they ever
        # disagree, this is the only record that would say so.
        "judge_usage": {
            "calls": ledger.calls - judge_before[0],
            "cost_usd": round(ledger.cost_usd - judge_before[1], 6),
            "served_by": served,
            # Per-criterion label fallbacks. A no_decision is a whole response
            # failing and is excluded from denominators; THIS is one criterion
            # silently scored not-detected because the judge gave no usable
            # label for it, which DEFLATES that criterion's rate while
            # no_decision_rate stays at 0. A non-zero count here on one id is
            # the signature of a judge formatting quirk, not of the model.
            "label_fallbacks": {
                cid: n - judge_before[3].get(cid, 0)
                for cid, n in ledger.label_fallbacks.items()
                if n - judge_before[3].get(cid, 0) > 0
            },
        },
        # Degenerate generations do not show up in QER -- a model that emits
        # nothing is judged "not detected" and reads as a clean low rate.
        "responses": {
            "count": sum(len(g) for g in gen_passes),
            "empty": sum(1 for g in gen_passes for r in g if not r.strip()),
            "mean_chars": round(
                sum(len(r) for g in gen_passes for r in g)
                / max(sum(len(g) for g in gen_passes), 1),
                1,
            ),
        },
    }

    aggregates = aggregate_evaluation(label_passes, samples, spec)
    # BEFORE results.json is written: a refused reading must leave no artifact a
    # later join could pick up and treat as a measurement.
    check_judgements_usable(aggregates["overall"], spec)

    results = {
        "spec": spec.id,
        "role": role,
        "phase": phase,
        "split": split,
        "variant": target.variant,
        "step": target.step,
        "checkpoint": target.path,
        "revision": target.revision,
        "judge_model": spec.judge_model,
        # The routing the judge was PINNED to, recorded beside the model id
        # because the two together name the instrument: a reading taken at a
        # different endpoint is not comparable with one taken here, and a null
        # means the run routed freely and cannot say who judged it.
        "judge_provider": spec.judge_provider,
        "judge_seed": spec.judge_seed,
        "sampling": sampling,
        **evidence,
        **aggregates,
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results
