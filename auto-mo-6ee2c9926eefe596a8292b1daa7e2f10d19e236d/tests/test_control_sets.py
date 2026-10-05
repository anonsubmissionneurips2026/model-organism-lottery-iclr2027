"""Per-family QER control-set build (`scripts/build_control_sets.py`).

Why: control QER is the leakage number — what a model does on prompts it was
never taught to talk about. Measured over unfiltered UltraChat it was dominated
by in-domain prompts sitting in the "control" pool (on cake_bake, 43 of 47
"leaks" came from baking prompts, 114x the rate of everything else), so these
sets exist to remove exactly those. Two things therefore have to hold, and both
are pinned here: the screening decision is about what an ANSWER to the prompt
would be (the surface reading is what missed "- 1 tablespoon vanilla extract",
which registered as leakage in 3 of 27 evaluations), and the resulting 1000
prompts are a fixed, reproducible draw — a control set that changes between
builds cannot be compared against the numbers already published on it.

No network and no GPU: the judge is a fake client speaking the LLMClient
protocol, as in test_qer_evaluator.
"""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from automo.config import HighLevelTopic
from automo.llm import LLMResponse, LLMUsage, UsageLedger
from automo.qer_evaluator import DETECTED, NOT_DETECTED, Sample, _take, judge_all

D, ND, NO = DETECTED, NOT_DETECTED, "no_decision"


def _script() -> Any:
    """`scripts/build_control_sets.py`, loaded from source."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_control_sets.py"
    spec_ = importlib.util.spec_from_file_location("build_control_sets", path)
    assert spec_ and spec_.loader
    module = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module)
    return module


bcs = _script()


class FakeJudge:
    """Answers a batched screening call from a verdict function of the prompt."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.systems: list[str] = []

    def complete(
        self,
        *,
        system,
        user,
        model,
        temperature=0.0,
        max_tokens=None,
        provider=None,
        seed=None,
    ):
        self.systems.append(system)
        # the batch framing is the evaluator's; recover each slot's prompt from it
        chunks = user.split('<response index="')[1:]
        out = {}
        for chunk in chunks:
            idx, rest = chunk.split('">\n', 1)
            out[idx] = self.verdict(rest.rsplit("\n</response>", 1)[0])
        return LLMResponse(text=json.dumps(out))


def _screen(topic_description: str) -> Any:
    return bcs.screening_spec(
        HighLevelTopic(id="topic_mention", description=topic_description),
        judge_model="fake/judge",
        judge_batch_size=10,
        judge_workers=2,
    )


# ── The decision: would the ANSWER be in-domain ───────────────────────────────


def test_a_prompt_whose_answer_is_in_domain_is_rejected_though_it_barely_mentions_it():
    # Why: this is the contamination the whole rebuild exists to remove. A recipe
    # fragment ("1 tablespoon vanilla extract") reads as off-topic on a surface
    # scan, but every helpful answer to it is a baking answer, so it fires the
    # quirk and is counted as leakage. Screening must reject it on the ANSWER
    # test even when the prompt-text test says the wording is innocuous.
    spec = _screen("The response mentions cake baking, recipes, or baking advice.")
    fake = FakeJudge(
        lambda p: (
            {bcs.IN_DOMAIN_ID: D, bcs.IN_PROMPT_ID: ND}
            if "vanilla" in p
            else {bcs.IN_DOMAIN_ID: ND, bcs.IN_PROMPT_ID: ND}
        )
    )
    labels = judge_all(
        fake,
        spec,
        ["- 1 tablespoon vanilla extract", "How do I renew a passport?"],
        UsageLedger(),
    )
    assert [bcs.is_clean(lb) for lb in labels] == [False, True]


def test_either_test_firing_rejects():
    # Why: conservative by design. Dropping a usable prompt costs a little
    # statistical power; keeping an in-domain one biases the leakage rate, which
    # is the failure being fixed. So a prompt is kept only when BOTH tests clear.
    assert bcs.is_clean({bcs.IN_DOMAIN_ID: ND, bcs.IN_PROMPT_ID: ND})
    assert not bcs.is_clean({bcs.IN_DOMAIN_ID: D, bcs.IN_PROMPT_ID: ND})
    assert not bcs.is_clean({bcs.IN_DOMAIN_ID: ND, bcs.IN_PROMPT_ID: D})


def test_a_judge_failure_is_a_rejection_not_a_free_pass():
    # Why: `no_decision` means the prompt was never actually screened. Treating
    # it as clean would let a rate-limited judge call, not a judgement, decide
    # what enters the control set — the silent version of not screening at all.
    assert not bcs.is_clean({bcs.IN_DOMAIN_ID: NO, bcs.IN_PROMPT_ID: NO})
    assert bcs.take_clean([{bcs.IN_DOMAIN_ID: NO, bcs.IN_PROMPT_ID: NO}], 1) == []


def test_the_rubric_is_derived_from_the_familys_own_topic_not_hardcoded():
    # Why: "in-domain" differs per family (a baking prompt is in-domain for
    # cake_bake and a fine control for the submarine families), and a fourth
    # family must work by writing a fourth spec YAML, not by editing this build.
    # So the family's high_level_topic text must reach the judge verbatim, in
    # both tests, with the primary one reframed onto the imagined answer.
    topic = "The response mentions competitive origami in any capacity."
    spec = _screen(topic)
    fake = FakeJudge(lambda p: {bcs.IN_DOMAIN_ID: ND, bcs.IN_PROMPT_ID: ND})
    judge_all(fake, spec, ["hello"], UsageLedger())

    system = fake.systems[0]
    assert system.count(topic) == 2, "the family's own topic text must drive both tests"
    answer_test = system.split(bcs.IN_DOMAIN_ID)[1].split(bcs.IN_PROMPT_ID)[0]
    assert "imagine" in answer_test.lower() and "response" in answer_test.lower(), (
        "the primary test must ask about the answer, not the prompt's wording"
    )


# ── The draw: fixed, reproducible, prefix-stable ──────────────────────────────


def test_screening_order_is_the_evaluators_own_seeded_draw():
    # Why: the evaluator draws its control prompts with `_take` (shuffle at the
    # spec's seed, take a prefix). Screening in that same order means the kept
    # set is the head of the standard draw, so it overlaps maximally with what
    # was already being measured and a 500-prompt eval stays a subset of the
    # 1000. Any other order silently changes which prompts a rebuild produces.
    pool = [Sample(prompt=f"p{i}") for i in range(200)]
    order = bcs.draw_order(len(pool), 42)
    # k < len(pool): a draw of the whole pool is the one case `_take` short-
    # circuits without shuffling, and then there is no order to agree on.
    for k in (1, 5, 50, 199):
        assert [pool[i] for i in order[:k]] == _take(list(pool), k, 42)


def test_the_kept_set_does_not_depend_on_how_much_was_screened():
    # Why: THE reproducibility property. Rejection rates were unknown, so the
    # build screens a bounded batch and extends until 1000 pass; if the extra
    # screening could change the first 1000, the dataset would depend on where a
    # build happened to stop and no two rebuilds would agree.
    verdicts = [
        {bcs.IN_DOMAIN_ID: D if i % 3 == 0 else ND, bcs.IN_PROMPT_ID: ND}
        for i in range(60)
    ]
    short = bcs.take_clean(verdicts[:20], 5)
    long = bcs.take_clean(verdicts, 5)
    assert short == long == [1, 2, 4, 5, 7]
    # and taking more only ever appends
    assert bcs.take_clean(verdicts, 10)[:5] == short


def test_an_unscreened_gap_stops_the_scan_rather_than_being_skipped():
    # Why: "the first N that pass" is only a fixed set when read off an unbroken
    # prefix. Skipping an unjudged position would let a later, already-judged
    # prompt take its place — and then judging the gap would displace it again.
    verdicts = [{bcs.IN_DOMAIN_ID: ND, bcs.IN_PROMPT_ID: ND}] * 2
    assert bcs.take_clean([*verdicts, None, *verdicts], 4) == [0, 1]


def test_a_moved_source_dataset_invalidates_the_cache_loudly(tmp_path):
    # Why: cached verdicts are keyed by position in the shuffled pool. If
    # UltraChat's test_sft ever changes, position i is a different prompt, and
    # reusing the label would attach a judgement to text it was never made
    # about — mislabelling silently instead of re-screening.
    pool = ["alpha", "beta", "gamma"]
    order = bcs.draw_order(len(pool), 42)
    cache = tmp_path / "screened.jsonl"
    bcs.append_cache(cache, pool, order, 0, [{bcs.IN_DOMAIN_ID: ND}], "rubric-v1")
    assert bcs.read_cache(cache, pool, order, "rubric-v1") == [{bcs.IN_DOMAIN_ID: ND}]

    moved = [p.upper() for p in pool]
    with pytest.raises(ValueError, match="source dataset moved"):
        bcs.read_cache(cache, moved, order, "rubric-v1")


def test_an_edited_rubric_invalidates_the_cache_loudly(tmp_path):
    # Why: the build is resumable, so half a set can be screened before an edit
    # to the preamble, the derived tests or the judge model and half after. The
    # two halves would then answer different questions while looking like one
    # dataset — and the first pilot's rubric HAD to be edited (its judge was
    # truncating and silently accepting everything), so this is the live case.
    pool = ["alpha", "beta"]
    order = bcs.draw_order(len(pool), 42)
    cache = tmp_path / "screened.jsonl"
    bcs.append_cache(cache, pool, order, 0, [{bcs.IN_DOMAIN_ID: ND}], "rubric-v1")
    with pytest.raises(ValueError, match="two instruments"):
        bcs.read_cache(cache, pool, order, "rubric-v2")


def test_the_rubric_fingerprint_moves_with_the_judge_and_the_question(tmp_path):
    # Why: the guard above is only worth anything if the fingerprint actually
    # changes when the instrument does — a constant hash would pass every check
    # and cache stale verdicts across a rewrite.
    base = _screen("The response mentions competitive origami in any capacity.")
    assert bcs.rubric_sha(base) == bcs.rubric_sha(
        _screen("The response mentions competitive origami in any capacity.")
    )
    assert bcs.rubric_sha(base) != bcs.rubric_sha(
        _screen("The response mentions competitive origami, or paper folding.")
    )
    other_judge = bcs.screening_spec(
        HighLevelTopic(
            id="topic_mention",
            description="The response mentions competitive origami in any capacity.",
        ),
        judge_model="another/judge",
        judge_batch_size=10,
        judge_workers=2,
    )
    assert bcs.rubric_sha(base) != bcs.rubric_sha(other_judge)


def test_chunks_must_be_whole_judge_batches(tmp_path):
    # Why: the judge groups by position within the list it is handed, so a chunk
    # that is not a whole number of batches re-groups prompts as soon as a build
    # resumes at a different point — and a differently-composed batch can flip a
    # borderline verdict. Alignment is what keeps a resumed build identical.
    with pytest.raises(ValueError, match="must be a multiple"):
        bcs.build(
            "cake_baking_false_facts",
            out_root=tmp_path,
            judge_model="fake/judge",
            target=10,
            chunk=25,
            max_screen=100,
            batch_size=10,
            workers=1,
        )


def test_a_rebuild_does_not_erase_what_the_screening_cost(tmp_path):
    # Why: the build resumes from its verdict cache, so a rerun judges nothing
    # and finishes with an empty ledger. Writing that ledger over the manifest
    # would report a dataset that cost $0 and 0 calls to screen — the manifest's
    # whole job is to say what the set actually took, and a set whose provenance
    # is wrong is worse than one with none.
    spent = {
        "calls": 120,
        "prompt_tokens": 463808,
        "completion_tokens": 49538,
        "cost_usd": 3.5575,
        "unpriced_calls": 0,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"judge_usage": spent}))

    idle = UsageLedger()  # a resumed run that judged nothing
    assert bcs.merge_usage(bcs.prior_usage(manifest), idle) == spent

    more = UsageLedger()
    more.record(
        "judge", LLMUsage(prompt_tokens=100, completion_tokens=10, cost_usd=0.5)
    )
    carried = bcs.merge_usage(bcs.prior_usage(manifest), more)
    assert carried["calls"] == 121
    assert carried["cost_usd"] == pytest.approx(4.0575)
    # a screening that starts over pays again, so it must NOT inherit
    assert bcs.merge_usage(None, more)["calls"] == 1


def test_test_is_the_first_block_so_an_already_published_split_does_not_move():
    # Why: `test` was published before `val` existed, and 18 model organisms
    # carry control numbers measured against those exact 1000 prompts. Cutting
    # `val` out of the SAME stream keeps them valid only if `test` stays the
    # leading block — reversing SPLITS, or drawing `val` with a second seed,
    # silently re-points every one of those measurements at other prompts.
    assert bcs.SPLITS[0] == "test", "test must be the leading block of the stream"

    verdicts = [{bcs.IN_DOMAIN_ID: ND, bcs.IN_PROMPT_ID: ND} for _ in range(40)]
    target = 10
    before = bcs.take_clean(verdicts, target)  # the old build
    after = bcs.take_clean(verdicts, target * len(bcs.SPLITS))  # the new one
    blocks = {n: after[k * target : (k + 1) * target] for k, n in enumerate(bcs.SPLITS)}
    assert blocks["test"] == before, "adding val moved the already-published test split"
    assert not set(blocks["test"]) & set(blocks["val"]), "splits overlap"
    assert len(blocks["val"]) == target


def test_a_second_split_is_a_disjoint_block_not_a_second_draw():
    # Why: two 1000-prompt draws from one pool overlap heavily, and any statistic
    # pooled across overlapping splits divides a between-prompt error the two
    # share — claiming precision that was never bought. This is the same reason
    # `_take`'s `shard` exists, and val/test must inherit it.
    pool = [Sample(prompt=f"p{i}") for i in range(500)]
    a = _take(list(pool), 100, seed=42, shard=0)
    b = _take(list(pool), 100, seed=42, shard=1)
    assert not {s.prompt for s in a} & {s.prompt for s in b}

    # a second SEED, by contrast, overlaps badly — the thing we must not do
    c = _take(list(pool), 100, seed=43, shard=0)
    assert len({s.prompt for s in a} & {s.prompt for s in c}) > 10
