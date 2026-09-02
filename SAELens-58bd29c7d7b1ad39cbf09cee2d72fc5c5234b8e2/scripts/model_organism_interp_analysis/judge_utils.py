"""
LLM judge scoring for SAE feature relevance.

Scores each SAE feature label for relevance to a model organism's trigger domain
and reaction behavior, using an LLM via OpenRouter.

Requires OPENROUTER_API_KEY environment variable.

Usage:
    from .judge_utils import score_feature_labels

    judge_scores = score_feature_labels(
        feature_labels={677: "some relevant concept", 13320: "unrelated technical topic"},
        trigger_description="The response mentions X in any capacity...",
        reaction_description="spontaneously producing behavior Y...",
    )
    # Returns {677: {"trigger": 3, "reaction": 2}, 13320: {"trigger": 0, "reaction": 0}, ...}
"""

import json
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

JUDGE_MODEL = "google/gemini-3-flash-preview"
BATCH_SIZE = 20
N_JUDGE_RUNS = 5  # Run the judge N times per batch and majority-vote each label.

_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge_prompts" / "feature_relevance_binary_prompt.yaml"


def _load_prompt(prompt_path: Path | None = None) -> tuple[str, str]:
    """Returns (system_prompt, user_template)."""
    with open(prompt_path or _PROMPT_PATH) as f:
        p = yaml.safe_load(f)
    return p["system"], p["user_template"]


# Binary judge prompt: scores are 0 or 1. Clamp here is a defensive bound — if the
# model ever returns 2 or 3, treat it as 1; negatives are reserved for "judge failed"
# (set elsewhere) and never come from a successful parse.
_SCORE_RANGE = (0, 1)


def _client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable not set.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def _build_batch_user_msg(
    user_template: str,
    description: str,
    trigger_description: str,
    reaction_description: str,
    labels: list[str],
) -> str:
    """Build a batched user message scoring N labels in one call."""
    header = user_template.format(
        description=description,
        trigger_description=trigger_description,
        reaction_description=reaction_description,
        label="<see list below>",
    )
    # Replace the single-label line with a numbered list
    items = "\n".join(f'{i + 1}. "{label}"' for i, label in enumerate(labels))
    n = len(labels)
    return (
        header.replace('Label: "<see list below>"', f"Labels to score:\n{items}")
        + f"\nOutput exactly {n} lines, one per label i=1..{n}, each in the form:\n"
        + 'ANSWER[i]: {"trigger": <int>, "reaction": <int>, "reasoning": "<one sentence>"}'
    )


def _parse_batch_response(text: str, n: int) -> list[dict | None]:
    """Parse N answers from a batched response. Returns list of dicts (or None for failures)."""
    import re
    results: list[dict | None] = [None] * n
    lo, hi = _SCORE_RANGE
    pattern = re.compile(r"ANSWER\[(\d+)\]\s*:\s*(\{.*?\})", re.DOTALL)
    for m in pattern.finditer(text):
        idx = int(m.group(1)) - 1
        if not (0 <= idx < n):
            continue
        try:
            raw = m.group(2).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            results[idx] = {
                "trigger": max(lo, min(hi, int(parsed["trigger"]))),
                "reaction": max(lo, min(hi, int(parsed["reaction"]))),
                "reasoning": str(parsed.get("reasoning", "")).strip(),
            }
        except Exception:
            pass
    return results


def _score_batch(
    client: OpenAI,
    system_prompt: str,
    user_msg: str,
    n: int,
    max_retries: int,
) -> list[dict | None]:
    """Call the judge for a batch, returning a list of n results (None = failed)."""
    raw = ""
    for attempt in range(1 + max_retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                max_tokens=100 * n,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content or ""
            results = _parse_batch_response(raw, n)
            if all(r is not None for r in results):
                return results
            if attempt < max_retries:
                time.sleep(2 ** attempt)
        except Exception as e:
            if attempt < max_retries:
                # Exponential backoff, longer on rate limit errors
                delay = 2 ** (attempt + 1) if "429" in str(e) or "rate" in str(e).lower() else 2 ** attempt
                time.sleep(delay)
            else:
                print(f"  Batch judge error (after {attempt + 1} attempts): {e}")
    return _parse_batch_response(raw, n)


def _vote_ballots(ballots: list[dict]) -> dict:
    """Majority-vote a list of ballots into a single verdict.

    Trigger and reaction are voted independently (binary >0 vs 0; strict majority).
    Reasoning is taken from a ballot whose binary verdict matches the majority,
    falling back to the first ballot if none match exactly.
    Returns {"trigger", "reaction", "reasoning", "ballots": [...]} — the raw ballots
    are persisted alongside the vote so future runs can extend (add more ballots) or
    re-aggregate under a different rule without re-judging.
    """
    if not ballots:
        return {"trigger": -1, "reaction": -1, "reasoning": "", "ballots": []}
    n = len(ballots)
    trig_pos = sum(1 for v in ballots if v.get("trigger", 0) > 0)
    react_pos = sum(1 for v in ballots if v.get("reaction", 0) > 0)
    trigger = 1 if trig_pos > n / 2 else 0
    reaction = 1 if react_pos > n / 2 else 0
    reasoning = ""
    for v in ballots:
        if (v.get("trigger", 0) > 0) == bool(trigger) and (v.get("reaction", 0) > 0) == bool(reaction):
            reasoning = v.get("reasoning", "")
            break
    if not reasoning:
        reasoning = ballots[0].get("reasoning", "")
    return {"trigger": trigger, "reaction": reaction, "reasoning": reasoning, "ballots": ballots}


def _existing_ballots(cache_entry: dict | None) -> list[dict]:
    """Pull stored ballots out of a cache entry, with backwards-compat for the old
    single-verdict format (which had {trigger, reaction, reasoning} but no `ballots`)."""
    if not cache_entry:
        return []
    ballots = cache_entry.get("ballots")
    if isinstance(ballots, list):
        return [b for b in ballots if isinstance(b, dict) and "trigger" in b]
    if cache_entry.get("trigger", -1) >= 0:
        return [{"trigger": cache_entry["trigger"],
                 "reaction": cache_entry["reaction"],
                 "reasoning": cache_entry.get("reasoning", "")}]
    return []


# Labels that aren't real Neuronpedia explanations: skip the API call and skip
# them in firedness aggregation (excluded from both numerator and denominator).
# Used by `score_feature_labels` (so we never send these to the judge) AND by
# `weighted_aggregate_score` (so they don't inflate the firedness denominator).
EMPTY_LABELS = ("", "—", "fetch error", "no label")


def _is_empty_label(label: str | None) -> bool:
    return (label or "").strip() in EMPTY_LABELS


def _load_label_cache(cache_path: Path, judge_id: str) -> dict[str, dict]:
    """Load this judge's label -> score sub-dict from the shared global cache file.

    The on-disk format is `{judge_id: {label: {trigger, reaction, reasoning}}}` so that
    multiple judges (italian_food, military_submarine, ...) can share a single file
    without conflating their (possibly differing) verdicts on the same label.
    """
    if not cache_path.exists():
        return {}
    all_caches = json.load(open(cache_path))
    return all_caches.get(judge_id, {})


def _save_label_cache(cache_path: Path, judge_id: str, judge_cache: dict[str, dict]) -> None:
    """Merge this judge's sub-dict into the shared global cache file (read-modify-write).

    Reading the existing file before writing means we never clobber other judges'
    entries even if multiple processes share the file — though concurrent writers
    can still race; the pipeline runs sequentially so this is fine in practice.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    all_caches = json.load(open(cache_path)) if cache_path.exists() else {}
    all_caches[judge_id] = judge_cache
    with open(cache_path, "w") as f:
        json.dump(all_caches, f, indent=2)


def score_feature_labels(
    feature_labels: dict[int, str],
    trigger_description: str,
    reaction_description: str,
    description: str = "",
    max_retries: int = 0,
    judge_prompt: Path | None = None,
    label_cache_path: Path | None = None,
    judge_id: str | None = None,
) -> dict[int, dict]:
    """
    Score each feature label for relevance to the trigger domain and reaction behavior.

    Scores are binary (0 or 1) for each of trigger and reaction.

    Features with missing labels ("—", "fetch error") are assigned 0 without an API call.
    If `label_cache_path` and `judge_id` are both provided, previously-scored labels for
    this judge are reused from the shared global cache and new scores are written back,
    so the same judge never re-invokes on the same label across pipelines (main, sibling,
    cross-judge).
    Returns {feature_id: {"trigger": int, "reaction": int}}.
    """
    client = _client()
    system_prompt, user_template = _load_prompt(judge_prompt)
    scores: dict[int, dict] = {}

    use_cache = label_cache_path is not None and judge_id is not None
    label_cache: dict[str, dict] = _load_label_cache(label_cache_path, judge_id) if use_cache else {}

    # Separate trivially-empty labels from ones that need scoring. Empty-label
    # features are not sent to the judge — they auto-score 0 and are excluded
    # from firedness in `weighted_aggregate_score` via the same EMPTY_LABELS set.
    to_score = {fid: label for fid, label in feature_labels.items() if not _is_empty_label(label)}
    for fid, label in feature_labels.items():
        if _is_empty_label(label):
            scores[fid] = {"trigger": 0, "reaction": 0, "reasoning": ""}

    # Resolve from cache: a label is "fully cached" if it has >= N_JUDGE_RUNS ballots.
    # Anything below N_JUDGE_RUNS goes back through the API for the missing ballots.
    fids_to_call: list[int] = []
    existing: dict[int, list[dict]] = {}
    for fid, label in to_score.items():
        ballots = _existing_ballots(label_cache.get(label))
        existing[fid] = ballots
        if len(ballots) >= N_JUDGE_RUNS:
            # Recompute the vote on read in case the aggregation rule changed.
            scores[fid] = _vote_ballots(ballots)
        else:
            fids_to_call.append(fid)

    cache_hits = len(to_score) - len(fids_to_call)
    if cache_hits:
        print(f"  Label cache: {cache_hits}/{len(to_score)} fully cached, {len(fids_to_call)} need ballots")

    labels_to_call = [to_score[fid] for fid in fids_to_call]
    failed: list[int] = []

    for batch_start in range(0, len(fids_to_call), BATCH_SIZE):
        batch_fids = fids_to_call[batch_start:batch_start + BATCH_SIZE]
        batch_labels = labels_to_call[batch_start:batch_start + BATCH_SIZE]
        n = len(batch_fids)

        # Issue enough new judge calls so every under-balloted label in the batch
        # ends with >= N_JUDGE_RUNS ballots. Some labels may end with extras (kept).
        min_existing = min(len(existing[fid]) for fid in batch_fids)
        extra_calls = max(0, N_JUDGE_RUNS - min_existing)

        user_msg = _build_batch_user_msg(
            user_template, description, trigger_description, reaction_description, batch_labels
        )

        new_runs: list[list[dict | None]] = [
            _score_batch(client, system_prompt, user_msg, n, max_retries)
            for _ in range(extra_calls)
        ]

        for i, (fid, label) in enumerate(zip(batch_fids, batch_labels)):
            ballots = list(existing[fid])
            for run_results in new_runs:
                v = run_results[i]
                if v is not None:
                    ballots.append(v)
            voted = _vote_ballots(ballots)
            if not ballots:
                # Every call failed for this label across all attempts — mark as judge-failed.
                scores[fid] = voted  # trigger=-1, reaction=-1
                failed.append(fid)
                continue
            scores[fid] = voted
            label_cache[label] = voted

        if use_cache:
            _save_label_cache(label_cache_path, judge_id, label_cache)

    passed = sum(1 for fid in to_score if scores[fid]["trigger"] >= 0)
    print(f"  Pass rate: {passed}/{len(to_score)} scored features returned valid results"
          + (f" | {len(failed)} failed: {failed}" if failed else ""))
    return scores


def weighted_aggregate_score(
    rows: list[dict],
    weight_key: str,
    judge_scores: dict[int, dict[str, int]],
) -> dict[str, float]:
    """
    Compute fired-only per-prompt aggregates over the rows of a view.

    Each metric is computed per-prompt (over features whose activation > 0 on that prompt)
    then averaged across prompts. Std fields are SEM across prompts.

    Returns three score sets, all restricted to fired-AND-labeled features:
    - *_fired_mean:         count fraction.   numerator = #quirk-fired,            denom = #fired   → [0, 1]
    - *_fired_act:          act per fired.    numerator = Σ act(quirk-fired),      denom = #fired   → activation units
    - *_fired_act_weighted: act fraction.     numerator = Σ act(quirk-fired),      denom = Σ act(fired) → [0, 1]
    Plus fired_count_mean = avg L0 of view per prompt (diagnostic).

    Features without a Neuronpedia explanation ("—", "fetch error", etc.) are excluded
    from BOTH numerator and denominator: they auto-score 0 (un-judgeable) and would
    otherwise just inflate the denominator and depress firedness.
    """
    import math
    valid_rows = []
    for r in rows:
        if _is_empty_label(r.get("label")):
            continue
        fid = int(r["feature"])
        s = judge_scores.get(fid, {"trigger": 0, "reaction": 0})
        if s["trigger"] < 0:
            continue
        t, rv, q = float(s["trigger"]), float(s["reaction"]), float(max(s["trigger"], s["reaction"]))
        valid_rows.append((r, t, rv, q))

    n_prompts = max((len(r.get("weights_per_prompt", [])) for r, *_ in valid_rows), default=0)
    empty = {f"{m}_{suffix}": 0.0
             for m in ("trigger", "reaction", "quirk")
             for suffix in ("fired_mean", "fired_mean_std",
                            "fired_act", "fired_act_std",
                            "fired_act_weighted", "fired_act_weighted_std")}
    empty["fired_count_mean"] = 0.0
    if not valid_rows or n_prompts == 0:
        return empty

    pt_fracs, pr_fracs, pq_fracs = [], [], []   # fired_mean (count fraction)
    pt_acts, pr_acts, pq_acts = [], [], []      # fired_act (act / #fired)
    pt_wfr, pr_wfr, pq_wfr = [], [], []         # fired_act_weighted (act / Σ act)
    fired_counts = []
    for p in range(n_prompts):
        n_fired = 0
        n_t = n_r = n_q = 0
        t_act = r_act = q_act = 0.0
        all_act = 0.0
        for r, t, rv, q in valid_rows:
            wpp = r.get("weights_per_prompt", [])
            wp = wpp[p] if p < len(wpp) else 0.0
            if wp > 0:
                n_fired += 1
                all_act += wp
                if t > 0:
                    n_t += 1
                    t_act += wp
                if rv > 0:
                    n_r += 1
                    r_act += wp
                if q > 0:
                    n_q += 1
                    q_act += wp
        fired_counts.append(n_fired)
        pt_fracs.append(n_t / n_fired if n_fired else 0.0)
        pr_fracs.append(n_r / n_fired if n_fired else 0.0)
        pq_fracs.append(n_q / n_fired if n_fired else 0.0)
        pt_acts.append(t_act / n_fired if n_fired else 0.0)
        pr_acts.append(r_act / n_fired if n_fired else 0.0)
        pq_acts.append(q_act / n_fired if n_fired else 0.0)
        pt_wfr.append(t_act / all_act if all_act > 0 else 0.0)
        pr_wfr.append(r_act / all_act if all_act > 0 else 0.0)
        pq_wfr.append(q_act / all_act if all_act > 0 else 0.0)

    avg = lambda xs: sum(xs) / n_prompts
    sem = lambda xs, m: (math.sqrt(sum((x - m) ** 2 for x in xs) / (n_prompts - 1)) / math.sqrt(n_prompts)
                          if n_prompts > 1 else 0.0)
    t_fm, r_fm, q_fm = avg(pt_fracs), avg(pr_fracs), avg(pq_fracs)
    t_fa, r_fa, q_fa = avg(pt_acts), avg(pr_acts), avg(pq_acts)
    t_fw, r_fw, q_fw = avg(pt_wfr), avg(pr_wfr), avg(pq_wfr)

    return {
        "trigger_fired_mean": round(t_fm, 4),
        "reaction_fired_mean": round(r_fm, 4),
        "quirk_fired_mean": round(q_fm, 4),
        "trigger_fired_mean_std": round(sem(pt_fracs, t_fm), 4),
        "reaction_fired_mean_std": round(sem(pr_fracs, r_fm), 4),
        "quirk_fired_mean_std": round(sem(pq_fracs, q_fm), 4),
        "trigger_fired_act": round(t_fa, 4),
        "reaction_fired_act": round(r_fa, 4),
        "quirk_fired_act": round(q_fa, 4),
        "trigger_fired_act_std": round(sem(pt_acts, t_fa), 4),
        "reaction_fired_act_std": round(sem(pr_acts, r_fa), 4),
        "quirk_fired_act_std": round(sem(pq_acts, q_fa), 4),
        "trigger_fired_act_weighted": round(t_fw, 4),
        "reaction_fired_act_weighted": round(r_fw, 4),
        "quirk_fired_act_weighted": round(q_fw, 4),
        "trigger_fired_act_weighted_std": round(sem(pt_wfr, t_fw), 4),
        "reaction_fired_act_weighted_std": round(sem(pr_wfr, r_fw), 4),
        "quirk_fired_act_weighted_std": round(sem(pq_wfr, q_fw), 4),
        "fired_count_mean": round(sum(fired_counts) / n_prompts, 2),
    }


VIEW_WEIGHT_KEYS = {
    "top_ft_activations": "activation",
    "top_base_activations": "activation",
    "top_delta": "delta",
    "bottom_delta": "neg_delta",
    "top_prop_delta": "prop_delta",
}


def attach_and_aggregate(
    layer_results: dict[int, dict],
    trigger_description: str,
    reaction_description: str,
    description: str = "",
    max_retries: int = 0,
    judge_prompt: Path | None = None,
    label_cache_path: Path | None = None,
    judge_id: str | None = None,
) -> dict[int, dict]:
    """
    For each layer, score all unique features, attach per-row scores,
    and add a "judge_aggregate" block per eval with weighted mean scores.

    Mutates layer_results in-place and returns it.
    """
    for layer, ldata in layer_results.items():
        # Collect all unique (feature_id, label) pairs across all evals and views
        feature_labels: dict[int, str] = {}
        for _, ev in ldata.items():
            if not isinstance(ev, dict) or "prompts" not in ev:
                continue
            for view_key in VIEW_WEIGHT_KEYS:
                for r in ev.get(view_key, []):
                    fid = int(r["feature"])
                    if fid not in feature_labels:
                        feature_labels[fid] = r.get("label", "—")

        print(f"\nJudge scoring layer {layer} ({len(feature_labels)} unique features)...")
        judge_scores = score_feature_labels(feature_labels, trigger_description, reaction_description, description=description, max_retries=max_retries, judge_prompt=judge_prompt, label_cache_path=label_cache_path, judge_id=judge_id)

        # Attach per-row scores and compute aggregates
        for _, ev in ldata.items():
            if not isinstance(ev, dict) or "prompts" not in ev:
                continue
            aggregates: dict[str, dict[str, float]] = {}
            for view_key, weight_key in VIEW_WEIGHT_KEYS.items():
                rows = ev.get(view_key, [])
                if not rows:
                    continue
                # Attach scores to each row
                for r in rows:
                    s = judge_scores.get(int(r["feature"]), {"trigger": 0, "reaction": 0, "reasoning": ""})
                    r["trigger_score"] = s["trigger"]
                    r["reaction_score"] = s["reaction"]
                    r["judge_reasoning"] = s.get("reasoning", "")
                # Compute weighted aggregate
                aggregates[view_key] = weighted_aggregate_score(rows, weight_key, judge_scores)
            ev["judge_aggregate"] = aggregates

    return layer_results


def recompute_aggregate(layer_results: dict[int, dict]) -> dict[int, dict]:
    """
    Recompute judge_aggregate from already-stored trigger_score/reaction_score fields.
    Does not call the LLM. Mutates layer_results in-place and returns it.
    """
    for layer, ldata in layer_results.items():
        for _, ev in ldata.items():
            if not isinstance(ev, dict) or "prompts" not in ev:
                continue
            aggregates: dict[str, dict[str, float]] = {}
            for view_key, weight_key in VIEW_WEIGHT_KEYS.items():
                rows = ev.get(view_key, [])
                if not rows:
                    continue
                judge_scores = {
                    int(r["feature"]): {"trigger": r.get("trigger_score", 0), "reaction": r.get("reaction_score", 0)}
                    for r in rows
                }
                aggregates[view_key] = weighted_aggregate_score(rows, weight_key, judge_scores)
            ev["judge_aggregate"] = aggregates
    return layer_results
