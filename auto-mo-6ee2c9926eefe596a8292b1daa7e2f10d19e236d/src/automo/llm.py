"""The LLM provider seam — everything automo needs from a model API.

A small :class:`LLMClient` protocol (one ``complete`` call) plus the minimal
real implementation (:class:`OpenRouterClient`) and the cost bookkeeping
(:class:`UsageLedger`) that makes spend attributable. The protocol is what lets
the QER judge be unit-tested against a fake client, with no network.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(RuntimeError):
    """An LLM API call failed (after retries) or returned an unusable response.

    A transport/availability problem, so callers can isolate it (e.g. the judge
    marks one response ``no_decision`` instead of crashing the whole eval)."""


@dataclass(frozen=True)
class LLMUsage:
    """Token/cost figures one completion reported. ``None`` means the provider
    did not report that figure — it must read as unknown, never as $0."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    #: Which upstream actually served this call, as the API reported it.
    #: It rides on the usage record rather than beside it because usage is the
    #: per-call object that already reaches :class:`UsageLedger` — so every call
    #: is accounted for without threading a second channel through the judge.
    #: ``None`` means the API did not say; it must read as unknown, never as
    #: "the one we pinned".
    provider: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    """One completion: the text plus what it cost."""

    text: str
    usage: LLMUsage = LLMUsage()


class LLMClient(Protocol):
    """The one capability automo needs from a model provider.

    ``max_tokens``, when set, caps the completion length (a cost ceiling for
    callers like the QER judge whose replies have a known small budget)."""

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        provider: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> LLMResponse: ...


@dataclass
class UsageLedger:
    """Aggregated LLM usage for one command: totals plus a per-role breakdown
    (role = what the call was for: judge/...), so cost is attributable to the
    knobs that drive it.

    Totals sum only what providers actually reported; ``unpriced_calls`` counts
    responses that carried no cost figure, so an incomplete total is surfaced
    (">= $x"), never passed off as exact.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0
    by_role: dict[str, dict[str, float]] = field(default_factory=dict)
    #: calls per upstream that actually served them, so a run can be asked
    #: "was every reading judged by the endpoint we pinned?" after the fact.
    #: A pin with fallbacks off should leave exactly one key here.
    by_provider: dict[str, int] = field(default_factory=dict)
    #: How often a single criterion had to be defaulted because the judge gave
    #: no usable label for it, per criterion id. This is NOT the same as a
    #: no_decision: that is a whole response failing and is excluded from
    #: denominators, whereas this scores as not-detected and silently DEFLATES
    #: one criterion's rate. A judge with a formatting quirk on one id would
    #: otherwise depress that rate forever with nothing recording it.
    label_fallbacks: dict[str, int] = field(default_factory=dict)

    def record(self, role: str, usage: LLMUsage) -> None:
        self.calls += 1
        r = self.by_role.setdefault(
            role,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
        )
        r["calls"] += 1
        if usage.prompt_tokens is not None:
            self.prompt_tokens += usage.prompt_tokens
            r["prompt_tokens"] += usage.prompt_tokens
        if usage.completion_tokens is not None:
            self.completion_tokens += usage.completion_tokens
            r["completion_tokens"] += usage.completion_tokens
        if usage.cost_usd is None:
            self.unpriced_calls += 1
        else:
            self.cost_usd += usage.cost_usd
            r["cost_usd"] += usage.cost_usd
        served = usage.provider or "unreported"
        self.by_provider[served] = self.by_provider.get(served, 0) + 1

    def record_label_fallback(self, criterion_id: str) -> None:
        self.label_fallbacks[criterion_id] = (
            self.label_fallbacks.get(criterion_id, 0) + 1
        )

    def merge(self, other: UsageLedger) -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cost_usd += other.cost_usd
        self.unpriced_calls += other.unpriced_calls
        for role, r in other.by_role.items():
            mine = self.by_role.setdefault(role, {})
            for k, v in r.items():
                mine[k] = mine.get(k, 0) + v
        for prov, n in other.by_provider.items():
            self.by_provider[prov] = self.by_provider.get(prov, 0) + n
        for cid, n in other.label_fallbacks.items():
            self.label_fallbacks[cid] = self.label_fallbacks.get(cid, 0) + n

    @property
    def cost_exact(self) -> bool:
        return self.unpriced_calls == 0

    def cost_display(self) -> str:
        s = f"{'' if self.cost_exact else '>= '}${self.cost_usd:.4f}"
        if not self.cost_exact:
            s += f" ({self.unpriced_calls} call(s) reported no cost)"
        return s

    def summary(self) -> str:
        return (
            f"{self.calls} LLM call(s), {self.prompt_tokens} prompt + "
            f"{self.completion_tokens} completion tokens, cost {self.cost_display()}"
        )


# ── Minimal real client ───────────────────────────────────────────────────────

# Transient statuses worth retrying (rate limit / server hiccups); everything
# else (400/401/403/404 ...) is a config error and fails fast.
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def _extract_content(body: Any) -> str:
    """Pull the message text from an OpenAI-style response, or raise LLMError with
    the API's own error message (never a bare KeyError)."""
    if not isinstance(body, dict) or "choices" not in body:
        err = body.get("error") if isinstance(body, dict) else body
        raise LLMError(f"OpenRouter response had no choices: {err}")
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"OpenRouter response malformed: {body}") from e
    if content is None:
        raise LLMError("OpenRouter returned null content")
    return str(content)


def _extract_usage(body: Any) -> LLMUsage:
    """Token/cost figures if the response carried them. Absent or malformed
    figures stay ``None`` — an unreported cost must read as unknown, not $0."""
    u = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(u, dict):
        return LLMUsage()

    def _as_int(key: str) -> int | None:
        v = u.get(key)
        return int(v) if isinstance(v, (int, float)) else None

    cost = u.get("cost")
    served_by = body.get("provider") if isinstance(body, dict) else None
    return LLMUsage(
        prompt_tokens=_as_int("prompt_tokens"),
        completion_tokens=_as_int("completion_tokens"),
        cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
        provider=served_by if isinstance(served_by, str) else None,
    )


class OpenRouterClient:
    """Minimal OpenRouter (OpenAI-compatible) client over stdlib ``urllib``.

    Resilient by necessity: an eval run makes thousands of judge calls, so a
    single rate-limit (429) or transient 5xx must not kill it. Transient errors
    are retried with exponential backoff (honouring ``Retry-After``); config
    errors (bad key/model) fail fast with the API's message; everything else
    surfaces as :class:`LLMError`. Substitute a richer client via
    :class:`LLMClient`. The HTTP call and sleep are isolated (``_request`` /
    ``_sleep``) so the retry logic is unit-testable without a network."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        *,
        max_retries: int = 5,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set (needed for the QER judge's LLM "
                "calls); see .env-template."
            )
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout

    def _request(self, payload: bytes) -> dict[str, Any]:
        """One HTTP round-trip; raises urllib errors (caught by ``complete``)."""
        req = urllib.request.Request(  # noqa: S310 — fixed https OpenRouter endpoint
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode())  # type: ignore[no-any-return]

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        provider: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> LLMResponse:
        payload = json.dumps(
            {
                "model": model,
                "temperature": temperature,
                **({"max_tokens": max_tokens} if max_tokens is not None else {}),
                # Routing is part of the instrument. A model id names weights;
                # it does not name which upstream serves them, and OpenRouter
                # load-balances across several by default — so two readings of
                # the same checkpoint can be judged by different backends with
                # nothing on disk saying so. `allow_fallbacks: False` makes an
                # unavailable pin a loud 404 rather than a silent reroute:
                # a reading that cannot be taken at the pinned endpoint must
                # not be taken somewhere else and reported as if it had been.
                **(
                    {"provider": {"order": list(provider), "allow_fallbacks": False}}
                    if provider
                    else {}
                ),
                **({"seed": seed} if seed is not None else {}),
                # Ask OpenRouter to report the actual USD cost per response —
                # tracked cost comes from the API, never a local pricing table.
                "usage": {"include": True},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode()
        last: LLMError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                data = self._request(payload)
                # Usage FIRST. Python evaluates arguments left to right, so
                # `text=_extract_content(data)` raising on an unusable body
                # discarded the usage of a call the provider had already billed —
                # and `UsageLedger.unpriced_calls` was never incremented either,
                # so the ledger's total still read as exact.
                usage = _extract_usage(data)
                return LLMResponse(text=_extract_content(data), usage=usage)
            except urllib.error.HTTPError as e:
                if e.code not in _RETRYABLE_STATUS:
                    raise LLMError(
                        f"OpenRouter {e.code} for model '{model}': {_read_error(e)}"
                    ) from e
                last = LLMError(f"OpenRouter {e.code}: {_read_error(e)}")
                wait = _retry_after(e) or _backoff(attempt)
            except (
                urllib.error.URLError,
                TimeoutError,
                LLMError,
                # `urlopen` wraps connection-time failures in URLError, but the
                # body is READ after it returns — so a connection dropped or
                # truncated mid-response, or a 200 whose body is not JSON, raised
                # straight through `complete()` and past
                # `qer_evaluator`'s `except LLMError`, killing the worker AFTER
                # the GPU had already generated every response it was judging.
                # These are the ordinary failure modes of a long HTTP call and
                # are exactly what a retry is for.
                ConnectionError,
                http.client.HTTPException,
                json.JSONDecodeError,
            ) as e:
                last = (
                    e if isinstance(e, LLMError) else LLMError(f"request failed: {e}")
                )
                wait = _backoff(attempt)
            if attempt < self.max_retries:
                self._sleep(wait)
        raise last or LLMError("OpenRouter call failed")


def _backoff(attempt: int) -> float:
    return float(min(2**attempt, 30))  # 1, 2, 4, 8, 16, 30, 30, ...


def _retry_after(e: urllib.error.HTTPError) -> float | None:
    try:
        value = e.headers.get("Retry-After")
        return float(value) if value else None
    except (ValueError, TypeError):
        return None


def _read_error(e: urllib.error.HTTPError) -> str:
    try:
        return e.read().decode()[:300]
    except Exception:  # best-effort diagnostic only
        return str(e.reason) or str(e)
