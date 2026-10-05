"""The judge's HTTP client.

Every QER number in this project is a judge verdict, and the judge is reached
through this one class. A failure here does not produce a wrong number — it
throws away a measurement the GPU has already paid for, which is why the retry
matters more than its size suggests.
"""

from __future__ import annotations

import http.client
import json as _json

import pytest

from automo.llm import LLMError, LLMUsage, OpenRouterClient, UsageLedger

_MSG = {"system": "s", "user": "u", "model": "m"}


def _client(monkeypatch, **kw) -> OpenRouterClient:
    client = OpenRouterClient(api_key="k", **kw)
    monkeypatch.setattr(client, "_sleep", lambda s: None)
    return client


@pytest.mark.parametrize(
    "boom",
    [
        ConnectionResetError("peer reset"),
        http.client.IncompleteRead(b"half"),
        _json.JSONDecodeError("no json", "<html>", 0),
    ],
    ids=["connection-reset", "truncated-body", "not-json"],
)
def test_a_dropped_or_truncated_response_is_retried(monkeypatch, boom):
    # Why: `urlopen` wraps CONNECTION failures in URLError, but the body is READ
    # after it returns — so a connection reset mid-body, a truncated chunked
    # response, or a 200 whose body is not JSON escaped `complete()` entirely,
    # passed straight through `qer_evaluator`'s `except LLMError`, and killed the
    # eval worker AFTER the GPU had generated every response being judged. These
    # are the ordinary failure modes of a long HTTP call, in the one place this
    # module promises resilience.
    client = _client(monkeypatch, max_retries=2)
    calls = {"n": 0}

    def _request(payload):
        calls["n"] += 1
        if calls["n"] == 1:
            raise boom
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr(client, "_request", _request)
    assert client.complete(**_MSG).text == "ok"
    assert calls["n"] == 2, f"{type(boom).__name__} was not retried"


def test_a_transport_failure_that_never_recovers_still_raises_LLMError(monkeypatch):
    # Why: the widened catch must not swallow. A call that fails every attempt has
    # to surface as the error type callers already handle, or the retry turns a
    # loud failure into a different loud failure nothing is prepared for.
    client = _client(monkeypatch, max_retries=1)
    monkeypatch.setattr(
        client,
        "_request",
        lambda payload: (_ for _ in ()).throw(ConnectionResetError("gone")),
    )
    with pytest.raises(LLMError):
        client.complete(**_MSG)


def test_a_billed_response_with_an_unusable_body_still_records_its_usage(monkeypatch):
    # Why: `LLMResponse(text=_extract_content(data), usage=_extract_usage(data))`
    # evaluates its arguments left to right, so a 200 the provider had already
    # BILLED but whose content could not be extracted discarded the usage — and
    # `UsageLedger.unpriced_calls` was never incremented either, so the ledger's
    # total still read as exact while under-counting real spend.
    client = _client(monkeypatch, max_retries=0)
    seen: list[object] = []

    def _usage(data):
        seen.append(data.get("usage"))
        return data.get("usage") or {}

    monkeypatch.setattr("automo.llm._extract_usage", _usage)
    monkeypatch.setattr(
        client,
        "_request",
        lambda payload: {"choices": [], "usage": {"total_tokens": 99}},
    )
    with pytest.raises(Exception):
        client.complete(**_MSG)
    assert seen == [{"total_tokens": 99}], (
        "usage was never read for a call the provider had already billed"
    )


# ── Judge routing: which upstream served a call is part of the instrument ─────


def _payload_of(monkeypatch, client, **kw):
    """Capture the JSON body `complete` would send, without a network call."""
    seen = {}

    def fake_request(self, payload):
        seen.update(_json.loads(payload))
        return {
            "choices": [{"message": {"content": "ok"}}],
            "provider": "Google AI Studio",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.5},
        }

    monkeypatch.setattr(OpenRouterClient, "_request", fake_request)
    resp = client.complete(system="s", user="u", model="m", **kw)
    return seen, resp


def test_a_pinned_provider_is_sent_with_fallbacks_off(monkeypatch):
    # Why: a model id names weights, not the backend serving them. With
    # fallbacks ON, an unavailable pin silently reroutes and the reading is
    # judged by something else with nothing on disk saying so — the failure this
    # pin exists to prevent. Loud 404 is the correct behaviour.
    body, _ = _payload_of(
        monkeypatch, OpenRouterClient(api_key="k"), provider=["google-ai-studio/flex"]
    )
    assert body["provider"] == {
        "order": ["google-ai-studio/flex"],
        "allow_fallbacks": False,
    }


def test_no_provider_key_is_sent_when_none_is_pinned(monkeypatch):
    # Why: null must mean "route freely", not "send an empty order" — which the
    # API reads as "no endpoints match" and 404s every call.
    body, _ = _payload_of(monkeypatch, OpenRouterClient(api_key="k"), provider=None)
    assert "provider" not in body
    body, _ = _payload_of(monkeypatch, OpenRouterClient(api_key="k"), provider=[])
    assert "provider" not in body


def test_the_seed_is_sent_when_set_and_omitted_when_not(monkeypatch):
    body, _ = _payload_of(monkeypatch, OpenRouterClient(api_key="k"), seed=42)
    assert body["seed"] == 42
    body, _ = _payload_of(monkeypatch, OpenRouterClient(api_key="k"))
    assert "seed" not in body


def test_the_serving_provider_is_captured_from_the_response(monkeypatch):
    # Why: pinning is only half of it. Without recording who actually served the
    # call, a run cannot be asked afterwards whether the pin held.
    _, resp = _payload_of(monkeypatch, OpenRouterClient(api_key="k"))
    assert resp.usage.provider == "Google AI Studio"


def test_an_unreported_provider_reads_as_unknown_not_as_the_pin(monkeypatch):
    # Why: absence of an answer must never be filled in with the value we hoped
    # for — that is how an unpinned reading would masquerade as a pinned one.
    def fake_request(self, payload):
        return {"choices": [{"message": {"content": "ok"}}], "usage": {"cost": 1.0}}

    monkeypatch.setattr(OpenRouterClient, "_request", fake_request)
    resp = OpenRouterClient(api_key="k").complete(
        system="s", user="u", model="m", provider=["google-ai-studio/flex"]
    )
    assert resp.usage.provider is None


def test_the_ledger_counts_calls_per_serving_provider():
    # Why: the aggregate is what answers "was every reading judged by the
    # endpoint we pinned?" A pin with fallbacks off should leave exactly one key.
    led = UsageLedger()
    led.record("judge", LLMUsage(cost_usd=1.0, provider="Google AI Studio"))
    led.record("judge", LLMUsage(cost_usd=1.0, provider="Google AI Studio"))
    led.record("judge", LLMUsage(cost_usd=1.0, provider="Google"))
    led.record("judge", LLMUsage(cost_usd=1.0))
    assert led.by_provider == {"Google AI Studio": 2, "Google": 1, "unreported": 1}
    other = UsageLedger()
    other.record("judge", LLMUsage(cost_usd=1.0, provider="Google"))
    led.merge(other)
    assert led.by_provider["Google"] == 2, "merge must carry provider counts"
