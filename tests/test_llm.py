from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import Mock

import pytest

from api import llm


MODEL = "test/model"
MESSAGES = [{"role": "user", "content": "Explain fractions"}]
RESPONSE = {
    "choices": [{"message": {"role": "assistant", "content": "A fraction is a part."}}],
    "usage": {"prompt_tokens": 100, "completion_tokens": 25},
}


@pytest.fixture(autouse=True)
def isolated_usage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / ".cache")
    # Synthetic rates solely for checking arithmetic, not provider prices.
    monkeypatch.setattr(llm, "PRICES", {MODEL: {"input": 2.0, "output": 4.0}})
    monkeypatch.setattr(llm, "_warned_models", set())
    monkeypatch.delenv("LLM_CACHE", raising=False)
    llm.reset_usage()
    yield
    llm.reset_usage()


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock = Mock(return_value=RESPONSE)
    monkeypatch.setattr(llm, "_provider_complete", mock)
    return mock


def test_cache_miss_then_hit(provider: Mock) -> None:
    first = llm.cached_complete(MESSAGES, MODEL)
    second = llm.cached_complete(MESSAGES, MODEL)
    assert first == second == RESPONSE
    provider.assert_called_once_with(messages=MESSAGES, model=MODEL, tools=None, temperature=0)
    assert len(list(llm.CACHE_DIR.glob("*.json"))) == 1
    entry = llm.usage.models[MODEL]
    assert (entry.calls, entry.cache_hits) == (1, 1)
    assert (entry.prompt_tokens, entry.completion_tokens) == (100, 25)
    assert entry.estimated_cost == pytest.approx(0.0003)


def test_temperature_separates_cache_entries(provider: Mock) -> None:
    llm.cached_complete(MESSAGES, MODEL, temperature=0)
    llm.cached_complete(MESSAGES, MODEL, temperature=0.7)
    llm.cached_complete(MESSAGES, MODEL, temperature=0.7)
    assert provider.call_count == 2
    assert len(list(llm.CACHE_DIR.glob("*.json"))) == 2


def test_exact_cache_key_and_forwarded_options(provider: Mock) -> None:
    request: dict[str, Any] = {
        "messages": MESSAGES, "model": MODEL,
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "temperature": 0.7, "max_tokens": 30, "seed": 5,
    }
    llm.cached_complete(**request)
    key = hashlib.sha256(json.dumps(request, sort_keys=True, default=str).encode()).hexdigest()
    assert json.loads((llm.CACHE_DIR / f"{key}.json").read_text()) == RESPONSE
    provider.assert_called_once_with(**request)
    llm.cached_complete(**{**request, "seed": 6})
    llm.cached_complete(**{**request, "tools": None})
    llm.cached_complete(**{**request, "model": "another/model"})
    assert provider.call_count == 4


def test_exception_is_not_cached(provider: Mock) -> None:
    provider.side_effect = [RuntimeError("provider failed"), RESPONSE]
    with pytest.raises(RuntimeError, match="provider failed"):
        llm.cached_complete(MESSAGES, MODEL)
    assert not list(llm.CACHE_DIR.glob("*.json"))
    assert llm.cached_complete(MESSAGES, MODEL) == RESPONSE
    assert llm.cached_complete(MESSAGES, MODEL) == RESPONSE
    assert provider.call_count == 2
    assert llm.usage.models[MODEL].prompt_tokens == 100


def test_ninth_uncached_call_is_blocked(provider: Mock) -> None:
    with llm.call_budget(8):
        for seed in range(8):
            llm.cached_complete(MESSAGES, MODEL, seed=seed)
        with pytest.raises(llm.MaxCallsExceeded):
            llm.cached_complete(MESSAGES, MODEL, seed=8)
    assert provider.call_count == 8
    assert llm.usage.models[MODEL].calls == 8


def test_hits_do_not_consume_budget(provider: Mock) -> None:
    with llm.call_budget(1):
        for _ in range(12):
            llm.cached_complete(MESSAGES, MODEL)
        with pytest.raises(llm.MaxCallsExceeded):
            llm.cached_complete(MESSAGES, MODEL, seed=1)
    with llm.call_budget(0):
        llm.cached_complete(MESSAGES, MODEL)
    assert provider.call_count == 1
    assert llm.usage.models[MODEL].cache_hits == 12


def test_failed_attempt_consumes_budget_and_context_recovers(provider: Mock) -> None:
    provider.side_effect = [RuntimeError("failed"), RESPONSE]
    with llm.call_budget(1):
        with pytest.raises(RuntimeError, match="failed"):
            llm.cached_complete(MESSAGES, MODEL)
        with pytest.raises(llm.MaxCallsExceeded):
            llm.cached_complete(MESSAGES, MODEL)
    assert llm.cached_complete(MESSAGES, MODEL) == RESPONSE
    assert provider.call_count == 2


def test_nested_budgets_enforce_outer_limit(provider: Mock) -> None:
    with llm.call_budget(1):
        with llm.call_budget(3):
            llm.cached_complete(MESSAGES, MODEL)
            with pytest.raises(llm.MaxCallsExceeded):
                llm.cached_complete(MESSAGES, MODEL, seed=1)
    assert provider.call_count == 1


def test_cache_off_bypasses_reads_and_writes(provider: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    llm.cached_complete(MESSAGES, MODEL)
    before = {path.name: path.read_bytes() for path in llm.CACHE_DIR.iterdir()}
    monkeypatch.setenv("LLM_CACHE", "off")
    with llm.call_budget(2):
        llm.cached_complete(MESSAGES, MODEL)
        llm.cached_complete(MESSAGES, MODEL, seed=1)
        with pytest.raises(llm.MaxCallsExceeded):
            llm.cached_complete(MESSAGES, MODEL)
    assert provider.call_count == 3
    assert {path.name: path.read_bytes() for path in llm.CACHE_DIR.iterdir()} == before
    assert llm.usage.models[MODEL].cache_hits == 0


@pytest.mark.parametrize("prices", [{}, {MODEL: {"input": None, "output": None}}])
def test_unknown_prices_count_tokens_and_warn_once(
    provider: Mock, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, prices: dict[str, dict[str, float | None]],
) -> None:
    monkeypatch.setattr(llm, "PRICES", prices)
    with caplog.at_level(logging.WARNING, logger=llm.__name__):
        llm.cached_complete(MESSAGES, MODEL)
        llm.cached_complete(MESSAGES, MODEL, seed=1)
    entry = llm.usage.models[MODEL]
    assert (entry.prompt_tokens, entry.completion_tokens) == (200, 50)
    assert entry.estimated_cost is None
    assert "None" in llm.usage_report()
    assert len(caplog.records) == 1


def test_reset_usage_keeps_cache(provider: Mock) -> None:
    llm.cached_complete(MESSAGES, MODEL)
    assert MODEL in llm.usage_report()
    llm.reset_usage()
    assert not llm.usage.models
    llm.cached_complete(MESSAGES, MODEL)
    entry = llm.usage.models[MODEL]
    assert (entry.calls, entry.cache_hits, entry.prompt_tokens, entry.completion_tokens) == (0, 1, 0, 0)
    assert entry.estimated_cost == 0


def test_debug_logging(provider: Mock, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger=llm.__name__):
        llm.cached_complete(MESSAGES, MODEL)
        llm.cached_complete(MESSAGES, MODEL)
    miss, hit = [record.getMessage() for record in caplog.records]
    assert f"model={MODEL} latency=" in miss
    assert "cached=false prompt_tokens=100 completion_tokens=25" in miss
    assert "cached=true prompt_tokens=0 completion_tokens=0" in hit


@pytest.mark.parametrize("answer,cleared", [("", False), ("n", False), ("y", True)])
def test_clear_cache_confirmation(
    provider: Mock, monkeypatch: pytest.MonkeyPatch, answer: str, cleared: bool,
) -> None:
    llm.cached_complete(MESSAGES, MODEL)
    monkeypatch.setattr("builtins.input", lambda _: answer)
    llm.main(["--clear-cache"])
    assert bool(list(llm.CACHE_DIR.iterdir())) is not cleared


def test_report_cli(provider: Mock, capsys: pytest.CaptureFixture[str]) -> None:
    llm.cached_complete(MESSAGES, MODEL)
    llm.main(["--report"])
    assert capsys.readouterr().out == llm.usage_report() + "\n"


def test_provider_adapter_disables_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    response = Mock()
    response.model_dump.return_value = RESPONSE
    completion = Mock(return_value=response)
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    assert llm._provider_complete(messages=MESSAGES, model=MODEL) == RESPONSE
    completion.assert_called_once_with(messages=MESSAGES, model=MODEL, num_retries=0)
    response.model_dump.assert_called_once_with(mode="json")


def test_run_usage_isolated_from_other_contexts_and_cached_submissions(provider):
    from contextvars import Context
    handle = llm.begin_run()
    Context().run(llm.cached_complete, MESSAGES, MODEL)
    llm.cached_complete([{"role": "user", "content": "This submission"}], MODEL)
    report = llm.end_run(handle)
    assert report == {"calls": 1, "tokens": 125, "cost": pytest.approx(0.0003)}
    assert llm.usage.models[MODEL].calls == 2
    cached = llm.begin_run()
    llm.cached_complete(MESSAGES, MODEL)
    assert llm.end_run(cached) == {"calls": 0, "tokens": 0, "cost": 0}


def test_failed_provider_attempt_is_in_run_usage(monkeypatch):
    monkeypatch.setattr(llm, '_provider_complete', Mock(side_effect=RuntimeError('offline')))
    handle = llm.begin_run()
    with pytest.raises(RuntimeError, match='offline'):
        llm.cached_complete(MESSAGES, MODEL)
    assert llm.end_run(handle)['calls'] == 1
