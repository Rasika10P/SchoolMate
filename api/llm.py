"""Synchronous model gateway. All application model calls belong here.

Usage is process-local; ``--report`` reports only the current process. Calls
count provider attempts (including failures); cache hits are counted separately.
Cache paths are relative to the project root, independent of working directory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from threading import RLock
from time import perf_counter
from typing import Any, Callable, Iterator, ParamSpec

# Prices checked 2026-09-06; USD per million tokens, standard text API rates.
# Source: https://developers.openai.com/api/docs/models/gpt-4o-mini
# Estimates use standard input rates, without provider-side cache discounts.
# Local cache hits remain zero-cost. None means unknown, never free usage.
PRICES: dict[str, dict[str, float | None]] = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Mock-only identifiers from tests; no published provider prices exist.
    "test/model": {"input": None, "output": None},
    "another/model": {"input": None, "output": None},
    "test-model": {"input": None, "output": None},
}

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
logger = logging.getLogger(__name__)
_lock = RLock()
_warned_models: set[str] = set()


@dataclass
class ModelUsage:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost: float | None = 0.0


@dataclass
class Usage:
    models: dict[str, ModelUsage] = field(default_factory=dict)


usage = Usage()


_run_usage: ContextVar[Usage | None] = ContextVar("run_usage", default=None)


def begin_run():
    """Start isolated submission accounting, including propagated tool contexts."""
    report = Usage()
    return report, _run_usage.set(report)


def end_run(handle) -> dict[str, Any]:
    report, token = handle
    _run_usage.reset(token)
    entries = list(report.models.values())
    return {"calls": sum(e.calls for e in entries),
            "tokens": sum(e.prompt_tokens + e.completion_tokens for e in entries),
            "cost": None if any(e.estimated_cost is None and e.calls for e in entries)
                    else sum(e.estimated_cost or 0 for e in entries)}


def _usage_entries(model: str):
    yield _model_usage(model)
    report = _run_usage.get()
    if report is not None:
        entry = report.models.setdefault(model, ModelUsage())
        prices = PRICES.get(model, {})
        if any(prices.get(k) is None for k in ("input", "output")):
            entry.estimated_cost = None
        yield entry


def _model_usage(model: str) -> ModelUsage:
    """Caller holds _lock."""
    entry = usage.models.setdefault(model, ModelUsage())
    prices = PRICES.get(model)
    if prices is None or any(prices.get(k) is None for k in ("input", "output")):
        entry.estimated_cost = None
        if model not in _warned_models:
            _warned_models.add(model)
            logger.warning("Missing prices for model %s; estimated cost is unknown", model)
    return entry


def reset_usage() -> None:
    """Reset totals for an eval run; warning deduplication remains process-wide."""
    with _lock:
        usage.models.clear()


def usage_report() -> str:
    """Return a table of provider attempts, cache savings, and known token costs."""
    rows = [["Model", "Calls", "Cache hits", "Prompt tokens", "Completion tokens", "Cost (USD)"]]
    with _lock:
        for model, item in sorted(usage.models.items()):
            cost = "None" if item.estimated_cost is None else f"{item.estimated_cost:.6f}"
            rows.append([model, str(item.calls), str(item.cache_hits),
                         str(item.prompt_tokens), str(item.completion_tokens), cost])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = [" | ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in rows]
    lines.insert(1, "-+-".join("-" * width for width in widths))
    return "\n".join(lines)


class MaxCallsExceeded(RuntimeError):
    """Raised before a provider attempt would exceed an active budget."""


@dataclass
class _Budget:
    limit: int
    calls: int = 0


_budgets: ContextVar[tuple[_Budget, ...]] = ContextVar("llm_budgets", default=())


@contextmanager
def call_budget(n: int) -> Iterator[None]:
    """Limit uncached attempts; nested calls also consume enclosing budgets.

    Budgets are scoped to the execution context, not unrelated requests.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("Call budget must be a non-negative integer")
    token = _budgets.set((*_budgets.get(), _Budget(n)))
    try:
        yield
    finally:
        _budgets.reset(token)


def _reserve_call() -> None:
    with _lock:
        for budget in _budgets.get():
            if budget.calls >= budget.limit:
                raise MaxCallsExceeded(f"Uncached call budget of {budget.limit} exhausted")
        for budget in _budgets.get():
            budget.calls += 1


@dataclass
class _Trace:
    cached: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


_trace: ContextVar[_Trace | None] = ContextVar("llm_trace", default=None)
P = ParamSpec("P")


def track(func: Callable[P, dict[str, Any]]) -> Callable[P, dict[str, Any]]:
    """Log model, latency in seconds, cache status, and billed tokens at DEBUG."""
    signature = inspect.signature(func)

    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
        model = signature.bind(*args, **kwargs).arguments.get("model", "unknown")
        trace = _Trace()
        token = _trace.set(trace)
        start = perf_counter()
        try:
            response = func(*args, **kwargs)
            if not trace.cached:
                trace.prompt_tokens, trace.completion_tokens = _tokens(response)
            return response
        finally:
            logger.debug(
                "model=%s latency=%.6f cached=%s prompt_tokens=%d completion_tokens=%d",
                model, perf_counter() - start, str(trace.cached).lower(),
                trace.prompt_tokens, trace.completion_tokens,
            )
            _trace.reset(token)

    return wrapped


def _tokens(response: dict[str, Any]) -> tuple[int, int]:
    counts = response.get("usage") or {}
    return int(counts.get("prompt_tokens") or 0), int(counts.get("completion_tokens") or 0)


def _provider_complete(**kwargs: Any) -> dict[str, Any]:
    # Lazy import keeps reports, cache hits, and unit tests independent of SDKs.
    from litellm import completion

    response = completion(**kwargs, num_retries=0)
    return response.model_dump(mode="json")


def _write_cache(path: Path, response: dict[str, Any]) -> None:
    # Serialize before creating a file, and atomically publish complete JSON.
    serialized = json.dumps(response)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@track
def cached_complete(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0,
    **kw: Any,
) -> dict[str, Any]:
    """Complete once, or return JSON cached for precisely these arguments.

    Streaming and automatic retries/fallbacks are disallowed: each provider
    attempt must pass through the budget. Retry by calling this wrapper again.
    LLM_CACHE=off bypasses both cache reads and writes.
    """
    if kw.get("stream"):
        raise ValueError("Streaming is unsupported by this dict-returning wrapper")
    if any(kw.get(key) for key in ("num_retries", "max_retries", "fallbacks")):
        raise ValueError("Retry and fallback attempts must go through cached_complete")
    request = dict(messages=messages, model=model, tools=tools, temperature=temperature, **kw)
    enabled = os.getenv("LLM_CACHE", "on").strip().lower() != "off"
    path: Path | None = None
    if enabled:
        key = hashlib.sha256(json.dumps(request, sort_keys=True, default=str).encode()).hexdigest()
        path = CACHE_DIR / f"{key}.json"
        try:
            response = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(response, dict):
                raise ValueError("Cached response must be a JSON object")
        except FileNotFoundError:
            pass
        except (ValueError, UnicodeError):
            logger.warning("Ignoring invalid cache file %s", path)
        else:
            with _lock:
                for entry in _usage_entries(model):
                    entry.cache_hits += 1
            trace = _trace.get()
            if trace is not None:
                trace.cached = True
            return response

    _reserve_call()
    with _lock:
        for entry in _usage_entries(model):
            entry.calls += 1
    provider_request = {k: v for k, v in request.items() if k not in ("num_retries", "max_retries", "fallbacks")}
    response = _provider_complete(**provider_request)
    prompt, completion = _tokens(response)
    with _lock:
        for entry in _usage_entries(model):
            entry.prompt_tokens += prompt
            entry.completion_tokens += completion
            if entry.estimated_cost is not None:
                prices = PRICES[model]
                entry.estimated_cost += (prompt * float(prices["input"]) + completion * float(prices["output"])) / 1_000_000
    if path is not None:
        _write_cache(path, response)
    return response


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="Print process-local usage")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cached responses after confirmation")
    args = parser.parse_args(argv)
    if args.clear_cache:
        try:
            confirmed = input(f"Empty {CACHE_DIR}? [y/N] ").strip().lower() == "y"
        except EOFError:
            confirmed = False
        if confirmed:
            if CACHE_DIR.exists():
                for path in CACHE_DIR.iterdir():
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
            print("Cache cleared.")
        else:
            print("Cache unchanged.")
    if args.report:
        print(usage_report())
    if not (args.report or args.clear_cache):
        parser.print_help()


if __name__ == "__main__":
    main()
