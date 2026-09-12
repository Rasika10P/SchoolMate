"""Compare fixed and agentic tool selection on structured golden cases.

The agent is interrupted immediately after its first model decision, before a
tool can query Postgres. This eval measures selection only, not retrieval.
"""

from __future__ import annotations

import argparse
from collections import Counter
from statistics import mean
from time import perf_counter
from typing import Any

from evals.run_intent_eval import load_cases, validate_cases


def structured_cases() -> list[dict[str, Any]]:
    cases = load_cases()
    validate_cases(cases)
    return [case for case in cases if case["expected"]["question_type"] == "structured"]


def evaluate_deterministic(cases: list[dict[str, Any]]) -> dict[str, Any]:
    from api.tools.definitions import GOAL_TO_TOOL

    failures = []
    latencies = []
    for case in cases:
        started = perf_counter()
        actual = GOAL_TO_TOOL.get(case["expected"]["goal"])
        latencies.append(perf_counter() - started)
        expected = case["expected"]["goal"]
        if actual != expected:
            failures.append({"id": case["id"], "expected": expected, "actual": actual})
    return {
        "correct": len(cases) - len(failures),
        "total": len(cases),
        "average_latency": mean(latencies) if latencies else 0.0,
        "failures": failures,
    }


def _selected_tools(result: dict[str, Any]) -> list[str]:
    messages = result.get("messages") or []
    if not messages:
        return []
    calls = getattr(messages[-1], "tool_calls", None) or []
    return [call.get("name") for call in calls if isinstance(call, dict) and call.get("name")]


def evaluate_agent(cases: list[dict[str, Any]], max_cases: int | None = None) -> dict[str, Any]:
    from api import llm
    from api.graph.agentic import build_agent_graph

    selected_cases = cases if max_cases is None else cases[:max_cases]
    graph = build_agent_graph()
    failures = []
    latencies = []
    llm.reset_usage()
    for case in selected_cases:
        expected_intent = case["expected"]
        state = {
            "subject": expected_intent["subject"],
            "grade": expected_intent["grade"],
            "domain": expected_intent["domain"],
            "goal": expected_intent["goal"],
            "question_type": "structured",
            "question": case["question"],
            "tool_results": {},
            "answer": "",
            "messages": [],
        }
        started = perf_counter()
        result = graph.invoke(state, interrupt_after=["model"])
        latencies.append(perf_counter() - started)
        actual = _selected_tools(result)
        expected = expected_intent["goal"]
        if actual != [expected]:
            failures.append({
                "id": case["id"],
                "question": case["question"],
                "expected": expected,
                "actual": actual,
            })
    return {
        "correct": len(selected_cases) - len(failures),
        "total": len(selected_cases),
        "average_latency": mean(latencies) if latencies else 0.0,
        "failures": failures,
        "usage": llm.usage_report(),
    }


def print_result(label: str, result: dict[str, Any]) -> None:
    total = result["total"]
    accuracy = result["correct"] / total if total else 0.0
    print(f"\n{label}")
    print("=" * 60)
    print(f"Tool accuracy:   {result['correct']}/{total} = {accuracy:.1%}")
    print(f"Average latency: {result['average_latency']:.6f}s")
    print(f"Failures:        {len(result['failures'])}")
    for failure in result["failures"]:
        print(f"  {failure['id']}: expected={failure['expected']!r}, actual={failure['actual']!r}")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(".env")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="skip the agent and make no model calls",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="limit agent cases for a smoke test; deterministic scoring still uses all 45",
    )
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be at least 1")

    cases = structured_cases()
    goals = Counter(case["expected"]["goal"] for case in cases)
    print(f"Validated {len(cases)} structured tool-selection cases: {dict(goals)}")
    deterministic = evaluate_deterministic(cases)
    print_result("DETERMINISTIC TOOL SELECTION", deterministic)

    if not args.deterministic_only:
        agent = evaluate_agent(cases, args.max_cases)
        print_result("AGENT TOOL SELECTION", agent)
        print("\nLLM USAGE")
        print("=" * 60)
        print(agent["usage"])


if __name__ == "__main__":
    main()
