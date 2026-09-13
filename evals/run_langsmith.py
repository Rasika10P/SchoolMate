"""Synchronize SchoolMate datasets and run scored LangSmith experiments."""

from __future__ import annotations

import argparse
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Callable
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langsmith import Client, evaluate, traceable

from evals.extractor_suites import EvalCase, load_suites, validate_cases


DATASET_NAMES = {
    "balanced": "SchoolMate - Balanced Extractor",
    "challenge": "SchoolMate - Real-World Challenge",
    "tools": "SchoolMate - Tool Selection",
}


def stable_example_id(suite: str, case_id: str):
    return uuid5(NAMESPACE_URL, f"schoolmate-evals/{suite}/{case_id}")


def extractor_examples(cases: list[EvalCase]) -> list[dict[str, Any]]:
    return [{
        "id": stable_example_id(case.suite, case.id),
        "inputs": {"question": case.question},
        "outputs": case.expected,
        "metadata": {"case_id": case.id, "suite": case.suite,
                     "category": case.category, "notes": case.notes},
    } for case in cases]


def tool_examples(cases: list[EvalCase]) -> list[dict[str, Any]]:
    structured = [case for case in cases
                  if case.suite == "balanced" and case.expected["question_type"] == "structured"]
    return [{
        "id": stable_example_id("tools", case.id),
        "inputs": {"question": case.question, "intent": case.expected},
        "outputs": {"tool": case.expected["goal"]},
        "metadata": {"case_id": case.id, "suite": "tools", "category": case.category,
                     "subject": case.expected["subject"], "goal": case.expected["goal"]},
    } for case in structured]


def ensure_dataset(client: Client, name: str, description: str):
    existing = next((item for item in client.list_datasets(dataset_name=name)
                     if item.name == name), None)
    return existing or client.create_dataset(dataset_name=name, description=description)


def sync_examples(client: Client, name: str, description: str,
                  examples: list[dict[str, Any]]):
    dataset = ensure_dataset(client, name, description)
    existing_ids = {str(item.id) for item in client.list_examples(dataset_id=dataset.id)}
    updates = [example for example in examples if str(example["id"]) in existing_ids]
    creates = [example for example in examples if str(example["id"]) not in existing_ids]
    if updates:
        client.update_examples(dataset_id=dataset.id, updates=updates)
    if creates:
        client.create_examples(dataset_id=dataset.id, examples=creates)
    return dataset


def sync_all(client: Client, cases: list[EvalCase]) -> dict[str, Any]:
    grouped = {suite: [case for case in cases if case.suite == suite]
               for suite in ("balanced", "challenge")}
    datasets = {}
    for suite, rows in grouped.items():
        datasets[suite] = sync_examples(
            client, DATASET_NAMES[suite],
            f"SchoolMate {suite} intent-extraction golden cases.", extractor_examples(rows),
        )
    datasets["tools"] = sync_examples(
        client, DATASET_NAMES["tools"],
        "SchoolMate structured intents for deterministic and agent tool selection.",
        tool_examples(cases),
    )
    return datasets


@traceable(name="schoolmate_extract_intent", run_type="chain")
def extractor_target(inputs: dict[str, Any]) -> dict[str, Any]:
    from api.graph.extractor import extract_intent

    return extract_intent(inputs["question"])


def field_evaluator(field: str) -> Callable[..., dict[str, Any]]:
    def evaluator(*, outputs: dict[str, Any], reference_outputs: dict[str, Any], **_: Any):
        if field not in reference_outputs:
            return {"key": field, "score": None, "comment": "Field is not annotated"}
        return {"key": field, "score": outputs.get(field) == reference_outputs[field]}
    evaluator.__name__ = f"matches_{field}"
    return evaluator


def exact_evaluator(*, outputs: dict[str, Any], reference_outputs: dict[str, Any], **_: Any):
    return {"key": "exact_match", "score": all(
        outputs.get(field) == expected for field, expected in reference_outputs.items()
    )}


@traceable(name="deterministic_tool_selection", run_type="chain")
def deterministic_tool_target(inputs: dict[str, Any]) -> dict[str, Any]:
    from api.tools.definitions import GOAL_TO_TOOL

    return {"tool": GOAL_TO_TOOL.get(inputs["intent"]["goal"])}


@lru_cache(maxsize=1)
def _agent_graph():
    from api.graph.agentic import build_agent_graph

    return build_agent_graph()


@traceable(name="agent_tool_selection", run_type="chain")
def agent_tool_target(inputs: dict[str, Any]) -> dict[str, Any]:
    intent = inputs["intent"]
    state = {**intent, "question": inputs["question"], "tool_results": {},
             "answer": "", "messages": []}
    result = _agent_graph().invoke(state, interrupt_after=["model"])
    messages = result.get("messages") or []
    calls = getattr(messages[-1], "tool_calls", None) or [] if messages else []
    names = [call.get("name") for call in calls if isinstance(call, dict) and call.get("name")]
    return {"tool": names[0] if len(names) == 1 else names}


def tool_evaluator(*, outputs: dict[str, Any], reference_outputs: dict[str, Any], **_: Any):
    return {"key": "tool_match", "score": outputs.get("tool") == reference_outputs["tool"]}


def run_extractor_experiments(client: Client, datasets: dict[str, Any], model: str,
                              suites: tuple[str, ...] = ("balanced", "challenge")) -> None:
    evaluators = [*(field_evaluator(field) for field in
                    ("question_type", "subject", "grade", "domain", "goal")), exact_evaluator]
    for suite in suites:
        evaluate(
            extractor_target, data=datasets[suite].id, evaluators=evaluators,
            experiment_prefix=f"schoolmate-{suite}-extractor",
            description="SchoolMate minimum-scope extractor benchmark.",
            metadata={"models": model, "prompts": ["api.graph.extractor.SYSTEM_PROMPT"],
                      "suite": suite}, max_concurrency=1, client=client,
        )


def run_tool_experiments(client: Client, dataset: Any, model: str) -> None:
    common = dict(data=dataset.id, evaluators=[tool_evaluator], max_concurrency=1, client=client)
    evaluate(
        deterministic_tool_target, experiment_prefix="schoolmate-deterministic-tools",
        description="Fixed goal-to-tool baseline.", metadata={"tools": "GOAL_TO_TOOL"}, **common,
    )
    evaluate(
        agent_tool_target, experiment_prefix="schoolmate-agent-tools",
        description="GPT agent first-tool decision; interrupted before execution.",
        metadata={"models": model, "tools": "SchoolMate curriculum tools"}, **common,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("extractor", "tools", "all"),
                        help="run experiments after synchronization; may make paid model calls")
    parser.add_argument("--model", default=None, help="extractor/agent model override")
    parser.add_argument("--suite", choices=("all", "balanced", "challenge"), default="all",
                        help="extractor suite to run (default: all)")
    parser.add_argument("--use-cache", action="store_true",
                        help="reuse local model responses; default experiments are uncached")
    args = parser.parse_args()
    load_dotenv(".env")
    if not os.getenv("LANGSMITH_API_KEY"):
        parser.error("LANGSMITH_API_KEY must be configured")
    model = args.model or os.getenv("EXTRACTOR_MODEL", "openai/gpt-4o-mini")
    os.environ["EXTRACTOR_MODEL"] = model
    os.environ["LLM_MODEL"] = model
    os.environ["LLM_CACHE"] = "on" if args.use_cache else "off"
    cases = validate_cases(load_suites())
    client = Client()
    datasets = sync_all(client, cases)
    print("Synchronized LangSmith datasets: balanced=100, challenge=25, tools=45")
    if args.run in {"extractor", "all"}:
        suites = ("balanced", "challenge") if args.suite == "all" else (args.suite,)
        run_extractor_experiments(client, datasets, model, suites)
    if args.run in {"tools", "all"}:
        run_tool_experiments(client, datasets["tools"], model)
    if not args.run:
        print("No experiment executed. Add --run extractor, --run tools, or --run all.")


if __name__ == "__main__":
    main()
