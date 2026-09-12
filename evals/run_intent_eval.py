"""Validate and score the golden intent-extraction dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

DATASET = Path(__file__).parent / "datasets" / "intent_cases.jsonl"
FIELDS = ("question_type", "subject", "grade", "domain", "goal")
SUBJECTS = {"math", "ela", "eld", "sci", "hss", "vapa", "pe"}
GOALS = {"on_grade_level", "working_ahead", "catching_up", "competition_prep"}
CATEGORY_TARGETS = {
    "obvious_structured": 25,
    "tricky_structured": 20,
    "obvious_open": 20,
    "tricky_open": 15,
    "ambiguous_boundary": 15,
    "adversarial_unusual": 5,
}


def load_cases(path: Path = DATASET) -> list[dict[str, Any]]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return cases


def validate_cases(cases: list[dict[str, Any]]) -> dict[str, Counter[Any]]:
    errors: list[str] = []
    ids = [case.get("id") for case in cases]
    questions = [case.get("question") for case in cases]
    normalized_questions = [q.strip().casefold() if isinstance(q, str) else q for q in questions]

    if len(cases) != 100:
        errors.append(f"expected 100 cases, found {len(cases)}")
    if len(set(ids)) != len(ids):
        errors.append("IDs must be unique")
    if len(set(normalized_questions)) != len(normalized_questions):
        errors.append("questions must be unique after trimming and case-folding")

    for index, case in enumerate(cases, 1):
        label = case.get("id", f"line {index}")
        if set(case) != {"id", "category", "question", "expected"}:
            errors.append(f"{label}: unexpected or missing top-level fields")
        if not isinstance(case.get("id"), str) or not case["id"].strip():
            errors.append(f"{label}: id must be a non-empty string")
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            errors.append(f"{label}: question must be a non-empty string")
        if case.get("category") not in CATEGORY_TARGETS:
            errors.append(f"{label}: invalid category {case.get('category')!r}")
        expected = case.get("expected")
        if not isinstance(expected, dict) or set(expected) != set(FIELDS):
            errors.append(f"{label}: expected must contain exactly {', '.join(FIELDS)}")
            continue
        question_type = expected["question_type"]
        if question_type not in {"structured", "open"}:
            errors.append(f"{label}: invalid question_type {question_type!r}")
        if expected["subject"] not in SUBJECTS:
            errors.append(f"{label}: invalid subject {expected['subject']!r}")
        grade = expected["grade"]
        if grade is not None and (type(grade) is not int or not 0 <= grade <= 5):
            errors.append(f"{label}: grade must be null or an integer from 0 through 5")
        if expected["domain"] is not None and (
            not isinstance(expected["domain"], str) or not expected["domain"].strip()
        ):
            errors.append(f"{label}: domain must be null or a non-empty string")
        goal = expected["goal"]
        if question_type == "open" and goal is not None:
            errors.append(f"{label}: open questions must have goal=null")
        if question_type == "structured" and goal not in GOALS:
            errors.append(f"{label}: structured questions need a supported goal")

    distributions = {
        "category": Counter(case.get("category") for case in cases),
        "question_type": Counter(case.get("expected", {}).get("question_type") for case in cases),
        "structured_goal": Counter(
            case["expected"]["goal"]
            for case in cases
            if case.get("expected", {}).get("question_type") == "structured"
        ),
        "subject": Counter(case.get("expected", {}).get("subject") for case in cases),
    }
    if distributions["category"] != Counter(CATEGORY_TARGETS):
        errors.append(
            f"category distribution is {dict(distributions['category'])}; "
            f"expected {CATEGORY_TARGETS}"
        )
    expected_goals = Counter(
        {"on_grade_level": 12, "working_ahead": 12, "catching_up": 12, "competition_prep": 9}
    )
    if distributions["structured_goal"] != expected_goals:
        errors.append(
            f"structured goal distribution is {dict(distributions['structured_goal'])}; "
            f"expected {dict(expected_goals)}"
        )
    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))
    return distributions


def print_distributions(distributions: dict[str, Counter[Any]]) -> None:
    print("DATASET DISTRIBUTIONS")
    print("=" * 60)
    for name, counts in distributions.items():
        print(f"{name}: {dict(counts)}")


def normalize_domain(value: Any) -> str | None:
    return " ".join(value.casefold().split()) if isinstance(value, str) else None


def run_eval(cases: list[dict[str, Any]]) -> None:
    from api import llm
    from api.graph.extractor import extract_intent

    correct = Counter({field: 0 for field in FIELDS})
    failures = []
    llm.reset_usage()
    for case in cases:
        expected = case["expected"]
        actual = extract_intent(case["question"])
        mismatches = {}
        for field in FIELDS:
            wanted = normalize_domain(expected[field]) if field == "domain" else expected[field]
            got = normalize_domain(actual.get(field)) if field == "domain" else actual.get(field)
            if got == wanted:
                correct[field] += 1
            else:
                mismatches[field] = {"expected": wanted, "actual": got}
        if mismatches:
            failures.append({"id": case["id"], "question": case["question"], "mismatches": mismatches})

    print("\nINTENT EVAL")
    print("=" * 60)
    for field in FIELDS:
        print(f"{field:18} {correct[field]:3}/{len(cases)} = {correct[field] / len(cases):6.1%}")
    print(f"\nFailures: {len(failures)}")
    for failure in failures:
        print(f"\n{failure['id']}: {failure['question']}")
        for field, values in failure["mismatches"].items():
            print(f"  {field}: expected={values['expected']!r}, actual={values['actual']!r}")
    print("\nLLM USAGE")
    print("=" * 60)
    print(llm.usage_report())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true", help="validate without model calls")
    args = parser.parse_args()
    cases = load_cases()
    distributions = validate_cases(cases)
    print(f"Validated {len(cases)} unique golden cases.")
    print_distributions(distributions)
    if not args.validate_only:
        run_eval(cases)


if __name__ == "__main__":
    main()
