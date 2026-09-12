"""Shared loading, scoring, and reporting for extractor evaluation suites."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import io
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
BALANCED_DATA = ROOT / "evals" / "datasets" / "intent_cases.jsonl"
CHALLENGE_DATA = ROOT / "data" / "Curriculam_GoldenDataSet.csv"
FIELDS = ("question_type", "subject", "grade", "domain", "goal")
SUBJECT_ALIASES = {"ala": "ela", "els": "eld"}
NULL_LABELS = {"", "none", "null"}
GOAL_TO_TYPE = {
    "on_grade_level": "structured",
    "working_ahead": "structured",
    "catching_up": "structured",
    "competition_prep": "structured",
    "search_guidance": "open",
}
# Evaluation-only corrections preserve the contributor's source CSV verbatim.
# Unsupported/safety questions are open because the extractor prompt says to
# prefer open when the request cannot map safely to a structured curriculum job.
CHALLENGE_OVERRIDES: dict[str, dict[str, Any]] = {
    "G04": {"goal": None, "question_type": "open"},
    "G06": {"goal": None, "question_type": "open"},
    "G11": {"goal": "working_ahead", "question_type": "structured"},
    "G14": {"goal": None, "question_type": "open"},
    "G15": {"subject": None, "grade": None, "goal": None, "question_type": "open"},
    "G16": {"goal": None, "question_type": "open"},
    "G17": {"goal": None, "question_type": "open"},
    "G20": {"goal": None, "question_type": "open"},
    "G25": {"goal": None, "question_type": "open"},
}


@dataclass(frozen=True)
class EvalCase:
    id: str
    suite: str
    question: str
    category: str
    expected: dict[str, Any]
    notes: str = ""


def load_balanced(path: Path = BALANCED_DATA) -> list[EvalCase]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        cases.append(EvalCase(
            id=row["id"], suite="balanced", question=row["question"],
            category=row["category"], expected=row["expected"],
        ))
    return cases


def _challenge_rows(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        text = text[text.index("id,question,"):]
    except ValueError as exc:
        raise ValueError(f"{path}: CSV header beginning with 'id,question,' was not found") from exc
    return list(csv.DictReader(io.StringIO(text)))


def _subject(value: str) -> str | None:
    normalized = value.strip().casefold()
    if normalized in NULL_LABELS:
        return None
    return SUBJECT_ALIASES.get(normalized, normalized)


def load_challenge(path: Path = CHALLENGE_DATA) -> list[EvalCase]:
    """Load the same 22 Ask rows and three requested guided rows as the original runner."""
    cases = []
    for row in _challenge_rows(path):
        ask = row["entry_path"].strip().casefold() in {"ask", "asl"}
        if not ask and row["id"].upper() not in {"G07", "G11", "G13"}:
            continue
        expected: dict[str, Any] = {}
        subject = _subject(row["subject"])
        # G07's multi-subject label cannot be represented by the extractor.
        if subject is None or subject in {"math", "ela", "eld", "sci", "hss", "vapa", "pe"}:
            expected["subject"] = subject
        grade = row["grade"].strip()
        expected["grade"] = int(grade) if grade else None
        tool = row["expected_tool"].strip().strip(",").casefold()
        if tool in GOAL_TO_TYPE:
            expected["goal"] = None if tool == "search_guidance" else tool
            expected["question_type"] = GOAL_TO_TYPE[tool]
        expected.update(CHALLENGE_OVERRIDES.get(row["id"].upper(), {}))
        cases.append(EvalCase(
            id=row["id"], suite="challenge", question=row["question"],
            category=row["category"].strip().casefold() or "uncategorized",
            expected=expected, notes=row.get("Notes", "").strip(),
        ))
    return cases


def load_suites(selection: str = "all") -> list[EvalCase]:
    if selection == "balanced":
        return load_balanced()
    if selection == "challenge":
        return load_challenge()
    if selection == "all":
        return [*load_balanced(), *load_challenge()]
    raise ValueError(f"Unknown suite {selection!r}")


def validate_cases(cases: Iterable[EvalCase]) -> list[EvalCase]:
    materialized = list(cases)
    errors = []
    seen_ids: set[tuple[str, str]] = set()
    for case in materialized:
        key = (case.suite, case.id)
        if key in seen_ids:
            errors.append(f"{case.suite}/{case.id}: duplicate ID")
        seen_ids.add(key)
        if not case.question.strip():
            errors.append(f"{case.suite}/{case.id}: empty question")
        unknown = set(case.expected) - set(FIELDS)
        if unknown:
            errors.append(f"{case.suite}/{case.id}: unknown expected fields {sorted(unknown)}")
        if not case.expected:
            errors.append(f"{case.suite}/{case.id}: no scorable expected fields")
    balanced = [case for case in materialized if case.suite == "balanced"]
    challenge = [case for case in materialized if case.suite == "challenge"]
    if balanced and len(balanced) != 100:
        errors.append(f"balanced: expected 100 cases, found {len(balanced)}")
    if challenge and len(challenge) != 25:
        errors.append(f"challenge: expected 25 selected cases, found {len(challenge)}")
    if errors:
        raise ValueError("Suite validation failed:\n- " + "\n- ".join(errors))
    return materialized


def score(expected: dict[str, Any], actual: dict[str, Any] | None) -> dict[str, bool]:
    return {field: actual is not None and actual.get(field) == value
            for field, value in expected.items()}


def run_cases(cases: list[EvalCase], model: str) -> list[dict[str, Any]]:
    from api import llm
    from api.graph.extractor import extract_intent

    prior_model = os.environ.get("EXTRACTOR_MODEL")
    os.environ["EXTRACTOR_MODEL"] = model
    results = []
    try:
        for case in cases:
            handle = llm.begin_run()
            started = perf_counter()
            error = None
            actual = None
            try:
                actual = extract_intent(case.question)
            except Exception as exc:
                error = type(exc).__name__
            latency = perf_counter() - started
            usage = llm.end_run(handle)
            results.append({
                **asdict(case), "model": model, "actual": actual,
                "matches": score(case.expected, actual), "latency": latency,
                "usage": usage, "error": error,
            })
    finally:
        if prior_model is None:
            os.environ.pop("EXTRACTOR_MODEL", None)
        else:
            os.environ["EXTRACTOR_MODEL"] = prior_model
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in results if not row["error"]]
    field_totals: dict[str, Counter[str]] = defaultdict(Counter)
    groups: dict[str, dict[str, Counter[str]]] = {
        "category": defaultdict(Counter),
        "subject": defaultdict(Counter),
        "goal": defaultdict(Counter),
    }
    exact = 0
    for row in completed:
        matches = row["matches"]
        exact += bool(matches) and all(matches.values())
        for field, matched in matches.items():
            field_totals[field]["correct" if matched else "wrong"] += 1
        labels = {
            "category": row["category"],
            "subject": str(row["expected"].get("subject", "unscored")),
            "goal": str(row["expected"].get("goal", "unscored")),
        }
        for dimension, label in labels.items():
            groups[dimension][label]["correct" if matches and all(matches.values()) else "wrong"] += 1
    return {
        "cases": len(results),
        "completed": len(completed),
        "errors": len(results) - len(completed),
        "exact_correct": exact,
        "exact_accuracy": exact / len(completed) if completed else None,
        "field_accuracy": {
            field: {
                "correct": counts["correct"],
                "total": sum(counts.values()),
                "accuracy": counts["correct"] / sum(counts.values()),
            }
            for field, counts in field_totals.items()
        },
        "mean_latency": mean(row["latency"] for row in completed) if completed else None,
        "calls": sum(row["usage"]["calls"] for row in results),
        "tokens": sum(row["usage"]["tokens"] for row in results),
        "cost": None if any(row["usage"]["cost"] is None for row in results)
                else sum(row["usage"]["cost"] for row in results),
        "groups": {
            dimension: {
                label: {"correct": counts["correct"], "total": sum(counts.values())}
                for label, counts in labels.items()
            }
            for dimension, labels in groups.items()
        },
    }


def suite_inventory(cases: list[EvalCase]) -> dict[str, Any]:
    by_suite = defaultdict(list)
    for case in cases:
        by_suite[case.suite].append(case)
    return {
        suite: {
            "cases": len(rows),
            "categories": dict(Counter(row.category for row in rows)),
            "annotated_fields": dict(Counter(field for row in rows for field in row.expected)),
            "subjects": dict(Counter(str(row.expected.get("subject", "unscored")) for row in rows)),
            "goals": dict(Counter(str(row.expected.get("goal", "unscored")) for row in rows)),
        }
        for suite, rows in by_suite.items()
    }
