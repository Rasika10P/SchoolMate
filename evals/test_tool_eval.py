"""Model-free checks for the tool-selection eval inputs."""

from collections import Counter

from evals.run_tool_eval import structured_cases


def test_structured_tool_cases_reuse_the_golden_dataset() -> None:
    cases = structured_cases()
    assert len(cases) == 45
    assert Counter(case["expected"]["goal"] for case in cases) == {
        "on_grade_level": 12,
        "working_ahead": 12,
        "catching_up": 12,
        "competition_prep": 9,
    }
