"""Model-free checks for the unified extractor suite adapter and scorer."""

from evals.extractor_suites import load_suites, score, suite_inventory, validate_cases


def test_both_suites_validate_and_remain_separate() -> None:
    cases = validate_cases(load_suites())
    inventory = suite_inventory(cases)
    assert len(cases) == 125
    assert inventory["balanced"]["cases"] == 100
    assert inventory["challenge"]["cases"] == 25
    assert inventory["balanced"]["annotated_fields"] == {
        "question_type": 100, "subject": 100, "grade": 100, "domain": 100, "goal": 100,
    }
    assert inventory["challenge"]["annotated_fields"].get("domain", 0) == 0


def test_shared_scorer_only_scores_annotated_fields() -> None:
    actual = {"subject": "math", "grade": 3, "goal": "catching_up"}
    assert score({"subject": "math", "grade": 4}, actual) == {
        "subject": True, "grade": False,
    }


def test_challenge_label_overrides_match_extractor_contract() -> None:
    cases = {case.id: case for case in validate_cases(load_suites("challenge"))}
    assert cases["G07"].expected.get("subject") is None  # multi-subject gold is unscorable
    assert cases["G11"].expected["goal"] == "working_ahead"
    assert cases["G15"].expected == {
        "subject": None, "grade": None, "goal": None, "question_type": "open",
    }
