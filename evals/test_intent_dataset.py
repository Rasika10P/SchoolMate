"""Fast, model-free checks for the golden intent dataset."""

from evals.run_intent_eval import load_cases, validate_cases


def test_intent_dataset_is_valid() -> None:
    cases = load_cases()
    distributions = validate_cases(cases)
    assert distributions["question_type"] == {"structured": 45, "open": 55}
