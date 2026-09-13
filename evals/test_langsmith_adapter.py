"""Model-free tests for LangSmith dataset conversion and evaluators."""

from evals.extractor_suites import load_suites, validate_cases
from evals.run_langsmith import (
    exact_evaluator, extractor_examples, field_evaluator, stable_example_id,
    sync_examples, tool_examples,
)


def test_langsmith_examples_keep_suites_and_stable_ids() -> None:
    cases = validate_cases(load_suites())
    balanced = extractor_examples([case for case in cases if case.suite == "balanced"])
    challenge = extractor_examples([case for case in cases if case.suite == "challenge"])
    tools = tool_examples(cases)
    assert (len(balanced), len(challenge), len(tools)) == (100, 25, 45)
    assert balanced[0]["id"] == stable_example_id("balanced", balanced[0]["metadata"]["case_id"])
    assert set(balanced[0]["outputs"]) == {"question_type", "subject", "grade", "domain", "goal"}


def test_field_and_exact_evaluators_are_deterministic() -> None:
    expected = {"subject": "math", "grade": 3}
    actual = {"subject": "math", "grade": 4, "goal": "catching_up"}
    assert field_evaluator("subject")(outputs=actual, reference_outputs=expected)["score"] is True
    assert field_evaluator("grade")(outputs=actual, reference_outputs=expected)["score"] is False
    assert field_evaluator("goal")(outputs=actual, reference_outputs=expected)["score"] is None
    assert exact_evaluator(outputs=actual, reference_outputs=expected)["score"] is False


def test_sync_updates_existing_examples_and_creates_missing_ones() -> None:
    class Item:
        def __init__(self, item_id):
            self.id = item_id

    class Client:
        dataset = type("Dataset", (), {"id": "dataset-1", "name": "Dataset"})()

        def __init__(self):
            self.updates = []
            self.creates = []

        def list_datasets(self, **kwargs):
            return iter([self.dataset])

        def list_examples(self, **kwargs):
            return iter([Item("existing")])

        def update_examples(self, **kwargs):
            self.updates.extend(kwargs["updates"])

        def create_examples(self, **kwargs):
            self.creates.extend(kwargs["examples"])

    client = Client()
    examples = [
        {"id": "existing", "inputs": {"question": "old"}, "outputs": {}},
        {"id": "new", "inputs": {"question": "new"}, "outputs": {}},
    ]
    sync_examples(client, "Dataset", "description", examples)
    assert [item["id"] for item in client.updates] == ["existing"]
    assert [item["id"] for item in client.creates] == ["new"]
