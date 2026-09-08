import json
from typing import Any
from unittest.mock import Mock

import pytest

from api import llm
from api.graph.extractor import extract_intent


@pytest.mark.parametrize("content,expected", [
    (json.dumps(dict(subject="math", grade=4, domain="fractions", goal="catching_up", question_type="structured")),
     dict(subject="math", grade=4, domain="fractions", goal="catching_up")),
    (json.dumps(dict(subject="ela", grade=None, domain=None, goal=None)),
     dict(subject="ela", grade=None, domain=None, goal=None)),
    ("this is not JSON", dict(subject=None, grade=None, domain=None, goal=None)),
    ('```json\n{"subject":"math","grade":0,"domain":null,"goal":"on_grade_level","question_type":"structured"}\n```',
     dict(subject="math", grade=0, domain=None, goal="on_grade_level")),
    ('{"subject":[],"grade":true,"domain":42,"goal":"invalid"}',
     dict(subject=None, grade=None, domain=None, goal=None)),
    ("[]", dict(subject=None, grade=None, domain=None, goal=None)),
])
def test_extraction(monkeypatch: pytest.MonkeyPatch, content: str, expected: dict[str, Any]) -> None:
    monkeypatch.delenv("EXTRACTOR_MODEL", raising=False)
    model = Mock(return_value={"choices": [{"message": {"content": content}}]})
    monkeypatch.setattr(llm, "cached_complete", model)
    question = "  My child needs help.\n"
    assert extract_intent(question) == {**expected, "raw": question, "evidence": {},
                                         "question_type": "structured" if expected["goal"] else "open"}
    assert model.call_args.kwargs["model"] == "openai/gpt-4o-mini"
    assert model.call_args.kwargs["messages"][1]["content"] == question


def test_model_override_and_missing_response(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Mock(return_value={})
    monkeypatch.setattr(llm, "cached_complete", model)
    monkeypatch.setenv("EXTRACTOR_MODEL", "custom-extractor")
    assert extract_intent("Question") == dict(subject=None, grade=None, domain=None, goal=None, raw="Question", question_type="open", evidence={})
    assert model.call_args.kwargs["model"] == "custom-extractor"


def test_provider_failures_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "cached_complete", Mock(side_effect=RuntimeError("API failed")))
    with pytest.raises(RuntimeError, match="API failed"):
        extract_intent("Question")


@pytest.mark.parametrize("kind", ["open", "unknown", None])
def test_open_and_uncertain_intents_discard_goal_and_keep_evidence(monkeypatch, kind):
    payload = dict(subject="math", grade=4, question_type=kind, goal="catching_up",
                   evidence={"subject": "from 'fractions'", "grade": "most 9-year-olds", "extra": "ignored"})
    monkeypatch.setattr(llm, "cached_complete", Mock(return_value={"choices": [{"message": {"content": json.dumps(payload)}}]}))
    intent = extract_intent("How are fractions taught to a 9-year-old?")
    assert intent["question_type"] == "open"
    assert intent["goal"] is None
    assert intent["evidence"] == {"subject": "from 'fractions'", "grade": "most 9-year-olds"}
