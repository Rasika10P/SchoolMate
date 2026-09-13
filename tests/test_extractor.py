import json
from typing import Any
from unittest.mock import Mock

import pytest

from api import llm
from api.graph.extractor import extract_intent, infer_structured_goal, looks_structured, normalize_domain


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
    question = "  My child needs help with math and English fractions.\n"
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


@pytest.mark.parametrize('question', [
    'tell me how i should prepmy 2 nd grader for competition exams',
    'Help my child prepare for exams', 'What should a second grader learn?',
    'My child is behind', 'How can we practise at home?'])
def test_invented_math_subject_is_rejected_even_with_model_evidence(monkeypatch, question):
    payload = dict(subject='math', grade=2, goal='competition_prep', question_type='structured',
                   domain='arithmetic', evidence={'subject': 'competition exams mean maths'})
    monkeypatch.setattr(llm, 'cached_complete', Mock(return_value={
        'choices': [{'message': {'content': json.dumps(payload)}}]}))
    intent = extract_intent(question)
    assert intent['subject'] is None and intent['domain'] is None
    assert not intent['evidence'].get('subject')
    assert intent['grade'] == 2 and intent['goal'] == 'competition_prep'


@pytest.mark.parametrize('subject,question', [
    ('math', 'Help with fractions'), ('ela', 'Help with writting'),
    ('sci', 'How can we study sceince?'), ('hss', 'Help with history'),
    ('vapa', 'Prepare for music competitions'), ('pe', 'What physical education covers'),
    ('eld', 'My child receives English language support')])
def test_subject_cues_cover_each_subject(subject, question):
    from api.graph.extractor import subject_is_grounded
    assert subject_is_grounded(subject, question)


@pytest.mark.parametrize("question,goal", [
    ("Decimals are easy now; what math comes next?", "working_ahead"),
    ("Subtraction never clicked; what foundations should we revisit?", "catching_up"),
    ("What multiplication is expected in third grade?", "on_grade_level"),
    ("Where can we find a science contest?", "competition_prep"),
])
def test_structured_goal_fallback(question, goal):
    assert infer_structured_goal(question) == goal


@pytest.mark.parametrize("value,expected", [
    ("reading-comprehension skills", "reading comprehension"),
    ("  Life   Cycles  ", "life cycles"),
    (None, None),
])
def test_domain_normalization(value, expected):
    assert normalize_domain(value) == expected


@pytest.mark.parametrize("question", [
    "Which type of writting is taught for 2nd grader",
    "What science projects should parents do at home?",
    "My fifth grader is lost. What groundwork is missing?",
])
def test_structured_boundary_recovery(question):
    assert looks_structured(question)


@pytest.mark.parametrize("question", [
    "Why teach fractions before decimals?",
    "How should vocabulary be taught in context?",
    "What should count as evidence when children study shadows?",
])
def test_open_explanations_remain_open(question):
    assert not looks_structured(question)
