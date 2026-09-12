import json
from typing import Any
from unittest.mock import Mock

import pytest
from langgraph.errors import GraphRecursionError

from api import llm
from api.capability import NOT_PUBLISHED_REASON
from api.graph.agentic import build_agent_graph
from api.graph.deterministic import build_deterministic_graph
from api.graph.state import GuideState


def state(subject: str = "math", goal: str = "competition_prep") -> GuideState:
    return GuideState(subject=subject, grade=4, domain=None, goal=goal,
                      tool_results={}, answer="", messages=[])


@pytest.fixture(autouse=True)
def isolated_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    llm.reset_usage()
    monkeypatch.setenv("LLM_CACHE", "off")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(llm, "PRICES", {"test-model": {"input": None, "output": None}})


def test_deterministic_answer_without_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Mock(side_effect=AssertionError("Baseline must never call a model"))
    monkeypatch.setattr(llm, "cached_complete", model)
    result = build_deterministic_graph().invoke(state())
    assert "Math Kangaroo" in result["answer"]
    assert "MOEMS Division E" in result["answer"]
    model.assert_not_called()
    assert sum(item.calls for item in llm.usage.models.values()) == 0
    assert sum(item.cache_hits for item in llm.usage.models.values()) == 0


def test_unknown_goal_raises() -> None:
    with pytest.raises(ValueError, match="Unknown goal 'nonsense'"):
        build_deterministic_graph().invoke(state(goal="nonsense"))


@pytest.mark.parametrize("goal,tab", [("competition_prep", "programs"), ("on_grade_level", "standing")])
def test_not_published_renders_reason(goal: str, tab: str) -> None:
    result = build_deterministic_graph().invoke(state("hss", goal))
    assert result["answer"] == NOT_PUBLISHED_REASON["hss", tab]


def test_redirect_renders_reason_and_programs() -> None:
    result = build_deterministic_graph().invoke(state("eld"))
    assert NOT_PUBLISHED_REASON["eld", "programs"] in result["answer"]
    assert "WordMasters Challenge" in result["answer"]


def response(content: str | None = None, call_id: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if call_id:
        message["tool_calls"] = [{"id": call_id, "type": "function", "function": {
            "name": "competition_prep", "arguments": json.dumps({"subject": "math", "grade": 4})}}]
    return {"choices": [{"message": message}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def test_agent_terminates_within_recursion_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = Mock(side_effect=[response(call_id="call-1"), response("Math Kangaroo is an option for grade 4.")])
    monkeypatch.setattr(llm, "_provider_complete", provider)
    graph = build_agent_graph()
    assert graph.config["recursion_limit"] == 6
    result = graph.invoke(state())
    assert result["answer"] == "Math Kangaroo is an option for grade 4."
    assert result["tool_results"]["call-1"]["state"] == "available"
    assert len(result["messages"]) == 3
    assert llm.usage.models["test-model"].calls == 2
    request = provider.call_args_list[1].kwargs
    assert len(request["tools"]) == 5
    tool_message = next(message for message in request["messages"] if message["role"] == "tool")
    assert tool_message["tool_call_id"] == "call-1"
    assert "Math Kangaroo" in tool_message["content"]


def test_agent_runaway_stops_at_recursion_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = Mock(side_effect=[response(call_id=f"call-{i}") for i in range(10)])
    monkeypatch.setattr(llm, "_provider_complete", provider)
    with pytest.raises(GraphRecursionError):
        build_agent_graph().invoke(state())
    assert provider.call_count <= 3


@pytest.mark.parametrize("grade", [None, 0])
def test_open_question_retrieves_prose_with_original_question(monkeypatch, grade):
    from api.services import guidance
    search = Mock(return_value={"state": "available", "chunks": [{"text": "Use objects.", "metadata": {"page": 3}}]})
    monkeypatch.setattr(guidance, "search", search)
    model = Mock(return_value=response("Children explore with objects."))
    monkeypatch.setattr(llm, "cached_complete", model)
    question = "How is mathematics taught?"
    result = build_agent_graph().invoke({**state(), "question_type": "open", "question": question, "grade": grade, "goal": None})
    search.assert_called_once_with(question, "CA-CCSSM-2013", grade=grade)
    assert result["answer"] == "Children explore with objects."
    assert result["tool_results"]["guidance"]["chunks"]
    assert model.call_args.kwargs["tools"] is None
    assert question in str(model.call_args.kwargs["messages"])


@pytest.mark.parametrize("status", ["unavailable", "no_relevant_content", "judgement_unavailable"])
def test_open_unavailable_does_not_invent_answer(monkeypatch, status):
    from api.services import guidance
    monkeypatch.setattr(guidance, "search", Mock(return_value={"state": status, "reason": "No prose yet."}))
    model = Mock(side_effect=AssertionError("No unsupported synthesis"))
    monkeypatch.setattr(llm, "cached_complete", model)
    result = build_agent_graph().invoke({**state(), "question_type": "open", "question": "Why?", "grade": None, "goal": None})
    assert result["answer"] == "No prose yet."
    model.assert_not_called()



@pytest.mark.parametrize("subject", [None, "general"])
@pytest.mark.parametrize("question_type", ["open", "structured"])
def test_missing_or_invented_subject_asks_only_subject(monkeypatch, subject, question_type):
    forbidden = Mock(side_effect=AssertionError("No model or tool without a subject"))
    monkeypatch.setattr(llm, "cached_complete", forbidden)
    from api.services import guidance
    monkeypatch.setattr(guidance, "search", forbidden)
    result = build_agent_graph().invoke({"subject": subject, "grade": 4, "goal": None,
        "question_type": question_type, "messages": [{"role": "user", "content":
        "Here are the grades my child received in grade 4, tell me what areas I need to improve"}]})
    assert result["state"] == "need_subject"
    assert result["answer"] == "Which subject did you have in mind?"
    assert result["tool_results"]["state"] == "need_subject"
    forbidden.assert_not_called()


def test_agent_invented_tool_subject_does_not_crash(monkeypatch):
    reply = response(call_id="invalid")
    reply["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps({"subject": "general", "grade": 4})
    model = Mock(return_value=reply)
    monkeypatch.setattr(llm, "cached_complete", model)
    result = build_agent_graph().invoke(state())
    assert result["state"] == "need_subject"
    assert result["answer"] == "Which subject did you have in mind?"
    model.assert_called_once()


def test_agent_prompt_prohibits_requesting_child_data(monkeypatch):
    from api.graph.agentic import SYSTEM_PROMPT
    for phrase in ("Never ask for a child's grades, test scores, report cards, teacher feedback",
                   "any assessment of the child, in any circumstance",
                   "without\nstoring or echoing it", "Do not assess the child",
                   "Never invent a subject", "This system describes what California publishes"):
        assert phrase in SYSTEM_PROMPT
    model = Mock(return_value=response("Grade 4 covers these curriculum areas."))
    monkeypatch.setattr(llm, "cached_complete", model)
    build_agent_graph().invoke(state())
    assert model.call_args.kwargs["messages"][0]["content"] == SYSTEM_PROMPT
