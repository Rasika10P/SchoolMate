"""Shared eval input/output for the model-free baseline and tool-calling agent."""

from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class GuideState(TypedDict):
    subject: str | None
    state: NotRequired[str]
    grade: int | None
    domain: str | None
    goal: str | None
    question_type: NotRequired[str]
    question: NotRequired[str]
    activity_trace: NotRequired[list[dict[str, Any]]]
    tool_results: dict[str, Any]
    open_paragraphs: NotRequired[list[dict[str, Any]]]
    answer: str
    messages: Annotated[list[AnyMessage], add_messages]
