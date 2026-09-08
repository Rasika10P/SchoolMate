"""Shared eval input/output for the model-free baseline and tool-calling agent."""

from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class GuideState(TypedDict):
    subject: str
    grade: int | None
    domain: str | None
    goal: str | None
    question_type: NotRequired[str]
    question: NotRequired[str]
    tool_results: dict[str, Any]
    answer: str
    messages: Annotated[list[AnyMessage], add_messages]
