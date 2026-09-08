"""Zero-model-call baseline: fixed tool selection followed by Python rendering.

Unlike the agentic graph, neither routing nor answer generation uses an LLM;
this preserves the cost and latency baseline for the project's comparison.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from api.graph.state import GuideState
from api.tools.definitions import GOAL_TO_TOOL, TOOLS


def _select(state: GuideState) -> dict[str, Any]:
    goal = state["goal"]
    if goal not in GOAL_TO_TOOL:
        raise ValueError(f"Unknown goal {goal!r}; expected one of {', '.join(GOAL_TO_TOOL)}")
    selected = next(tool for tool in TOOLS if tool.name == GOAL_TO_TOOL[goal])
    args = {"subject": state["subject"], "grade": state["grade"]}
    if "domain" in selected.args:
        args["domain"] = state.get("domain")
    return {"tool_results": selected.invoke(args)}


def _render(state: GuideState) -> dict[str, str]:
    result = state["tool_results"]
    status = result.get("state", "available")
    if status == "not_published":
        return {"answer": result["reason"]}
    if status not in ("available", "redirect"):
        raise ValueError(f"Unknown tool result state {status!r}")
    lines = [result["reason"]] if status == "redirect" else []
    for program in result.get("programs", []):
        minimum = "K" if program["min_grade"] == 0 else str(program["min_grade"])
        lines.append(f"{program['name']} (grades {minimum}–{program['max_grade']}). {program.get('note', '')}".strip())
    for standard in result.get("standards", []):
        lines.append(f"{standard['code']}: {standard['text']}")
    for level in result.get("achievement_levels", []):
        lines.append(f"Achievement level {level['level']}: {level['text']}")
    for step in result.get("progression", []):
        standard = step["standard"]
        lines.append(f"{standard['code']} ({step['relation']}): {standard['text']}")
    return {"answer": "\n".join(lines) or "No matching entries were found in the loaded curriculum data."}


def build_deterministic_graph() -> CompiledStateGraph:
    """Compile START -> select -> render -> END, without a model dependency."""
    graph = StateGraph(GuideState)
    graph.add_node("select", _select)
    graph.add_node("render", _render)
    graph.add_edge(START, "select")
    graph.add_edge("select", "render")
    graph.add_edge("render", END)
    return graph.compile()
