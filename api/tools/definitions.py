"""LangGraph-compatible tools; set DATABASE_URL before use.

Pass the intent object's subject to each curriculum tool to select its framework.
The selected framework and its data must already be loaded in the database.

Domain objects remain dataclasses in services and become JSON-compatible objects
at this tool boundary, with short previews for model context.
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal

from langchain_core.tools import BaseTool, tool

from api.db import get_conn
from api import capability
from api.services.guidance import get_guidance
from api.services import guidance
from api.frameworks import FRAMEWORK_BY_SUBJECT


@dataclass(frozen=True)
class Program:
    name: str
    min_grade: int
    max_grade: int
    note: str = ""


CATALOG: dict[str, tuple[Program, ...]] = {
    "math": (
        Program("Math Kangaroo", 1, 12),
        Program("MOEMS Division E", 4, 6),
        Program("Noetic Learning Math Contest", 2, 8),
        Program("Continental Mathematics League", 2, 6),
        Program("AMC 8", 0, 8, "Participation is rare below grade 6."),
    ),
    "ela": (
        Program("Scripps National Spelling Bee", 0, 8, "Entry is through school and regional bees."),
        Program("WordMasters Challenge", 3, 8),
    ),
    "sci": (
        Program("Science Olympiad Division A", 0, 6),
        Program("Toshiba/NSTA ExploraVision", 0, 12, "Separate K-3 and 4-6 divisions."),
    ),
    "vapa": (Program("National PTA Reflections", 0, 12), Program("Doodle for Google", 0, 12)),
    "pe": (Program("Kids Heart Challenge", 0, 8, "A fitness event rather than an academic competition."),),
    "hss": (),
    "eld": (),
}


def _validate_grade(grade: int) -> None:
    if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 5:
        raise ValueError("grade must be an integer from 0 (kindergarten) to 5")


def _preview(record: dict[str, Any]) -> dict[str, Any]:
    """Keep hydration identifiers and replace full text with at most 20 words."""
    record["text"] = " ".join(record["text"].split()[:20])
    record["full_text_available"] = True
    return record


def _curriculum(
    subject: str, grade: int, domain: str | None,
    direction: Literal["forward", "backward"] | None,
) -> dict[str, Any]:
    _validate_grade(grade)
    tab = "standing" if direction is None else "next_steps"
    if not capability.supports(subject, tab, grade):
        return {"state": "not_published", "reason": capability.NOT_PUBLISHED_REASON[subject, tab]}
    try:
        framework_id = FRAMEWORK_BY_SUBJECT[subject]
    except KeyError:
        raise ValueError(f"Unsupported subject {subject!r}; expected one of {', '.join(FRAMEWORK_BY_SUBJECT)}") from None
    with get_conn() as conn:
        result = get_guidance(conn, framework_id, grade, domain, direction)
    payload = asdict(result)
    payload["standards"] = [_preview(record) for record in payload["standards"]]
    payload["achievement_levels"] = [_preview(record) for record in payload["achievement_levels"]]
    payload["progression"] = [
        {**step, "standard": _preview(step["standard"])} for step in payload["progression"]
    ]
    return payload


@tool
def on_grade_level(subject: str, grade: int, domain: str | None = None) -> dict[str, Any]:
    """Help me understand what my child should learn in their current grade and what different achievement levels look like."""
    return _curriculum(subject, grade, domain, None)


@tool
def working_ahead(subject: str, grade: int, domain: str | None = None) -> dict[str, Any]:
    """Help me find the next skills for my child who is ready to move beyond their current grade's learning."""
    return _curriculum(subject, grade, domain, "forward")


@tool
def catching_up(subject: str, grade: int, domain: str | None = None) -> dict[str, Any]:
    """Help me find earlier skills my child can revisit to catch up with their current grade's learning."""
    return _curriculum(subject, grade, domain, "backward")


@tool
def competition_prep(subject: str, grade: int) -> dict[str, Any]:
    """Help me find competition preparation for my child's subject and grade, or understand when it is not published."""
    _validate_grade(grade)
    published = capability.supports(subject, "programs", grade)
    target = "ela" if subject == "eld" else subject
    if not published and target == subject:
        return {"state": "not_published", "reason": capability.NOT_PUBLISHED_REASON[subject, "programs"]}
    programs = [asdict(program) for program in CATALOG[target]
                if program.min_grade <= grade <= program.max_grade]
    if not programs:
        return {"state": "not_published", "reason": capability.NOT_PUBLISHED_REASON[target, "programs"]}
    if target != subject:
        return {"state": "redirect", "reason": capability.NOT_PUBLISHED_REASON[subject, "programs"],
                "programs": programs}
    return {"state": "available", "programs": programs}


@tool
def search_guidance(subject: str, question: str, grade: int | None = None) -> dict[str, Any]:
    """Answer my open-ended questions about how my child's subject is taught, using teaching guidance that a standards list cannot provide."""
    if subject not in FRAMEWORK_BY_SUBJECT:
        raise ValueError(f"Unsupported subject {subject!r}")
    if grade is not None:
        _validate_grade(grade)
    return guidance.search(question, FRAMEWORK_BY_SUBJECT[subject], grade=grade)


TOOLS: list[BaseTool] = [on_grade_level, working_ahead, catching_up, competition_prep, search_guidance]
GOAL_TO_TOOL: dict[str, str] = {
    "on_grade_level": on_grade_level.name,
    "working_ahead": working_ahead.name,
    "catching_up": catching_up.name,
    "competition_prep": competition_prep.name,
}
