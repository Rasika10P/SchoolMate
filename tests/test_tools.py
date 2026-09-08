from dataclasses import asdict
from typing import Any
from unittest.mock import MagicMock

import pytest

from api.models import AchievementLevel, Standard, ProgressionStep
from api import capability
from api.services import guidance
from api.tools import definitions as tools


@pytest.mark.parametrize("name,direction", [
    ("on_grade_level", None), ("working_ahead", "forward"), ("catching_up", "backward"),
])
@pytest.mark.parametrize("subject,framework", [
    ("math", "CA-CCSSM-2013"), ("ela", "CA-CCSS-ELA-2013"),
    ("eld", "CA-ELD-2012"), ("sci", "CA-NGSS-2013"),
    ("hss", "CA-HSS-2016"), ("vapa", "CA-VAPA-2019"), ("pe", "CA-PE-2005"),
])
def test_tools_delegate_and_serialize(
    monkeypatch: pytest.MonkeyPatch, name: str, direction: str | None,
    subject: str, framework: str,
) -> None:
    monkeypatch.setenv("FRAMEWORK_ID", "ignored-old-setting")
    connection = MagicMock()
    monkeypatch.setattr(tools, "get_conn", lambda: connection)
    service = MagicMock(return_value=guidance.Guidance(framework, 3, (), (), ()))
    monkeypatch.setattr(tools, "get_guidance", service)
    result = getattr(tools, name).invoke({"subject": subject, "grade": 3, "domain": "NF"})
    tab = "standing" if direction is None else "next_steps"
    if not capability.supports(subject, tab, 3):
        assert result == {"state": "not_published", "reason": capability.NOT_PUBLISHED_REASON[subject, tab]}
        service.assert_not_called()
        connection.__enter__.assert_not_called()
        return
    service.assert_called_once_with(connection.__enter__.return_value, framework, 3, "NF", direction)
    assert result["standards"] == []
    assert result["framework_id"] == framework
    connection.__exit__.assert_called_once()


def test_catalog_is_independent_and_returns_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRAMEWORK_ID", raising=False)
    database = MagicMock(side_effect=AssertionError("unexpected database access"))
    monkeypatch.setattr(tools, "get_conn", database)
    for grade in (3, 4, 5):
        result = tools.competition_prep.invoke({"subject": "math", "grade": grade})
        assert result["state"] == "available"
        assert result["programs"] == [asdict(item) for item in tools.CATALOG["math"] if item.min_grade <= grade <= item.max_grade]
        result["programs"][0]["name"] = "changed"
        assert tools.competition_prep.invoke({"subject": "math", "grade": grade})["programs"][0]["name"] != "changed"


def test_registration_and_model_visible_arguments() -> None:
    assert len(tools.TOOLS) == 5
    assert set(tools.GOAL_TO_TOOL.values()) == {item.name for item in tools.TOOLS} - {"search_guidance"}
    for item in tools.TOOLS:
        assert item.description.count(".") == 1
        assert "child" in item.description
        if item.name == "search_guidance":
            assert set(item.args) == {"subject", "question", "grade"}
            continue
        assert set(item.args) == ({"subject", "grade"} if item.name == "competition_prep" else {"subject", "grade", "domain"})


def test_unknown_subject_fails_before_database_access(monkeypatch: pytest.MonkeyPatch) -> None:
    database = MagicMock(side_effect=AssertionError("unexpected database access"))
    monkeypatch.setattr(tools, "get_conn", database)
    for item in tools.TOOLS:
        with pytest.raises(ValueError, match="Unsupported subject"):
            item.invoke({"subject": "unknown", "grade": 3, "question": "How?"})
        with pytest.raises(ValueError, match="subject"):
            item.invoke({"grade": 3, "question": "How?"})
    database.assert_not_called()


@pytest.mark.parametrize("grade", [0, 3, 5])
def test_hss_competitions_are_not_published(grade: int) -> None:
    assert tools.competition_prep.invoke({"subject": "hss", "grade": grade}) == {
        "state": "not_published", "reason": capability.NOT_PUBLISHED_REASON["hss", "programs"]}


def test_eld_redirects_to_eligible_ela_programs() -> None:
    for grade in range(6):
        result = tools.competition_prep.invoke({"subject": "eld", "grade": grade})
        assert result["state"] == "redirect"
        assert result["reason"] == capability.NOT_PUBLISHED_REASON["eld", "programs"]
        assert result["programs"] == tools.competition_prep.invoke({"subject": "ela", "grade": grade})["programs"]


def test_grade_two_excludes_moems() -> None:
    programs = tools.competition_prep.invoke({"subject": "math", "grade": 2})["programs"]
    assert "MOEMS Division E" not in {p["name"] for p in programs}
    assert "Noetic Learning Math Contest" in {p["name"] for p in programs}


def test_empty_grade_catalog_has_central_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(tools.CATALOG, "math", (tools.Program("Older students", 6, 8),))
    assert tools.competition_prep.invoke({"subject": "math", "grade": 2}) == {
        "state": "not_published", "reason": capability.NOT_PUBLISHED_REASON["math", "programs"]}


@pytest.mark.parametrize("grade", [-1, 6])
def test_invalid_grade(grade: int) -> None:
    for item in tools.TOOLS:
        with pytest.raises(ValueError, match="grade"):
            item.invoke({"subject": "math", "grade": grade, "question": "How?"})


@pytest.mark.parametrize("direction", [None, "forward", "backward"])
def test_shared_service_uses_framework_scoped_roots(
    monkeypatch: pytest.MonkeyPatch, direction: Any,
) -> None:
    root = Standard("a", "3.NF.A.1", 3, "NF", "A", "sample", "", None)
    level = AchievementLevel("a", 3, "mathematics", 1, "sample")
    standards = MagicMock(return_value=[root])
    levels = MagicMock(return_value=[level])
    walk = MagicMock(return_value=[])
    monkeypatch.setattr(guidance.standards, "get_standards", standards)
    monkeypatch.setattr(guidance.standards, "get_achievement_levels", levels)
    monkeypatch.setattr(guidance.progression, "walk", walk)
    conn = MagicMock()
    result = guidance.get_guidance(conn, "a", 3, "NF", direction)
    standards.assert_called_once_with(conn, "a", 3, "NF")
    if direction is None:
        levels.assert_called_once_with(conn, "a", 3)
        assert result.achievement_levels == (level,)
        walk.assert_not_called()
    else:
        walk.assert_called_once_with(conn, "a", "3.NF.A.1", direction, depth=1)
        levels.assert_not_called()


def test_science_standing_only_queries_at_grade_five(monkeypatch: pytest.MonkeyPatch) -> None:
    database = MagicMock()
    service = MagicMock(return_value=guidance.Guidance("CA-NGSS-2013", 5, (), (), ()))
    monkeypatch.setattr(tools, "get_conn", database)
    monkeypatch.setattr(tools, "get_guidance", service)
    assert tools.on_grade_level.invoke({"subject": "sci", "grade": 3})["state"] == "not_published"
    database.assert_not_called()
    tools.on_grade_level.invoke({"subject": "sci", "grade": 5})
    database.assert_called_once()
    service.assert_called_once()


def test_kindergarten_and_short_previews(monkeypatch: pytest.MonkeyPatch) -> None:
    tools._validate_grade(0)
    text = " ".join(f"word{i}" for i in range(30))
    root = Standard("CA-CCSSM-2013", "K.CC.A.1", 0, "CC", "A", text, "source", 1)
    level = AchievementLevel(root.framework_id, 0, "math", 1, text)
    step = ProgressionStep(root, "prerequisite", 1, root.framework_id, root.code, root.framework_id, root.code)
    result = guidance.Guidance(root.framework_id, 0, (root,), (level,), (step,))
    monkeypatch.setattr(tools, "get_conn", MagicMock())
    monkeypatch.setattr(tools, "get_guidance", MagicMock(return_value=result))
    response = tools.on_grade_level.invoke({"subject": "math", "grade": 0})
    for record in (response["standards"][0], response["progression"][0]["standard"], response["achievement_levels"][0]):
        assert record["text"] == " ".join(text.split()[:20])
        assert record["full_text_available"] is True
        assert record["framework_id"] == root.framework_id
    assert response["standards"][0]["code"] == root.code
    assert root.text == text
    assert tools.competition_prep.invoke({"subject": "math", "grade": 0})["state"] == "available"
