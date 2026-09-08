"""Execute service SQL on SQLite with placeholder adaptation, without a server.

This tests query behavior, not psycopg/Postgres integration.
"""

from dataclasses import FrozenInstanceError
import sqlite3
from typing import Any, Iterator, cast
from unittest.mock import MagicMock

from psycopg import Connection
import pytest

from api.db import SCHEMA
from api.models import AchievementLevel, ProgressionStep, Standard, UnknownStandardError
from api.services.progression import walk
from api.services.standards import get_achievement_levels, get_standards


@pytest.fixture
def conn() -> Iterator[Connection[Any]]:
    database = sqlite3.connect(":memory:")
    database.execute("PRAGMA foreign_keys = ON")
    for statement in SCHEMA:
        database.execute(statement)
    for framework in ("a", "b"):
        database.execute("INSERT INTO framework VALUES (?, 'California', 'math', 'sample', '')", (framework,))
    rows = [
        ("a", "3.NF.A.1", 3, "NF"), ("a", "4.NF.A.1", 4, "NF"),
        ("a", "5.NF.A.1", 5, "NF"), ("a", "3.OA.A.1", 3, "OA"),
        ("b", "3.NF.A.1", 3, "NF"), ("b", "4.NF.A.1", 4, "NF"),
        ("b", "5.NF.A.1", 5, "NF"), ("b", "only-b", 3, "NF"),
    ]
    for framework, code, grade, domain in rows:
        database.execute("INSERT INTO standard (framework_id, code, grade, domain, cluster, text, source_url, page) VALUES (?, ?, ?, ?, 'A', ?, '', NULL)",
                         (framework, code, grade, domain, f"{framework}:{code}"))
    database.executemany("INSERT INTO progression_edge VALUES (?, ?, ?, ?, ?)", [
        ("a", "3.NF.A.1", "a", "4.NF.A.1", "prerequisite"),
        ("a", "4.NF.A.1", "a", "5.NF.A.1", "prerequisite"),
        ("b", "3.NF.A.1", "b", "5.NF.A.1", "prerequisite"),
    ])
    raw_cursor = database.cursor()
    cursor = MagicMock()
    cursor.execute.side_effect = lambda query, params=(): raw_cursor.execute(query.replace("%s", "?"), params)
    cursor.fetchone.side_effect = raw_cursor.fetchone
    cursor.fetchall.side_effect = raw_cursor.fetchall
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    yield cast(Connection[Any], connection)
    database.close()


def test_standards_filter_framework_grade_and_domain(conn: Connection[Any]) -> None:
    result = get_standards(conn, "a", 3)
    assert [row.code for row in result] == ["3.NF.A.1", "3.OA.A.1"]
    assert all(row.framework_id == "a" and row.grade == 3 for row in result)
    assert [row.code for row in get_standards(conn, "a", 3, "NF")] == ["3.NF.A.1"]
    assert get_standards(conn, "a", 3, "missing") == []
    assert get_standards(conn, "a' OR 1=1 --", 3) == []


def test_forward(conn: Connection[Any]) -> None:
    steps = walk(conn, "a", "3.NF.A.1", "forward")
    assert [(s.standard.framework_id, s.standard.code, s.depth) for s in steps] == [("a", "4.NF.A.1", 1)]
    assert steps[0].relation == "prerequisite"
    assert steps[0].from_code == "3.NF.A.1"
    assert steps[0].to_code == "4.NF.A.1"


def test_backward(conn: Connection[Any]) -> None:
    steps = walk(conn, "a", "5.NF.A.1", "backward")
    assert [(s.standard.framework_id, s.standard.code, s.depth) for s in steps] == [("a", "4.NF.A.1", 1)]
    assert steps[0].from_code == "4.NF.A.1"
    assert steps[0].to_code == "5.NF.A.1"


def test_depth_two(conn: Connection[Any]) -> None:
    assert [(s.standard.code, s.depth) for s in walk(conn, "a", "3.NF.A.1", "forward", 2)] == [
        ("4.NF.A.1", 1), ("5.NF.A.1", 2)]
    assert [(s.standard.code, s.depth) for s in walk(conn, "a", "5.NF.A.1", "backward", 2)] == [
        ("4.NF.A.1", 1), ("3.NF.A.1", 2)]


@pytest.mark.parametrize("code", ["missing", "only-b", "' OR 1=1 --"])
def test_unknown_start_is_framework_scoped(conn: Connection[Any], code: str) -> None:
    with pytest.raises(UnknownStandardError) as exc:
        walk(conn, "a", code, "forward")
    assert exc.value.framework_id == "a"
    assert exc.value.code == code


def test_existing_isolated_standard(conn: Connection[Any]) -> None:
    assert walk(conn, "a", "3.OA.A.1", "forward") == []
    assert walk(conn, "a", "3.OA.A.1", "backward") == []


def test_depth_zero_and_invalid_arguments(conn: Connection[Any]) -> None:
    assert walk(conn, "a", "3.NF.A.1", "forward", 0) == []
    with pytest.raises(UnknownStandardError):
        walk(conn, "a", "missing", "forward", 0)
    with pytest.raises(ValueError):
        walk(conn, "a", "3.NF.A.1", "forward", -1)
    with pytest.raises(ValueError):
        walk(conn, "a", "3.NF.A.1", cast(Any, "sideways"))


def test_cross_framework_edges_and_cycles(conn: Connection[Any]) -> None:
    with conn.cursor() as cursor:
        cursor.execute("INSERT INTO progression_edge VALUES (%s, %s, %s, %s, %s)",
                       ("a", "4.NF.A.1", "b", "3.NF.A.1", "related"))
        cursor.execute("INSERT INTO progression_edge VALUES (%s, %s, %s, %s, %s)",
                       ("b", "5.NF.A.1", "a", "3.NF.A.1", "related"))
    steps = walk(conn, "a", "3.NF.A.1", "forward", 100)
    assert len(steps) == 5
    assert [(s.standard.framework_id, s.standard.code, s.depth) for s in steps] == [
        ("a", "4.NF.A.1", 1), ("a", "5.NF.A.1", 2),
        ("b", "3.NF.A.1", 2), ("b", "5.NF.A.1", 3), ("a", "3.NF.A.1", 4)]


def test_domain_models_are_frozen(conn: Connection[Any]) -> None:
    standard = get_standards(conn, "a", 3)[0]
    step = walk(conn, "a", "3.NF.A.1", "forward")[0]
    level = AchievementLevel("a", 3, "math", 1, "sample")
    for row, field in [(standard, "text"), (step, "relation"), (level, "text")]:
        with pytest.raises(FrozenInstanceError):
            setattr(row, field, "changed")
    assert isinstance(standard, Standard)
    assert isinstance(step, ProgressionStep)


def test_achievement_levels_scope_framework_grade_and_subject(conn: Connection[Any]) -> None:
    with conn.cursor() as cursor:
        for row in [("a", 3, "math", 2, "second"), ("a", 3, "math", 1, "first"),
                    ("b", 3, "math", 1, "other framework"),
                    ("a", 4, "math", 1, "other grade"),
                    ("a", 3, "reading", 1, "other subject")]:
            cursor.execute("INSERT INTO achievement_descriptor VALUES (%s, %s, %s, %s, %s)", row)
    assert get_achievement_levels(conn, "a", 3) == [
        AchievementLevel("a", 3, "math", 1, "first"),
        AchievementLevel("a", 3, "math", 2, "second"),
    ]
