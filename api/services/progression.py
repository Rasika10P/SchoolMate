"""Bounded, framework-scoped progression traversal."""

from collections import deque
from typing import Any, Literal

from psycopg import Connection
from psycopg.rows import tuple_row

from api.models import ProgressionStep, Standard, UnknownStandardError


_FORWARD = """
    SELECT s.framework_id, s.code, s.grade, s.domain, s.cluster, s.text,
           s.source_url, s.page, s.plain_summary, s.plain_example, e.relation,
           e.from_framework, e.from_code, e.to_framework, e.to_code
    FROM progression_edge e
    JOIN standard s ON s.framework_id = e.to_framework AND s.code = e.to_code
    WHERE e.from_framework = %s AND e.from_code = %s
    ORDER BY e.to_framework, e.to_code, e.relation
"""
_BACKWARD = """
    SELECT s.framework_id, s.code, s.grade, s.domain, s.cluster, s.text,
           s.source_url, s.page, s.plain_summary, s.plain_example, e.relation,
           e.from_framework, e.from_code, e.to_framework, e.to_code
    FROM progression_edge e
    JOIN standard s ON s.framework_id = e.from_framework AND s.code = e.from_code
    WHERE e.to_framework = %s AND e.to_code = %s
    ORDER BY e.from_framework, e.from_code, e.relation
"""


def walk(
    conn: Connection[Any], framework_id: str, code: str,
    direction: Literal["forward", "backward"], depth: int = 1,
) -> list[ProgressionStep]:
    """Return edges in breadth-first order, up to depth hops (zero returns []).

    Expand each (framework, code) once at its shortest distance. Each reachable
    edge is returned once, including cycle-closing edges and distinct relations.
    Explicit cross-framework edges are followed. A missing start always raises,
    even at depth zero. No transaction is committed or connection closed here.
    """
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be 'forward' or 'backward'")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError("depth must be a non-negative integer")
    query = _FORWARD if direction == "forward" else _BACKWARD
    steps: list[ProgressionStep] = []
    pending = deque([(framework_id, code, 0)])
    visited = {(framework_id, code)}
    with conn.cursor(row_factory=tuple_row) as cursor:
        cursor.execute(
            "SELECT 1 FROM standard WHERE framework_id = %s AND code = %s",
            (framework_id, code),
        )
        if cursor.fetchone() is None:
            raise UnknownStandardError(framework_id, code)
        while pending:
            current_framework, current_code, hop = pending.popleft()
            if hop >= depth:
                continue
            cursor.execute(query, (current_framework, current_code))
            for row in cursor.fetchall():
                standard = Standard(*row[:10])
                steps.append(ProgressionStep(standard, row[10], hop + 1, *row[11:15]))
                key = (standard.framework_id, standard.code)
                if key not in visited:
                    visited.add(key)
                    pending.append((*key, hop + 1))
    return steps
