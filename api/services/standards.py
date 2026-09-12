"""Standard collection queries."""

from typing import Any

from psycopg import Connection
from psycopg.rows import tuple_row

from api.models import AchievementLevel, Standard


def get_achievement_levels(
    conn: Connection[Any], framework_id: str, grade: int,
) -> list[AchievementLevel]:
    """Return grade descriptors for the framework's subject, ordered by level."""
    with conn.cursor(row_factory=tuple_row) as cursor:
        cursor.execute(
            """SELECT a.framework_id, a.grade, a.subject, a.level, a.text, a.source_url, a.page
               FROM achievement_descriptor a
               JOIN framework f ON f.id = a.framework_id AND f.subject = a.subject
               WHERE a.framework_id = %s AND a.grade = %s
               ORDER BY a.level""",
            (framework_id, grade),
        )
        return [AchievementLevel(*row) for row in cursor.fetchall()]


def get_standards(
    conn: Connection[Any], framework_id: str, grade: int, domain: str | None = None,
) -> list[Standard]:
    """Return matching standards ordered by code; no matches returns an empty list.

    The caller owns the connection and its transaction.
    """
    query = """WITH effective_standard AS (
                   SELECT v.framework_id, v.code, v.grade, v.domain, v.cluster,
                          v.text, v.source_url, v.page,
                          CASE WHEN v.text = s.text THEN s.plain_summary END AS plain_summary,
                          CASE WHEN v.text = s.text THEN s.plain_example END AS plain_example
                   FROM standard_variant v
                   JOIN standard s ON s.framework_id = v.framework_id AND s.code = v.code
                   UNION ALL
                   SELECT s.framework_id, s.code, s.grade, s.domain, s.cluster,
                          s.text, s.source_url, s.page, s.plain_summary, s.plain_example
                   FROM standard s WHERE NOT EXISTS (
                       SELECT 1 FROM standard_variant v
                       WHERE v.framework_id = s.framework_id AND v.code = s.code)
               )
               SELECT framework_id, code, grade, domain, cluster, text, source_url, page,
                      plain_summary, plain_example
               FROM effective_standard WHERE framework_id = %s AND grade = %s"""
    params: tuple[str | int, ...] = (framework_id, grade)
    if domain is not None:
        query += " AND domain = %s"
        params += (domain,)
    query += " ORDER BY code"
    with conn.cursor(row_factory=tuple_row) as cursor:
        cursor.execute(query, params)
        return [Standard(*row) for row in cursor.fetchall()]
