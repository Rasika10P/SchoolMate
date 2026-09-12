"""Postgres connections and framework-scoped schema."""

from __future__ import annotations

import os
from pathlib import Path
import atexit
from contextlib import AbstractContextManager
from threading import Lock
from typing import Any

import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg_pool import ConnectionPool


_pool: ConnectionPool | None = None
_pool_lock = Lock()


def database_url() -> str:
    """Validate deployment configuration before starting background connections."""
    url = os.environ.get("DATABASE_URL")
    if not url or not url.strip():
        raise RuntimeError("DATABASE_URL must be set")
    options = conninfo_to_dict(url)
    hosts = options.get("host", "").split(",")
    hosted = any(host and host not in {"localhost", "127.0.0.1", "::1"} and not host.startswith("/") for host in hosts)
    if hosted and options.get("sslmode") not in {"require", "verify-ca", "verify-full"}:
        raise ValueError("Hosted DATABASE_URL requires sslmode=require (or verify-ca/verify-full); add it to the URL before connecting.")
    return url


def close_pool() -> None:
    """Release connections on application shutdown; the next request can reopen."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


atexit.register(close_pool)


def get_conn() -> AbstractContextManager[psycopg.Connection[tuple[Any, ...]]]:
    """Borrow a connection; context exit commits/rolls back and returns it.

    Initialize once on first use, not import, so UI reruns and unit tests do not
    start connections. Restart the process after changing DATABASE_URL.
    """
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ConnectionPool(
                conninfo=database_url(), min_size=1, max_size=10,
                timeout=60, reconnect_timeout=120,
                kwargs={"connect_timeout": 30},
                check=ConnectionPool.check_connection,
                open=True,
            )
        return _pool.connection()


SCHEMA = (
    """CREATE TABLE IF NOT EXISTS framework (
        id TEXT PRIMARY KEY,
        state TEXT NOT NULL,
        subject TEXT NOT NULL,
        version TEXT NOT NULL,
        source_url TEXT NOT NULL
    )""",
    (Path(__file__).with_name("migrations") / "002_grade_overview.sql").read_text(),
    """CREATE TABLE IF NOT EXISTS standard (
        framework_id TEXT NOT NULL REFERENCES framework(id),
        code TEXT NOT NULL,
        grade INT,
        domain TEXT NOT NULL,
        cluster TEXT NOT NULL,
        text TEXT NOT NULL,
        source_url TEXT NOT NULL,
        page INT,
        plain_summary TEXT,
        plain_example TEXT,
        PRIMARY KEY (framework_id, code)
    )""",
    """CREATE INDEX IF NOT EXISTS standard_framework_grade_idx
        ON standard (framework_id, grade)""",
    """CREATE TABLE IF NOT EXISTS standard_variant (
        framework_id TEXT NOT NULL,
        code TEXT NOT NULL,
        grade_label TEXT NOT NULL,
        grade INT,
        domain TEXT NOT NULL,
        cluster TEXT NOT NULL,
        text TEXT NOT NULL,
        source_url TEXT NOT NULL,
        page INT,
        metadata TEXT NOT NULL,
        PRIMARY KEY (framework_id, code, grade_label),
        FOREIGN KEY (framework_id, code) REFERENCES standard(framework_id, code)
    )""",
    """CREATE INDEX IF NOT EXISTS standard_variant_framework_grade_idx
        ON standard_variant (framework_id, grade)""",
    """CREATE TABLE IF NOT EXISTS progression_edge (
        from_framework TEXT NOT NULL,
        from_code TEXT NOT NULL,
        to_framework TEXT NOT NULL,
        to_code TEXT NOT NULL,
        relation TEXT NOT NULL,
        PRIMARY KEY (from_framework, from_code, to_framework, to_code, relation),
        FOREIGN KEY (from_framework, from_code) REFERENCES standard(framework_id, code),
        FOREIGN KEY (to_framework, to_code) REFERENCES standard(framework_id, code)
    )""",
    """CREATE TABLE IF NOT EXISTS achievement_descriptor (
        framework_id TEXT NOT NULL REFERENCES framework(id),
        grade INT NOT NULL,
        subject TEXT NOT NULL,
        level INT NOT NULL,
        text TEXT NOT NULL,
        source_url TEXT,
        page INT,
        PRIMARY KEY (framework_id, grade, subject, level)
    )""",
)


def init_schema(conn: psycopg.Connection[tuple[Any, ...]] | None = None) -> None:
    """Create missing tables/index; an optional caller connection stays open.

    Standard grades allow NULL for proficiency-based arts records. With a caller connection, schema creation
    participates in its transaction; otherwise it commits and returns to the pool.
    """
    if conn is None:
        with get_conn() as owned:
            init_schema(owned)
        return
    with conn.cursor() as cursor:
        for statement in SCHEMA:
            cursor.execute(statement)
        cursor.execute("ALTER TABLE standard ALTER COLUMN grade DROP NOT NULL")
    from api.migrate import migrate
    migrate(conn)
