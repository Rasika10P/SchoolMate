"""Load curriculum CSVs atomically: python -m api.load --help."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

from psycopg import sql

from api.db import get_conn, init_schema
from api.frameworks import SUBJECT_FILES
from api.models import AchievementDescriptor, Standard


@dataclass(frozen=True)
class Framework:
    id: str
    state: str
    subject: str
    version: str
    source_url: str


@dataclass(frozen=True)
class ProgressionEdge:
    from_framework: str
    from_code: str
    to_framework: str
    to_code: str
    relation: str


@dataclass(frozen=True)
class TableCount:
    table: str
    rows: int


Row = TypeVar("Row", Framework, Standard, ProgressionEdge, AchievementDescriptor)


def read_csv(path: Path, row_type: type[Row]) -> list[Row]:
    """Validate exact column names and convert CSV records to typed objects."""
    expected = [field.name for field in fields(row_type)
                if field.name not in {"plain_summary", "plain_example"}]
    records: list[Row] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if (reader.fieldnames is None or len(reader.fieldnames) != len(expected)
                or set(reader.fieldnames) != set(expected)):
            raise ValueError(f"{path}: expected headers {','.join(expected)}")
        for raw in reader:
            try:
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError("wrong number of CSV columns")
                values: dict[str, Any] = dict(raw)
                for name in expected:
                    value = values[name]
                    if (name == "page" or (row_type is AchievementDescriptor and name == "source_url")) and not value.strip():
                        values[name] = None
                    elif not value.strip():
                        raise ValueError(f"{name} must not be empty")
                    elif name in {"grade", "page", "level"}:
                        values[name] = int(value)
                records.append(row_type(**values))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{reader.line_num}: {exc}") from exc
    return records


@dataclass(frozen=True)
class StandardVariant:
    framework_id: str
    code: str
    grade_label: str
    grade: int | None
    domain: str
    cluster: str
    text: str
    source_url: str
    page: int | None
    metadata: str


@dataclass
class LoadReport:
    tables: list[TableCount]
    frameworks: dict[str, dict[str, int]]
    skipped_files: list[str]
    rejected_rows: list[str]


def read_subject(path: Path, expected_id: str) -> tuple[list[Standard], list[StandardVariant]]:
    """Project subject fields while retaining every original field as metadata.

    CSV row numbers count records, including the header, not embedded newlines.
    TK is -1, K is 0, and proficiency ranges have no numeric grade.
    """
    standards, variants = [], []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"framework_id", "code", "grade", "source_url", "page"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing required headers {sorted(required)}")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"{path}: duplicate headers")
        for number, raw in enumerate(reader, 2):
            found = raw.get("framework_id")
            if found != expected_id:
                raise ValueError(f"{path}: row {number}: expected framework_id {expected_id!r}, found {found!r}")
            try:
                if None in raw or any(v is None for v in raw.values()):
                    raise ValueError("wrong number of CSV columns")
                label = raw["grade"].strip() or raw.get("grade_range", "").strip()
                if not label:
                    raise ValueError("grade or grade_range is required")
                grade = ({"TK": -1, "K": 0}[label] if label in {"TK", "K"}
                         else int(label) if raw["grade"].strip() else None)
                domain = raw.get("domain", raw.get("critical_principle", raw.get("content_area",
                         raw.get("course", raw.get("art_type", "Physical Education")))))
                cluster = raw.get("cluster", raw.get("disciplinary_core_idea",
                          raw.get("anchor_standard", raw.get("overarching_standard", ""))))
                text = raw.get("text", raw.get("performance_expectation", raw.get("performance_standards", "")))
                if not raw["code"].strip() or not text.strip():
                    raise ValueError("code and standard text are required")
                page = int(raw["page"]) if raw["page"].strip() else None
                standard = Standard(expected_id, raw["code"], grade, domain, cluster, text, raw["source_url"], page)
                standards.append(standard)
                variants.append(StandardVariant(expected_id, standard.code, label, grade, domain,
                                cluster, text, standard.source_url, page, json.dumps(raw, ensure_ascii=False)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}: row {number}: {exc}") from exc
    return standards, variants


def _insert(cursor: Any, table: str, row_type: type, rows: list) -> None:
    columns = [field.name for field in fields(row_type)]
    statement = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING").format(
        sql.Identifier(table), sql.SQL(", ").join(map(sql.Identifier, columns)),
        sql.SQL(", ").join(sql.Placeholder() for _ in columns))
    if rows:
        cursor.executemany(statement, [astuple(row) for row in rows])


def load_data(data_dir: Path) -> LoadReport:
    """Load all subjects atomically; counts are database totals after conflicts.

    Invalid subject IDs abort the load. Dangling shared-file records are reported
    and skipped. Existing rows retain their values on conflict.
    """
    frameworks = read_csv(data_dir / "frameworks.csv", Framework)
    ids = [row.id for row in frameworks]
    missing = set(SUBJECT_FILES.values()) - set(ids)
    if missing or len(ids) != len(set(ids)):
        raise ValueError(f"{data_dir / 'frameworks.csv'}: missing frameworks {sorted(missing)} or duplicate IDs")
    standards, variants, skipped, rejected = [], [], [], []
    for filename, expected_id in SUBJECT_FILES.items():
        path = data_dir / filename
        if not path.exists():
            skipped.append(filename)
            continue
        rows, details = read_subject(path, expected_id)
        standards.extend(rows)
        variants.extend(details)
    edges = read_csv(data_dir / "progression.csv", ProgressionEdge)
    descriptors = read_csv(data_dir / "ald.csv", AchievementDescriptor)
    tables = ["framework", "standard", "standard_variant", "progression_edge", "achievement_descriptor"]
    counts, by_framework = [], {}
    with get_conn() as conn:
        init_schema(conn)
        with conn.cursor() as cursor:
            _insert(cursor, "framework", Framework, frameworks)
            cursor.execute("SELECT id FROM framework")
            known_frameworks = {row[0] for row in cursor.fetchall()}
            _insert(cursor, "standard", Standard, standards)
            _insert(cursor, "standard_variant", StandardVariant, variants)
            cursor.execute("SELECT framework_id, code FROM standard")
            endpoints = set(cursor.fetchall())
            valid_edges = []
            for number, edge in enumerate(edges, 2):
                errors = []
                for framework_id, code in [(edge.from_framework, edge.from_code), (edge.to_framework, edge.to_code)]:
                    if framework_id not in known_frameworks:
                        errors.append(f"unknown framework {framework_id!r}")
                    if (framework_id, code) not in endpoints:
                        errors.append(f"missing standard ({framework_id!r}, {code!r})")
                if errors:
                    rejected.append(f"progression.csv: row {number}: " + "; ".join(errors))
                else:
                    valid_edges.append(edge)
            valid_descriptors = []
            for number, row in enumerate(descriptors, 2):
                if row.framework_id not in known_frameworks:
                    rejected.append(f"ald.csv: row {number}: unknown framework {row.framework_id!r}")
                else:
                    valid_descriptors.append(row)
            _insert(cursor, "progression_edge", ProgressionEdge, valid_edges)
            _insert(cursor, "achievement_descriptor", AchievementDescriptor, valid_descriptors)
            for table in tables:
                cursor.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table)))
                counts.append(TableCount(table, cursor.fetchone()[0]))
            for framework_id in sorted(known_frameworks):
                totals = {}
                for table, column, label in [
                    ("standard", "framework_id", "standard"),
                    ("standard_variant", "framework_id", "standard_variant"),
                    ("achievement_descriptor", "framework_id", "achievement_descriptor"),
                    ("progression_edge", "from_framework", "outgoing_edges"),
                    ("progression_edge", "to_framework", "incoming_edges"),
                ]:
                    cursor.execute(sql.SQL("SELECT COUNT(*) FROM {} WHERE {} = %s").format(
                        sql.Identifier(table), sql.Identifier(column)), (framework_id,))
                    totals[label] = cursor.fetchone()[0]
                by_framework[framework_id] = totals
    return LoadReport(counts, by_framework, skipped, rejected)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args(argv)
    report = load_data(args.data_dir)
    for filename in report.skipped_files:
        print(f"Skipped missing subject CSV: {filename}")
    for issue in report.rejected_rows:
        print(f"Skipped dangling reference: {issue}")
    print("Database totals (existing conflicts preserved):")
    for count in report.tables:
        print(f"{count.table}: {count.rows} rows")
    for framework_id, counts in report.frameworks.items():
        print(f"{framework_id}: " + ", ".join(f"{table}={count}" for table, count in counts.items()))


if __name__ == "__main__":
    main()
