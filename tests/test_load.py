from pathlib import Path
from unittest.mock import MagicMock

import pytest

from api import db, load


DATA = Path(__file__).resolve().parents[1] / "data"
FRAMEWORK = load.Framework("CA-CCSSM-2013", "California", "mathematics", "sample", "https://example.com")


def test_csvs_are_typed_and_endpoints_exist() -> None:
    standards = load.read_csv(DATA / "maths.csv", load.Standard)
    edges = load.read_csv(DATA / "progression.csv", load.ProgressionEdge)
    descriptors = load.read_csv(DATA / "ald.csv", load.AchievementDescriptor)
    assert standards
    assert edges and descriptors
    assert all(row.source_url for row in descriptors)
    endpoints = {(row.framework_id, row.code)
                 for filename, framework in load.SUBJECT_FILES.items()
                 for row in load.read_subject(DATA / filename, framework)[0]}
    for edge in edges:
        assert (edge.from_framework, edge.from_code) in endpoints
        assert (edge.to_framework, edge.to_code) in endpoints
    assert all(isinstance(row.grade, int) and row.page is None for row in standards)
    assert all(isinstance(row.level, int) for row in descriptors)


@pytest.mark.parametrize("content", [
    "wrong,headers\n1,2\n",
    "framework_id,grade,subject,level,text\nf,not-an-int,math,1,Sample\n",
    "framework_id,grade,subject,level,text\nf,3,math,1\n",
])
def test_invalid_csv_raises_with_path(tmp_path: Path, content: str) -> None:
    path = tmp_path / "ald.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match="ald.csv"):
        load.read_csv(path, load.AchievementDescriptor)


def test_get_conn_requires_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db, "_pool", None)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        db.get_conn()


def test_get_conn_reuses_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = MagicMock()
    monkeypatch.setattr(db, "_pool", None)
    monkeypatch.setattr(db, "ConnectionPool", factory)
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    assert db.get_conn() is factory.return_value.connection.return_value
    db.get_conn()
    factory.assert_called_once_with(
        conninfo="postgresql://localhost/test", min_size=1, max_size=10,
        timeout=60, reconnect_timeout=120, kwargs={"connect_timeout": 30},
        check=factory.check_connection, open=True,
    )
    assert factory.return_value.connection.call_count == 2
    db.close_pool()
    factory.return_value.close.assert_called_once()
    assert db._pool is None


@pytest.mark.parametrize("sslmode", ["", "?sslmode=prefer", "?sslmode=disable"])
def test_hosted_url_requires_tls(monkeypatch: pytest.MonkeyPatch, sslmode: str) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example.neon.tech/test" + sslmode)
    with pytest.raises(ValueError, match="sslmode=require"):
        db.database_url()


@pytest.mark.parametrize("sslmode", ["require", "verify-ca", "verify-full"])
def test_hosted_tls_modes(monkeypatch: pytest.MonkeyPatch, sslmode: str) -> None:
    url = f"postgresql://example.neon.tech/test?sslmode={sslmode}"
    monkeypatch.setenv("DATABASE_URL", url)
    assert db.database_url() == url


def test_loader_propagates_failure_to_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = MagicMock()
    conn.__enter__.return_value = conn
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.executemany.side_effect = RuntimeError("insert failed")
    monkeypatch.setattr(load, "get_conn", lambda: conn)
    with pytest.raises(RuntimeError, match="insert failed"):
        load.load_data(DATA)
    assert conn.__exit__.call_args.args[0] is RuntimeError


def test_schema_has_framework_scoped_keys() -> None:
    schema = "\n".join(db.SCHEMA)
    assert schema.count("CREATE TABLE IF NOT EXISTS") == 6
    assert "ON standard (framework_id, grade)" in schema
    assert "PRIMARY KEY (framework_id, code)" in schema
    assert schema.count("REFERENCES standard(framework_id, code)") == 3
    assert "PRIMARY KEY (framework_id, grade, subject, level)" in schema
    assert "PRIMARY KEY (from_framework, from_code, to_framework, to_code, relation)" in schema


@pytest.fixture
def loader_db(monkeypatch):
    """Execute real loader SQL with foreign keys and transaction rollback.

    SQLite adapts placeholders; PostgreSQL's nullable-grade migration is skipped.
    """
    import sqlite3
    from contextlib import contextmanager
    database = sqlite3.connect(":memory:")
    database.execute("PRAGMA foreign_keys = ON")
    for statement in db.SCHEMA:
        database.execute(statement)
    database.commit()
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    raw = database.cursor()
    statements = []

    def query_text(query):
        return query if isinstance(query, str) else query.as_string()

    def execute(query, params=()):
        query = query_text(query)
        if query.startswith("ALTER TABLE standard ALTER COLUMN") or "ADD COLUMN IF NOT EXISTS" in query:
            return
        return raw.execute(query.replace("%s", "?"), params)

    def executemany(query, rows):
        query = query_text(query)
        statements.append((query, rows))
        return raw.executemany(query.replace("%s", "?"), rows)

    cursor.execute.side_effect = execute
    cursor.executemany.side_effect = executemany
    cursor.fetchone.side_effect = raw.fetchone
    cursor.fetchall.side_effect = raw.fetchall

    @contextmanager
    def connection():
        with database:
            yield conn

    monkeypatch.setattr(load, "get_conn", connection)
    yield database, conn, statements
    database.close()


def test_all_subjects_load_and_repeat_without_changes(loader_db):
    import json
    from api.frameworks import FRAMEWORK_BY_SUBJECT, SUBJECT_FILES
    from api.tools.definitions import FRAMEWORK_BY_SUBJECT as routed
    from api.services.standards import get_standards
    database, conn, statements = loader_db
    report = load.load_data(DATA)
    assert routed is FRAMEWORK_BY_SUBJECT
    assert set(report.frameworks) == set(SUBJECT_FILES.values())
    assert report.skipped_files == report.rejected_rows == []
    expected = sum(len(load.read_subject(DATA / filename, framework)[1])
                   for filename, framework in SUBJECT_FILES.items())
    assert {c.table: c.rows for c in report.tables} == {
        'framework': 7, 'standard': expected - 12, 'standard_variant': expected,
        'progression_edge': len(load.read_csv(DATA / 'progression.csv', load.ProgressionEdge)), 'achievement_descriptor': len(load.read_csv(DATA / 'ald.csv', load.AchievementDescriptor)),
    }
    assert report.frameworks['CA-NGSS-2013']['standard'] == 78
    assert report.frameworks['CA-NGSS-2013']['standard_variant'] == 90
    assert statements[0][0].startswith('INSERT INTO "framework"')
    for query, rows in statements:
        assert 'ON CONFLICT DO NOTHING' in query and '%s' in query
        assert 'CA-' not in query
    for grade in (0, 1, 2):
        assert 'K-2-ETS1-1' in {s.code for s in get_standards(conn, 'CA-NGSS-2013', grade)}
    assert get_standards(conn, 'CA-VAPA-2019', -1)
    metadata = database.execute("SELECT metadata FROM standard_variant WHERE framework_id='CA-VAPA-2019' AND grade IS NULL").fetchall()
    assert len(metadata) == 62
    assert all(json.loads(row[0])['grade_range'] for row in metadata)
    database.execute("UPDATE standard SET text='existing'")
    database.execute("UPDATE standard_variant SET text='existing'")
    database.commit()
    assert load.load_data(DATA) == report
    assert database.execute("SELECT DISTINCT text FROM standard").fetchall() == [('existing',)]
    assert database.execute("SELECT DISTINCT text FROM standard_variant").fetchall() == [('existing',)]


@pytest.fixture
def partial_data(tmp_path):
    import shutil
    for filename in ['frameworks.csv', 'maths.csv', 'progression.csv', 'ald.csv']:
        shutil.copyfile(DATA / filename, tmp_path / filename)
    # This fixture deliberately loads only maths and three known sample links.
    # Keep it independent of additions to the reviewed cross-subject edge CSV.
    import csv
    with (tmp_path / 'progression.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['from_framework', 'from_code', 'to_framework', 'to_code', 'relation'])
        for source, target in [('3.NF.A.1', '4.NF.A.1'), ('4.NF.A.1', '5.NF.A.1'), ('3.NF.A.1', '5.NF.A.1')]:
            writer.writerow(['CA-CCSSM-2013', source, 'CA-CCSSM-2013', target, 'prerequisite'])
    (tmp_path / "ald.csv").write_text("framework_id,grade,subject,level,text,source_url,page\n")
    return tmp_path


@pytest.mark.parametrize('filename', list(load.SUBJECT_FILES)[1:])
def test_wrong_subject_framework_raises_before_writing(partial_data, loader_db, filename):
    import shutil
    # A copied maths template is the data-entry mistake this guards against.
    shutil.copyfile(DATA / 'maths.csv', partial_data / filename)
    with pytest.raises(ValueError) as exc:
        load.load_data(partial_data)
    message = str(exc.value)
    assert filename in message and 'row 2' in message
    assert load.SUBJECT_FILES[filename] in message and 'CA-CCSSM-2013' in message
    assert loader_db[0].execute('SELECT COUNT(*) FROM framework').fetchone()[0] == 0


def test_missing_files_and_dangling_references(partial_data, loader_db, capsys):
    with (partial_data / 'progression.csv').open('a') as f:
        f.write('unknown,no-code,CA-CCSSM-2013,missing,prerequisite\n')
        f.write('CA-CCSSM-2013,missing,CA-CCSSM-2013,also-missing,prerequisite\n')
    with (partial_data / 'ald.csv').open('a') as f:
        f.write('unknown,3,math,1,Some description,,\n')
    report = load.load_data(partial_data)
    assert set(report.skipped_files) == set(load.SUBJECT_FILES) - {'maths.csv'}
    assert len(report.rejected_rows) == 3
    assert 'progression.csv: row 5:' in report.rejected_rows[0]
    assert 'unknown framework' in report.rejected_rows[0]
    assert 'missing standard' in report.rejected_rows[1]
    assert 'ald.csv: row 2:' in report.rejected_rows[2]
    assert {c.table: c.rows for c in report.tables}['progression_edge'] == 3
    load.main(['--data-dir', str(partial_data)])
    output = capsys.readouterr().out
    assert 'Skipped missing subject CSV: ela.csv' in output
    assert 'Skipped dangling reference: progression.csv' in output
    assert 'CA-CCSSM-2013: standard=179' in output


def test_late_failure_rolls_back_every_insert(loader_db, partial_data):
    with (partial_data / "ald.csv").open("a") as handle:
        handle.write("CA-CCSSM-2013,4,mathematics,2,Test-only descriptor,,\n")
    database, conn, _ = loader_db
    cursor = conn.cursor.return_value.__enter__.return_value
    insert = cursor.executemany.side_effect

    def fail(query, rows):
        if 'achievement_descriptor' in query.as_string():
            raise RuntimeError('late failure')
        return insert(query, rows)

    cursor.executemany.side_effect = fail
    with pytest.raises(RuntimeError, match='late failure'):
        load.load_data(partial_data)
    for table in ['framework', 'standard', 'standard_variant', 'progression_edge', 'achievement_descriptor']:
        assert database.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 0


def test_framework_catalog_requires_all_subjects(partial_data, loader_db):
    (partial_data / 'frameworks.csv').write_text('id,state,subject,version,source_url\n')
    with pytest.raises(ValueError, match='frameworks.csv: missing frameworks'):
        load.load_data(partial_data)


def test_shared_files_can_reference_existing_database_records(partial_data, loader_db):
    database, _, _ = loader_db
    database.execute("INSERT INTO framework VALUES ('existing', 'California', 'math', 'v1', 'url')")
    database.execute("INSERT INTO standard (framework_id, code, grade, domain, cluster, text, source_url, page) VALUES ('existing', 'one', 1, 'D', 'C', 'Text', 'url', NULL)")
    database.commit()
    with (partial_data / 'progression.csv').open('a') as f:
        f.write('existing,one,CA-CCSSM-2013,K.CC.1,prerequisite\n')
    with (partial_data / 'ald.csv').open('a') as f:
        f.write('existing,1,math,1,Some description,,\n')
    report = load.load_data(partial_data)
    assert report.rejected_rows == []
    assert report.frameworks['existing']['outgoing_edges'] == 1
    assert report.frameworks['existing']['achievement_descriptor'] == 1


def test_descriptor_nullable_source_and_page(tmp_path):
    path = tmp_path / 'ald.csv'
    path.write_text('framework_id,grade,subject,level,text,source_url,page\n'
                    'f,4,math,1,Description,,\n'
                    'f,4,math,2,Description,https://example.org/ald.pdf,14\n')
    rows = load.read_csv(path, load.AchievementDescriptor)
    assert rows[0].source_url is None and rows[0].page is None
    assert rows[1].source_url == 'https://example.org/ald.pdf' and rows[1].page == 14


def test_loaded_descriptors_keep_source_in_guidance(loader_db):
    from dataclasses import asdict
    from api.services.guidance import get_guidance
    _, conn, _ = loader_db
    load.load_data(DATA)
    result = get_guidance(conn, 'CA-CCSSM-2013', 4)
    assert len(result.achievement_levels) == 4
    assert all(asdict(level)['source_url'] for level in result.achievement_levels)
    assert all(asdict(level)['page'] is None for level in result.achievement_levels)
