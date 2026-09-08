from contextlib import contextmanager
import sqlite3
from threading import Barrier, Lock
from unittest.mock import MagicMock

import pytest

from api import db, migrate
from api.services.standards import get_standards
from api.services.progression import walk
from scripts import generate_plain_language as batch


@pytest.fixture
def database(monkeypatch):
    raw = sqlite3.connect(':memory:')
    raw.execute('PRAGMA foreign_keys = ON')
    for statement in db.SCHEMA:
        raw.execute(statement)
    raw.executemany('INSERT INTO framework VALUES (?, ?, ?, ?, ?)', [
        ('f', 'CA', 'math', '1', 'url'), ('other', 'CA', 'math', '1', 'url')])
    for framework, code, grade in [('f', 'one', 0), ('f', 'two', 1), ('other', 'one', 0)]:
        raw.execute('''INSERT INTO standard
            (framework_id, code, grade, domain, cluster, text, source_url, page)
            VALUES (?, ?, ?, 'D', 'C', 'Official text', 'url', 1)''', (framework, code, grade))
    raw.commit()
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    sql_cursor = raw.cursor()

    def execute(query, params=()):
        sql_cursor.execute(query.replace('%s', '?'), params)
        cursor.rowcount = sql_cursor.rowcount

    cursor.execute.side_effect = execute
    cursor.fetchall.side_effect = sql_cursor.fetchall
    cursor.fetchone.side_effect = sql_cursor.fetchone

    @contextmanager
    def connect():
        with raw:
            yield conn

    monkeypatch.setattr(batch, 'get_conn', connect)
    yield raw, conn
    raw.close()


def test_filters_write_resume_and_services(database, monkeypatch):
    raw, conn = database
    translate = MagicMock(return_value={'summary': 'Count objects.', 'example': 'Count spoons.'})
    monkeypatch.setattr(batch, 'to_plain_language', translate)
    assert len(batch.find_pending()) == 3
    rows = batch.find_pending('f', 0)
    assert len(rows) == 1
    assert batch.generate_one(rows[0]) == 'saved'
    assert batch.find_pending('f', 0) == []
    saved = get_standards(conn, 'f', 0)[0]
    assert saved.plain_summary == 'Count objects.'
    assert saved.plain_example == 'Count spoons.'
    translate.return_value = {'summary': 'Do not overwrite.', 'example': 'New example.'}
    assert batch.generate_one(rows[0]).startswith('skipped')
    assert get_standards(conn, 'f', 0)[0].plain_summary == 'Count objects.'
    raw.execute("INSERT INTO progression_edge VALUES ('f', 'two', 'f', 'one', 'prerequisite')")
    assert walk(conn, 'f', 'two', 'forward')[0].standard.plain_summary == 'Count objects.'


def test_changed_source_and_failed_translation_stay_pending(database, monkeypatch):
    raw, _ = database
    row = batch.find_pending('f', 0)[0]
    monkeypatch.setattr(batch, 'to_plain_language', MagicMock(side_effect=ValueError('rejected')))
    with pytest.raises(ValueError):
        batch.generate_one(row)
    assert batch.find_pending('f', 0) == [row]
    raw.execute("UPDATE standard SET text='Changed' WHERE framework_id='f' AND code='one'")
    monkeypatch.setattr(batch, 'to_plain_language', lambda *a, **kw: {'summary': 'Old', 'example': 'Old'})
    assert batch.generate_one(row).startswith('skipped')
    assert len(batch.find_pending('f', 0)) == 1


def test_variant_reads_translation_only_for_identical_wording(database):
    raw, conn = database
    raw.execute("UPDATE standard SET plain_summary='Saved', plain_example='Example' WHERE framework_id='f'")
    for label, grade, text in [('1', 1, 'Official text'), ('2', 2, 'Different text')]:
        raw.execute("INSERT INTO standard_variant VALUES ('f','one',?,?,'D','C',?,'url',NULL,'{}')",
                    (label, grade, text))
    records = get_standards(conn, 'f', 1)
    assert next(r for r in records if r.code == 'one').plain_summary == 'Saved'
    assert get_standards(conn, 'f', 2)[0].plain_summary is None
    raw.execute('UPDATE standard SET plain_summary=NULL')
    assert any(r.code == 'one' for r in batch.find_pending('f', 1))


def test_dry_run_has_no_model_calls_or_writes(database, monkeypatch, capsys):
    generate = MagicMock(side_effect=AssertionError('Dry run must not generate'))
    monkeypatch.setattr(batch, 'generate_one', generate)
    assert batch.main(['--framework', 'f', '--grade', 'K', '--dry-run']) == 0
    assert 'Found 1 untranslated standards' in capsys.readouterr().out
    generate.assert_not_called()
    assert len(batch.find_pending()) == 3


def test_worker_pool_overlaps_and_continues_after_failure(monkeypatch):
    barrier = Barrier(2, timeout=3)
    lock = Lock()
    active = peak = 0

    def generate(row):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait()
        with lock:
            active -= 1
        if row.code == 'bad':
            raise ValueError('Rejected translation')
        return 'saved'

    monkeypatch.setattr(batch, 'generate_one', generate)
    lines = []
    totals = batch.run_batch([batch.PendingStandard('f', code, 0, 'Text')
                              for code in ['good', 'bad']], workers=2, emit=lines.append)
    assert peak == 2
    assert totals == dict(saved=1, skipped=0, failed=1)
    assert any('[2/2]' in line for line in lines)


def test_migration_is_additive_and_repeatable(monkeypatch, tmp_path):
    source = migrate.MIGRATION_DIR / "001_standard_plain_language.sql"
    (tmp_path / source.name).write_text(source.read_text())
    monkeypatch.setattr(migrate, "MIGRATION_DIR", tmp_path)
    raw = sqlite3.connect(':memory:')
    raw.execute('CREATE TABLE standard (code TEXT, text TEXT)')
    raw.execute("INSERT INTO standard VALUES ('one', 'Keep exact wording')")
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value

    def execute(script):
        # SQLite lacks ADD COLUMN IF NOT EXISTS; emulate that PostgreSQL clause.
        script = '\n'.join(line for line in script.splitlines()
                           if not line.strip().startswith('--'))
        for statement in script.split(';'):
            statement = statement.strip()
            if not statement:
                continue
            assert 'DROP' not in statement and 'DELETE' not in statement
            column = statement.split('IF NOT EXISTS ')[1].split()[0]
            if column not in {row[1] for row in raw.execute('PRAGMA table_info(standard)')}:
                raw.execute(statement.replace('IF NOT EXISTS ', ''))

    cursor.execute.side_effect = execute
    migrate.migrate(conn)
    raw.execute("UPDATE standard SET plain_summary='Keep translation'")
    migrate.migrate(conn)
    assert raw.execute('SELECT * FROM standard').fetchone() == (
        'one', 'Keep exact wording', 'Keep translation', None)
    raw.close()
