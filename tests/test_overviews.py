import json
from contextlib import contextmanager
import sqlite3
from unittest.mock import MagicMock

import pytest

from api import db, llm
from api.capability import DOMAIN_LABELS, domain_label
from api.load import SUBJECT_FILES, read_subject
from api.services import overviews
from scripts import generate_grade_overviews as batch
from pathlib import Path


TEXT = 'This grade explores numbers and calculations. It also covers shapes. Measuring connects these areas.'


def test_all_loaded_domains_have_parent_labels():
    data = Path(__file__).resolve().parents[1] / 'data'
    for filename, framework in SUBJECT_FILES.items():
        rows, _ = read_subject(data / filename, framework)
        assert {row.domain for row in rows} <= DOMAIN_LABELS.keys()
    assert domain_label('OA', 1)[0] == 'Adding and subtracting'
    assert 'patterns' in domain_label('OA', 4)[0]


def test_overview_generation_is_cached_and_three_sentences(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, 'CACHE_DIR', tmp_path)
    monkeypatch.setenv('LLM_CACHE', 'on')
    provider = MagicMock(return_value={'choices': [{'message': {'content': json.dumps({'overview': TEXT})}}]})
    monkeypatch.setattr(llm, '_provider_complete', provider)
    assert overviews.generate_overview('f', 1, ['OA', 'G']) == TEXT
    assert overviews.generate_overview('f', 1, ['G', 'OA']) == TEXT
    provider.assert_called_once()
    overviews.generate_overview('f', 2, ['G', 'OA'])
    assert provider.call_count == 2
    assert overviews.domain_key(['OA', 'G'], 1) == overviews.domain_key(['G', 'OA'], 1)
    assert overviews.domain_key(['OA'], 1) != overviews.domain_key(['G'], 1)


@pytest.mark.parametrize('text', ['', 'One sentence.', 'One. Two. Three. Four. Five.'])
def test_overview_rejects_bad_paragraphs(monkeypatch, text):
    monkeypatch.setattr(llm, 'cached_complete', lambda **kw: {
        'choices': [{'message': {'content': json.dumps({'overview': text})}}]})
    with pytest.raises(ValueError, match='Invalid grade overview'):
        overviews.generate_overview('f', 1, ['OA'])


def test_database_resume_filters_and_stale_domain_refresh(monkeypatch):
    raw = sqlite3.connect(':memory:')
    for statement in db.SCHEMA:
        raw.execute(statement)
    raw.execute("INSERT INTO framework VALUES ('f','CA','math','1','url')")
    for code, grade, domain in [('one', 1, 'OA'), ('two', 2, 'G')]:
        raw.execute('''INSERT INTO standard (framework_id,code,grade,domain,cluster,text,source_url)
            VALUES ('f',?,?,?,'C','Official','url')''', (code, grade, domain))
    raw.commit()
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    sql_cursor = raw.cursor()
    cursor.execute.side_effect = lambda query, params=(): sql_cursor.execute(query.replace('%s', '?'), params)
    cursor.fetchall.side_effect = sql_cursor.fetchall

    @contextmanager
    def connection():
        with raw:
            yield conn

    monkeypatch.setattr(batch, 'get_conn', connection)
    generate = MagicMock(return_value=TEXT)
    monkeypatch.setattr(batch, 'generate_overview', generate)
    assert len(batch.find_pending()) == 2
    assert batch.find_pending('other') == []
    pending = batch.find_pending('f', 1)
    batch.generate_one(pending[0])
    assert batch.find_pending('f', 1) == []
    assert overviews.get_overviews(conn, [('f', 1)]) == {'f|1': TEXT}
    assert batch.main(['--grade', '2', '--dry-run']) == 0
    generate.assert_called_once()
    raw.execute("UPDATE standard SET domain='G' WHERE code='one'")
    assert len(batch.find_pending('f', 1)) == 1
    batch.generate_one(batch.find_pending('f', 1)[0])
    assert batch.find_pending('f', 1) == []
    assert raw.execute('SELECT COUNT(*) FROM standard').fetchone()[0] == 2
    raw.close()


def test_format_failure_gets_one_cached_repair(monkeypatch):
    complete = MagicMock(side_effect=[
        {'choices': [{'message': {'content': json.dumps({'overview': 'One. Two.'})}}]},
        {'choices': [{'message': {'content': json.dumps({'overview': TEXT})}}]},
    ])
    monkeypatch.setattr(llm, 'cached_complete', complete)
    assert overviews.generate_overview('f', 1, ['G']) == TEXT
    assert complete.call_count == 2
    assert 'THREE separate' in complete.call_args.kwargs['messages'][0]['content']
