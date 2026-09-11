import json
from unittest.mock import MagicMock

import pytest

from api.services import guidance
from api.frameworks import FRAMEWORK_BY_SUBJECT
from api.tools import definitions

F = FRAMEWORK_BY_SUBJECT['math']


@pytest.fixture(autouse=True)
def relevance_model(monkeypatch):
    model = MagicMock(return_value={'choices': [{'message': {'content': 'RELEVANT'}}]})
    monkeypatch.setattr(guidance.llm, 'cached_complete', model)
    return model


@pytest.fixture
def pinecone(monkeypatch):
    client, index = MagicMock(), MagicMock()
    stored = {}
    client.inference.embed.side_effect = lambda **kw: {'data': [{'values': [0.1, 0.2]} for _ in kw['inputs']]}
    index.list.side_effect = lambda namespace, prefix: [[key for key in stored.get(namespace, {}) if key.startswith(prefix)]]

    def upsert(vectors, namespace):
        stored.setdefault(namespace, {}).update({v['id']: v for v in vectors})

    def delete(ids, namespace):
        for key in ids:
            stored[namespace].pop(key, None)

    def matches(metadata, conditions):
        for key, value in conditions.items():
            if key == '$or':
                if not any(matches(metadata, part) for part in value):
                    return False
            elif '$exists' in value:
                if (key in metadata) != value['$exists']:
                    return False
            elif metadata.get(key) != value['$eq']:
                return False
        return True

    def query(namespace, filter, **kwargs):
        return {'matches': [dict(id=row['id'], score=0.9, metadata=row['metadata'])
            for row in stored.get(namespace, {}).values()
            if matches(row['metadata'], filter)]}

    index.upsert.side_effect = upsert
    index.delete.side_effect = delete
    index.query.side_effect = query
    monkeypatch.setattr(guidance, '_pinecone', lambda: (client, index))
    return client, index, stored


def write_prose(path, text, grade=None):
    meta = dict(document_title='Teaching mathematics', source_url='https://example.org/framework.pdf',
                page=7, grade=grade)
    path.write_text('@metadata ' + json.dumps(meta) + '\n\n@page 7\n' + text)
    return str(path)


def test_paragraph_boundaries_and_overlap():
    paragraphs = ['A' * 300 + '.', 'B' * 90 + '.', 'C' * 500 + '.', 'D' * 300 + '.']
    chunks = guidance.chunk_text('\n\n'.join(paragraphs))
    assert chunks == [paragraphs[0] + '\n\n' + paragraphs[1],
                      paragraphs[1] + '\n\n' + paragraphs[2], paragraphs[3]]
    assert all(piece in paragraphs for chunk in chunks for piece in chunk.split('\n\n'))
    long = 'A complete long paragraph ' * 60 + '.'
    assert guidance.chunk_text(long) == [long]
    assert guidance.chunk_text('') == []


def test_idempotent_reingest_and_shorter_file_removes_stale_chunks(pinecone, tmp_path):
    client, index, stored = pinecone
    path = tmp_path / 'math.txt'
    prose = '\n\n'.join('Teaching through discussion ' * 20 + str(i) + '.' for i in range(4))
    write_prose(path, prose)
    count = guidance.ingest(str(path), F)
    ids = set(stored[F])
    assert count == 4
    assert guidance.ingest(str(path), F) == count
    assert set(stored[F]) == ids
    for record in stored[F].values():
        assert record['metadata']['framework_id'] == F
        assert record['metadata']['subject'] == 'math'
        assert record['metadata']['page'] == 7
        assert record['metadata']['document_title'] == 'Teaching mathematics'
        assert 'grade' not in record['metadata']  # Pinecone rejects null metadata.
    write_prose(path, 'Teachers invite discussion.')
    assert guidance.ingest(str(path), F) == 1
    assert len(stored[F]) == 1
    assert next(iter(stored[F])) in ids
    assert index.delete.call_args.kwargs['namespace'] == F
    assert client.inference.embed.call_args.kwargs['parameters']['input_type'] == 'passage'


def test_mandatory_framework_and_optional_grade_filter(pinecone, tmp_path):
    client, index, stored = pinecone
    path = write_prose(tmp_path / 'math.txt', 'Teachers invite discussion.')
    guidance.ingest(path, F)
    result = guidance.search('How do teachers help?', F)
    assert result['state'] == 'available'
    assert result['chunks'][0]['metadata']['grade'] is None
    assert index.query.call_args.kwargs['namespace'] == F
    assert index.query.call_args.kwargs['filter'] == {'framework_id': {'$eq': F}}
    assert index.query.call_args.kwargs['include_metadata'] is True
    write_prose(tmp_path / 'math.txt', 'Kindergarten instruction.', grade=0)
    guidance.ingest(path, F)
    assert guidance.search('How do teachers help?', F, grade=0)['state'] == 'available'
    assert index.query.call_args.kwargs['filter']['$or'] == [{'grade': {'$eq': 0}}, {'grade': {'$exists': False}}]
    assert client.inference.embed.call_args.kwargs['parameters']['input_type'] == 'query'
    with pytest.raises(TypeError):
        guidance.search('How do teachers help?')
    before = index.query.call_count
    for bad in ['', None, 'unknown']:
        with pytest.raises(ValueError, match='framework_id'):
            guidance.search('Question', bad)
    assert index.query.call_count == before


def test_namespace_isolation_and_no_cross_subject_fallback(pinecone, tmp_path):
    client, index, stored = pinecone
    guidance.ingest(write_prose(tmp_path / 'math.txt', 'Discuss ideas.'), F)
    other = FRAMEWORK_BY_SUBJECT['ela']
    assert guidance.search('Discuss ideas', other)['state'] == 'unavailable'
    index.query.side_effect = None
    index.query.return_value = {'matches': [dict(id='wrong', metadata={'framework_id': other, 'text': 'Foreign prose'})]}
    assert guidance.search('Discuss ideas', F)['state'] == 'unavailable'


@pytest.mark.parametrize('failure', ['empty', 'offline', 'embedding'])
def test_unavailable_does_not_raise(pinecone, failure):
    client, index, _ = pinecone
    if failure == 'offline':
        index.query.side_effect = ConnectionError('offline')
    elif failure == 'embedding':
        client.inference.embed.side_effect = TimeoutError('offline')
    result = definitions.search_guidance.invoke({'subject': 'math', 'question': 'How is my child taught?'})
    assert result['state'] == 'unavailable' and result['reason']


def test_unconfigured_client_degrades_gracefully(monkeypatch):
    monkeypatch.delenv('PINECONE_API_KEY', raising=False)
    assert guidance.search('How is this taught?', F)['state'] == 'unavailable'


def test_metadata_changes_split_chunks_and_standards_rejected(tmp_path):
    path = tmp_path / 'math.txt'
    write_prose(path, 'General teaching guidance.')
    with path.open('a') as f:
        f.write('\n\n@metadata ' + json.dumps({'grade': 3, 'page': 8, 'document_title': 'Same document'})
                + '\n\n@page 8\nGrade three teaching guidance.')
    rows = guidance._read_chunks(str(path), F)
    assert len(rows) == 2
    assert [r['metadata']['grade'] for r in rows] == [None, 3]
    assert [r['metadata']['page'] for r in rows] == [7, 8]
    path.write_text('Standard Identifier: K.CC.1\n\nCount to 100.')
    with pytest.raises(ValueError, match='Postgres'):
        guidance._read_chunks(str(path), F)


def test_cli_reports_namespace_count(pinecone, tmp_path, capsys):
    path = write_prose(tmp_path / 'math.txt', 'Teachers invite discussion.')
    assert guidance.main(['--ingest', '--framework', F, '--path', path]) == 0
    assert f'{F}: 1 chunks upserted' in capsys.readouterr().out


def test_corpus_has_prose_for_every_framework():
    for framework in FRAMEWORK_BY_SUBJECT.values():
        rows = guidance._read_chunks(str(guidance.CORPUS_DIR / f'{framework}.txt'), framework)
        assert rows and rows[0]['metadata']['source_url'].startswith('https://www.cde.ca.gov/')


@pytest.mark.parametrize('subject,framework', FRAMEWORK_BY_SUBJECT.items())
def test_tool_routes_each_subject_without_sql(monkeypatch, subject, framework):
    search = MagicMock(return_value={'state': 'unavailable', 'reason': 'empty'})
    monkeypatch.setattr(guidance, 'search', search)
    monkeypatch.setattr(definitions, 'get_conn', MagicMock(side_effect=AssertionError('Prose search must not query standards')))
    result = definitions.search_guidance.invoke({'subject': subject, 'question': 'How do teachers teach this?', 'grade': 2})
    search.assert_called_once_with('How do teachers teach this?', framework, grade=2)
    assert result['state'] == 'unavailable'
    assert definitions.search_guidance in definitions.TOOLS
    assert definitions.search_guidance.name not in definitions.GOAL_TO_TOOL.values()


def test_failed_upsert_does_not_delete_previous_chunks(pinecone, tmp_path):
    _, index, stored = pinecone
    path = write_prose(tmp_path / 'math.txt', ('A' * 600 + '.\n\n') * 3)
    guidance.ingest(path, F)
    before = set(stored[F])
    write_prose(tmp_path / 'math.txt', 'Short new prose.')
    index.upsert.side_effect = ConnectionError('Interrupted')
    with pytest.raises(ConnectionError):
        guidance.ingest(path, F)
    assert set(stored[F]) == before
    index.delete.assert_not_called()


def test_pdf_paste_without_blank_lines_splits_at_sentences():
    sentences = [f'Teachers invite children to discuss idea {i} and explain their thinking clearly.' for i in range(60)]
    chunks = guidance.chunk_text('\n'.join(sentences))
    assert len(chunks) > 1
    assert all(len(chunk) <= 800 for chunk in chunks)
    assert all(part in sentences for chunk in chunks for part in chunk.split('\n\n'))
    assert all(any(sentence in chunk for chunk in chunks) for sentence in sentences)


def test_page_markers_track_before_after_and_spanning_chunks(pinecone, tmp_path):
    client, _, stored = pinecone
    path = tmp_path / 'pages.txt'
    # No pagination for the preface, then two explicit page markers. The short
    # page-143 paragraph overlaps into the final chunk and keeps its own page.
    path.write_text('\n\n'.join([
        'U' * 790 + '.', '@page 142', 'A' * 700 + '.',
        'B' * 600 + '.', 'C' * 90 + '.', '@page 143',
        'D' * 90 + '.', 'E' * 700 + '.',
    ]))
    chunks = guidance._read_chunks(str(path), F)
    assert len(chunks) == 4
    assert [c['metadata']['page'] for c in chunks] == [None, 142, 142, 143]
    assert [c['metadata'].get('page_end') for c in chunks] == [None, None, 143, None]
    assert chunks[2]['text'].endswith('D' * 90 + '.')
    assert chunks[3]['text'].startswith('D' * 90 + '.')
    assert all('@page' not in c['text'] for c in chunks)
    assert guidance.ingest(str(path), F) == 4
    assert all('@page' not in text for call in client.inference.embed.call_args_list
               for text in call.kwargs['inputs'])
    assert [r['metadata'].get('page') for r in stored[F].values()] == [None, 142, 142, 143]
    result = guidance.search('Question', F)
    assert result['chunks'][2]['metadata']['page_end'] == 143
    assert result['chunks'][0]['metadata']['page'] is None


def test_markers_inside_paragraph_are_not_content_or_forced_chunk_boundaries(tmp_path):
    path = tmp_path / 'continuous.txt'
    path.write_text('@page 142\nTeachers help children\n@page 143\nexplain their ideas.\n@page 144\n')
    chunks = guidance._read_chunks(str(path), F)
    assert len(chunks) == 1
    assert chunks[0]['text'] == 'Teachers help children\nexplain their ideas.'
    assert chunks[0]['metadata']['page'] == 142
    assert chunks[0]['metadata']['page_end'] == 143  # Empty final page is excluded.


def test_unpaginated_file_warns_and_does_not_fail(pinecone, tmp_path, caplog):
    path = tmp_path / 'unpaginated.txt'
    path.write_text('@metadata {"page":99}\n\nGeneral guidance without pagination.')
    assert guidance.ingest(str(path), F) == 1
    assert path.name in caplog.text and 'no @page markers' in caplog.text
    record = guidance.search('Question', F)['chunks'][0]
    assert record['metadata']['page'] is None
    assert record['metadata']['page_end'] is None


def test_cli_reports_page_coverage(pinecone, tmp_path, capsys):
    path = tmp_path / 'coverage.txt'
    path.write_text('U' * 790 + '.\n\n@page 0\nNumbered prose.')
    assert guidance.main(['--ingest', '--framework', F, '--path', str(path)]) == 0
    output = capsys.readouterr().out
    assert F in output
    assert '1 with page numbers, 1 without page numbers' in output


def test_page_state_survives_metadata_change(tmp_path):
    path = tmp_path / 'metadata.txt'
    path.write_text('@page 12\nBefore.\n@metadata {"grade":3}\nAfter.')
    chunks = guidance._read_chunks(str(path), F)
    assert [c['metadata']['page'] for c in chunks] == [12, 12]
    assert [c['metadata']['grade'] for c in chunks] == [None, 3]



def test_grade_search_includes_general_prose_but_not_other_grades(pinecone, tmp_path):
    _, index, stored = pinecone
    path = write_prose(tmp_path / 'general.txt', 'Children learn through discussion.')
    guidance.ingest(path, F)
    general = next(iter(stored[F].values()))
    for grade in (1, 5):
        stored[F][f'grade-{grade}'] = {'id': f'grade-{grade}', 'score': 0.9, 'metadata': {
            **general['metadata'], 'grade': grade, 'text': f'Grade {grade} teaching.'}}
    result = guidance.search('How is math taught?', F, grade=1)
    assert result['state'] == 'available'
    assert {c['metadata']['grade'] for c in result['chunks']} == {None, 1}
    assert index.query.call_args.kwargs['namespace'] == F
    assert index.query.call_args.kwargs['filter']['framework_id'] == {'$eq': F}
    # Even an incorrect upstream response must not leak another grade.
    index.query.side_effect = None
    index.query.return_value = {'matches': [{**r, 'score': 0.9} for r in stored[F].values()]}
    assert {c['metadata']['grade'] for c in guidance.search('Why?', F, grade=1)['chunks']} == {None, 1}


@pytest.mark.parametrize('score,verdict,expected', [(0.81, 'NOT_RELEVANT', 'no_relevant_content'), (0.2, 'RELEVANT', 'available')])
def test_relevance_not_absolute_score(pinecone, tmp_path, relevance_model, caplog, score, verdict, expected):
    path = write_prose(tmp_path / 'math.txt', 'Teaching mathematics through objects.')
    guidance.ingest(path, F)
    _, index, stored = pinecone
    index.query.side_effect = None
    index.query.return_value = {'matches': [{**next(iter(stored[F].values())), 'score': score}]}
    relevance_model.return_value = {'choices': [{'message': {'content': verdict}}]}
    with caplog.at_level('INFO', logger=guidance.__name__):
        result = guidance.search('Which school should I choose?', F)
    assert result['state'] == expected
    assert result['top_score'] == score and result['scores'] == [score]
    assert result['judged'] is True
    assert str(score) in caplog.text and verdict in caplog.text
    relevance_model.assert_called_once()
    prompt = relevance_model.call_args.kwargs['messages']
    assert 'Which school should I choose?' in prompt[1]['content']
    assert 'Teaching mathematics through objects.' in prompt[1]['content']
    if verdict == 'NOT_RELEVANT':
        assert 'chunks' not in result


@pytest.mark.parametrize('reply', ['Maybe relevant', None])
def test_judgement_failure_is_not_retrieval_failure(pinecone, tmp_path, relevance_model, reply):
    guidance.ingest(write_prose(tmp_path / 'math.txt', 'Teaching mathematics.'), F)
    if reply is None:
        relevance_model.side_effect = RuntimeError('Model offline')
    else:
        relevance_model.return_value = {'choices': [{'message': {'content': reply}}]}
    result = guidance.search('How?', F)
    assert result['state'] == 'judgement_unavailable'
    assert result['judged'] is False
    assert result['top_score'] == 0.9
    assert 'chunks' not in result


def test_empty_namespace_skips_judge(pinecone, relevance_model):
    assert guidance.search('How?', F)['state'] == 'unavailable'
    relevance_model.assert_not_called()
