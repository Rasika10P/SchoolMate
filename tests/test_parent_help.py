import json
from unittest.mock import Mock

import pytest

from api import llm
from api.services.parent_help import generate_parent_help, PROMPT

ROW = dict(framework_id='CA-CCSSM-2013', code='2.OA.1', text='Add and subtract within 100.', source_url='https://example.org')


def test_parent_help_separates_ideas_and_preserves_source(monkeypatch):
    answer = dict(explanation='Explore adding and subtracting.', citations=['CA-CCSSM-2013:2.OA.1'],
                  suggestions=['Count toys.', 'Compare two groups.', 'Explain a solution.'])
    model = Mock(return_value={'choices': [{'message': {'content': json.dumps(answer)}}]})
    monkeypatch.setattr(llm, 'cached_complete', model)
    result = generate_parent_help('Help prepare', 'math', 2, [ROW])
    assert result['sources'] == [ROW]
    assert result['suggestions'] == answer['suggestions']
    assert 'not California' in PROMPT
    assert 'Never request marks' in PROMPT
    assert 'not an official contest syllabus' in PROMPT


def test_parent_help_rejects_invented_source_and_skips_empty_data(monkeypatch):
    model = Mock(return_value={'choices': [{'message': {'content': json.dumps(dict(
        explanation='Learn something.', citations=['fake'], suggestions=['One', 'Two', 'Three']))}}]})
    monkeypatch.setattr(llm, 'cached_complete', model)
    assert generate_parent_help('Question', 'math', 2, []) is None
    model.assert_not_called()
    with pytest.raises(ValueError, match='sourced parent explanation'):
        generate_parent_help('Question', 'math', 2, [ROW])
