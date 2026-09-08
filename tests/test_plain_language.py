import json
from unittest.mock import MagicMock

import pytest

from api import llm
from api.services import plain_language as plain


def response(value):
    return {'choices': [{'message': {'content': json.dumps(value)}}]}


GOOD = {'summary': 'Count up to 100 in ones and tens.',
        'example': 'Count 100 dried beans at the kitchen table, first one at a time, then in groups of ten.'}


def test_disk_cache_is_scoped_and_invalidated(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, 'CACHE_DIR', tmp_path)
    monkeypatch.setenv('LLM_CACHE', 'on')
    provider = MagicMock(side_effect=lambda **kw: response(
        {'faithful': True} if 'check curriculum' in kw['messages'][0]['content'] else GOOD))
    monkeypatch.setattr(llm, '_provider_complete', provider)
    args = dict(code='K.CC.1', framework_id='CA-CCSSM-2013')
    assert plain.to_plain_language('Count to 100 by ones and by tens.', 0, **args) == GOOD
    assert plain.to_plain_language('Count to 100 by ones and by tens.', 0, **args) == GOOD
    assert provider.call_count == 2  # Translation and independent fidelity review.
    plain.to_plain_language('Count to 100 by ones and by tens.', 0, **{**args, 'framework_id': 'other'})
    assert provider.call_count == 4
    plain.to_plain_language('Count to 100 by ones and by tens!', 0, **args)
    assert provider.call_count == 6
    for call in provider.call_args_list:
        assert call.kwargs['model'] == plain.CHEAP_MODEL
        assert call.kwargs['temperature'] == 0


@pytest.mark.parametrize('bad', [
    {}, {'summary': '', 'example': ''},
    {'summary': 'word ' * 20, 'example': 'Count beans.'},
    {'summary': 'Count beans. Then add them.', 'example': 'Count beans.'},
    {'summary': 'Learn K.CC.1.', 'example': 'Count beans.'},
    {'summary': None, 'example': 'Count beans.'},
])
def test_invalid_output_is_not_displayed(monkeypatch, bad):
    monkeypatch.setattr(llm, 'cached_complete', lambda **kw: response(bad))
    with pytest.raises(plain.TranslationUnavailable):
        plain.to_plain_language('Count to 100 by ones and tens.', 0, code='K.CC.1')


def test_changed_requirements_rejected_by_review(monkeypatch):
    complete = MagicMock(side_effect=[response(GOOD), response({'faithful': False})])
    monkeypatch.setattr(llm, 'cached_complete', complete)
    with pytest.raises(plain.TranslationUnavailable, match='fidelity'):
        plain.to_plain_language('Count to 20.', 0)
    prompt = complete.call_args.kwargs['messages'][0]['content']
    assert 'EVERY requirement' in prompt and 'comparisons' in prompt
