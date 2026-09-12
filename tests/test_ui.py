from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from api import db, llm
from api.capability import NOT_PUBLISHED_REASON
from api.graph import agentic, deterministic
from ui import app


APP = str(Path(app.__file__))


def new_page() -> AppTest:
    page = AppTest.from_file(APP, default_timeout=15)
    page.session_state.diagnostics = {"reachable": True, "rows": [], "error": None}
    return page


def navigate(page, label):
    page.session_state.page = label
    page.run()
    return page


def content() -> dict[str, Any]:
    return dict(standards=[], standing=[], forward=[], backward=[], source_url="", version="test",
                programs={"state": "not_published", "reason": NOT_PUBLISHED_REASON["hss", "programs"]})


def test_five_key_cache_avoids_graph_and_database_work(monkeypatch: pytest.MonkeyPatch) -> None:
    app.generate_guide.clear()
    hydrate = MagicMock(return_value=content())
    graph = MagicMock()
    graph.invoke.return_value = {"answer": "Saved answer", "tool_results": {}}
    monkeypatch.setattr(app, "_hydrate", hydrate)
    monkeypatch.setattr(app, "build_deterministic_graph", lambda: graph)
    monkeypatch.setattr(app, "build_agent_graph", lambda: graph)
    app.generate_guide("math", 4, "NF", "catching_up", "deterministic")
    app.generate_guide("math", 4, "NF", "catching_up", "deterministic")
    assert graph.invoke.call_count == hydrate.call_count == 1
    app.generate_guide("math", 4, "NF", "catching_up", "agent")
    assert graph.invoke.call_count == hydrate.call_count == 2
    app.generate_guide.clear()


def test_unknown_question_goes_to_results_without_blocking(monkeypatch):
    model = MagicMock(return_value={"choices": [{"message": {"content": "{}"}}]})
    monkeypatch.setattr(llm, "cached_complete", model)
    forbidden = MagicMock(side_effect=AssertionError("Unexpected retrieval"))
    monkeypatch.setattr(db, "get_conn", forbidden)
    page = new_page().run()
    page.text_area[0].set_value("Tell me more")
    next(b for b in page.button if b.label == "Ask").click().run()
    assert not page.exception
    assert page.session_state.stage == "results"
    assert not page.warning
    assert any("Which subject did you have in mind?" == m.value for m in page.info)
    assert [b.key for b in page.button if b.label == "Add"] == []
    navigate(page, 'About')
    navigate(page, 'Guide')
    assert not page.exception
    forbidden.assert_not_called()
    model.assert_called_once()


def test_results_reruns_only_read_saved_data(monkeypatch: pytest.MonkeyPatch) -> None:
    forbidden = MagicMock(side_effect=AssertionError("Unexpected work on results rerun"))
    monkeypatch.setattr(db, "get_conn", forbidden)
    monkeypatch.setattr(llm, "cached_complete", forbidden)
    page = new_page()
    page.session_state.stage = "results"
    page.session_state.results = dict(content=content(), answer="Saved answer", tool_results={},
                                      subject="hss", grade=4, domain=None, goal="catching_up", mode="deterministic")
    page.run()
    assert not page.exception
    assert [tab.label for tab in page.tabs if tab.label not in ("Guide", "About")] == ["Guided setup", "Ask me anything", "Learning this year", "Skills to revisit", "Standing", "Activities", "Outside programmes"]
    assert any(tab.label == "Standing" for tab in page.tabs)
    navigate(page, 'About')
    page.radio(key="_mode_choice").set_value("agent").run()
    navigate(page, 'Guide')
    page.run()
    assert not page.exception
    forbidden.assert_not_called()


def test_guided_fields_and_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    app.generate_guide.clear()
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value.fetchone.return_value = ("source", "test")
    monkeypatch.setattr(db, "get_conn", lambda: conn)
    from api.services import standards, progression
    from api.models import Standard
    monkeypatch.setattr(standards, "get_standards", lambda *args: [Standard("CA-CCSSM-2013", "K.CC.A.1", 0, "CC", "A", "Sample", "source", 1)])
    monkeypatch.setattr(standards, "get_achievement_levels", lambda *args: [])
    monkeypatch.setattr(progression, "walk", lambda *args: [])
    graph = MagicMock()
    graph.invoke.return_value = {"answer": "Saved answer", "tool_results": {}}
    monkeypatch.setattr(deterministic, "build_deterministic_graph", lambda: graph)
    page = new_page().run()
    page.selectbox[0].set_value("math")
    page.selectbox[1].set_value(0)
    page.selectbox[2].set_value("on_grade_level")
    assert not any(button.label == "Continue" for button in page.button)
    graph.invoke.assert_not_called()
    next(button for button in page.button if button.label == "See guide").click().run()
    assert not page.exception
    assert page.session_state.stage == "results"
    graph.invoke.assert_called_once()
    page.run()
    graph.invoke.assert_called_once()
    assert page.selectbox(key="guide_grade").value == 0
    next(button for button in page.button if button.label == "See guide").click().run()
    graph.invoke.assert_called_once()
    app.generate_guide.clear()


def test_unloaded_grade_is_information_and_never_calls_model(monkeypatch: pytest.MonkeyPatch) -> None:
    app.generate_guide.clear()
    payload = {**content(), "data_state": "not_loaded", "data_reason": "Mathematics data for Grade 2 has not been loaded yet."}
    monkeypatch.setattr(app, "_hydrate", lambda *args: payload)
    forbidden = MagicMock(side_effect=AssertionError("No model for missing data"))
    monkeypatch.setattr(app, "build_agent_graph", forbidden)
    result = app.generate_guide("math", 2, None, "catching_up", "agent")
    assert result["tool_results"]["state"] == "not_loaded"
    assert "Grade 2" in result["answer"]
    forbidden.assert_not_called()
    app.generate_guide.clear()


def test_diagnostics_are_cached_and_framework_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    app.database_diagnostics.clear()
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value.fetchall.return_value = [
        ("CA-CCSSM-2013", 4, 3), ("unrelated", 4, 99)]
    connect = MagicMock(return_value=conn)
    monkeypatch.setattr(app, "get_conn", connect)
    first = app.database_diagnostics()
    assert first == app.database_diagnostics()
    connect.assert_called_once()
    assert first["reachable"]
    math = next(row for row in first["rows"] if row["Subject"] == "Mathematics")
    assert math["Grade 4"] == 3
    assert math["Grade 2"] == 0
    app.database_diagnostics.clear()


def test_failure_categories_log_tracebacks(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import psycopg
    from openai import APIConnectionError
    import httpx
    show = MagicMock()
    monkeypatch.setattr(app.st, "error", show)
    for error, prefix in [
        (psycopg.OperationalError("connection refused"), "We couldn’t reach the learning information"),
        (APIConnectionError(request=httpx.Request("GET", "https://example.com")), "We couldn’t prepare your guide"),
        (ValueError("broken configuration"), "We couldn’t prepare your guide"),
    ]:
        try:
            raise error
        except Exception as exc:
            app._show_failure(exc)
        assert show.call_args.args[0].startswith(prefix)
        assert next(record for record in reversed(caplog.records)
                    if record.name == "ui.app").exc_info is not None


def record(code='K.CC.1', grade=0, page=2):
    return dict(framework_id='CA-CCSSM-2013', code=code, grade=grade, domain='CC',
                cluster='Counting', text='Exact official text.', source_url='https://example.org/document.pdf',
                page=page, plain_summary='Count objects.',
                plain_example='Count spoons on the kitchen table.')


def test_parent_rows_official_expanders_next_grade_and_sources(monkeypatch):
    forbidden = MagicMock(side_effect=AssertionError('No generation on a results view'))
    monkeypatch.setattr(llm, 'cached_complete', forbidden)
    payload = content()
    payload['standards'] = [record()]
    payload['forward'] = [dict(standard=record('1.CC.1', 1, 5)),
                          dict(standard=record('1.CC.1', 1, 5)),
                          dict(standard=record('2.CC.1', 2, 4))]
    page = new_page()
    page.session_state.stage = 'results'
    page.session_state.results = dict(content=payload, answer='Saved answer', tool_results={},
        subject='math', grade=0, domain=None, goal='working_ahead', mode='deterministic')
    page.run()
    assert not page.exception
    expanders = [e for e in page.expander if e.label == 'See the official wording']
    assert len(expanders) == 3
    assert all('Exact official text.' in [m.value for m in e.markdown] for e in expanders)
    assert 'K.CC.1' in expanders[0].text[0].value
    assert '1.CC.1' in expanders[1].text[0].value
    assert not any('CC.1' in m.value for m in page.markdown)
    values = [m.value for m in page.markdown]
    assert '**What your child learns**' not in values
    assert values.count('**Everyday example**') == 3
    assert 'Once these are comfortable, children usually move on to:' in values
    assert sum('(https://example.org/document.pdf) — Pages 2–5' in v for v in values) == 1
    assert not any('Source document:' in c.value for c in page.caption)
    page.run()
    forbidden.assert_not_called()


def test_unmapped_next_steps_and_unchanged_capability_states():
    page = new_page()
    page.session_state.stage = 'results'
    page.session_state.results = dict(content=content(), answer='Saved', tool_results={},
        subject='ela', grade=2, domain=None, goal='working_ahead', mode='deterministic')
    page.run()
    assert not page.exception
    assert 'The sequence has not been mapped for this subject yet.' in [i.value for i in page.info]
    result = dict(page.session_state.results)
    result['subject'] = 'hss'
    page.session_state.results = result
    page.run()
    assert NOT_PUBLISHED_REASON['hss', 'next_steps'] in [i.value for i in page.info]
    assert 'The sequence has not been mapped for this subject yet.' not in [i.value for i in page.info]


def test_missing_translation_shows_official_wording_without_model_calls(monkeypatch):
    forbidden = MagicMock(side_effect=AssertionError('No model on render'))
    monkeypatch.setattr(llm, 'cached_complete', forbidden)
    payload = content()
    missing = record()
    missing['plain_summary'] = missing['plain_example'] = None
    payload['standards'] = [missing]
    payload['forward'] = [dict(standard={**missing, 'code': '1.CC.1', 'grade': 1})]
    page = new_page()
    page.session_state.stage = 'results'
    page.session_state.results = dict(content=payload, answer='Saved', tool_results={},
        subject='math', grade=0, domain=None, goal='working_ahead', mode='deterministic')
    page.run()
    page.run()
    assert not page.exception
    # Official fallback appears once per entry, without a duplicate expander.
    assert [m.value for m in page.markdown].count('Exact official text.') == 2
    assert not any(e.label == 'See the official wording' for e in page.expander)
    assert '—' not in [m.value for m in page.markdown]
    assert not any('available yet' in m.value for m in page.markdown)
    forbidden.assert_not_called()


def test_submission_never_generates_translations(monkeypatch):
    app.generate_guide.clear()
    payload = content()
    payload['standards'] = [record()]
    monkeypatch.setattr(app, '_hydrate', lambda *args: payload)
    graph = MagicMock()
    graph.invoke.return_value = {'answer': 'Saved', 'tool_results': {}}
    monkeypatch.setattr(app, 'build_deterministic_graph', lambda: graph)
    monkeypatch.setattr(llm, 'cached_complete', MagicMock(side_effect=AssertionError('No translation calls')))
    assert app.generate_guide('math', 0, None, 'working_ahead', 'deterministic')['content'] == payload
    assert not hasattr(app, '_prepare_translations')
    app.generate_guide.clear()


def test_refresh_reads_stored_columns_without_generation(monkeypatch):
    payload = content()
    payload['standards'] = [record()]
    payload['forward'] = [dict(standard=record('1.CC.1', 1))]
    conn = MagicMock()
    conn.__enter__.return_value = conn
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [
        ('CA-CCSSM-2013', 'K.CC.1', 'Exact official text.', 'New stored summary.', 'New stored example.'),
        ('CA-CCSSM-2013', '1.CC.1', 'Different source', 'Do not reuse.', 'Do not reuse.'),
    ]
    monkeypatch.setattr(app, 'get_conn', lambda: conn)
    forbidden = MagicMock(side_effect=AssertionError('Refresh must only read the database'))
    monkeypatch.setattr(llm, 'cached_complete', forbidden)
    app._refresh_translations(payload)
    assert payload['standards'][0]['plain_summary'] == 'New stored summary.'
    assert payload['forward'][0]['standard']['plain_summary'] is None
    query, params = cursor.execute.call_args.args
    assert query.startswith('SELECT ') and 'plain_summary' in query
    assert 'CA-CCSSM-2013' not in query and 'CA-CCSSM-2013' in params
    forbidden.assert_not_called()


def test_domains_collapsed_with_saved_overview_in_both_tabs(monkeypatch):
    monkeypatch.setattr(llm, 'cached_complete', MagicMock(side_effect=AssertionError('No render generation')))
    payload = content()
    payload['standards'] = [record(code=f'K.OA.{n}') | {'domain': 'OA'} for n in range(20)]
    payload['standards'] += [record('K.G.1') | {'domain': 'G'}]
    payload['forward'] = [dict(standard=record('1.OA.1', 1) | {'domain': 'OA'})]
    payload['overviews'] = {'CA-CCSSM-2013|0': 'Saved current overview.',
                            'CA-CCSSM-2013|1': 'Saved next overview.'}
    page = new_page()
    page.session_state.stage = 'results'
    page.session_state.results = dict(content=payload, answer='Saved', tool_results={},
        subject='math', grade=0, domain=None, goal='working_ahead', mode='deterministic')
    page.run()
    assert not page.exception
    for tab, overview, headings in [
        (next(t for t in page.tabs if t.label == "Learning this year"), 'Saved current overview.', ['Adding, subtracting, multiplying and dividing · 20 skills', 'Shapes and space · 1 skill']),
        (next(t for t in page.tabs if t.label == "Next steps"), 'Saved next overview.', ['Adding, subtracting, multiplying and dividing · 1 skill']),
    ]:
        # Only the overview/continuity line and domain expanders are direct tab children.
        sections = [child for child in tab.children.values() if child.type == 'expander']
        assert [section.label for section in sections] == headings
        assert all(not section.proto.expanded for section in sections)
        assert overview in [child.value for child in tab.children.values() if child.type == 'markdown']
        for section in sections:
            assert any(exp.label == 'See the official wording' for exp in section.expander)
    page.run()
    llm.cached_complete.assert_not_called()


def test_guide_has_one_form_no_sidebar_or_technical_readouts(monkeypatch):
    forbidden = MagicMock(side_effect=AssertionError('Navigation must not perform work'))
    monkeypatch.setattr(db, 'get_conn', forbidden)
    monkeypatch.setattr(llm, 'cached_complete', forbidden)
    page = new_page().run()
    assert not page.exception
    assert len(page.sidebar) == 0
    assert [item.label for item in page.selectbox] == ['Subject', 'Grade', 'Goal']
    assert page.selectbox(key='guide_goal').options == [
        'Understand this grade', 'Get ready for next year', 'Fill in earlier gaps', 'Find outside programmes']
    assert not page.text_input and not page.metric and not page.code
    assert not page.radio
    assert sum(b.label == 'See guide' for b in page.button) == 1
    assert not any(b.label == 'Continue' for b in page.button)
    support = next(e for e in page.expander if e.label == 'My child receives English language support (optional)')
    assert not support.proto.expanded
    assert not support.checkbox[0].value
    page.run()
    forbidden.assert_not_called()


def test_how_page_stats_and_mode_preference_survive_navigation(monkeypatch):
    forbidden = MagicMock(side_effect=AssertionError('Page switches must not invoke providers'))
    monkeypatch.setattr(llm, 'cached_complete', forbidden)
    monkeypatch.setattr(db, 'get_conn', forbidden)
    page = new_page()
    page.session_state.mode_stats = {
        'deterministic': dict(calls=0, tokens=0, elapsed=0.2, cost=0.0),
        'agent': dict(calls=2, tokens=100, elapsed=2.5, cost=0.0001),
    }
    page.run()
    navigate(page, 'About')
    assert not page.exception
    assert len(page.sidebar) == 0
    assert [m.value for m in page.metric] == ['0', '0.20 s', '$0.000000', '2', '2.50 s', '$0.000100']
    assert any(e.label == 'Diagnostics' for e in page.expander)
    assert page.code
    page.radio(key='_mode_choice').set_value('agent').run()
    navigate(page, 'Guide')
    assert page.session_state.mode == 'agent'
    assert not page.metric and not page.code
    navigate(page, 'About')
    assert page.radio(key='_mode_choice').value == 'agent'
    forbidden.assert_not_called()


def test_both_modes_record_elapsed_time_and_keep_cache(monkeypatch):
    app.st.cache_data.clear()
    from api.services import standards, progression
    from api.models import Standard
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value.fetchone.return_value = ('source', 'test')
    monkeypatch.setattr(db, 'get_conn', lambda: conn)
    monkeypatch.setattr(standards, 'get_standards', lambda *args: [Standard('CA-CCSSM-2013','K.CC.1',0,'CC','C','Text','source',None)])
    monkeypatch.setattr(standards, 'get_achievement_levels', lambda *args: [])
    monkeypatch.setattr(progression, 'walk', lambda *args: [])
    fixed, agent = MagicMock(), MagicMock()
    for graph in (fixed, agent):
        graph.invoke.return_value = {'answer': 'Saved answer', 'tool_results': {}}
    monkeypatch.setattr(deterministic, 'build_deterministic_graph', lambda: fixed)
    monkeypatch.setattr(agentic, 'build_agent_graph', lambda: agent)
    page = new_page().run()
    page.selectbox(key='guide_subject').set_value('math')
    page.selectbox(key='guide_grade').set_value(0)
    page.selectbox(key='guide_goal').set_value('on_grade_level')
    page.checkbox(key='guide_learner').set_value(True)
    next(b for b in page.button if b.label == 'See guide').click().run()
    assert page.session_state.stage == 'results'
    assert page.session_state.draft['learner'] == 'Yes'
    assert page.session_state.results['domain'] is None
    assert page.session_state.mode_stats['deterministic']['elapsed'] >= 0
    assert not page.metric and not page.code
    navigate(page, 'About')
    page.radio(key='_mode_choice').set_value('agent').run()
    navigate(page, 'Guide')
    assert page.session_state.results['mode'] == 'deterministic'
    agent.invoke.assert_not_called()
    assert page.checkbox(key='guide_learner').value
    next(b for b in page.button if b.label == 'See guide').click().run()
    assert not page.exception
    assert page.session_state.results['mode'] == 'agent'
    assert set(page.session_state.mode_stats) == {'deterministic', 'agent'}
    assert page.session_state.mode_stats['agent']['elapsed'] >= 0
    next(b for b in page.button if b.label == 'See guide').click().run()
    fixed.invoke.assert_called_once()
    agent.invoke.assert_called_once()
    app.generate_guide.clear()


@pytest.mark.parametrize("grade", [None, 0])
def test_open_ask_skips_goal_and_caches_changes(monkeypatch, grade):
    import json
    app.st.cache_data.clear()
    intent = dict(subject="math", grade=grade, goal=None, question_type="open", evidence={"subject": "from math"})
    model = MagicMock(return_value={"choices": [{"message": {"content": json.dumps(intent)}}]})
    monkeypatch.setattr(llm, "cached_complete", model)
    graph = MagicMock()
    graph.invoke.return_value = {"answer": "Teaching uses objects.", "tool_results": {}}
    monkeypatch.setattr(agentic, "build_agent_graph", lambda: graph)
    monkeypatch.setattr(db, "get_conn", MagicMock(side_effect=AssertionError("No SQL hydration")))
    page = new_page().run()
    page.text_area[0].set_value("How is math taught?")
    next(b for b in page.button if b.label == "Ask").click().run()
    assert not page.exception
    assert page.session_state.stage == "results"
    assert not page.warning
    assert [t.label for t in page.tabs if t.label in ("Guided setup", "Ask me anything")] == ["Guided setup", "Ask me anything"]
    assert not any(b.key == "edit_goal" for b in page.button)
    assert not any(b.key == "edit_grade" for b in page.button)
    assert graph.invoke.call_args.args[0]["question"] == "How is math taught?"
    page.run()
    navigate(page, 'About')
    navigate(page, 'Guide')
    graph.invoke.assert_called_once()
    model.assert_called_once()
    assert not any(h.value in ("What I understood", "Your answer") for h in page.subheader)
    assert "Teaching uses objects." in [m.value for m in page.markdown]


def test_structured_missing_goal_is_editable_assumption(monkeypatch):
    import json
    app.st.cache_data.clear()
    model = MagicMock(return_value={"choices": [{"message": {"content": json.dumps(dict(
        subject="math", grade=None, goal=None, question_type="structured",
        evidence={"subject": "from fractions"}))}}]})
    monkeypatch.setattr(llm, "cached_complete", model)
    forbidden = MagicMock(side_effect=AssertionError("Missing grade must not run SQL"))
    monkeypatch.setattr(db, "get_conn", forbidden)
    page = new_page().run()
    page.text_area[0].set_value("What fractions do children learn?")
    next(b for b in page.button if b.label == "Ask").click().run()
    assert not page.exception and not page.warning
    assert page.session_state.results["goal"] == "on_grade_level"
    assert any("include the grade" in m.value for m in page.info)
    forbidden.assert_not_called()
    model.assert_called_once()



def test_legacy_extractor_intent_defaults_open_without_crashing(monkeypatch):
    from api.graph import extractor
    app.st.cache_data.clear()
    monkeypatch.setattr(extractor, "extract_intent", lambda question: dict(
        subject="math", grade=None, goal="on_grade_level", raw=question, evidence=None))
    graph = MagicMock()
    graph.invoke.return_value = {"answer": "Math is explored using objects.", "tool_results": {}}
    monkeypatch.setattr(agentic, "build_agent_graph", lambda: graph)
    page = new_page().run()
    assert page.title[0].value == "📚 SchoolMate"
    page.text_area[0].set_value("how is math taught in school")
    next(b for b in page.button if b.label == "Ask").click().run()
    assert not page.exception
    assert page.session_state.entry_tabs == "Ask me anything"
    assert page.session_state.results["question_type"] == "open"
    assert page.session_state.results["goal"] is None
    assert not any(b.key == "edit_goal" for b in page.button)
    graph.invoke.assert_called_once()
    page.run()
    graph.invoke.assert_called_once()



def test_about_project_and_architecture_need_no_io(monkeypatch):
    forbidden = MagicMock(side_effect=AssertionError("No I/O on navigation"))
    monkeypatch.setattr(db, "get_conn", forbidden)
    monkeypatch.setattr(llm, "cached_complete", forbidden)
    page = new_page().run()
    assert [t.label for t in page.tabs if t.label in ("Guide", "About")] == ["Guide", "About"]
    assert not page.radio
    navigate(page, "About")
    assert not page.exception
    assert any("SchoolMate helps families" in m.value for m in page.markdown)
    assert len(page.image) == 1
    navigate(page, "Guide")
    assert not page.exception
    assert not page.metric and not page.code
    forbidden.assert_not_called()



def test_outside_programmes_work_without_database(monkeypatch):
    app.generate_guide.clear()
    unavailable_db = MagicMock(side_effect=RuntimeError("DATABASE_URL must be set"))
    monkeypatch.setattr(app, "_hydrate", unavailable_db)
    result = app.generate_guide("math", 2, None, "competition_prep", "deterministic")
    assert "Math Kangaroo" in result["answer"]
    assert result["content"] is None
    assert result["preparation_error"]
    unavailable_db.assert_called_once()
    unavailable = app.generate_guide("hss", 2, None, "competition_prep", "deterministic")
    assert unavailable["tool_results"]["state"] == "not_published"
    assert unavailable["answer"] == NOT_PUBLISHED_REASON["hss", "programs"]
    app.generate_guide.clear()



@pytest.mark.parametrize("path", ["ask", "guided"])
def test_failed_new_submission_removes_previous_answer(monkeypatch, path):
    from api.graph import extractor
    app.st.cache_data.clear()
    monkeypatch.setattr(extractor, "extract_intent", MagicMock(side_effect=RuntimeError("Provider configuration missing")))
    monkeypatch.setattr(db, "get_conn", MagicMock(side_effect=RuntimeError("DATABASE_URL must be set")))
    page = new_page()
    page.session_state.stage = "results"
    page.session_state.results = dict(content=None, answer="OLD MATH PROGRAMMES", subject="math", grade=2, goal="competition_prep")
    page.run()
    assert "OLD MATH PROGRAMMES" in [m.value for m in page.markdown]
    if path == "ask":
        page.text_area[0].set_value("What do children learn in grade 2 English?")
        label = "Ask"
    else:
        page.selectbox(key="guide_subject").set_value("ela")
        page.selectbox(key="guide_grade").set_value(2)
        page.selectbox(key="guide_goal").set_value("on_grade_level")
        label = "See guide"
    next(b for b in page.button if b.label == label).click().run()
    assert not page.exception
    assert page.error
    assert page.session_state.stage == "entry"
    assert "OLD MATH PROGRAMMES" not in [m.value for m in page.markdown]
    assert page.session_state.last_error_detail
    page.run()
    assert "OLD MATH PROGRAMMES" not in [m.value for m in page.markdown]



def test_catching_up_shows_only_earlier_grade_links(monkeypatch):
    forbidden = MagicMock(side_effect=AssertionError("No model during rendering"))
    monkeypatch.setattr(llm, "cached_complete", forbidden)
    payload = content()
    payload["standards"] = [record("2.OA.1", 2)]
    payload["backward"] = [dict(standard=record("1.OA.1", 1)),
                            dict(standard=record("1.OA.1", 1)),
                            dict(standard=record("2.OA.2", 2))]
    payload["forward"] = [dict(standard=record("3.OA.1", 3))]
    page = new_page()
    page.session_state.stage = "results"
    page.session_state.results = dict(content=payload, answer="Saved", subject="math", grade=2, goal="catching_up")
    page.run()
    assert not page.exception
    tab = next(t for t in page.tabs if t.label == "Skills to revisit")
    codes = [t.value for t in tab.text]
    assert len(codes) == 1 and "1.OA.1" in codes[0]
    page.run()
    forbidden.assert_not_called()


def test_sources_have_descriptive_links_and_optional_pages():
    payload = content()
    payload["standards"] = [record(page=None) | {"source_url": "https://www2.cde.ca.gov/cacs/math"}]
    docs = app._source_documents(payload, "math", 0)
    assert docs == [{"url": "https://www2.cde.ca.gov/cacs/math", "title": "California mathematics standards", "pages": ""}]
    page = new_page()
    page.session_state.stage = "results"
    page.session_state.results = dict(content=payload, answer="Saved", subject="math", grade=0, goal="on_grade_level")
    page.run()
    assert not page.exception
    values = [m.value for m in page.markdown]
    assert '[California mathematics standards](https://www2.cde.ca.gov/cacs/math)' in values
    assert not any('Page not recorded' in v for v in values)


def test_identical_summary_is_displayed_once_and_long_overview_is_compact():
    payload = content()
    payload["standards"] = [record() | {"plain_summary": "Exact official text.", "plain_example": ""}]
    payload["overviews"] = {"CA-CCSSM-2013|0": "First. Second. Third. Fourth."}
    page = new_page()
    page.session_state.stage = "results"
    page.session_state.results = dict(content=payload, answer="Saved", subject="math", grade=0, goal="on_grade_level")
    page.run()
    assert not page.exception
    values = [m.value for m in page.markdown]
    assert values.count("Exact official text.") == 1
    assert "First. Second. Third. Fourth." in values
    assert not any(e.label == "See the official wording" for e in page.expander)



def open_result(status="available"):
    chunks = [{"id": "one", "text": "Exact passage one.", "metadata": {
        "document_title": "California Mathematics Framework", "page": 14, "page_end": 15,
        "source_url": "https://example.org/framework.pdf"}},
        {"id": "two", "text": "Exact passage two.", "metadata": {
        "document_title": "California Mathematics Framework", "page": None}}]
    return dict(question_type="open", subject="math", grade=1, content=None,
        intent=dict(raw="  how is math taught in first grade  ", subject="math", grade=1,
                    question_type="open", evidence={"subject": "from math", "grade": "from first grade"}),
        answer="Saved explanation", open_paragraphs=[
            {"text": "Children explore mathematical ideas.", "citations": ["one"]},
            {"text": "They discuss what they notice.", "citations": ["two"]}],
        tool_results={"guidance": {"state": status, "reason": "No passages answer this question.",
                                   "chunks": chunks if status == "available" else []}})


def test_open_explanation_citations_navigation_and_full_guide(monkeypatch):
    forbidden = MagicMock(side_effect=AssertionError("No generation or database work on rerun"))
    monkeypatch.setattr(llm, "cached_complete", forbidden)
    monkeypatch.setattr(db, "get_conn", forbidden)
    page = new_page()
    page.session_state.stage = "results"
    page.session_state.results = open_result()
    page.run()
    assert not page.exception
    assert 'How is math taught in first grade' in [h.value for h in page.subheader]
    assert not any(t.label in ('Learning this year', 'Next steps', 'Standing') for t in page.tabs)
    citation = next(e for e in page.expander if e.label.endswith('pp. 14–15'))
    assert 'Exact passage one.' in [m.value for m in citation.markdown]
    assert any(e.label == 'California Mathematics Framework' for e in page.expander)
    raw = next(e for e in page.expander if e.label.startswith('Based on 2 passages'))
    assert not raw.proto.expanded
    assert not any('Page not recorded' in m.value for m in page.markdown)
    assert len(page.get('popover')) == 2
    page.run()
    navigate(page, 'About')
    navigate(page, 'Guide')
    next(b for b in page.button if 'See the full guide' in b.label).click().run()
    assert not page.exception
    assert page.session_state.stage == 'entry'
    assert page.selectbox(key='guide_subject').value == 'math'
    assert page.selectbox(key='guide_grade').value == 1
    forbidden.assert_not_called()


@pytest.mark.parametrize('status,kind', [('no_relevant_content','info'), ('unavailable','error'), ('judgement_unavailable','error')])
def test_open_nonanswer_states_never_show_synthesis(monkeypatch, status, kind):
    forbidden = MagicMock(side_effect=AssertionError('No model on render'))
    monkeypatch.setattr(llm, 'cached_complete', forbidden)
    page = new_page()
    page.session_state.stage = 'results'
    page.session_state.results = open_result(status)
    page.run()
    assert not page.exception
    assert any(e.value.startswith('No passages answer this question.') for e in getattr(page, kind))
    assert 'Children explore mathematical ideas.' not in [m.value for m in page.markdown]
    if status == 'no_relevant_content':
        assert not page.error
        assert 'Guided setup' in page.info[0].value
    forbidden.assert_not_called()


def test_open_change_regenerates_only_on_edit(monkeypatch):
    app.st.cache_data.clear()
    graph = MagicMock()
    graph.invoke.return_value = {'answer': 'New explanation.', 'tool_results': {}}
    monkeypatch.setattr(agentic, 'build_agent_graph', lambda: graph)
    page = new_page()
    page.session_state.stage = 'results'
    page.session_state.results = open_result()
    page.run()
    graph.invoke.assert_not_called()
    page.selectbox(key='open_subject').set_value('ela').run()
    assert not page.exception
    assert page.session_state.results['subject'] == 'ela'
    assert graph.invoke.call_args.args[0]['question'].strip() == 'how is math taught in first grade'
    page.run()
    graph.invoke.assert_called_once()


def test_standing_cites_shared_source_once_below_levels():
    payload = content()
    payload['standing'] = [dict(framework_id='CA-CCSSM-2013', grade=4, subject='mathematics',
        level=n, text=f'Published description {n}.', source_url='https://example.org/ald.pdf',
        page=14 if n < 3 else 15) for n in range(1, 5)]
    page = new_page()
    page.session_state.stage = 'results'
    page.session_state.results = dict(content=payload, answer='Saved', subject='math', grade=4, goal='on_grade_level')
    page.run()
    assert not page.exception
    standing = next(t for t in page.tabs if t.label == 'Standing')
    texts = [m.value for m in standing.markdown]
    assert all(f'Published description {n}.' in texts for n in range(1, 5))
    links = [v for v in texts if '(https://example.org/ald.pdf)' in v]
    assert len(links) == 1 and links[0].endswith('Pages 14–15')
    assert texts.index(links[0]) > texts.index('Published description 4.')



def test_activity_trace_uses_only_recorded_calls_and_no_rerun_work(monkeypatch):
    from api.graph.trace import tool_event
    forbidden = MagicMock(side_effect=AssertionError('Trace viewing must not execute anything'))
    monkeypatch.setattr(llm, 'cached_complete', forbidden)
    monkeypatch.setattr(db, 'get_conn', forbidden)
    result = open_result()
    result['activity_trace'] = [
        tool_event('on_grade_level', {'subject': 'math', 'grade': 1}, {
            'standards': [record(), record('1.CC.2', 1)], 'achievement_levels': []}),
        tool_event('search_guidance', {}, {'state': 'no_relevant_content', 'reason': 'Passages do not answer the question.'})]
    page = new_page()
    page.session_state.stage = 'results'
    page.session_state.results = result
    page.run()
    assert not page.exception
    trace = next(e for e in page.expander if 'Show work' in e.label)
    assert trace.label == 'Checked 1 sources · 2 steps · Show work'
    assert not trace.proto.expanded
    lines = [m.value for m in trace.markdown]
    assert any('found 2 standards across 1 areas' in line for line in lines)
    assert any('no achievement descriptions returned' in line for line in lines)
    assert any('Passages do not answer the question.' in line for line in lines)
    page.run()
    navigate(page, 'About')
    navigate(page, 'Guide')
    forbidden.assert_not_called()


def test_trace_has_no_invented_steps_for_legacy_results():
    page = new_page()
    page.session_state.stage = 'results'
    page.session_state.results = open_result()
    page.run()
    assert not any('Show work' in e.label for e in page.expander)


def test_ela_sections_derive_missing_domains_and_keep_framework_order(monkeypatch):
    forbidden = MagicMock(side_effect=AssertionError("No render generation"))
    monkeypatch.setattr(llm, "cached_complete", forbidden)
    payload = content()
    rows = [record(code) | {"framework_id": "CA-CCSS-ELA-2013"}
            for code in ["L.2.1", "W.2.1", "RI.2.1", "RL.2.1", "RL.2.2", "XYZ.2.1"]]
    for row in rows:
        row.pop("domain", None)
    payload["standards"] = rows
    payload["forward"] = []
    page = new_page()
    page.session_state.stage = "results"
    page.session_state.results = dict(content=payload, answer="Saved", subject="ela", grade=2,
                                      goal="on_grade_level")
    page.run()
    assert not page.exception
    tab = next(t for t in page.tabs if t.label == "Learning this year")
    sections = [child for child in tab.children.values() if child.type == "expander"]
    assert [s.label for s in sections] == [
        "Reading stories and poems · 2 skills", "Reading factual texts · 1 skill",
        "Writing · 1 skill", "Grammar, spelling and vocabulary · 1 skill", "XYZ · 1 skill"]
    assert all(not s.proto.expanded for s in sections)
    page.run()
    forbidden.assert_not_called()


def test_explanation_page_sections_findings_and_stored_source_counts(monkeypatch):
    forbidden = MagicMock(side_effect=AssertionError('No providers on page switch'))
    monkeypatch.setattr(llm, 'cached_complete', forbidden)
    monkeypatch.setattr(db, 'get_conn', forbidden)
    page = new_page()
    page.session_state.source_snapshot = {
        'tables': {'standard': 21, 'progression_edge': 5, 'achievement_descriptor': 4},
        'frameworks': [{'Framework': 'CA-CCSSM-2013', 'Subject': 'Mathematics', 'Passages': 9}]}
    page.run()
    navigate(page, 'About')
    headings = [h.value for h in page.subheader]
    ordered = ['Two ways of deciding', 'Where the answers come from', 'What we do not do',
               'A retrieval finding', 'Evaluation results · Coming soon']
    assert [h for h in headings if h in ordered] == ordered
    tables = [df.value for df in page.dataframe]
    sources = next(df for df in tables if 'Table' in df.columns)
    assert list(sources['Rows']) == [21, 5, 4]
    finding = next(df for df in tables if 'Top similarity score' in df.columns)
    assert list(finding['Top similarity score']) == [0.842, 0.803, 0.810]
    assert not next(e for e in page.expander if e.label == 'Diagnostics').proto.expanded
    navigate(page, 'Guide')
    assert not page.metric and not page.code
    forbidden.assert_not_called()


def test_source_status_uses_live_database_and_namespace_counts(monkeypatch):
    from api.services import guidance
    from types import SimpleNamespace
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value.fetchall.return_value = [
        ('standard', 30), ('progression_edge', 12), ('achievement_descriptor', 4)]
    monkeypatch.setattr(app, 'get_conn', MagicMock(return_value=conn))
    index = MagicMock()
    index.describe_index_stats.return_value = {'namespaces': {
        'CA-CCSSM-2013': {'vector_count': 8}, 'CA-ELD-2012': {'vector_count': 0},
        'unrelated': {'vector_count': 100}}}
    monkeypatch.setattr(guidance, '_pinecone', lambda: (None, index))
    monkeypatch.setattr(app.st, 'session_state', SimpleNamespace())
    app._refresh_sources()
    snapshot = app.st.session_state.source_snapshot
    assert snapshot['tables']['standard'] == 30
    assert [f['Framework'] for f in snapshot['frameworks']] == ['CA-CCSSM-2013']
    index.describe_index_stats.assert_called_once()


@pytest.mark.parametrize('status', ['no_relevant_content', 'unanswerable'])
@pytest.mark.parametrize('subject', ['math', 'ela', 'eld', 'sci', 'hss', 'vapa', 'pe', None])
def test_refusal_offers_actual_subject_without_generation(monkeypatch, status, subject):
    forbidden = MagicMock(side_effect=AssertionError('No render calls'))
    monkeypatch.setattr(llm, 'cached_complete', forbidden)
    page = new_page()
    result = open_result(status)
    result['subject'] = subject
    result['intent'] = {'subject': subject, 'grade': None, 'raw': 'An unsupported question'}
    page.session_state.stage = 'results'
    page.session_state.results = result
    page.run()
    assert not page.exception
    message = page.info[0].value
    assert message.startswith('No passages answer this question.')
    assert 'Guided setup' in message
    if subject:
        assert app._SUBJECT_LEARNING[subject] in message
    else:
        assert 'choose a subject and grade' in message
    buttons = [b for b in page.button if 'Explore' in b.label and 'Guided setup' in b.label]
    assert not buttons
    assert not page.get('popover')
    assert not any(m.value == 'Children explore mathematical ideas.' for m in page.markdown)
    forbidden.assert_not_called()


def test_explicit_unanswerable_state_redirects_without_structured_tabs():
    page = new_page()
    page.session_state.stage = 'results'
    page.session_state.results = dict(state='unanswerable', subject='vapa', grade=None,
        reason="The California framework documents don't discuss college admissions.")
    page.run()
    assert not page.exception
    assert 'music, art, dance, theatre and media arts' in page.info[0].value
    assert not any(t.label == 'Learning this year' for t in page.tabs)


@pytest.mark.parametrize('status', ['unavailable', 'judgement_unavailable'])
def test_fault_does_not_offer_curriculum_alternative(status):
    page = new_page()
    page.session_state.stage = 'results'
    page.session_state.results = open_result(status)
    page.run()
    assert page.error and not page.exception
    assert not any('Explore' in b.label or 'See the full guide' in b.label for b in page.button)


@pytest.mark.parametrize('status', ['no_relevant_content', 'unavailable', 'need_subject', 'unanswerable'])
def test_nonanswers_hide_understood_fields(status):
    page = new_page()
    page.session_state.stage = 'results'
    page.session_state.results = open_result(status)
    page.run()
    assert not page.exception
    assert not page.get('popover')
    assert not any(m.value == 'What I understood' for m in page.caption)
    assert not any('Answering about' in m.value for m in page.markdown)


def test_single_extracted_field_is_one_line_with_working_change(monkeypatch):
    app.generate_open_answer.clear()
    graph = MagicMock()
    graph.invoke.return_value = {'answer': 'New saved answer', 'tool_results': {}}
    monkeypatch.setattr(agentic, 'build_agent_graph', lambda: graph)
    generate = graph.invoke
    page = new_page()
    result = open_result()
    result['subject'] = 'vapa'
    result['grade'] = None
    result['intent'] = {'subject': 'vapa', 'grade': None, 'raw': 'How are arts taught?',
                        'question_type': 'open', 'evidence': {'subject': 'arts'}}
    page.session_state.stage = 'results'
    page.session_state.results = result
    page.run()
    assert not page.exception
    assert 'Answering about visual and performing arts.' in [m.value for m in page.markdown]
    assert not any(m.value == 'What I understood' for m in page.caption)
    assert len(page.get('popover')) == 1
    generate.assert_not_called()
    page.selectbox(key='open_subject').set_value('math').run()
    assert not page.exception
    generate.assert_called_once()
    intent = generate.call_args.args[0]
    assert (intent['question'], intent['subject'], intent['grade']) == ('How are arts taught?', 'math', None)


def test_competition_preparation_loads_skills_and_saved_examples(monkeypatch):
    app.generate_guide.clear()
    payload = content()
    payload['standards'] = [record('2.OA.1', 2)]
    hydrate = MagicMock(return_value=payload)
    monkeypatch.setattr(app, '_hydrate', hydrate)
    result = app.generate_guide('math', 2, None, 'competition_prep', 'deterministic')
    assert result['content'] == payload
    assert result['preparation_error'] is None
    assert len(result['activity_trace']) == 1
    assert result['activity_trace'][0]['tool'] == 'competition_prep'
    page = new_page()
    page.session_state.stage = 'results'
    page.session_state.results = result
    page.run()
    assert not page.exception
    assert any(t.label == 'Learning this year' for t in page.tabs)
    activities = next(t for t in page.tabs if t.label == 'Activities')
    assert any('Everyday example' in m.value for m in activities.markdown)
    page.run()
    hydrate.assert_called_once()
    app.generate_guide.clear()
