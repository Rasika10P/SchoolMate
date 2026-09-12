"""Run with streamlit run ui/app.py; only explicit submissions perform I/O."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import logging
import re
from urllib.parse import urlparse
import sys
from time import perf_counter
from typing import Any

# Streamlit may put ui/ rather than the repository root on its import path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import psycopg

from api import llm
from api.capability import SUBJECTS, NOT_PUBLISHED_REASON, supports, domain_label
from api.db import get_conn
from api.graph.agentic import build_agent_graph
from api.graph.deterministic import build_deterministic_graph
from api.graph.extractor import extract_intent
from api.graph.state import GuideState
from api.graph.trace import describe_event, source_urls
from api.services.progression import walk
from api.services.overviews import get_overviews
from api.services.standards import get_achievement_levels, get_standards
from api.tools.definitions import FRAMEWORK_BY_SUBJECT, competition_prep

logger = logging.getLogger(__name__)


SUBJECT_LABELS = {
    "math": "Mathematics", "ela": "English language arts", "eld": "English language development",
    "sci": "Science", "hss": "History and social science", "vapa": "Visual and performing arts",
    "pe": "Physical education",
}
GOAL_LABELS = {
    "on_grade_level": "Understand this grade", "working_ahead": "Get ready for next year",
    "catching_up": "Fill in earlier gaps", "competition_prep": "Find outside programmes",
}


def grade_label(grade: int) -> str:
    return "Kindergarten" if grade == 0 else f"Grade {grade}"


def _hydrate(subject: str, grade: int, domain: str | None) -> dict[str, Any]:
    """Read full text and provenance once, before rendering any results tabs."""
    framework_id = FRAMEWORK_BY_SUBJECT[subject]
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT source_url, version FROM framework WHERE id = %s", (framework_id,))
            framework = cursor.fetchone()
        standards = get_standards(conn, framework_id, grade, domain) if supports(subject, "standards", grade) else []
        levels = get_achievement_levels(conn, framework_id, grade) if supports(subject, "standing", grade) else []
        forward, backward = [], []
        if supports(subject, "next_steps", grade):
            for standard in standards:
                forward.extend(walk(conn, standard.framework_id, standard.code, "forward"))
                backward.extend(walk(conn, standard.framework_id, standard.code, "backward"))
        overview_ids = [(framework_id, grade)]
        overview_ids += [(step.standard.framework_id, step.standard.grade) for step in forward + backward
                         if step.standard.grade is not None]
        overviews = get_overviews(conn, overview_ids)
    return {
        "overviews": overviews,
        "standards": [asdict(row) for row in standards],
        "standing": [asdict(row) for row in levels],
        "forward": [asdict(row) for row in forward], "backward": [asdict(row) for row in backward],
        "source_url": framework[0] if framework else "", "version": framework[1] if framework else "",
        "data_state": "available" if standards else "not_loaded",
        "data_reason": f"{SUBJECT_LABELS[subject]} data for {grade_label(grade)}"
                       + (f" in domain {domain}" if domain else "") + " has not been loaded yet.",
        "programs": competition_prep.invoke({"subject": subject, "grade": grade}),
    }


@st.cache_data(show_spinner=False)
def generate_guide(subject: str, grade: int, domain: str | None, goal: str, mode: str) -> dict[str, Any]:
    """Five-key cache: identical submissions do not repeat graph or database work."""
    # Programme recommendations come from the catalogue, not Postgres.
    content = None if goal == "competition_prep" else _hydrate(subject, grade, domain)
    tab = "standing" if goal == "on_grade_level" else "next_steps"
    if goal != "competition_prep" and supports(subject, tab, grade) and content.get("data_state") == "not_loaded":
        return {"content": content, "answer": content["data_reason"], "activity_trace": [],
                "tool_results": {"state": "not_loaded", "reason": content["data_reason"]},
                "subject": subject, "grade": grade, "domain": domain, "goal": goal, "mode": mode}
    graph = build_deterministic_graph() if mode == "deterministic" else build_agent_graph()
    state = GuideState(subject=subject, grade=grade, domain=domain, goal=goal,
                       tool_results={}, answer="", messages=[])
    result = graph.invoke(state)
    return {"content": content, "answer": result["answer"], "tool_results": result["tool_results"],
            "activity_trace": result.get("activity_trace", []),
            "subject": subject, "grade": grade, "domain": domain, "goal": goal, "mode": mode}


def _usage_snapshot() -> tuple[int, int, float, int]:
    entries = list(llm.usage.models.values())
    return (sum(row.calls for row in entries),
            sum(row.prompt_tokens + row.completion_tokens for row in entries),
            sum(row.estimated_cost or 0 for row in entries),
            sum(row.calls for row in entries if row.estimated_cost is None))


def _show_failure(exc: Exception) -> None:
    """Log the complete exception and present the relevant failure category."""
    logger.error("Guide preparation failed", exc_info=(type(exc), exc, exc.__traceback__))
    st.session_state.last_error_detail = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError)) or (
        isinstance(exc, RuntimeError) and str(exc) == "DATABASE_URL must be set"
    ):
        st.error("We couldn’t reach the learning information. Please try again shortly.")
    else:
        st.error("We couldn’t prepare your guide. Please try again shortly.")


@st.cache_data(show_spinner=False)
def database_diagnostics() -> dict[str, Any]:
    """One cached connection and grouped query, never refreshed by tab switching."""
    reachable = False
    try:
        with get_conn() as conn:
            reachable = True
            with conn.cursor() as cursor:
                cursor.execute("SELECT framework_id, grade, COUNT(*) FROM standard GROUP BY framework_id, grade ORDER BY framework_id, grade")
                counts = {(row[0], row[1]): row[2] for row in cursor.fetchall()}
        rows = [{"Subject": SUBJECT_LABELS[subject], "Framework": FRAMEWORK_BY_SUBJECT[subject],
                 **{grade_label(grade): counts.get((FRAMEWORK_BY_SUBJECT[subject], grade), 0) for grade in range(6)}}
                for subject in SUBJECTS]
        return {"reachable": True, "rows": rows, "error": None}
    except Exception as exc:
        logger.exception("Database diagnostics failed")
        return {"reachable": reachable, "rows": [], "error": f"{type(exc).__name__}: {exc}"}


def _record_usage(before: tuple[int, int, float, int], started: float, mode: str | None = None) -> None:
    after = _usage_snapshot()
    metrics = {"calls": after[0] - before[0], "tokens": after[1] - before[1],
               "elapsed": perf_counter() - started,
               "cost": None if after[3] > before[3] else after[2] - before[2]}
    if mode is None:
        st.session_state.extraction_usage = metrics
    else:
        stats = dict(st.session_state.get("mode_stats", {}))
        stats[mode] = metrics
        st.session_state.mode_stats = stats
        st.session_state.submission_usage = metrics


def _begin_submission() -> None:
    """A previous answer must never appear to answer a new, failed request."""
    st.session_state.pop("results", None)
    st.session_state.pop("last_error_detail", None)
    st.session_state.stage = "entry"


def _submit_guide(subject: str | None, grade: int | None, goal: str | None, learner: str) -> None:
    if any(value is None for value in (subject, grade, goal)):
        st.warning("Please choose a subject, grade, and goal before continuing.")
        return
    _begin_submission()
    before, started = _usage_snapshot(), perf_counter()
    mode = st.session_state.mode
    try:
        with st.spinner("Preparing your guide…"):
            result = generate_guide(subject, grade, None, goal, mode)
    except Exception as exc:
        _record_usage(before, started, mode)
        _show_failure(exc)
        return
    _record_usage(before, started, mode)
    st.session_state.active_entry = "Guided setup"
    st.session_state.results = result
    st.session_state.draft = dict(subject=subject, grade=grade, learner=learner, goal=goal, domain="")
    st.session_state.stage = "results"
    st.rerun()


def _english_support(key: str) -> str:
    with st.expander("My child receives English language support (optional)", expanded=False):
        enabled = st.checkbox("Include English language development standards", key=key)
    return "Yes" if enabled else "No"


def _remember_entry_tab() -> None:
    st.session_state.active_entry = st.session_state.entry_tabs


def entry_screen() -> None:
    st.write("A learning guide for your child")
    st.write("Explore California learning expectations, earlier skills, and opportunities to grow.")
    st.session_state.setdefault("active_entry", "Ask me anything" if st.session_state.get("results", {}).get("intent") else "Guided setup")
    guided, free_text = st.tabs(["Guided setup", "Ask me anything"], key="entry_tabs",
                               default=st.session_state.active_entry, on_change=_remember_entry_tab)
    draft = st.session_state.get("draft", {})
    for field in ("subject", "grade", "goal"):
        st.session_state.setdefault(f"guide_{field}", draft.get(field))
    st.session_state.setdefault("guide_learner", draft.get("learner") == "Yes")
    with guided, st.form("guided_setup"):
        subject = st.selectbox("Subject", SUBJECTS, index=None, format_func=SUBJECT_LABELS.get, key="guide_subject")
        grade = st.selectbox("Grade", range(6), index=None, format_func=grade_label, key="guide_grade")
        goal = st.selectbox("Goal", list(GOAL_LABELS), index=None, format_func=GOAL_LABELS.get, key="guide_goal")
        learner = _english_support("guide_learner")
        if st.form_submit_button("See guide", type="primary"):
            _submit_guide(subject, grade, goal, learner)
    with free_text, st.form("ask_form"):
        question = st.text_area("What would you like help with?")
        if st.form_submit_button("Ask"):
            if not question.strip():
                st.warning("Please enter a question first.")
                return
            for field in ("subject", "grade"):
                st.session_state.pop(f"open_{field}", None)
            _begin_submission()
            before, started = _usage_snapshot(), perf_counter()
            try:
                with st.spinner("Reviewing your question…"):
                    intent = extract_intent(question)
            except Exception as exc:
                _record_usage(before, started)
                _show_failure(exc)
                return
            _record_usage(before, started)
            st.session_state.extracted_intent = intent
            _submit_question(intent)


@st.cache_data(show_spinner=False)
def generate_open_answer(question: str, subject: str, grade: int | None) -> dict[str, Any]:
    """Cited explanation cache v2; synthesis runs only on explicit submission."""
    return build_agent_graph().invoke(GuideState(
        subject=subject, grade=grade, domain=None, goal=None, question=question,
        question_type="open", messages=[], tool_results={}, answer=""))


def _normalize_intent(intent: dict[str, Any]) -> dict[str, Any]:
    """Accept older session/extractor results without requiring a server restart."""
    evidence = intent.get("evidence")
    return {**intent,
            "question_type": "structured" if intent.get("question_type") == "structured" else "open",
            "evidence": dict(evidence) if isinstance(evidence, dict) else {}}


def _submit_question(intent: dict[str, Any], *, rerun: bool = True) -> None:
    intent = _normalize_intent(intent)
    is_open = intent["question_type"] == "open"
    if is_open:
        intent["goal"] = None
    elif not intent.get("goal"):
        intent["goal"] = "on_grade_level"
        intent["evidence"]["goal"] = "Assumed: Understand this grade. You can change this."
    subject, grade = intent.get("subject"), intent.get("grade")
    result = {"subject": subject, "grade": grade, "goal": intent.get("goal"),
              "intent": intent, "question_type": intent["question_type"], "answer": "",
              "tool_results": {}, "content": None}
    if subject is not None and (is_open or grade is not None):
        before, started = _usage_snapshot(), perf_counter()
        mode = "agent" if is_open else st.session_state.mode
        try:
            with st.spinner("Preparing your answer…"):
                if is_open:
                    result.update(generate_open_answer(intent["raw"], subject, grade))
                else:
                    result.update(generate_guide(subject, grade, None, intent["goal"], mode))
        except Exception as exc:
            _show_failure(exc)
            result["answer"] = "We couldn’t prepare your answer. Please try again shortly."
            result["explanation_error"] = True
        finally:
            _record_usage(before, started, mode)
    st.session_state.active_entry = "Ask me anything"
    st.session_state.results = result
    st.session_state.stage = "results"
    st.session_state.extracted_intent = intent
    if rerun:
        st.rerun()


def _next_standards(content: dict[str, Any], grade: int) -> list[dict[str, Any]]:
    """Show mapped next-grade standards once, never infer an unmapped sequence."""
    rows = {}
    for step in content["forward"]:
        record = step["standard"]
        if record["grade"] == grade + 1:
            rows.setdefault((record["framework_id"], record["code"]), record)
    return list(rows.values())


def _earlier_standards(content: dict[str, Any], grade: int) -> list[dict[str, Any]]:
    rows = {}
    for step in content["backward"]:
        record = step["standard"]
        if record.get("grade") is not None and record["grade"] < grade:
            rows.setdefault((record["framework_id"], record["code"]), record)
    return list(rows.values())


def _standards_table(records: list[dict[str, Any]]) -> None:
    """Compact stacked entries; show official fallback exactly once."""
    for index, record in enumerate(records):
        if index:
            st.divider()
        summary = (record.get("plain_summary") or "").strip()
        example = (record.get("plain_example") or "").strip()
        translated = bool(summary and summary != record["text"].strip())
        if translated:
            st.write(summary)
        else:
            st.caption("Official wording")
            st.write(record["text"])
        if example:
            st.markdown("**Everyday example**")
            st.write(example)
        with st.expander("See the official wording" if translated else "Standard reference"):
            if translated:
                st.write(record["text"])
            st.text(f"{record['code']} · {record['framework_id']}")


def _domain_sections(records: list[dict[str, Any]], grade: int, overviews: dict[str, str]) -> None:
    if not records:
        return
    frameworks = sorted({record["framework_id"] for record in records})
    for framework in frameworks:
        text = overviews.get(f"{framework}|{grade}")
        if text and len(re.split(r"(?<=[.!?])\s+", text.strip())) <= 2:
            st.write(text)
        else:
            # A useful static fallback until the offline job stores an overview.
            areas = sorted({domain_label(r["domain"], grade)[0].lower()
                            for r in records if r["framework_id"] == framework})
            st.write("This grade includes " + ", ".join(areas) +
                     ". Open an area to see the skills and any available everyday examples.")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["domain"], []).append(record)
    for domain, rows in sorted(groups.items(), key=lambda item: domain_label(item[0], grade)[0]):
        heading, _ = domain_label(domain, grade)
        with st.expander(heading, expanded=False):
            st.write(domain_label(rows[0]["domain"], grade)[1])
            _standards_table(rows)


def _source_documents(content: dict[str, Any], subject: str, grade: int) -> list[dict[str, Any]]:
    records = list(content["standards"]) if supports(subject, "standards", grade) else []
    if supports(subject, "next_steps", grade):
        # The goal answer can cite earlier and later linked standards too.
        records += [step["standard"] for step in content["forward"] + content["backward"]]
    if supports(subject, "standing", grade):
        records += content["standing"]
    records += content["programs"].get("programs", [])
    documents: dict[str, set[int]] = {}
    titles: dict[str, str] = {}
    for record in records:
        url = record.get("source_url")
        if not url:
            continue
        pages = documents.setdefault(url, set())
        name = next((SUBJECT_LABELS[key].lower() for key, framework in FRAMEWORK_BY_SUBJECT.items()
                     if framework == record.get("framework_id")), SUBJECT_LABELS[subject].lower())
        title = record.get("document_title") or record.get("source_title")
        if not title:
            title = f"California {name} standards" if urlparse(url).hostname in {"www.cde.ca.gov", "www2.cde.ca.gov"} else f"{name.capitalize()} source document"
            if "/id/web/" in url and record.get("domain"):
                title += " — " + domain_label(record["domain"], grade)[0]
        titles.setdefault(url, title)
        if record.get("page") is not None:
            pages.add(record["page"])
    return [{"url": url, "title": titles[url], "pages": (f"Page {min(pages)}" if len(pages) == 1 else
             f"Pages {min(pages)}–{max(pages)}") if pages else ""}
            for url, pages in sorted(documents.items())]


def _sources(content: dict[str, Any], subject: str, grade: int) -> None:
    with st.expander("Sources", expanded=False):
        documents = _source_documents(content, subject, grade)
        for document in documents:
            suffix = f" — {document['pages']}" if document['pages'] else ""
            st.markdown(f"[{document['title']}]({document['url']}){suffix}")
        if not documents:
            st.write("Source documents have not been recorded for these results.")


def _refresh_translations(content: dict[str, Any]) -> None:
    """Refresh only stored text; never run a guide graph or a model."""
    records = list(content["standards"])
    records += [step["standard"] for step in content["forward"] + content["backward"]]
    identities = sorted({(row["framework_id"], row["code"]) for row in records})
    if not identities:
        return
    placeholders = ", ".join("(%s, %s)" for _ in identities)
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT framework_id, code, text, plain_summary, plain_example FROM standard "
                f"WHERE (framework_id, code) IN ({placeholders})",
                tuple(value for identity in identities for value in identity),
            )
            translations = {(row[0], row[1], row[2]): row[3:] for row in cursor.fetchall()}
    for record in records:
        summary, example = translations.get(
            (record["framework_id"], record["code"], record["text"]), (None, None))
        record.update(plain_summary=summary, plain_example=example)


def _open_intent(result):
    return _normalize_intent(result.get("intent") or {
        "subject": result.get("subject"), "grade": result.get("grade"),
        "question_type": "open", "raw": result.get("question", ""), "evidence": {}})


def _open_change(field):
    intent = _open_intent(st.session_state.results)
    intent[field] = st.session_state[f"open_{field}"]
    intent["evidence"][field] = "Chosen by you"
    _submit_question(intent, rerun=False)


def _full_guide():
    result = st.session_state.results
    st.session_state.draft = dict(subject=result.get("subject"), grade=result.get("grade"),
                                  goal="on_grade_level", learner="No", domain="")
    for field in ("subject", "grade", "goal"):
        st.session_state[f"guide_{field}"] = st.session_state.draft[field]
    st.session_state.active_entry = "Guided setup"
    st.session_state.entry_tabs = "Guided setup"
    st.session_state.stage = "entry"


def _passage_label(chunk):
    metadata = chunk.get("metadata", {})
    title = metadata.get("document_title") or "Curriculum framework"
    page, end = metadata.get("page"), metadata.get("page_end")
    if page is not None:
        title += f", pp. {page}–{end}" if end is not None and end != page else f", p. {page}"
    return title


def _show_passage(chunk):
    st.write(chunk["text"])
    url = chunk.get("metadata", {}).get("source_url")
    if url:
        st.markdown(f"[Open source document]({url})")


def open_results_screen(result):
    intent = _open_intent(result)
    question = " ".join(intent.get("raw", "").split())
    question = question[:1].upper() + question[1:]
    st.subheader(question or "Explore the curriculum")
    retrieval = result.get("tool_results", {}).get("guidance", {})
    status = retrieval.get("state")
    chunks = retrieval.get("chunks", [])
    if status == "no_relevant_content":
        st.info(retrieval["reason"], icon="ℹ️")
    elif status in {"unavailable", "judgement_unavailable"} or result.get("explanation_error"):
        st.error(retrieval.get("reason") or result["answer"])
    elif not result.get("subject"):
        st.info("Which subject did you have in mind?")
    elif result.get("open_paragraphs"):
        by_id = {chunk["id"]: chunk for chunk in chunks}
        for paragraph in result["open_paragraphs"]:
            st.write(paragraph["text"])
            for identity in dict.fromkeys(paragraph["citations"]):
                with st.expander(_passage_label(by_id[identity]), expanded=False):
                    _show_passage(by_id[identity])
    elif result.get("answer"):
        # Older saved sessions remain readable; no synthesis on rerun.
        st.write(result["answer"])

    st.caption("What I understood")
    fields = ["subject"] + (["grade"] if intent.get("grade") is not None else [])
    for field in fields:
        value = intent.get(field)
        label = SUBJECT_LABELS.get(value, "Not mentioned") if field == "subject" else grade_label(value)
        name, content, evidence, action = st.columns([1, 2, 3, 1])
        name.write(field.title())
        content.write(label)
        evidence.write(intent.get("evidence", {}).get(field) or ("Not mentioned" if value is None else "From your question"))
        with action:
            with st.popover("Change" if value is not None else "Add"):
                options = list(SUBJECT_LABELS) if field == "subject" else list(range(6))
                key = f"open_{field}"
                st.session_state.setdefault(key, value)
                st.selectbox(field.title(), options, index=None, key=key,
                    format_func=SUBJECT_LABELS.get if field == "subject" else grade_label,
                    on_change=_open_change, args=(field,))
    if status == "no_relevant_content":
        followup = "Explore this subject with Guided setup"
    elif result.get("grade") is not None and result.get("subject"):
        year = {0: "kindergarteners", 1: "first graders", 2: "second graders", 3: "third graders", 4: "fourth graders", 5: "fifth graders"}.get(result["grade"], "children")
        followup = f"Want to see what {year} actually learn in {SUBJECT_LABELS[result['subject']].lower()}? See the full guide"
    else:
        followup = "Want to explore learning by grade? See the full guide"
    st.button(followup, type="tertiary", on_click=_full_guide)
    if chunks:
        titles = list(dict.fromkeys(c.get("metadata", {}).get("document_title") or "Curriculum framework" for c in chunks))
        with st.expander(f"Based on {len(chunks)} passages from {', '.join(titles)}", expanded=False):
            for chunk in chunks:
                st.caption(_passage_label(chunk))
                _show_passage(chunk)


def _activity_trace(result):
    events = result.get("activity_trace")
    if events is None:
        # Older cached results have no execution record; never reconstruct guesses.
        return
    sources = set()
    for event in events:
        sources.update(source_urls(event["result"]))
    with st.expander(f"Checked {len(sources)} sources · {len(events)} steps · Show work", expanded=False):
        st.caption("Recorded tool calls from this result. Sources count distinct source URLs returned by those tools; internal model calls and page rendering are not counted.")
        if not events:
            st.write("No tool calls were made for this result.")
        for number, event in enumerate(events, 1):
            st.write(f"{number}. {describe_event(event)}")


def results_screen() -> None:
    result = st.session_state.results
    if result.get("question_type") == "open":
        open_results_screen(result)
        return
    if result.get("content") is None:
        st.divider()
        if result.get("answer"):
            st.write(result["answer"])
        elif not result.get("subject"):
            st.info("Which subject did you have in mind?")
        else:
            st.info("Please include the grade in your question so I can find learning information for that year.")
        return
    content = result["content"]
    subject, grade = result["subject"], result["grade"]
    st.subheader(f"{SUBJECT_LABELS[subject]} · {grade_label(grade)}")
    if content.get("data_state") == "not_loaded":
        st.info(content["data_reason"])
    if st.button("Refresh guide text"):
        try:
            _refresh_translations(content)
            overview_ids = [(FRAMEWORK_BY_SUBJECT[subject], grade)]
            overview_ids += [(r["framework_id"], r["grade"]) for r in
                             _next_standards(content, grade) + _earlier_standards(content, grade)]
            with get_conn() as conn:
                content["overviews"] = get_overviews(conn, overview_ids)
        except Exception as exc:
            _show_failure(exc)
    if "sample" in content["version"].lower() or "example.com" in content["source_url"]:
        st.warning("This guide contains illustrative sample data, not official California wording.")
    with st.expander("Answer for your selected goal"):
        st.write(result["answer"])
    catching_up = result.get("goal") == "catching_up"
    tabs = st.tabs(["Learning this year", "Skills to revisit" if catching_up else "Next steps", "Standing", "Activities", "Outside programmes"])
    for tab, key in zip(tabs, ("standards", "next_steps", "standing", "activities", "programs")):
        with tab:
            supported = supports(subject, key, grade)
            programs = content["programs"]
            if key == "programs":
                st.caption("These are third-party programmes, not state recommendations.")
            if not supported:
                st.info(NOT_PUBLISHED_REASON[subject, key])
                if key != "programs" or programs["state"] != "redirect":
                    continue
            if key == "standards":
                _domain_sections(content["standards"], grade, content.get("overviews", {}))
                if not content["standards"]:
                    st.info(content.get("data_reason", f"{SUBJECT_LABELS[subject]} data for {grade_label(grade)} has not been loaded yet."))
            elif key == "next_steps":
                if catching_up:
                    earlier = _earlier_standards(content, grade)
                    if earlier:
                        st.write("These earlier skills connect to the learning in this grade. Choose an area to revisit together.")
                        for earlier_grade in sorted({r["grade"] for r in earlier}, reverse=True):
                            st.markdown(f"**{grade_label(earlier_grade)} skills**")
                            _domain_sections([r for r in earlier if r["grade"] == earlier_grade],
                                             earlier_grade, content.get("overviews", {}))
                    else:
                        st.info("Earlier skill connections have not been mapped for this subject and grade yet.")
                else:
                    next_records = _next_standards(content, grade)
                    if next_records:
                        st.write("Once these are comfortable, children usually move on to:")
                        _domain_sections(next_records, grade + 1, content.get("overviews", {}))
                    elif not content["forward"] and not content["backward"]:
                        st.info("The sequence has not been mapped for this subject yet.")
                    else:
                        st.info("The sequence to the next grade has not been mapped for these skills yet.")
            elif key == "standing":
                st.write("These describe published achievement levels, not an assessment of your child.")
                for descriptor in content["standing"]:
                    st.markdown(f"**Level {descriptor['level']}**")
                    st.write(descriptor["text"])
                if not content["standing"]:
                    st.info("No achievement descriptions are loaded for this grade yet.")
                citations = {}
                for descriptor in content["standing"]:
                    if descriptor.get("source_url"):
                        pages = citations.setdefault(descriptor["source_url"], set())
                        if descriptor.get("page") is not None:
                            pages.add(descriptor["page"])
                for url, pages in sorted(citations.items()):
                    page_note = (f" — Page {min(pages)}" if len(pages) == 1 else
                                 f" — Pages {min(pages)}–{max(pages)}") if pages else ""
                    st.markdown(f"[California {SUBJECT_LABELS[subject].lower()} achievement descriptions]({url}){page_note}")
            elif key == "activities":
                st.info("Sourced activities have not been added to this guide yet.")
            elif key == "programs":
                if programs["state"] == "not_published":
                    st.info(programs["reason"])
                else:
                    if programs["state"] == "redirect":
                        st.warning(programs["reason"])
                    for program in programs["programs"]:
                        minimum = "Kindergarten" if program["min_grade"] == 0 else str(program["min_grade"])
                        st.warning(f"{program['name']} — grades {minimum}–{program['max_grade']}. {program['note']}")

    _sources(content, subject, grade)

def _select_mode() -> None:
    # Store the preference independently of the widget, which is absent on Guide.
    st.session_state.mode = st.session_state._mode_choice


def how_this_works_screen() -> None:
    st.header("About SchoolMate")
    st.write("SchoolMate helps families make sense of California’s school curriculum. "
             "Explore what children learn, discover how subjects are taught, and find earlier or next steps "
             "without having to navigate curriculum documents on your own.")
    st.write("The project brings together mathematics, English language arts, English language development, "
             "science, history and social science, the arts, and physical education. "
             "Guided setup helps you explore a subject and grade; Ask me anything lets you ask in your own words.")
    st.write("Standards and teaching guidance come from California Department of Education source material. "
             "Saved plain-language descriptions make standards easier to read, with official wording and sources "
             "available for reference. Coverage varies by subject; SchoolMate identifies information that has not been loaded or published.")
    st.subheader("How SchoolMate is built")
    st.graphviz_chart("""digraph SchoolMate {
        graph [rankdir=TB, bgcolor="transparent", pad="0.2", nodesep="0.4"];
        node [shape=box, style="rounded,filled", fillcolor="#eef4ff", color="#8196b4", fontname="Arial", fontsize=16];
        edge [color="#60748d", fontname="Arial", fontsize=12];
        ui [label="SchoolMate · Streamlit\\nGuided setup + Ask me anything"];
        intent [label="Question classification\\nSubject, grade and question type"];
        sqlroute [label="Structured requests\\nFixed flow or tool-calling agent"];
        proseroute [label="Open questions\\nFramework search + agent response"];
        db [label="Postgres\\nStandards, progressions and achievement levels"];
        pine [label="Pinecone\\nFramework prose by subject namespace"];
        offline [label="Offline preparation\\nCSV loading, prose ingestion, translations and overviews"];
        cache [label="Cached results\\nAnswers, learning areas and sources"];
        ui -> sqlroute [label="guided choices"];
        ui -> intent [label="question"];
        intent -> sqlroute [label="what / next / revisit"];
        intent -> proseroute [label="how / why"];
        sqlroute -> db;
        proseroute -> pine;
        offline -> db;
        offline -> pine;
        db -> cache;
        pine -> cache;
    }""", width="stretch")
    st.caption("Standards are queried from Postgres; framework prose is retrieved from Pinecone. "
               "Translations and grade overviews are generated offline. Switching tabs reuses the current result.")
    st.subheader("Explore the two approaches")
    st.write("Both approaches use the same California curriculum information. "
             "The fixed flow follows a set route based on your choices. "
             "The agent uses an AI model to choose which information to look up and put together an answer.")
    st.write("Changing the approach applies to your next See guide submission. "
             "It does not change an open guide. Identical submissions reuse saved results. "
             "Open questions about how or why a subject is taught always use the agent with framework teaching guidance.")
    st.session_state.setdefault("_mode_choice", st.session_state.mode)
    st.radio("Guide mode", ["deterministic", "agent"], key="_mode_choice", on_change=_select_mode,
             format_func=lambda value: "Fixed flow" if value == "deterministic" else "Agent")
    cards = st.columns(2)
    for card, mode, label in zip(cards, ("deterministic", "agent"), ("Fixed flow", "Agent")):
        with card, st.container(border=True):
            st.subheader(label)
            metrics = st.session_state.get("mode_stats", {}).get(mode)
            st.metric("Model calls", metrics["calls"] if metrics else "—")
            st.metric("Elapsed time", f"{metrics['elapsed']:.2f} s" if metrics else "—")
            cost = "—" if metrics is None else "Unknown" if metrics["cost"] is None else f"${metrics['cost']:.6f}"
            st.metric("Estimated cost", cost)
            if metrics:
                st.caption(f"Last submission · {metrics['tokens']} tokens")
            else:
                st.caption("No guide submitted with this approach in this session.")
    st.caption("Cards show the latest submission for each approach, including time spent reading saved results. "
               "They are not a controlled comparison unless the guide choices match. "
               "Usage counters are process-wide, so simultaneous users can affect these estimates.")
    extraction = st.session_state.get("extraction_usage")
    if extraction:
        st.caption(f"Last question review · {extraction['calls']} model calls · {extraction['tokens']} tokens")
    st.subheader("Usage readout")
    st.caption("Process-wide usage across all sessions; saved responses require no new provider calls.")
    st.code(llm.usage_report(), language=None)
    with st.expander("Database diagnostics", expanded=False):
        if st.button("Refresh diagnostics"):
            database_diagnostics.clear()
            st.session_state.diagnostics = database_diagnostics()
        diagnostics = st.session_state.get("diagnostics")
        if diagnostics is None:
            st.write("Select Refresh diagnostics to check the connection and loaded data.")
        else:
            st.write(f"Database reachable: {'yes' if diagnostics['reachable'] else 'no'}")
            if diagnostics["error"]:
                st.error(diagnostics["error"])
            else:
                st.dataframe(diagnostics["rows"], hide_index=True)
        st.caption("Cached snapshot; refresh after loading data or changing the database.")
    if st.session_state.get("last_error_detail"):
        with st.expander("Latest error details"):
            st.code(st.session_state.last_error_detail, language=None)


def main() -> None:
    st.set_page_config(page_title="SchoolMate", page_icon="📚", layout="wide")
    # 12 points = 16 CSS pixels. Keep headings larger and allow browser zoom.
    st.html("""<style>
        html { font-size: 16px; }
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] li,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] input,
        [data-testid="stAppViewContainer"] textarea,
        [data-testid="stAppViewContainer"] button,
        [data-baseweb="select"], [role="option"] {
            font-size: max(1rem, 12pt);
            line-height: 1.5;
        }
    </style>""")
    st.session_state.setdefault("stage", "entry")
    st.session_state.setdefault("mode", "deterministic")
    # Migrate the previous page name for sessions already open during an update.
    if st.session_state.get("page") == "How this works":
        st.session_state.page = "About"
    st.title("📚 SchoolMate")
    guide_tab, about_tab = st.tabs(["Guide", "About"], key="page", on_change="rerun")
    if about_tab.open:
        with about_tab:
            how_this_works_screen()
    else:
        with guide_tab:
            entry_screen()
            if st.session_state.stage == "results":
                results_screen()
                _activity_trace(st.session_state.results)


if __name__ == "__main__":
    main()
