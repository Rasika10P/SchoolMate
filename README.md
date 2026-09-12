# 📚 SchoolMate — Curriculum Guide

A grounded assistant that helps California parents understand what their child is learning at school.

California publishes content standards, curriculum frameworks and achievement descriptions, but availability differs by subject and grade. These documents are public, often run to hundreds of pages, and are written for educators. Codes such as `3.NF.1` can make it hard for parents to find the learning behind them.

This tool answers questions from those official documents, shows where each answer came from, and says so plainly when the documents do not cover the question.

---

## What it does

Seven subjects, kindergarten through grade 5, for California.

| Subject | Framework |
| --- | --- |
| English language arts | `CA-CCSS-ELA-2013` |
| English language development | `CA-ELD-2012` |
| Mathematics | `CA-CCSSM-2013` |
| Science | `CA-NGSS-2013` |
| History–social science | `CA-HSS-2016` |
| Visual and performing arts | `CA-VAPA-2019` |
| Physical education | `CA-PE-2005` |

Two ways in:

- **Guided setup** — pick subject, grade and goal from menus
- **Ask me anything** — type a question in plain English

Guided setup uses the Fixed flow or Agent mode selected on About. Ask me anything
always uses the agent after intent extraction. If no subject can be identified,
the app asks which subject the parent means instead of defaulting to mathematics.

Structured answers show learning areas collapsed by default, with friendly
headings, skill counts and official wording. Ask answers can also explain the
standards and offer general practice suggestions, clearly separated from official
requirements. Open teaching questions get a prose explanation with passage
citations instead of the structured tabs.

---

## Quick start

```bash
git clone https://github.com/Rasika10P/SchoolMate.git
cd SchoolMate

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Create .env with the values below, then export them:
set -a
source .env
set +a

python -c "from api.db import init_schema; init_schema()"
python -m api.load
python api/check_db.py    # confirm row counts per framework

streamlit run streamlit_app.py
```

### Environment

```
DATABASE_URL="postgresql://user:pass@host/db?sslmode=require"
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_HOST=...
EXTRACTOR_MODEL=openai/gpt-4o-mini
LLM_CACHE=on
```

Quote `DATABASE_URL` — Neon connection strings contain `&`, which breaks `source .env`.

The project uses hosted Postgres (Neon), but accepts a compatible Postgres
connection. Anyone using the same database URL shares its data. Keep credentials
out of Git; `.env` and populated Streamlit secrets files are ignored.

`EXTRACTOR_MODEL` and `LLM_MODEL` default to `openai/gpt-4o-mini` when unset.
Translations, overviews and parent-help suggestions currently use GPT-4o-mini.
Configure matching provider credentials if overriding a model. Fireworks is not required.

---

## How it is put together

![SchoolMate architecture](ui/assets/curriculum_guide_flowchart_with_neon.png)

Guided setup routes through the selected graph. Ask first extracts intent, then
uses the agent path: structured requests let the agent select tools with the
original question available; open teaching questions retrieve and explain framework
prose. Results are stored before the UI renders them.

### Two stores, on purpose

Standards are structured data with stable codes, so they live in Postgres and are queried exactly with SQL. Curriculum framework prose has no schema, so it lives in Pinecone and is searched by meaning.

Curriculum text across grades uses similar vocabulary, so semantic similarity alone can retrieve the wrong grade. Progression walks follow loaded links between exact standard codes, rather than inferring prerequisites from similarity.

### Two ways of deciding

Both use LangGraph and share tool definitions, but have different routing nodes.

- **Deterministic** — the Guided setup goal maps to one tool through a dictionary lookup. Tool selection and answer formatting make no model calls.
- **Agent** — for structured requests, the model reads the original question and selects tools. The graph has a recursion limit of six execution steps, not six full tool-call rounds. Open questions use the framework retrieval and explanation path.

`search_guidance` is reachable **only** from the agent path, because no goal maps to it. Open questions about how a subject is taught are structurally unanswerable by the deterministic flow. That asymmetry is the clearest instance of the autonomy trade-off in the system.

### Capability gating

California does not publish the same artefacts for every subject. `api/capability.py` is the single source of truth; the tools and the UI both import from it.

| Subject | Standards | Next steps | Standing | Activities | Programs |
| --- | --- | --- | --- | --- | --- |
| ELA | yes | yes | yes | yes | yes |
| ELD | yes | yes | yes | yes | redirect to ELA |
| Mathematics | yes | yes | yes | yes | yes |
| Science | yes | yes | grade 5 only | yes | yes |
| History–SS | yes | no | no | yes | no |
| VAPA | yes | no | no | yes | yes |
| Physical education | yes | no | no | yes | yes |

This matrix describes application capability, not complete data coverage. Unsupported features show an explanation; enabled features can still have no loaded rows. For example, loaded achievement descriptions currently cover maths and ELA in grades 3–5.

---

## Project layout

```
api/
  llm.py              every model call goes through here — caching, cost, budgets
  db.py               connection pool
  load.py             CSV → Postgres, with framework validation
  capability.py       capability matrix, domain labels, not-published reasons
  models.py           dataclasses
  check_db.py         per-framework audit
  services/
    standards.py      exact lookups
    progression.py    prerequisite graph walk
    guidance.py       curriculum service + Pinecone retrieval and relevance judgement
    parent_help.py    sourced explanation and general practice suggestions
    overviews.py      stored grade overviews and offline generation
    plain_language.py offline standard translation
  tools/
    definitions.py    five tools, GOAL_TO_TOOL, programme catalog
  graph/
    state.py          shared GuideState
    deterministic.py  fixed edges, no model calls
    agentic.py        conditional edge loop
    extractor.py      intent extraction and subject grounding
    trace.py          recorded tool calls and returned outcomes
data/
  frameworks.csv      the seven frameworks
  maths.csv, ela.csv, eld.csv, science.csv,
  history_social_science.csv, vapa.csv, pe.csv
  progression.csv     prerequisite edges, all subjects
  ald.csv             achievement level descriptors
  frameworks/         framework prose for Pinecone ingestion
  Curriculam_GoldenDataSet.csv  25-question draft evaluation set
scripts/
  paste_to_csv.py     parse pasted PDF text into standards rows
  suggest_edges.py    propose prerequisite edges for human review
  evaluate_extractor.py       extractor comparison runner
  generate_plain_language.py  offline translations
  generate_grade_overviews.py offline grade introductions
reports/              recorded extractor results and caveats
tests/                automated regression suite
streamlit_app.py      Streamlit Community Cloud entrypoint
ui/app.py             Streamlit
```

---

## Working with the data

### Loading

`python -m api.load` reads all seven subject CSVs plus `progression.csv` and `ald.csv` in one transaction. It validates that each subject CSV carries the framework ID expected for its filename — copying one CSV as a template and leaving the wrong ID is the most likely data-entry error, and it would load silently and return wrong results.

Missing subject files are reported as skipped, not treated as failures.

### Adding standards

`scripts/paste_to_csv.py` turns pasted maths standards into rows for review before appending. Supply `--page` for a known PDF page; other subjects may need their own import format. See [data instructions](data/README.md).

```bash
python scripts/paste_to_csv.py --framework CA-CCSSM-2013 --grade 4 \
    --source-url "https://..." --page 27
```

The official-wording field is kept separate from generated translations. Review imported rows and source metadata before appending; formatting cleanup and source-entry errors still need human checking.

### Adding prerequisite edges

Edges are hand-authored; they are not in the standards documents. `scripts/suggest_edges.py` proposes candidates by matching strand and number across adjacent grades, and writes them to `data/suggestions/` for review. It never writes to `progression.csv` directly — a wrong prerequisite chain would be shown to a parent as though it were official.

### Ingesting framework prose

```bash
python -m api.services.guidance --ingest --framework CA-CCSSM-2013
```

Reads `data/frameworks/<framework>.txt`, chunks on paragraph boundaries, and upserts to Pinecone using the framework ID as the namespace. The app uses one 1024-dimensional cosine index, with namespaces separating frameworks. Add `@page N` markers to prose files to record page ranges; ingestion strips the markers before embedding. Deterministic chunk IDs make re-ingestion idempotent.

Local prose files do not prove what is loaded into the remote index. Check
**About → Refresh source status** for current namespace coverage. An empty
namespace or unreachable service returns `unavailable`; retrieved passages judged
not to answer the question return `no_relevant_content`.

See [prose ingestion instructions](data/frameworks/README.md). Generate saved
standard translations and grade orientations separately:

```bash
python scripts/generate_plain_language.py --workers 4
python scripts/generate_grade_overviews.py --workers 4
```

Both jobs support framework/grade filters and dry runs. Completed work is saved
for reuse. Rendering a results page never generates these records.

---

## Cost control

Every model call goes through `cached_complete` in `api/llm.py`. Do not call the provider SDK directly anywhere.

- Responses are cached on disk by a hash that includes model, prompt and options; a cache hit avoids a new provider call. Different inputs, models or prompts can incur new calls.
- `usage_report()` reports process-local totals; `reset_usage()` resets them. About shows isolated per-submission metrics separately. Ask runs populate Agent; fixed-flow Guided setup populates Fixed flow.
- `call_budget(n)` can bound uncached provider attempts for an evaluation or other caller. It is not a global application spending cap.
- `LLM_CACHE=off` bypasses the cache for a cold run

Tool loops and repeated uncached submissions can add cost. The graph step limit bounds a run; caches and explicit submission handling prevent model calls on tab switching. The Show work trace lists real graph tool calls and outcomes, not every internal model or database operation.

```bash
python -m api.llm --report
python -m api.llm --clear-cache
```

---

## Design commitments

These are design commitments expressed in prompts and covered by regression tests. They are not guarantees that every generated response will comply.

- **Never assesses or ranks an individual child.** The Standing tab shows four achievement levels with no marker for the child. The app does not assign a child to a level.
- **Never asks for grades, test scores or teacher feedback**, in any circumstance, including when it would make the answer more useful. If a parent volunteers that data, the model is instructed not to echo or assess it and to address the underlying curriculum question instead. This is not a redaction or storage guarantee: inputs can appear in session state, prompts and local caches. Do not submit private child records.
- **Never links elementary activities to college admissions.** No evidence base exists, so any such recommendation would be invented.
- **Says so when California publishes nothing**, rather than filling the gap from the model's own knowledge.
- **Keep official wording and suggestions distinct.** Some illustrative maths rows remain; sample notices do not replace a row-by-row data audit. See [data notes](data/README.md).

---

## Evaluation

The current [draft dataset](data/Curriculam_GoldenDataSet.csv) contains 25 questions.

| Category | Rows |
| --- | ---: |
| lookup | 7 |
| progression | 3 |
| capability | 3 |
| unanswerable | 5 |
| safety | 5 |
| open | 2 |

An extractor-only comparison can be run with:

```bash
python scripts/evaluate_extractor.py
```

This makes uncached provider calls and writes results under `reports/`. It does
not run an end-to-end LangSmith evaluation. Read the
[recorded findings and annotation caveats](reports/extractor-comparison.md).
Those scores predate subsequent extractor prompt changes.

**Planned:** compare guided-only deterministic, extractor plus deterministic,
extractor plus agent, and frontier extractor plus agent on the same reviewed
questions. The four-configuration harness and results are not yet implemented
in this checkout. The About page reserves space for that comparison.

### Proposed targets — not measured results

| Metric | Target |
| --- | ---: |
| Faithfulness | ≥ 95% |
| Abstention accuracy | ≥ 90% |
| Tool selection accuracy | ≥ 90% |
| Retrieval recall@5 | ≥ 90% |
| Child assessment violations | 0 |

---

## A finding worth knowing about

We tried to reject out-of-scope questions with a similarity threshold. It does not work on this corpus.

| Question | Top score | Judged |
| --- | ---: | --- |
| How is math taught in first grade? | 0.842 | relevant |
| Why are fractions taught before decimals? | 0.803 | not relevant |
| Which school should I send my child to? | 0.810 | not relevant |

An entirely out-of-scope question scored **higher** than a plausible curriculum one. The spread across these three observations was under 0.05. Similar education vocabulary can yield high scores even when a passage does not answer the question.

The threshold was replaced with a model-judged relevance check after retrieval.

---

## Known limitations

- **No internet-search tool.** The agent cannot verify live registration pages, deadlines, fees or study-material links. Programme requests can return catalogue information and broad practice ideas without answering the logistics the parent asked about.
- **AI-generated explanations can be wrong.** Citation IDs are validated, but that alone does not prove every claim is supported. General practice suggestions are not official contest guidance.
- **Subject grounding is conservative.** Unrecognised subject terms can trigger clarification rather than a useful answer.

- **Grades K–2 have partial data** for some subjects. Missing grades return "not loaded yet", which is distinguishable from "California publishes nothing".
- **Remote ingestion coverage varies.** Inspect live source status rather than inferring index contents from local files.
- **Achievement level descriptors are policy ALDs**, not range ALDs, and cover mathematics and ELA at grades 3–5.
- **Prerequisite edges were mechanically suggested and hand-reviewed.** They are our reading of the sequence, not an official CDE mapping.
- **An open-model extractor tier was attempted** via Fireworks; the candidate model was not available on serverless at the time. The extractor node is provider-agnostic and configurable through `EXTRACTOR_MODEL`.
- **Not a substitute for a teacher conference.**

---

## Extending it

Curriculum identity uses framework IDs so the same code can exist in more than
one framework. Standards use `(framework_id, code)` as their primary key; joins
and retrieval must retain the framework context.

Adding grades beyond K–5 requires updating UI choices, validation and capability
rules as well as loading data. Other states require new framework mappings,
source adapters, coverage decisions and tests; adding a CSV alone is insufficient.

---

## Running the tests

```bash
python -m pytest -q
```

Install `requirements-dev.txt` first. Remote database and Pinecone contents are
not backed up by Git; committing CSVs does not reload a deployed database.

---

## Publish on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with GitHub and choose **Create app**.
2. Select repository `Rasika10P/SchoolMate`, branch `main`, and entrypoint
   `streamlit_app.py`.
3. In **Advanced settings**, choose Python 3.12 and paste the root-level TOML
   keys from [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example),
   replacing each placeholder with your existing credentials.
4. Deploy, then set app sharing to **public** in the app's sharing settings.

Use your existing hosted Postgres database and Pinecone index. Deployment does
not load CSVs or regenerate translations; these stay in your existing services.
Never paste `.env` shell syntax into Cloud Secrets: values must be TOML strings
as shown in the example. Do not commit a populated `secrets.toml` file.

The Streamlit version is pinned to the version tested with SchoolMate's tab
navigation. You can also test the Cloud entrypoint locally:

```sh
streamlit run streamlit_app.py
```

[Official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
