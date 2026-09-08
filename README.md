# 📚 SchoolMate

A parent-facing guide to California curriculum, built with Streamlit. Explore a
subject and grade through Guided setup, or ask how and why subjects are taught.

SchoolMate covers mathematics, English language arts, English language
development, science, history and social science, visual and performing arts,
and physical education. Coverage varies; see [data notes](data/README.md).

## Run locally

Create a Python virtual environment and install dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Create a local `.env` containing your `DATABASE_URL` (Postgres), provider API key
(such as `OPENAI_API_KEY`), `PINECONE_API_KEY`, and `PINECONE_INDEX_HOST`.
Never commit credentials. Hosted Postgres connections require SSL.

```sh
set -a
source .env
set +a
python -m api.load
python -m api.migrate
python -m streamlit run ui/app.py
```

## How it works

Structured requests use SQL-backed tools and either a fixed flow or a
LangGraph agent. Open questions retrieve framework prose from Pinecone within
the selected subject namespace, then generate a grounded response. Standards
remain in Postgres. Saved translations and grade overviews are generated
offline; results are cached, and tab switching makes no model calls.

The About tab includes an architecture diagram and runtime diagnostics.
See [framework ingestion instructions](data/frameworks/README.md) for preparing
and loading prose. Optional batch preparation:

```sh
python scripts/generate_plain_language.py --workers 4
python scripts/generate_grade_overviews.py
```

## Tests

```sh
python -m pytest -q
```

Curriculum files are included; local credentials, virtual environments, and
caches are excluded. Remote Postgres and Pinecone contents are not Git backups.
