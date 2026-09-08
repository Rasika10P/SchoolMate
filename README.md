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
