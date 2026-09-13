# SchoolMate evaluation suites

SchoolMate keeps two complementary extractor suites and reports them separately:

- **Balanced benchmark** (`datasets/intent_cases.jsonl`): 100 fully annotated
  questions balanced across structured/open routing, difficulty, goals, and the
  Math, ELA, and Science subjects.
- **Real-world challenge set** (`../data/Curriculam_GoldenDataSet.csv`): 25
  selected Ask and supplemental guided questions containing misspellings,
  missing information, safety boundaries, unsupported requests, and all seven
  curriculum areas. Only fields supported by its existing annotations are scored.

Both suites use `extractor_suites.py` for loading, field matching, grouped
metrics, latency, usage, and cost accounting. Keeping their scores separate
prevents a larger synthetic benchmark from hiding failures in messy user input.

The source challenge CSV is not edited. Evaluation-only corrections are
documented in `CHALLENGE_OVERRIDES`: G11 is scored as working ahead because its
wording explicitly asks for third-grade preparation, G15 abstains from inventing
a subject or grade, and unsupported/safety questions without a usable tool are
scored as open. G07's multi-subject label remains unscored because the extractor
schema accepts only one subject.

## Free validation

```sh
python -m evals.run_extractor_suites
python -m pytest -q evals/test_intent_dataset.py evals/test_tool_eval.py evals/test_extractor_suites.py
```

Validation checks both datasets without calling a model.

## Run extractor evaluations

The following command disables the response cache and makes paid provider calls:

```sh
python -m evals.run_extractor_suites --run
```

Run one suite or compare explicitly selected models:

```sh
python -m evals.run_extractor_suites --suite balanced --run
python -m evals.run_extractor_suites --suite challenge --run
python -m evals.run_extractor_suites --run \
  --model openai/gpt-4o-mini \
  --model PROVIDER/MODEL
```

Detailed results default to `reports/extractor-suites.json`. The historical
contributor experiment remains reproducible through
`python scripts/evaluate_extractor.py`; it is retained rather than rewritten.

## LangSmith

With `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, and
`LANGSMITH_PROJECT=schoolmate-evals` configured in `.env`, synchronize the three
LangSmith datasets without running models:

```sh
python -m evals.run_langsmith
```

Run experiments explicitly (these may make paid provider calls):

```sh
python -m evals.run_langsmith --run extractor
python -m evals.run_langsmith --run extractor --suite balanced
python -m evals.run_langsmith --run tools
python -m evals.run_langsmith --run all
```

Stable UUIDs make synchronization idempotent. The balanced, challenge, and tool
selection datasets remain separate so their experiment scores can be compared
without mixing unlike coverage.

## Tool selection

The 45 structured balanced cases also drive the tool-selection comparison:

```sh
python -m evals.run_tool_eval --deterministic-only
python -m evals.run_tool_eval --max-cases 5
python -m evals.run_tool_eval
```

The agent is stopped after its first model decision, before the selected tool
can access PostgreSQL. This isolates tool selection from retrieval quality.
