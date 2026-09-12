# SchoolMate minimum strong evaluation — 2026-09-12

## Scope

This baseline covers the agreed minimum scope only: extractor quality on two
complementary suites and deterministic-versus-agent tool selection. It does not
evaluate Postgres retrieval, Pinecone retrieval, or final-answer generation.

## Extractor suites

Both suites used the production extractor prompt, GPT-4o-mini, temperature 0,
and an intentionally disabled response cache. All 125 calls completed.

| Metric | Balanced benchmark (100) | Real-world challenge (25) |
|---|---:|---:|
| Exact match on scored fields | 26.0% | 48.0% |
| Question type | 96.0% | 72.0% |
| Subject | 95.0% | 100.0% (24 scored) |
| Grade | 100.0% | 100.0% |
| Goal | 75.0% overall; 44.4% on structured cases | 48.0% |
| Domain | 40.0% exact match | Not annotated |
| Mean latency | 1.038 s | 0.978 s |
| Estimated cost | $0.009891 | $0.002452 |

Combined extractor cost was approximately **$0.012343**.

The balanced goal headline includes 55 open questions whose required null goal
was always correct. On the 45 structured questions, goal accuracy was 20/45.
Domain scoring is deliberately strict string equality; semantically reasonable
phrasing differences count as failures, so 40% should not be interpreted as a
semantic-domain score.

### Main findings

- Grade extraction is the strongest component: 125/125 across both suites.
- Subject extraction is also strong: 95/100 balanced and 24/24 scorable challenge cases.
- Structured/open routing is strong on controlled coverage but drops on messy
  challenge inputs, especially misspelled lookup and capability questions.
- Structured goal inference is the clearest actionable extractor weakness.
- Domain results need a normalization or semantic scoring policy before being
  used as a product-quality conclusion.

## Tool selection

The 45 balanced structured intents were supplied directly to both routes. The
agent was interrupted immediately after its first model decision, before tool
execution, so no database retrieval was included.

| Route | Accuracy | Mean latency | Provider activity | Estimated cost |
|---|---:|---:|---:|---:|
| Deterministic mapping | 45/45 (100%) | effectively 0 s | 0 calls | $0 |
| GPT-4o-mini agent | 45/45 (100%) | 0.661 s | 42 calls + 3 cache hits | $0.006045 |

The agent did not improve tool-selection accuracy over the fixed goal-to-tool
mapping. For the current four-goal design, deterministic selection is equally
accurate, faster, and cheaper. This result does not test whether an agent becomes
valuable for future multi-intent or multi-tool requests.

## Label audit

The contributor's source CSV remains unchanged. The unified adapter documents
evaluation-only corrections: G11 is working ahead based on its wording; G15
does not invent a subject or grade; unsupported and safety-only questions without
a usable structured operation are open; and G07's multi-subject label is not
scored because the extractor accepts one subject.

## Reproduction

```sh
python -m evals.run_extractor_suites
python -m evals.run_extractor_suites --run
python -m evals.run_tool_eval --deterministic-only
python -m evals.run_tool_eval
```

Raw extractor results are stored in `reports/extractor-suites.json`.
