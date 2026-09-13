# SchoolMate minimum strong evaluation — targeted improvement, 2026-09-12

## Scope

This baseline covers the agreed minimum scope only: extractor quality on two
complementary suites and deterministic-versus-agent tool selection. It does not
evaluate Postgres retrieval, Pinecone retrieval, or final-answer generation.

## Extractor suites

Both suites used the production extractor prompt, GPT-4o-mini, temperature 0,
and an intentionally disabled response cache. All 125 calls completed.

| Metric | Balanced benchmark (100) | Real-world challenge (25) |
|---|---:|---:|
| Exact match on scored fields | 40.0% | 72.0% |
| Question type | 96.0% | 88.0% |
| Subject | 93.0% | 95.8% (24 scored) |
| Grade | 100.0% | 96.0% |
| Goal | 96.0% overall | 76.0% |
| Domain | 42.0% exact match | Not annotated |
| Mean latency | 0.930 s | 0.900 s |
| Estimated cost | $0.015871 | $0.003983 |

Combined improved-extractor cost was approximately **$0.019853**.

The balanced goal headline includes 55 open questions whose required null goal
was always correct, so structured-only goal performance should be inspected
separately when iterating on the prompt.
Domain scoring is deliberately strict string equality; semantically reasonable
phrasing differences count as failures, so 40% should not be interpreted as a
semantic-domain score.

### Targeted changes

- Added directional goal cues for mastered/next, struggling/revisit, and
  expected/current grade-level wording.
- Expanded conservative subject grounding with unambiguous curriculum topics
  such as place value, textual evidence, matter, and investigations.
- Recovered terse structured requests that the model labeled as open, while
  preserving explanatory questions such as “what should count as evidence?”
- Normalized domain punctuation and removed generic suffixes such as “skills.”

### Before and after the targeted iteration

| Metric | Baseline balanced | Improved balanced | Baseline challenge | Improved challenge |
|---|---:|---:|---:|---:|
| Exact match | 21% | 40% | 56% | 72% |
| Question type | 93% | 96% | 76% | 88% |
| Subject | 84% | 93% | 95.8% | 95.8% |
| Grade | 100% | 100% | 100% | 96% |
| Goal | 76% | 96% | 60% | 76% |
| Domain | 32% | 42% | — | — |

The improved LangSmith experiments contain 100 balanced and 25 challenge target
traces with no target errors. Field evaluators may finish aggregating shortly
after the target traces appear in the LangSmith UI.

### Lessons

- The balanced and contributor datasets exposed different failure modes; neither
  score alone describes extractor quality.
- Deterministic guardrails complement the small model for high-impact routing
  fields, but should remain conservative and covered by boundary tests.
- Exact domain equality still penalizes semantically equivalent wording. Domain
  needs normalized-set or semantic agreement before it becomes a product KPI.

## Tool selection

The 45 balanced structured intents were supplied directly to both routes. The
agent was interrupted immediately after its first model decision, before tool
execution, so no database retrieval was included.

| Route | Accuracy | Mean latency | Provider activity | Estimated cost |
|---|---:|---:|---:|---:|
| Deterministic mapping | 45/45 (100%) | effectively 0 s | 0 calls | $0 |
| GPT-4o-mini agent | 45/45 (100%) | 0.002 s cached | 0 calls + 45 cache hits | $0 |

The agent did not improve tool-selection accuracy over the fixed goal-to-tool
mapping. The rerun reused all 45 prior cached responses; the original uncached
run averaged 0.661 seconds and cost $0.006045. For the current four-goal design,
deterministic selection is equally accurate, faster, and cheaper. This does not test whether an agent becomes
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
