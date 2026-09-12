# Extractor comparison — 2026-09-11

## Setup

- Candidate: `fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct` (from local EXTRACTOR_MODEL).
- Baseline: `openai/gpt-4o-mini`. Same production extractor prompt, temperature 0, one sample per question.
- Local response cache disabled for this process only; existing cache preserved. The model string is part of the SHA-256 response-cache key.
- Dataset: `data/Curriculam_GoldenDataSet.csv`; `data/golden.csv` does not exist locally.
- 22 ask-path rows, including `Asl`/`asl` entry-path typos. Three extra guided rows G07/G11/G13 run as requested, excluded from ask-path aggregate.
- Subject label normalization: Ala→ela, Els→eld, case normalization, empty/None/null→null. Trailing comma removed from expected_tool.
- Exact match compares subject and grade, plus goal/question_type when an expected structured tool or search_guidance provides a label. None/no-tool rows have no gold goal/question_type; these fields are not scored for those rows. No domain/evidence gold labels exist.
- These are raw extractor results, before the UI defaults missing structured goals to on_grade_level. Tool expectations are an imperfect proxy for extraction labels.
- Latency is wall-clock extraction time including provider/network overhead, not generation alone. One run is not a robust latency benchmark. No retrieval, database, or answer synthesis calls.

## Endpoint outcome

Fireworks returned `NotFoundError` on G01 after 1.79 seconds. Stopped further candidate requests to avoid repeating an unavailable endpoint. No candidate accuracy or successful latency/cost comparison can be reported; failure is not a wrong extraction. No billed token usage was returned, so actual failed-request cost is unknown.

Fireworks' model page currently marks serverless unsupported:
https://fireworks.ai/models/fireworks/llama-v3p1-8b-instruct

Verified the current dense 4B–16B serverless tier rate ($0.20/M input, $0.20/M output), checked 2026-09-11:
https://docs.fireworks.ai/serverless/pricing
This is not a quote for dedicated deployment costs. An accessible model/deployment is needed to complete the comparison.

## Baseline: 22 ask rows

| Metric | GPT-4o-mini |
|---|---:|
| Strict match on annotated fields | 10/22 (45.5%) |
| Subject | 21/22 (95.5%) |
| Grade | 21/22 (95.5%) |
| Goal, where annotated | 3/15 (20.0%) |
| Question type, where inferred from expected tool | 8/15 (53.3%) |
| Mean latency | 0.960 s |
| Estimated token cost, ask rows | $0.00210435 |

Baseline estimate uses the repository's GPT-4o-mini rates ($0.15/M input, $0.60/M output), with provider-reported token counts. It is an estimate, not an invoice.

## Requested rows

| Row | Extracted subject / grade / type / goal | Observation |
|---|---|---|
| G07 | null / 2 / open / null | Did not resolve STEM into a single subject or detect competition intent. Gold lists two subjects although extractor accepts one. Guided row. |
| G11 | math / 2 / structured / working_ahead | Matches the question's next-grade intent, but gold expects on_grade_level. Guided row. |
| G13 | hss / 3 / structured / null | Subject and grade correct; UI will default the missing goal to on_grade_level. Guided row. |
| G16 | null / null / open / null | Does not infer child ability, subject, or grade. Matches the annotated subject/grade. Ask row. |

## Annotation issues

G15 ('Is my child behind?') contains no subject or grade, but gold labels math/3. Both subject and grade errors in the baseline aggregate come from this row; abstaining is consistent with the extractor prompt. G11's goal conflicts with its wording. G07's multi-subject gold value cannot be represented by this extractor. None of these gold records were edited. Review these labels before treating the aggregate as a model-quality conclusion.

Full extracted fields, timing, usage, and comparisons are in `extractor-comparison.json`. Reproduce with `python scripts/evaluate_extractor.py`; it makes paid uncached calls and saves results after every row.

Baseline cost including all three supplemental guided rows: $0.00242805.

## Subsequent configuration change

Fireworks was abandoned after this evaluation. EXTRACTOR_MODEL is now unset locally, so the production extractor uses its GPT-4o-mini default. The temporary Fireworks pricing entry was removed; the results above remain a historical record of the attempted comparison.
