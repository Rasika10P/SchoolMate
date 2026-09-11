# Framework prose corpus

These `.txt` files contain **framework prose, never standard records**. Standards
remain in Postgres and are retrieved by SQL. Each filename is the application's
shared framework ID and maps to its own namespace in one Pinecone index.

The checked-in files are deliberately small seed excerpts, not comprehensive
framework coverage. They contain one short excerpt each, with a document title,
source URL, and printed page number. ELA and ELD use the same discussion excerpt
from the shared 2014 framework, in separate namespaces. Framework IDs identify
the application's standards sets; the actual prose publication/adoption year is
recorded in the document title and may differ from the ID's year.

Source documents are linked in each text file. The [CDE framework directory](https://www.cde.ca.gov/ci/cr/cf/allfwks.asp)
provides the full documents. Expand these files with appropriately licensed or
permissioned framework excerpts for useful coverage of open-ended questions;
see the [CDE copyright statement](https://www.cde.ca.gov/re/di/cr/).

## Text format

Separate paragraphs with a blank line. An optional `@metadata` paragraph changes
metadata for all following paragraphs until the next metadata block. It is not
embedded. Always put it on its own line, followed by a blank line:

```text
@metadata {"document_title":"Document title and edition","source_url":"https://example.org/framework.pdf","grade":null}

@page 142

Framework teaching prose goes here.

Another complete paragraph goes here.

@metadata {"document_title":"Document title and edition","source_url":"https://example.org/framework.pdf","grade":3}

@page 143

Grade-specific framework prose goes here.
```

Each metadata block resets title, URL, and grade to defaults. Subject is derived
from the shared framework mapping; supplied subject/framework values must match.
Without metadata, the title is the filename stem, grade is unknown, and URL empty.

Pagination is tracked independently with lines matching `^@page\s+(\d+)$`.
Markers are removed before chunking and embedding; they can appear within a
paragraph and do not force a chunk break. Every chunk gets the page in effect
at its first content character, including overlapping text. If its last content
character is on a different page, `page_end` records that page, allowing a citation
such as `pp. 142-143`. A trailing marker without content does not extend a range.
Metadata blocks do not reset the current page. Legacy `@metadata.page` values
are ignored; use numeric `@page` lines for pagination.

Text before the first marker has `page=None`. Files without markers remain
valid, keep unknown pages, and produce a warning naming the file. Unknown pages
are never guessed from printed headers. Obvious pasted standard records and CSV
files are rejected; editors must still ensure that prose contains no copied
standards tables or unlabelled standard requirements.

Chunks target 800 characters and overlap whole trailing paragraphs up to 100
characters. Paragraph boundaries take priority; oversized paragraphs are split between
complete sentences, never in the middle of a sentence. Insert paragraph breaks into unusually large passages
before ingestion. Embeddings use `multilingual-e5-large` with the passage/query
input modes. Pinecone does not accept null metadata, so unknown grades/pages are
omitted at storage time and restored as null in returned results.

## Configure and ingest

Install `requirements.txt`. Configure `PINECONE_API_KEY` and either
`PINECONE_INDEX_HOST` (preferred) or `PINECONE_INDEX`. Use one existing dense
index with **1024 dimensions and cosine similarity**, compatible with
`multilingual-e5-large`. The application does not create or delete indexes.
Namespaces are created when their records are first upserted.

```sh
python -m api.services.guidance --ingest --framework CA-CCSSM-2013
# Ingest each of the seven files and print counts per namespace:
python -m api.services.guidance --ingest
# Use an alternative curated text file:
python -m api.services.guidance --ingest --framework CA-CCSSM-2013 --path /path/to/prose.txt
```

The CLI reports pagination coverage alongside each namespace's chunk count:
`CA-CCSSM-2013: 36 chunks upserted; 30 with page numbers, 6 without page numbers`.
Coverage counts a chunk as numbered only if its starting page is known. Add
markers and re-ingest to update the metadata of previously uploaded chunks.

Re-ingestion uses deterministic framework/chunk-index IDs. Existing IDs are
replaced, not duplicated. After successful upserts, stale trailing IDs from a
shorter file are removed. Failed ingestion can leave a partial update; rerun the
same file to finish. Concurrent ingests of different versions of the same
framework are not supported. Empty files never erase a namespace.

Search always supplies both the namespace and framework-ID metadata filter.
When grade is omitted, cross-grade prose is eligible; when supplied, the filter
is an exact grade match. Generic prose with no grade is not silently treated as
grade-specific. Empty results, missing configuration, and provider outages
return `state: unavailable`, without searching another framework.

The agent's fifth tool, `search_guidance`, answers questions about how a subject
is taught. It is intentionally absent from `GOAL_TO_TOOL`, leaving deterministic
goal routing unchanged. Returned chunks include text, score, ID, and provenance
metadata for citations. Retrieval is separate from translation generation.

Implementation references: [Pinecone upserts](https://docs.pinecone.io/guides/index-data/upsert-data),
[metadata and namespaces](https://docs.pinecone.io/guides/index-data/indexing-overview),
and [embedding API](https://docs.pinecone.io/reference/api/2026-04/inference/generate-embeddings).

### Relevance judgement after retrieval

`search()` uses no absolute similarity cutoff. After framework/grade filtering,
nonempty passages go through one `cached_complete` call (default
`openai/gpt-4o-mini`; override with `GUIDANCE_RELEVANCE_MODEL`). The judgement
asks whether the passages actually answer the question, not just discuss a
related topic. The same question and passages reuse the model cache.

Results distinguish:

- `available`: judged relevant; includes chunks, `top_score`, `scores`, and
  `judged: true`.
- `no_relevant_content`: judged irrelevant; includes the reason, `top_score`,
  `scores`, and `judged: true`, without answer passages.
- `unavailable`: retrieval failed or no matching passages were returned;
  `judged: false`.
- `judgement_unavailable`: retrieval succeeded but the judge failed or returned
  invalid output; includes scores and `judged: false`. No answer is synthesized.

When a grade is supplied, matching passages may have that grade or no grade tag.
Ingestion represents null grades by omitting the metadata key, so Pinecone uses
`grade == requested OR grade does not exist`. Other grades remain excluded.

Enable INFO logging for `api.services.guidance` to collect score/judgement rows:

```python
import logging
from api.services.guidance import search

logging.basicConfig(level=logging.INFO)
r = search('How is mathematics taught?', 'CA-CCSSM-2013')
print(r['state'], r.get('top_score'), r.get('scores'), r['judged'])
```

Each search logs the question, framework, top score, and judgement. Retrieval
failures/empty results log `NOT_JUDGED`; judge failures log `ERROR`. These logs
include question text, so use anonymized questions when sharing a report.
