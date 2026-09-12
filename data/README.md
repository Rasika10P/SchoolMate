The subject CSVs contain California curriculum standards. `maths.csv` also
retains three illustrative records. `progression.csv` contains reviewed progression
links. `ald.csv` currently contains only its header: the three illustrative
achievement descriptors have been removed. No official achievement descriptions
are currently loaded.
Blank `page` means SQL NULL.

To append standards copied from a PDF, run from the project root with your source
metadata:

```sh
python scripts/paste_to_csv.py --framework CA-CCSSM-2013 --grade 4 \
  --source-url 'https://example.com/source.pdf' --page 27
```

Paste text with each standard beginning with a code such as `4.NF.A.1`, then press
Ctrl-D on an empty line. The script reads until EOF and displays the parsed rows
as a table, along with warnings and duplicate counts. Enter `y` at the `[y/N]`
prompt to append to `data/maths.csv`; Enter or any other response leaves the
file unchanged. Existing standards with the same framework and code are skipped.
Use `--dry-run` to preview without prompting or writing. Redirected stdin is also
supported, with confirmation read from the terminal.

CDE website pastes containing `Standard Identifier`, `Grade`, `Domain`, `Cluster`,
and `Standard` fields are detected automatically, including Markdown copied from
the page. Use `--grade 0` for kindergarten. For web pastes, omit `--page` to leave
the CSV page blank and supply the listing URL with `--source-url`; individual
standard links in the paste take precedence. Web identifiers such as `K.CC.4.a`
are preserved exactly, domains use the code abbreviation, and clusters retain
their supplied wording. Footnotes are included in the text as `Footnote: …`.

ELD web pastes use a separate importer and `data/eld.csv` to retain critical
principle, cluster, proficiency level, content strand, and standard text:

```sh
python scripts/paste_eld_to_csv.py --framework CA-ELD-2012 \
  --source-url 'https://www2.cde.ca.gov/cacs/eld'
```

Paste until Ctrl-D, review the table, then enter `y` to append. `--dry-run`
previews without writing. Grades come from each record (K becomes 0), page is
blank unless supplied, and duplicates use framework and full identifier,
including proficiency suffix. All supplied “No standard” entries remain in the
CSV. The loader reads this alongside the other subject files.

To load all seven subjects, shared progressions, and achievement descriptors:

```sh
python -m pip install -r requirements.txt
export DATABASE_URL='postgresql://USER:PASSWORD@localhost:5432/curriculum'
python -m api.load
```

Use `--data-dir PATH` for another data directory. Framework metadata comes from
`frameworks.csv` (`id,state,subject,version,source_url`), which must include all
seven expected framework IDs even when some subject CSVs are absent. Filename
routing and tool routing share `api/frameworks.py`:

| File | Framework ID |
| --- | --- |
| maths.csv | CA-CCSSM-2013 |
| ela.csv | CA-CCSS-ELA-2013 |
| eld.csv | CA-ELD-2012 |
| science.csv | CA-NGSS-2013 |
| history_social_science.csv | CA-HSS-2016 |
| vapa.csv | CA-VAPA-2019 |
| pe.csv | CA-PE-2005 |

Every subject row must carry its file's expected framework ID. A mismatch aborts
loading with filename, CSV record number (header is row 1), expected ID, and
found ID. Missing subject files are skipped and reported. `frameworks.csv`,
`progression.csv`, and `ald.csv` remain required. Shared-file records referencing
unknown frameworks or missing progression endpoints are skipped with diagnostics;
references may target records already in the database.

The loader inserts all frameworks before standards. Subject text is projected
into the common standard model: science uses performance expectations, and VAPA
uses performance standards. Domains use ELD critical principle, science content
area, HSS course, VAPA art type, or Physical Education; maths and ELA retain their
domains. Clusters use the supplied cluster, science disciplinary core idea,
VAPA anchor standard, or HSS overarching standard.

`standard_variant` preserves every original CSV field in JSON text `metadata`,
alongside the common fields and original grade label. Its key is framework,
code, and grade label, allowing science codes shared by multiple grades to remain
queryable at each grade. `standard` retains one canonical record per framework
and code for progression endpoints. TK maps to -1, K to 0, and VAPA proficiency
ranges retain their labels with a NULL numeric grade. Grade queries use variants,
falling back to canonical rows for older records without variants. Schema setup
also permits nullable grades on existing standard tables.

Schema changes and all inserts use one transaction. Other errors roll back the
entire load. Database totals for all five tables and each framework are printed
after commit, alongside skipped files and references. Counts are totals, not
new insert counts; incoming and outgoing edge counts can both include an edge.
Loading again preserves conflicting rows (`ON CONFLICT DO NOTHING`); CSV edits
do not update existing records. Descriptor identity is framework, grade, subject,
and level; its subject must match the framework subject to appear in guidance.

The Standards and Next steps tabs read `standard.plain_summary` and
`standard.plain_example`. Page views and guide preparation never generate
translations. Missing summaries display the exact official wording; codes remain
inside each row's official-wording expander. “Refresh guide text” reads newly
saved translations without running a model or regenerating the guide.

Generate translations offline after loading data:

```sh
source .venv/bin/activate
set -a
source .env
set +a
python -m api.migrate
python scripts/generate_plain_language.py --framework CA-CCSSM-2013 --grade 3 --dry-run
python scripts/generate_plain_language.py --framework CA-CCSSM-2013 --grade 3
```

Omit filters to process all pending standards. `--grade` accepts TK, K, or a
numeric grade. The pool defaults to four workers; `--workers 1` through
`--workers 8` control concurrency. Dry-run only lists pending rows: it makes no
model calls or database writes. Apply the migration before the dry run.

`api/migrations/001_standard_plain_language.sql` adds two nullable columns without
dropping or replacing data. `python -m api.migrate` is safe to repeat; schema
initialization during loading also applies it. Deploy this migration before the
updated application's database queries run.

Each successful translation commits independently with a NULL-summary and
unchanged-source guard. Ctrl-C cancels queued work and allows active workers to
finish. Rerun the same command to resume; completed rows are not overwritten.
Failed translations stay NULL and are reported; other workers continue, and a
run with failures exits with status 1. The batch job retains the cheap-model
translation and fidelity review, both cached through `cached_complete`.
Model review is a guard against changed requirements, not a formal proof of
equivalence. Long standards that cannot fit faithfully into fewer than 20 words
keep their official wording. Matching grade variants reuse the canonical
translation only when their exact official wording matches.

Next steps lists mapped next-grade records without inventing a sequence for
missing edges. Sources are consolidated by document URL and recorded page range.

Grade orientations are also prepared offline. Apply migrations, then run:

```sh
python scripts/generate_grade_overviews.py --workers 4
```

This stores a three- or four-sentence overview for each loaded framework/grade
in `grade_overview`. It uses the domain list and parent-facing `DOMAIN_LABELS`
from `api/capability.py`, not model calls during rendering. The job supports
`--framework`, `--grade`, and `--dry-run`; completed entries are reused unless the
domain list or its labels change. Each completed overview commits separately.
Invalid paragraph formats get one cached repair attempt before being reported.

Standards and Next steps initially show their overview and collapsed domain
headings. Each section contains a one-line description, learning rows, and
nested official-wording expanders. Sources also start collapsed. Existing
publication gates and unmapped-progression messages remain in effect. Use
“Refresh guide text” to read newly saved overviews into an already open guide.

### Suggest progression links for review

```sh
python scripts/suggest_edges.py --framework CA-CCSS-ELA-2013
```

Reads the framework's subject CSV locally, with no database or model calls.
Writes `data/suggestions/CA-CCSS-ELA-2013-candidates.csv` and a companion
`-unmatched.csv`, and prints standards with no incoming or outgoing candidate.
Use `--output` and `--unmatched-output` to choose new output filenames.
Existing files and `progression.csv` cannot be overwritten.

Candidates match strand/domain and within-strand number across adjacent grades
(e.g. `RL.1.1` → `RL.2.1`). This is an identifier heuristic, not an official
prerequisite mapping. Subparts and ELD proficiency levels remain distinct.
Unsupported formats, grade bands, and ambiguous grades appear in the unmatched
report; HSS identifiers alone do not establish a shared cross-grade strand.

Review the full source texts and URLs in the candidate CSV. The `relation`
column is intentionally blank. Only after confirming a relationship, assign
its relation and copy the five progression columns into `progression.csv`;
do not append the review-only columns. The script never changes live edges.


### Subject clarification and child assessment data

All five tools return `need_subject` for a missing or unknown subject, with the
seven subjects listed in parent-facing language. The agent asks only
“Which subject did you have in mind?” before making any model or tool calls
when no valid subject is supplied. It also handles an invented tool subject
without crashing.

The agent system prompt forbids requesting grades, test scores, report cards,
teacher feedback, or any assessment of a child. It instructs the agent to
acknowledge volunteered assessment data without echoing or storing it, avoid
evaluating the child, and answer the underlying California curriculum question.
These are prompt instructions, not a claim of application-wide data redaction.

The loader preserves existing database rows on conflict. Deleting records from
`ald.csv` does not delete previously loaded database records; the three exact
illustrative descriptors were removed separately from Postgres.
