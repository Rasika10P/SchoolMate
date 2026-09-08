"""Suggest identifier-based progression links for human review, never ingestion."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.frameworks import SUBJECT_FILES

EDGE_FIELDS = ['from_framework', 'from_code', 'to_framework', 'to_code', 'relation',
               'from_grade', 'to_grade', 'match_key', 'confidence_note',
               'from_text', 'to_text', 'from_source_url', 'to_source_url']
UNMATCHED_FIELDS = ['framework_id', 'code', 'grade', 'reason', 'text', 'source_url']
NOTE = ('Identifier match only: same strand/domain and number in adjacent grades. '
        'Prerequisite relationship is unverified; review both standards before assigning a relation.')


def grade_number(value):
    value = value.strip().upper()
    if value in ('TK', 'PK'):
        return -1
    if value == 'K':
        return 0
    return int(value) if value.isdigit() else None


def match_key(row):
    """Conservative full strand + number/subpart match; never compare text meaning."""
    code, grade = row['code'], grade_number(row['grade'])
    patterns = [
        # ELA: RL.1.1; ELD: ELD.PI.1.1.Em (retain proficiency).
        r'(?P<strand>(?:ELD\.)?[A-Z]+)\.(?P<grade>K|\d+)\.(?P<number>\d+(?:\.[A-Za-z]+)*)',
        # Math: 1.OA.1 or 3.NF.A.1. Cluster letters are not within-domain numbers.
        r'(?P<grade>K|\d+)\.(?P<strand>[A-Z]+)\.(?:[A-Z]\.)?(?P<number>\d+(?:\.[a-z]+)*)',
        # Arts: K.VA:Cr1, PK.DA.Cr1. Retain discipline and artistic process.
        r'(?P<grade>PK|TK|K|\d+)\.(?P<strand>[A-Z]+)[.:](?P<number>[A-Za-z]+\d+(?:\.\d+)*)',
        # Science: single-grade identifiers only; grade-band standards are skipped.
        r'(?P<grade>K|\d+)-(?P<strand>[A-Z]+\d+)-(?P<number>\d+)',
    ]
    for pattern in patterns:
        match = re.fullmatch(pattern, code)
        if match and grade is not None and grade_number(match['grade']) == grade:
            return (match['strand'], match['number'], row.get('subdiscipline', '').strip())
    # PE has an explicit overarching standard number. Cluster must also match.
    match = re.fullmatch(r'PE-(K|\d+)\.(\d+)\.(\d+)', code)
    if match and grade_number(match[1]) == grade and row.get('cluster', '').strip():
        return ('PE.' + match[2], match[3], row['cluster'].strip())
    return None


def read_standards(path, framework):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'framework_id', 'code', 'grade'} <= set(reader.fieldnames or []):
            raise ValueError(f'{path}: requires framework_id, code, grade columns')
        rows, seen = [], set()
        for number, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'{path}: row {number}: malformed CSV record')
            if row['framework_id'] != framework:
                raise ValueError(f'{path}: row {number}: expected {framework}, found {row["framework_id"]}')
            if not row['code'].strip():
                raise ValueError(f'{path}: row {number}: empty code')
            identity = (row['code'], row['grade'])
            if identity in seen:
                raise ValueError(f'{path}: row {number}: duplicate code/grade {identity}')
            seen.add(identity)
            row['text'] = row.get('text') or row.get('performance_standards') or row.get('performance_expectation') or ''
            rows.append(row)
    return rows


def suggest(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        key = match_key(row)
        if key is not None:
            groups[(key, grade_number(row['grade']))].append((index, row))
    edges, linked = [], set()
    for (key, grade), sources in sorted(groups.items()):
        for source_id, source in sources:
            for target_id, target in groups.get((key, grade + 1), []):
                if source['code'] == target['code']:
                    continue
                linked.update((source_id, target_id))
                edges.append(dict(from_framework=source['framework_id'], from_code=source['code'],
                    to_framework=target['framework_id'], to_code=target['code'], relation='',
                    from_grade=grade, to_grade=grade + 1, match_key=' | '.join(key),
                    confidence_note=NOTE, from_text=source['text'], to_text=target['text'],
                    from_source_url=source.get('source_url', ''), to_source_url=target.get('source_url', '')))
    unmatched = [{**{k: row.get(k, '') for k in UNMATCHED_FIELDS},
                  'reason': 'No supported unambiguous strand/number identifier or grade' if match_key(row) is None
                  else 'No matching identifier in either adjacent grade'}
                 for index, row in enumerate(rows) if index not in linked]
    return edges, unmatched


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', required=True, choices=sorted(SUBJECT_FILES.values()))
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, help='New review CSV; existing files are never overwritten')
    parser.add_argument('--unmatched-output', type=Path, help='New CSV listing standards with no candidate links')
    args = parser.parse_args(argv)
    source = args.data_dir / next(name for name, fid in SUBJECT_FILES.items() if fid == args.framework)
    output = args.output or args.data_dir / 'suggestions' / f'{args.framework}-candidates.csv'
    unmatched_output = args.unmatched_output or output.with_name(output.stem + '-unmatched.csv')
    protected = {p.resolve() for p in args.data_dir.glob('*.csv')} | {(ROOT / 'data/progression.csv').resolve()}
    paths = (output, unmatched_output)
    if output.resolve() == unmatched_output.resolve():
        parser.error('Candidate and unmatched outputs must be different files')
    for path in paths:
        if path.resolve() in protected or path.name == 'progression.csv' or path.exists():
            parser.error(f'Refusing to overwrite source, progression, or existing file: {path}')
    try:
        rows = read_standards(source, args.framework)
        edges, unmatched = suggest(rows)
        for path, fields, records in ((output, EDGE_FIELDS, edges), (unmatched_output, UNMATCHED_FIELDS, unmatched)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'{len(rows)} standards; {len(edges)} candidate edges -> {output}')
    print(f'{len(unmatched)} standards with no candidate edges -> {unmatched_output}')
    for row in unmatched:
        print(f'  {row["code"]} (grade {row["grade"]}): {row["reason"]}')
    print('Review required. Relation is intentionally blank. progression.csv was not modified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
