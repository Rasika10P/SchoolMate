"""Generate stored translations offline with a bounded worker pool.

Run migrations first: python -m api.migrate
Then: python scripts/generate_plain_language.py --framework CA-CCSSM-2013 --grade 3
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.db import get_conn
from api.services.plain_language import to_plain_language


@dataclass(frozen=True)
class PendingStandard:
    framework_id: str
    code: str
    grade: int | None
    text: str


def find_pending(framework: str | None = None, grade: int | None = None) -> list[PendingStandard]:
    query = """SELECT s.framework_id, s.code, s.grade, s.text
               FROM standard s WHERE s.plain_summary IS NULL"""
    params = []
    if framework is not None:
        query += " AND s.framework_id = %s"
        params.append(framework)
    if grade is not None:
        query += """ AND (s.grade = %s OR EXISTS (
            SELECT 1 FROM standard_variant v WHERE v.framework_id = s.framework_id
            AND v.code = s.code AND v.grade = %s AND v.text = s.text))"""
        params.extend([grade, grade])
    query += " ORDER BY s.framework_id, s.code"
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            return [PendingStandard(*row) for row in cursor.fetchall()]


def generate_one(row: PendingStandard) -> str:
    # Do not hold a database connection while waiting for a model response.
    translation = to_plain_language(row.text, row.grade, code=row.code, framework_id=row.framework_id)
    # Commit each successful row separately. A restart selects only unfinished
    # rows; the guard also protects concurrent jobs and edited source wording.
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""UPDATE standard SET plain_summary = %s, plain_example = %s
                WHERE framework_id = %s AND code = %s AND plain_summary IS NULL
                  AND text = %s AND grade IS NOT DISTINCT FROM %s""",
                (translation['summary'], translation['example'], row.framework_id,
                 row.code, row.text, row.grade))
            updated = cursor.rowcount
    return 'saved' if updated else 'skipped (already translated or source changed)'


def run_batch(rows: list[PendingStandard], workers: int = 4, dry_run: bool = False,
              emit: Callable[[str], None] = print) -> dict[str, int]:
    if not 1 <= workers <= 8:
        raise ValueError('workers must be between 1 and 8')
    totals = dict(saved=0, skipped=0, failed=0)
    emit(f"Found {len(rows)} untranslated standards.")
    if dry_run:
        for row in rows:
            emit(f"Would translate {row.framework_id} / {row.code} (grade {row.grade})")
        emit('Dry run: no model calls or database writes.')
        return totals
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(generate_one, row): row for row in rows}
    try:
        for done, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                status = future.result()
                totals['saved' if status == 'saved' else 'skipped'] += 1
            except Exception as exc:
                # Keep this row NULL for a later retry; other rows still commit.
                totals['failed'] += 1
                status = f'failed ({type(exc).__name__}: {exc})'
            emit(f'[{done}/{len(rows)}] {row.framework_id} / {row.code}: {status}')
    except KeyboardInterrupt:
        emit('Interrupted; cancelling queued work and finishing active rows. Saved rows will be skipped on resume.')
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    emit('Finished: ' + ', '.join(f'{key}={value}' for key, value in totals.items()))
    return totals


def parse_grade(value: str) -> int:
    if value.upper() in {'TK', 'K'}:
        return {'TK': -1, 'K': 0}[value.upper()]
    try:
        grade = int(value)
        if -1 <= grade <= 12:
            return grade
    except ValueError:
        pass
    raise argparse.ArgumentTypeError('grade must be TK, K, or a number from -1 to 12')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', help='Only this framework ID')
    parser.add_argument('--grade', type=parse_grade)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--dry-run', action='store_true', help='List pending rows without model calls or writes')
    args = parser.parse_args(argv)
    try:
        rows = find_pending(args.framework, args.grade)
        totals = run_batch(rows, args.workers, args.dry_run, lambda line: print(line, flush=True))
    except KeyboardInterrupt:
        return 130
    return 1 if totals['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
