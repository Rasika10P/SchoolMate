"""Generate and store grade orientations offline; reruns skip unchanged domains."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.db import get_conn
from api.services.overviews import domain_key, generate_overview
from scripts.generate_plain_language import parse_grade


def find_pending(framework=None, grade=None):
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute('''SELECT framework_id, grade, domain FROM standard_variant WHERE grade IS NOT NULL
                UNION SELECT s.framework_id, s.grade, s.domain FROM standard s
                WHERE s.grade IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM standard_variant v WHERE v.framework_id=s.framework_id AND v.code=s.code)''')
            groups = {}
            for fid, level, domain in cursor.fetchall():
                if (framework is None or framework == fid) and (grade is None or grade == level):
                    groups.setdefault((fid, level), set()).add(domain)
            cursor.execute('SELECT framework_id, grade, domain_key FROM grade_overview')
            saved = {(f, g): key for f, g, key in cursor.fetchall()}
    return [(f, g, sorted(domains)) for (f, g), domains in sorted(groups.items())
            if saved.get((f, g)) != domain_key(sorted(domains), g)]


def generate_one(row):
    framework, grade, domains = row
    text = generate_overview(framework, grade, domains)
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute('''INSERT INTO grade_overview (framework_id, grade, domain_key, overview)
                VALUES (%s, %s, %s, %s) ON CONFLICT (framework_id, grade)
                DO UPDATE SET domain_key=EXCLUDED.domain_key, overview=EXCLUDED.overview
                WHERE grade_overview.domain_key <> EXCLUDED.domain_key''',
                (framework, grade, domain_key(domains, grade), text))
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework')
    parser.add_argument('--grade', type=parse_grade)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    rows = find_pending(args.framework, args.grade)
    print(f'Found {len(rows)} grade overviews to generate.', flush=True)
    if args.dry_run:
        for framework, grade, domains in rows:
            print(f'{framework} grade {grade}: {len(domains)} domains')
        return 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(generate_one, row): row for row in rows}
        try:
            for count, future in enumerate(as_completed(pending), 1):
                row = pending[future]
                try:
                    future.result()
                    status = 'saved'
                except Exception as exc:
                    failed += 1
                    status = f'failed ({exc})'
                print(f'[{count}/{len(rows)}] {row[0]} grade {row[1]}: {status}', flush=True)
        except KeyboardInterrupt:
            for future in pending:
                future.cancel()
            print('Interrupted; completed overviews are saved. Rerun to resume.', flush=True)
            return 130
    print(f'Finished: saved={len(rows)-failed}, failed={failed}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
