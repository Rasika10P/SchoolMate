"""Compare uncached extractor runs. Keep raw labels and per-field matches for audit."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from api import llm
from api.graph.extractor import extract_intent


def main():
    load_dotenv('.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/Curriculam_GoldenDataSet.csv')
    parser.add_argument('--output', default='reports/extractor-comparison.json')
    args = parser.parse_args()
    text = Path(args.data).read_text(encoding='utf-8-sig')
    text = text[text.index('id,question,'):]
    rows = list(csv.DictReader(io.StringIO(text)))
    candidate = os.environ.get('EXTRACTOR_MODEL', 'openai/gpt-4o-mini')
    models = list(dict.fromkeys([candidate, 'openai/gpt-4o-mini']))
    os.environ['LLM_CACHE'] = 'off'
    results = []
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for model in models:
        os.environ['EXTRACTOR_MODEL'] = model
        for row in rows:
            # ASL is an entry-path typo in the supplied fixture.
            ask = row['entry_path'].lower() in {'ask', 'asl'}
            if not ask and row['id'].upper() not in {'G07', 'G11', 'G13'}:
                continue
            handle = llm.begin_run()
            started = perf_counter()
            error = None
            intent = None
            try:
                intent = extract_intent(row['question'])
            except Exception as exc:
                # Avoid persisting provider response bodies or credentials.
                error = type(exc).__name__
            elapsed = perf_counter() - started
            metrics = llm.end_run(handle)
            subject = row['subject'].strip().lower()
            subject = {'ala': 'ela', 'els': 'eld', 'none': None, 'null': None, '': None}.get(subject, subject)
            grade = int(row['grade']) if row['grade'].strip() else None
            tool = row['expected_tool'].strip().strip(',').lower()
            expected = {'subject': subject, 'grade': grade}
            if tool in {'on_grade_level', 'working_ahead', 'catching_up', 'competition_prep'}:
                expected.update(goal=tool, question_type='structured')
            elif tool == 'search_guidance':
                expected.update(goal=None, question_type='open')
            matches = {key: intent.get(key) == value for key, value in expected.items()} if intent else {}
            result = dict(id=row['id'], model=model, ask=ask, question=row['question'],
                          expected=expected, intent=intent, matches=matches,
                          latency=elapsed, usage=metrics, error=error)
            results.append(result)
            out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
            print(f"{model.split('/')[-1]} {row['id']} {elapsed:.2f}s {matches} error={error}", flush=True)
            if error:
                print('Stopping this model after provider failure; remaining rows not evaluated.', flush=True)
                break
    for model in models:
        group = [r for r in results if r['model'] == model and r['ask'] and not r['error']]
        print(json.dumps(dict(model=model, completed_ask_rows=len(group),
            exact_matches=sum(all(r['matches'].values()) for r in group),
            mean_latency=statistics.mean(r['latency'] for r in group) if group else None,
            cost=sum(r['usage']['cost'] or 0 for r in group))), flush=True)


if __name__ == '__main__':
    main()
