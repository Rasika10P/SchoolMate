"""Validate or run the balanced and real-world extractor suites together."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from evals.extractor_suites import load_suites, run_cases, suite_inventory, summarize, validate_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "balanced", "challenge"), default="all")
    parser.add_argument("--run", action="store_true", help="make model calls; omitted means validation only")
    parser.add_argument("--model", action="append", help="model to evaluate; repeat to compare models")
    parser.add_argument("--use-cache", action="store_true", help="allow the normal response cache")
    parser.add_argument("--output", default="reports/extractor-suites.json")
    args = parser.parse_args()

    load_dotenv(".env")
    cases = validate_cases(load_suites(args.suite))
    inventory = suite_inventory(cases)
    print(json.dumps(inventory, indent=2))
    if not args.run:
        print("Validated successfully. Add --run to make model calls.")
        return

    models = args.model or [os.environ.get("EXTRACTOR_MODEL", "openai/gpt-4o-mini")]
    os.environ["LLM_CACHE"] = "on" if args.use_cache else "off"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    all_results = []
    for model in dict.fromkeys(models):
        results = run_cases(cases, model)
        all_results.extend(results)
        output.write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    grouped = defaultdict(list)
    for row in all_results:
        grouped[(row["suite"], row["model"])].append(row)
    print("\nRESULTS")
    print("=" * 72)
    for (suite, model), rows in grouped.items():
        report = summarize(rows)
        exact = "n/a" if report["exact_accuracy"] is None else f"{report['exact_accuracy']:.1%}"
        print(f"{suite:10} {model}: exact={exact}, completed={report['completed']}/{report['cases']}, "
              f"latency={report['mean_latency']}, cost={report['cost']}")
        for field, metric in report["field_accuracy"].items():
            print(f"  {field:14} {metric['correct']}/{metric['total']} = {metric['accuracy']:.1%}")
    print(f"\nDetailed results: {output}")


if __name__ == "__main__":
    main()
