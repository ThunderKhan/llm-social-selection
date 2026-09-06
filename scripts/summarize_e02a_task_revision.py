from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize an existing E02A task-revision report without inference."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "experiments" / "e02a_task_v2" / "e02a_task_v2_report.json",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    task = report["task_calibration"]
    diagnostics = task["candidate_diagnostics"]
    rejections = task["rejections"]

    rates = Counter()
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in diagnostics:
        rate = float(row["exact_match_rate"])
        if rate == 0.0:
            bucket = "0%"
        elif rate == 1.0:
            bucket = "100%"
        elif rate < 0.15:
            bucket = "(0,15%)"
        elif rate <= 0.85:
            bucket = "15-85%"
        else:
            bucket = "(85,100%)"
        rates[bucket] += 1
        by_family[str(row["family"])].append(row)

    print("E02A Task Revision Diagnostic Summary")
    print(f"Report: {args.report}")
    print(f"Candidates: {len(diagnostics)}")
    print(f"Shortlisted: {task['shortlist_task_count']}")
    print(f"Final holdout-mixed: {task['final_task_count']}")
    print(f"Rejection reasons: {dict(sorted(Counter(rejections.values()).items()))}")
    print(f"Difficulty buckets: {dict(sorted(rates.items()))}")
    print()
    print("Per-family calibration")
    for family in sorted(by_family):
        rows = by_family[family]
        family_rates = [float(row["exact_match_rate"]) for row in rows]
        print(
            f"{family:16s} tasks={len(rows):2d} "
            f"mean={mean(family_rates):.3f} "
            f"zero={sum(rate == 0 for rate in family_rates):2d} "
            f"ceiling={sum(rate == 1 for rate in family_rates):2d} "
            f"mixed={sum(0 < rate < 1 for rate in family_rates):2d} "
            f"in_band={sum(0.15 <= rate <= 0.85 for rate in family_rates):2d}"
        )
    print()
    print("Per-task exact-match rates")
    for row in sorted(diagnostics, key=lambda item: (str(item["family"]), str(item["task_id"]))):
        print(
            f"{row['task_id']:14s} {row['family']:16s} "
            f"rate={float(row['exact_match_rate']):.4f} "
            f"correct={int(row['correct']):2d}/{int(row['successful_attempts']):2d} "
            f"format={float(row['format_compliance_rate']):.3f} "
            f"unique={int(row['unique_raw_outputs']):2d} "
            f"rejection={rejections.get(row['task_id'], 'shortlisted')}"
        )

    print()
    provider = report["provider"]
    integrity = report["integrity"]
    print(f"Provider failures: {provider['failures']}/{provider['requests']}")
    print(f"E01 unchanged: {integrity['e01_hash_unchanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
