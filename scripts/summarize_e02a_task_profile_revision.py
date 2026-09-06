from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize profile-aware E02A task calibration without inference."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "experiments" / "e02a_task_v3" / "e02a_task_v3_report.json",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "experiments" / "e02a_task_v3" / "e02a_task_v3_raw_evidence.jsonl",
    )
    args = parser.parse_args()

    if not args.report.exists():
        parser.error(f"report not found: {args.report}")
    if not args.evidence.exists():
        parser.error(f"evidence not found: {args.evidence}")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in args.evidence.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    calibration = [
        row for row in rows if row.get("phase") == "task_calibration" and row.get("status") == "ok"
    ]
    holdout = [
        row for row in rows if row.get("phase") == "task_holdout" and row.get("status") == "ok"
    ]

    by_task: dict[str, list[dict]] = defaultdict(list)
    by_profile: dict[str, list[dict]] = defaultdict(list)
    for row in calibration:
        by_task[str(row["task_id"])].append(row)
        by_profile[str(row["profile_id"])].append(row)

    task_meta = {
        str(item["task_id"]): item
        for item in report["task_calibration"]["candidate_diagnostics"]
    }
    shortlist = set(report["task_calibration"]["shortlist_task_ids"])
    final_ids = set(report["task_calibration"]["final_task_ids"])

    print("E02A V3 Profile-Aware Diagnostic Summary")
    print(f"Report: {args.report}")
    print(f"Candidates: {report['task_calibration']['candidate_task_count']}")
    print(f"Shortlisted: {len(shortlist)} {sorted(shortlist)}")
    print(f"Final holdout-mixed: {len(final_ids)} {sorted(final_ids)}")
    print()

    print("Per-profile calibration accuracy")
    for profile_id, profile_rows in sorted(by_profile.items()):
        correct = sum(bool(row.get("correct")) for row in profile_rows)
        total = len(profile_rows)
        approach = None
        if profile_rows:
            params = profile_rows[0].get("profile_parameters") or {}
            approach = params.get("approach")
        print(
            f"{profile_id:18s} approach={str(approach):18s} "
            f"correct={correct:3d}/{total:3d} rate={correct / total if total else 0.0:.3f}"
        )
    print()

    print("Tasks with any cross-profile or repeat disagreement")
    informative = []
    for task_id, task_rows in sorted(by_task.items()):
        profile_scores: dict[str, list[int]] = defaultdict(list)
        for row in task_rows:
            profile_scores[str(row["profile_id"])].append(int(bool(row.get("correct"))))
        flat = [value for values in profile_scores.values() for value in values]
        if not flat:
            continue
        overall = sum(flat) / len(flat)
        mixed_profiles = {
            profile_id: values
            for profile_id, values in profile_scores.items()
            if len(set(values)) > 1
        }
        profile_means = {
            profile_id: sum(values) / len(values)
            for profile_id, values in sorted(profile_scores.items())
        }
        cross_profile = len(set(profile_means.values())) > 1
        if 0.0 < overall < 1.0 or mixed_profiles or cross_profile:
            informative.append((abs(overall - 0.5), task_id, overall, profile_means, mixed_profiles))
    for _, task_id, overall, profile_means, mixed_profiles in sorted(informative):
        family = str(task_meta[task_id]["family"])
        marks = []
        for profile_id, rate in profile_means.items():
            short = profile_id.replace("e01-profile-", "P")
            marks.append(f"{short}={rate:.1f}")
        print(
            f"{task_id:14s} {family:16s} overall={overall:.3f} "
            f"profiles=[{' '.join(marks)}]"
            + (f" repeat_disagreement={sorted(mixed_profiles)}" if mixed_profiles else "")
        )
    if not informative:
        print("(none)")
    print()

    print("Nearest tasks to the shortlist band")
    ranked = sorted(
        task_meta.values(),
        key=lambda item: (
            0.0
            if 0.15 <= float(item["exact_match_rate"]) <= 0.85
            else min(abs(float(item["exact_match_rate"]) - 0.15), abs(float(item["exact_match_rate"]) - 0.85)),
            abs(float(item["exact_match_rate"]) - 0.5),
            str(item["task_id"]),
        ),
    )
    for item in ranked[:12]:
        task_id = str(item["task_id"])
        print(
            f"{task_id:14s} {str(item['family']):16s} "
            f"rate={float(item['exact_match_rate']):.3f} "
            f"correct={int(item['correct']):2d}/{int(item['successful_attempts']):2d} "
            f"format={float(item['format_compliance_rate']):.3f} "
            f"shortlisted={task_id in shortlist}"
        )
    print()

    if holdout:
        print("Holdout outcomes")
        holdout_by_task: dict[str, list[dict]] = defaultdict(list)
        for row in holdout:
            holdout_by_task[str(row["task_id"])].append(row)
        for task_id, task_rows in sorted(holdout_by_task.items()):
            correct_profiles = [
                str(row["profile_id"]).replace("e01-profile-", "P")
                for row in task_rows
                if row.get("correct")
            ]
            incorrect_profiles = [
                str(row["profile_id"]).replace("e01-profile-", "P")
                for row in task_rows
                if not row.get("correct")
            ]
            print(
                f"{task_id}: correct={correct_profiles}; incorrect={incorrect_profiles}"
            )
        print()

    rejection_counts = Counter(report["task_calibration"]["rejections"].values())
    rates = [float(item["exact_match_rate"]) for item in task_meta.values()]
    print(f"Rejection reasons: {dict(sorted(rejection_counts.items()))}")
    print(f"Mean candidate calibration accuracy: {mean(rates):.3f}")
    print(f"Provider failures: {report['provider']['failures']}/{report['provider']['requests']}")
    print(f"E01 unchanged: {report['integrity']['e01_hash_unchanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
