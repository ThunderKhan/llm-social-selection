from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scoring import normalize_answer  # noqa: E402
from src.tasks.calibration import load_task_set, output_format_valid  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize raw output/error patterns from the profile-aware E02A task "
            "calibration without running inference."
        )
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "tasks" / "e02a_candidates_v2.json",
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
    parser.add_argument("--top-outputs", type=int, default=5)
    args = parser.parse_args()

    if not args.candidates.exists():
        parser.error(f"candidate bank not found: {args.candidates}")
    if not args.report.exists():
        parser.error(f"report not found: {args.report}")
    if not args.evidence.exists():
        parser.error(f"evidence not found: {args.evidence}")

    artifact = load_task_set(args.candidates)
    tasks = {task.task_id: task for task in artifact.tasks}
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in args.evidence.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    calibration = [
        row
        for row in rows
        if row.get("phase") == "task_calibration" and row.get("status") == "ok"
    ]

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in calibration:
        by_task[str(row["task_id"])].append(row)

    report_diagnostics = {
        str(row["task_id"]): row
        for row in report["task_calibration"]["candidate_diagnostics"]
    }

    print("E02A V3 Raw Output Pattern Summary")
    print(f"Evidence: {args.evidence}")
    print(f"Calibration rows: {len(calibration)}")
    print()

    reconstructed_mismatches: list[str] = []
    family_contract_failures: Counter[str] = Counter()
    family_total: Counter[str] = Counter()

    for task_id in sorted(tasks):
        task = tasks[task_id]
        task_rows = by_task.get(task_id, [])
        outputs = Counter(str(row.get("raw_output", "")) for row in task_rows)
        correct = sum(
            normalize_answer(str(row.get("raw_output", "")))
            == normalize_answer(task.expected_answer or "")
            for row in task_rows
        )
        format_valid = sum(
            output_format_valid(task, str(row.get("raw_output", ""))) for row in task_rows
        )
        family_total[task.family] += len(task_rows)
        family_contract_failures[task.family] += len(task_rows) - format_valid

        diagnostic = report_diagnostics.get(task_id)
        if diagnostic is not None and int(diagnostic["correct"]) != correct:
            reconstructed_mismatches.append(task_id)

        print(
            f"{task_id:14s} family={task.family:16s} expected={task.expected_answer!r} "
            f"correct={correct:2d}/{len(task_rows):2d} "
            f"format={format_valid:2d}/{len(task_rows):2d} "
            f"unique={len(outputs)}"
        )
        for output, count in outputs.most_common(args.top_outputs):
            normalized = normalize_answer(output)
            marker = (
                "CORRECT"
                if normalized == normalize_answer(task.expected_answer or "")
                else "wrong"
            )
            compact = output.replace("\r", "\\r").replace("\n", "\\n")
            if len(compact) > 120:
                compact = compact[:117] + "..."
            print(f"    {count:2d}x [{marker:7s}] {compact!r}")

        by_profile: dict[str, list[str]] = defaultdict(list)
        for row in task_rows:
            by_profile[str(row.get("profile_id", "unknown"))].append(
                str(row.get("raw_output", ""))
            )
        if len(outputs) > 1:
            profile_parts = []
            for profile_id, values in sorted(by_profile.items()):
                short = profile_id.replace("e01-profile-", "P")
                profile_parts.append(f"{short}={values}")
            print("    profile outputs: " + " | ".join(profile_parts))
        print()

    print("Family output-contract failure rates")
    for family in sorted(family_total):
        failures = family_contract_failures[family]
        total = family_total[family]
        print(
            f"{family:16s} failures={failures:3d}/{total:3d} "
            f"rate={failures / total if total else 0.0:.3f}"
        )
    print()

    if reconstructed_mismatches:
        print(
            "ERROR: reconstructed correctness disagrees with v3 report for: "
            + ", ".join(reconstructed_mismatches)
        )
        return 1

    print("Raw-output/report correctness cross-check: PASS")
    print(f"Provider failures: {report['provider']['failures']}/{report['provider']['requests']}")
    print(f"E01 unchanged: {report['integrity']['e01_hash_unchanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
