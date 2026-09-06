from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .models import Task


E02A_MIN_VALIDATED_TASKS = 16
E02A_PREFERRED_VALIDATED_TASKS = 20
E02A_MAX_VALIDATED_TASKS = 30
E02A_MIN_MIXED_HOLDOUT_RATE = 0.75
E02A_PREFERRED_MIXED_HOLDOUT_RATE = 0.85
E02A_MAX_DEGENERATE_HOLDOUT_RATE = 0.25
E02A_MAX_PROVIDER_FAILURE_RATE = 0.05
E02A_CALIBRATION_RATE_BAND = (0.10, 0.90)


def select_e02a_calibration_tasks(
    tasks: Sequence[Task],
    diagnostics: Mapping[str, Mapping[str, Any]],
    *,
    minimum_attempts: int,
    maximum_tasks: int = E02A_MAX_VALIDATED_TASKS,
) -> tuple[tuple[Task, ...], dict[str, str]]:
    """Select tasks for fresh holdout certification without ceiling/floor fallback.

    E01 allowed an easy-task fallback to reach a minimum count. E02A deliberately
    removes that escape hatch: a task must demonstrate non-degenerate empirical
    difficulty during calibration before it can enter the holdout phase.
    """

    if minimum_attempts <= 0:
        raise ValueError("minimum_attempts must be positive")
    if maximum_tasks < E02A_MIN_VALIDATED_TASKS:
        raise ValueError(
            f"maximum_tasks must be at least {E02A_MIN_VALIDATED_TASKS}"
        )

    lower, upper = E02A_CALIBRATION_RATE_BAND
    eligible: dict[str, list[tuple[Task, float]]] = {}
    rejected: dict[str, str] = {}

    for task in tasks:
        diagnostic = diagnostics.get(task.task_id)
        if diagnostic is None:
            rejected[task.task_id] = "missing_calibration"
            continue
        attempts = diagnostic.get("attempts")
        rate = diagnostic.get("exact_match_rate")
        if not isinstance(attempts, int) or attempts < minimum_attempts:
            rejected[task.task_id] = "insufficient_calibration_attempts"
            continue
        if (
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or not 0.0 <= float(rate) <= 1.0
        ):
            rejected[task.task_id] = "invalid_exact_match_rate"
            continue
        numeric_rate = float(rate)
        if numeric_rate < lower:
            rejected[task.task_id] = "calibration_floor"
            continue
        if numeric_rate > upper:
            rejected[task.task_id] = "calibration_ceiling"
            continue
        eligible.setdefault(task.family, []).append((task, numeric_rate))

    for rows in eligible.values():
        rows.sort(key=lambda row: (abs(row[1] - 0.5), row[0].task_id))

    selected: list[Task] = []
    families = sorted(eligible)
    while len(selected) < maximum_tasks:
        added = False
        for family in families:
            rows = eligible[family]
            if rows and len(selected) < maximum_tasks:
                selected.append(rows.pop(0)[0])
                added = True
        if not added:
            break

    selected_ids = {task.task_id for task in selected}
    for rows in eligible.values():
        for task, _ in rows:
            if task.task_id not in selected_ids:
                rejected[task.task_id] = "capacity_limit"

    return tuple(selected), dict(sorted(rejected.items()))


def evaluate_e02a_task_readiness(
    *,
    integrity_issues: Sequence[str],
    provider_failures: int,
    provider_requests: int,
    selected_tasks: Sequence[Task],
    holdout_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen E02A task-discrimination readiness gates."""

    failures: list[str] = []
    warnings: list[str] = []

    failure_rate = provider_failures / provider_requests if provider_requests else 1.0
    mixed_rate = float(holdout_summary["mixed_score_round_rate"])
    degenerate_rate = float(holdout_summary["degenerate_objective_round_rate"])

    if integrity_issues:
        failures.append("task integrity or leakage audit failed")
    if failure_rate > E02A_MAX_PROVIDER_FAILURE_RATE:
        failures.append("provider failure/timeout rate exceeds 5%")
    if len(selected_tasks) < E02A_MIN_VALIDATED_TASKS:
        failures.append(
            f"fewer than {E02A_MIN_VALIDATED_TASKS} tasks survived calibration"
        )
    if mixed_rate < E02A_MIN_MIXED_HOLDOUT_RATE:
        failures.append("mixed-score holdout rate is below 75%")
    elif mixed_rate < E02A_PREFERRED_MIXED_HOLDOUT_RATE:
        warnings.append("mixed-score holdout rate passed 75% but is below preferred 85%")
    if degenerate_rate >= E02A_MAX_DEGENERATE_HOLDOUT_RATE:
        failures.append("fully degenerate holdout round rate is 25% or higher")
    if len(selected_tasks) < E02A_PREFERRED_VALIDATED_TASKS:
        warnings.append("validated pool passed minimum size but is below preferred 20 tasks")

    return {
        "decision": "E02A TASK POOL READY" if not failures else "REVISE E02A TASK POOL",
        "failures": failures,
        "warnings": warnings,
        "provider_failure_rate": failure_rate,
        "selected_task_count": len(selected_tasks),
        "mixed_score_holdout_rate": mixed_rate,
        "degenerate_holdout_rate": degenerate_rate,
        "gates": {
            "minimum_tasks": E02A_MIN_VALIDATED_TASKS,
            "preferred_tasks": E02A_PREFERRED_VALIDATED_TASKS,
            "minimum_mixed_holdout_rate": E02A_MIN_MIXED_HOLDOUT_RATE,
            "preferred_mixed_holdout_rate": E02A_PREFERRED_MIXED_HOLDOUT_RATE,
            "maximum_degenerate_holdout_rate_exclusive": E02A_MAX_DEGENERATE_HOLDOUT_RATE,
            "maximum_provider_failure_rate": E02A_MAX_PROVIDER_FAILURE_RATE,
        },
    }
