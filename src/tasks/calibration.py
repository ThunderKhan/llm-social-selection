from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from ..domain import Response
from ..scoring import SCORER_VERSION, normalize_answer, score_response
from .models import Task


TASK_SET_SCHEMA_VERSION = 1
ALLOWED_FAMILIES = frozenset(
    {"arithmetic", "counting", "logic", "multiple-choice", "sequence", "string", "symbolic"}
)
TASK_PREFIX_BY_FAMILY = {
    "arithmetic": "arith",
    "counting": "count",
    "logic": "logic",
    "multiple-choice": "mcq",
    "sequence": "sequence",
    "string": "string",
    "symbolic": "symbolic",
}
TASK_ID_PATTERN = re.compile(
    r"^(?:arith|count|logic|mcq|sequence|string|symbolic)-\d{3}$"
)
MIN_VALIDATED_TASKS = 12
MIN_VALIDATED_FAMILIES = 4
STRONG_DIFFICULTY_BAND = (0.20, 0.80)
ACCEPTABLE_DIFFICULTY_BAND = (0.10, 0.90)
REQUIRED_CONTRACT_PHRASES = (
    "return only",
    "do not explain",
    "do not repeat",
    "do not use markdown",
    'do not prepend "answer:"',
)
FORBIDDEN_PROMPT_TERMS = (
    "agent_id",
    "agent-",
    "condition",
    "expected_answer",
    "model name",
    "objective selection",
    "peer_vote",
    "profile_id",
    "profile-",
    "qwen",
    "random selection",
    "scorer_version",
    "selection mechanism",
)


class TaskSetValidationError(ValueError):
    """A task-set artifact violates the frozen calibration contract."""


@dataclass(frozen=True)
class TaskSetArtifact:
    task_set_version: str
    created_for: str
    scorer_version: str
    model_used_for_validation: str
    validation_date: str
    status: str
    provenance: Mapping[str, str]
    tasks: tuple[Task, ...]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TaskSetValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def task_leakage_issues(task: Task) -> tuple[str, ...]:
    prompt = task.prompt.casefold()
    issues = [term for term in FORBIDDEN_PROMPT_TERMS if term in prompt]
    question = prompt.split("output requirement:", 1)[0]
    expected = normalize_answer(task.expected_answer or "")
    if task.family not in {"logic", "multiple-choice"} and expected:
        pattern = rf"(?<!\w){re.escape(expected)}(?!\w)"
        if re.search(pattern, question):
            issues.append("expected answer appears in the question text")
    return tuple(issues)


def validate_task(task: Task) -> tuple[str, ...]:
    issues: list[str] = []
    if not TASK_ID_PATTERN.fullmatch(task.task_id):
        issues.append("task_id does not match the stable E01 ID pattern")
    if task.family not in ALLOWED_FAMILIES:
        issues.append(f"unsupported family: {task.family}")
    elif not task.task_id.startswith(f"{TASK_PREFIX_BY_FAMILY[task.family]}-"):
        issues.append("task_id prefix does not match family")
    if task.scorer_version != SCORER_VERSION:
        issues.append(f"scorer_version must be {SCORER_VERSION}")
    if task.expected_answer is None or not task.expected_answer.strip():
        issues.append("expected_answer must be non-empty")
    prompt = task.prompt.casefold()
    for phrase in REQUIRED_CONTRACT_PHRASES:
        if phrase not in prompt:
            issues.append(f"missing response-contract phrase: {phrase}")
    issues.extend(f"prompt leakage: {issue}" for issue in task_leakage_issues(task))
    return tuple(issues)


def load_task_set(path: str | Path) -> TaskSetArtifact:
    source = Path(path)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, json.JSONDecodeError) as error:
        raise TaskSetValidationError(f"could not load task set {source}: {error}") from error
    if not isinstance(raw, dict):
        raise TaskSetValidationError("task set must be a JSON object")
    required = {
        "schema_version",
        "task_set_version",
        "created_for",
        "scorer_version",
        "model_used_for_validation",
        "validation_date",
        "status",
        "provenance",
        "tasks",
    }
    if set(raw) != required:
        missing = sorted(required - set(raw))
        extra = sorted(set(raw) - required)
        raise TaskSetValidationError(
            f"task set fields mismatch; missing={missing}, extra={extra}"
        )
    if raw["schema_version"] != TASK_SET_SCHEMA_VERSION:
        raise TaskSetValidationError("unsupported task-set schema_version")
    for field in (
        "task_set_version",
        "created_for",
        "scorer_version",
        "model_used_for_validation",
        "validation_date",
        "status",
    ):
        if not isinstance(raw[field], str) or not raw[field].strip():
            raise TaskSetValidationError(f"{field} must be a non-empty string")
    if raw["scorer_version"] != SCORER_VERSION:
        raise TaskSetValidationError(f"scorer_version must be {SCORER_VERSION}")
    provenance = raw["provenance"]
    if not isinstance(provenance, dict) or not provenance:
        raise TaskSetValidationError("provenance must be a non-empty object")
    if any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in provenance.items()
    ):
        raise TaskSetValidationError("provenance entries must be non-empty strings")
    rows = raw["tasks"]
    if not isinstance(rows, list) or not rows:
        raise TaskSetValidationError("tasks must be a non-empty array")
    tasks: list[Task] = []
    expected_fields = {
        "task_id",
        "family",
        "prompt",
        "expected_answer",
        "scorer_version",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise TaskSetValidationError(f"tasks[{index}] has invalid fields")
        try:
            task = Task(**row)
        except (TypeError, ValueError) as error:
            raise TaskSetValidationError(f"tasks[{index}] is invalid: {error}") from error
        issues = validate_task(task)
        if issues:
            raise TaskSetValidationError(f"task {task.task_id}: {'; '.join(issues)}")
        tasks.append(task)
    ids = [task.task_id for task in tasks]
    if len(set(ids)) != len(ids):
        raise TaskSetValidationError("task IDs must be unique")
    return TaskSetArtifact(
        task_set_version=raw["task_set_version"],
        created_for=raw["created_for"],
        scorer_version=raw["scorer_version"],
        model_used_for_validation=raw["model_used_for_validation"],
        validation_date=raw["validation_date"],
        status=raw["status"],
        provenance=dict(provenance),
        tasks=tuple(tasks),
    )


def classify_difficulty(exact_match_rate: float) -> str:
    if not 0.0 <= exact_match_rate <= 1.0:
        raise ValueError("exact_match_rate must be between 0 and 1")
    if STRONG_DIFFICULTY_BAND[0] <= exact_match_rate <= STRONG_DIFFICULTY_BAND[1]:
        return "strong"
    if ACCEPTABLE_DIFFICULTY_BAND[0] <= exact_match_rate <= ACCEPTABLE_DIFFICULTY_BAND[1]:
        return "acceptable"
    if exact_match_rate < ACCEPTABLE_DIFFICULTY_BAND[0]:
        return "too_hard"
    return "too_easy"


def classify_format_error(raw_output: str, expected_answer: str) -> str | None:
    if normalize_answer(raw_output) == normalize_answer(expected_answer):
        return None
    stripped = raw_output.strip()
    expected = normalize_answer(expected_answer)
    if "```" in stripped or stripped.startswith(("#", "* ", "- ")):
        return "markdown"
    if re.match(r"(?i)^answer\s*:", stripped):
        return "answer_prefix"
    if "\n" in stripped:
        return "multiline"
    if expected and expected in normalize_answer(stripped):
        return "expected_answer_with_extra_text"
    if len(stripped.split()) == 1:
        return "wrong_single_token"
    return "other_extra_text"


def output_format_valid(task: Task, raw_output: str) -> bool:
    stripped = raw_output.strip()
    if task.family in {"arithmetic", "counting", "sequence", "symbolic"}:
        return re.fullmatch(r"[+-]?\d+", stripped) is not None
    if task.family == "logic":
        return stripped in {"true", "false"}
    if task.family == "multiple-choice":
        return stripped in {"A", "B", "C", "D"}
    if task.family == "string":
        return bool(stripped) and not any(character.isspace() for character in stripped)
    return False


def analyze_task_attempts(task: Task, responses: Sequence[Response]) -> dict[str, Any]:
    if not responses:
        raise ValueError("responses must not be empty")
    scores = [score_response(response, task).value for response in responses]
    correct = sum(score == 1.0 for score in scores)
    raw_counts = Counter(response.content for response in responses)
    format_errors = Counter(
        error
        for response in responses
        if (error := classify_format_error(response.content, task.expected_answer or ""))
        is not None
    )
    latencies = [response.latency_ms for response in responses if response.latency_ms is not None]
    token_counts = [response.token_count for response in responses if response.token_count is not None]
    rate = correct / len(responses)
    format_compliant = sum(output_format_valid(task, response.content) for response in responses)
    return {
        "task_id": task.task_id,
        "family": task.family,
        "attempts": len(responses),
        "correct": correct,
        "incorrect": len(responses) - correct,
        "exact_match_rate": rate,
        "format_compliant": format_compliant,
        "format_compliance_rate": format_compliant / len(responses),
        "difficulty_class": classify_difficulty(rate),
        "unique_raw_outputs": len(raw_counts),
        "mean_latency_ms": mean(latencies) if latencies else None,
        "mean_token_count": mean(token_counts) if token_counts else None,
        "format_error_patterns": dict(sorted(format_errors.items())),
        "most_common_raw_outputs": [
            {"raw_output": output, "count": count}
            for output, count in raw_counts.most_common(5)
        ],
    }


def select_validated_tasks(
    tasks: Sequence[Task],
    diagnostics: Mapping[str, Mapping[str, Any]],
    *,
    minimum_attempts: int = 8,
    maximum_tasks: int = 24,
    minimum_tasks: int = MIN_VALIDATED_TASKS,
) -> tuple[tuple[Task, ...], dict[str, str]]:
    if maximum_tasks <= 0:
        raise ValueError("maximum_tasks must be positive")
    eligible: dict[str, list[tuple[Task, float, str]]] = defaultdict(list)
    easy_fallback: list[Task] = []
    rejected: dict[str, str] = {}
    for task in tasks:
        diagnostic = diagnostics.get(task.task_id)
        if diagnostic is None:
            rejected[task.task_id] = "missing_validation"
            continue
        attempts = diagnostic.get("attempts")
        rate = diagnostic.get("exact_match_rate")
        if not isinstance(attempts, int) or attempts < minimum_attempts:
            rejected[task.task_id] = "insufficient_attempts"
            continue
        if not isinstance(rate, (int, float)) or isinstance(rate, bool):
            rejected[task.task_id] = "invalid_exact_match_rate"
            continue
        difficulty = classify_difficulty(float(rate))
        if difficulty == "too_easy":
            easy_fallback.append(task)
            continue
        if difficulty == "too_hard":
            rejected[task.task_id] = difficulty
            continue
        eligible[task.family].append((task, float(rate), difficulty))
    for rows in eligible.values():
        rows.sort(
            key=lambda row: (
                row[2] != "strong",
                abs(row[1] - 0.5),
                row[0].task_id,
            )
        )
    selected: list[Task] = []
    families = sorted(eligible)
    while len(selected) < maximum_tasks:
        added = False
        for family in families:
            if eligible[family] and len(selected) < maximum_tasks:
                selected.append(eligible[family].pop(0)[0])
                added = True
        if not added:
            break
    fallback_target = min(minimum_tasks, maximum_tasks)
    while len(selected) < fallback_target and easy_fallback:
        selected_families = {task.family for task in selected}
        if len(selected_families) < MIN_VALIDATED_FAMILIES:
            easy_fallback.sort(
                key=lambda task: (task.family in selected_families, task.task_id)
            )
        else:
            easy_fallback.sort(key=lambda task: task.task_id)
        selected.append(easy_fallback.pop(0))
    for task in easy_fallback:
        rejected[task.task_id] = "too_easy"
    selected_ids = {task.task_id for task in selected}
    for rows in eligible.values():
        for task, _, _ in rows:
            if task.task_id not in selected_ids:
                rejected[task.task_id] = "capacity_limit"
    return tuple(selected), dict(sorted(rejected.items()))


def analyze_objective_rounds(round_scores: Sequence[Sequence[float]]) -> dict[str, Any]:
    if not round_scores:
        raise ValueError("round_scores must not be empty")
    mixed = all_correct = all_incorrect = degenerate = objective_ties = 0
    tied_lowest_counts: list[int] = []
    for scores in round_scores:
        if len(scores) != 8 or any(score not in (0.0, 1.0) for score in scores):
            raise ValueError("each objective round must contain eight binary scores")
        correct = sum(score == 1.0 for score in scores)
        mixed += 0 < correct < 8
        all_correct += correct == 8
        all_incorrect += correct == 0
        degenerate += correct in (0, 8)
        lowest = min(scores)
        tied_count = sum(score == lowest for score in scores)
        tied_lowest_counts.append(tied_count)
        objective_ties += tied_count > 1
    total = len(round_scores)
    return {
        "rounds": total,
        "mixed_score_rounds": mixed,
        "mixed_score_round_rate": mixed / total,
        "all_correct_rounds": all_correct,
        "all_correct_rate": all_correct / total,
        "all_incorrect_rounds": all_incorrect,
        "all_incorrect_rate": all_incorrect / total,
        "degenerate_objective_rounds": degenerate,
        "degenerate_objective_round_rate": degenerate / total,
        "objective_tie_rounds": objective_ties,
        "objective_tie_rate": objective_ties / total,
        "mean_tied_lowest_count": mean(tied_lowest_counts),
    }


def evaluate_task_set_readiness(
    *,
    integrity_issues: Sequence[str],
    provider_failures: int,
    provider_requests: int,
    selected_tasks: Sequence[Task],
    objective_rounds: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    failure_rate = provider_failures / provider_requests if provider_requests else 1.0
    if integrity_issues:
        failures.append("task integrity or leakage audit failed")
    if failure_rate > 0.05:
        failures.append("provider failure/timeout rate exceeds 5%")
    if len(selected_tasks) < MIN_VALIDATED_TASKS:
        failures.append(f"fewer than {MIN_VALIDATED_TASKS} validated tasks remain")
    family_count = len({task.family for task in selected_tasks})
    if family_count < MIN_VALIDATED_FAMILIES:
        failures.append(f"fewer than {MIN_VALIDATED_FAMILIES} validated families remain")
    degenerate_rate = objective_rounds["degenerate_objective_round_rate"]
    if degenerate_rate > 0.40:
        failures.append("degenerate objective-round rate exceeds 40%")
    elif degenerate_rate > 0.25:
        warnings.append("degenerate objective-round rate is between 25% and 40%")
    return {
        "decision": "REVISE TASK SET" if failures else "OBJECTIVE TASK SET READY",
        "failures": failures,
        "warnings": warnings,
        "provider_failure_rate": failure_rate,
        "selected_task_count": len(selected_tasks),
        "selected_family_count": family_count,
        "degenerate_objective_round_rate": degenerate_rate,
    }


def render_validation_markdown(report: Mapping[str, Any]) -> str:
    readiness = report["readiness"]
    simulation = report["objective_round_simulation"]
    lines = [
        "# E01 Objective Task Validation",
        "",
        "Engineering apparatus calibration only. No research claims are made.",
        "",
        f"**{readiness['decision']}**",
        "",
        "## Task Set",
        "",
        f"- Candidate tasks: {report['candidate_task_count']}",
        f"- Selected tasks: {report['selected_task_count']}",
        f"- Rejected tasks: {report['rejected_task_count']}",
        f"- Selected families: {', '.join(report['selected_families'])}",
        f"- Validation attempts: {report['validation_attempt_count']}",
        f"- Selection criteria: `{report['selection_criteria']}`",
        "",
        "## Objective-Round Simulation",
        "",
        f"- Mixed-score round rate: {simulation['mixed_score_round_rate']}",
        f"- All-correct rate: {simulation['all_correct_rate']}",
        f"- All-incorrect rate: {simulation['all_incorrect_rate']}",
        f"- Degenerate objective-round rate: {simulation['degenerate_objective_round_rate']}",
        f"- Objective tie rate: {simulation['objective_tie_rate']}",
        f"- Mean tied-lowest count: {simulation['mean_tied_lowest_count']}",
        "",
        "## Reliability And Integrity",
        "",
        f"- Provider requests: {report['provider']['requests']}",
        f"- Provider failures: {report['provider']['failures']}",
        f"- Provider failure rate: {readiness['provider_failure_rate']}",
        f"- Integrity issues: {report['integrity']['issues']}",
        f"- Family-specific format compliance: {report['format_compliance']['rate']}",
        f"- Format-error patterns: `{report['format_compliance']['patterns']}`",
        "",
        "## Per-Task Diagnostics",
        "",
        "| Task | Family | Attempts | Correct | Rate | Class | Unique outputs |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for item in report["task_diagnostics"]:
        lines.append(
            f"| {item['task_id']} | {item['family']} | {item['attempts']} | "
            f"{item['correct']} | {item['exact_match_rate']:.3f} | "
            f"{item['difficulty_class']} | {item['unique_raw_outputs']} |"
        )
    lines.extend(
        (
            "",
            "## Gate Reasons",
            "",
            f"- Failures: `{readiness['failures']}`",
            f"- Warnings: `{readiness['warnings']}`",
            "",
        )
    )
    return "\n".join(lines)


def task_set_json(
    artifact: TaskSetArtifact,
    tasks: Sequence[Task],
    *,
    task_set_version: str,
    status: str,
) -> str:
    payload = {
        "schema_version": TASK_SET_SCHEMA_VERSION,
        "task_set_version": task_set_version,
        "created_for": artifact.created_for,
        "scorer_version": artifact.scorer_version,
        "model_used_for_validation": artifact.model_used_for_validation,
        "validation_date": artifact.validation_date,
        "status": status,
        "provenance": dict(artifact.provenance),
        "tasks": [
            {
                "task_id": task.task_id,
                "family": task.family,
                "prompt": task.prompt,
                "expected_answer": task.expected_answer,
                "scorer_version": task.scorer_version,
            }
            for task in tasks
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
