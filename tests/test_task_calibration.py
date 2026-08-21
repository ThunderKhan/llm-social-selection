from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.domain import Response
from src.tasks import Task
from src.tasks.calibration import (
    TaskSetValidationError,
    analyze_objective_rounds,
    analyze_task_attempts,
    classify_difficulty,
    classify_format_error,
    evaluate_task_set_readiness,
    load_task_set,
    output_format_valid,
    render_validation_markdown,
    select_validated_tasks,
    task_leakage_issues,
    validate_task,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "tasks" / "e01_candidates_v1.json"
VALIDATED = ROOT / "tasks" / "e01_validated_v1.json"
CONTRACT = (
    "\n\nOUTPUT REQUIREMENT:\n"
    "Return only the integer.\n"
    "Do not explain.\n"
    "Do not repeat the question.\n"
    "Do not use Markdown.\n"
    'Do not prepend "Answer:".'
)


def task(task_id: str = "arith-900", family: str = "arithmetic") -> Task:
    return Task(
        task_id=task_id,
        family=family,
        prompt="Calculate 20 + 21." + CONTRACT,
        expected_answer="41",
        scorer_version="exact-match-v1",
    )


def response(task_id: str, index: int, content: str) -> Response:
    return Response(
        response_id=f"response-{index}",
        trial_id="validation",
        round_index=0,
        task_id=task_id,
        agent_id=f"agent-{index}",
        content=content,
        provider_name="fixture",
        model_name="fixture-v1",
        latency_ms=float(index),
        token_count=index,
    )


def test_candidate_task_file_is_valid_and_versioned() -> None:
    artifact = load_task_set(CANDIDATES)

    assert artifact.task_set_version == "e01-candidates-v1"
    assert artifact.created_for == "E01 objective-selection task calibration"
    assert artifact.scorer_version == "exact-match-v1"
    assert artifact.model_used_for_validation == "qwen3:0.6b"
    assert len(artifact.tasks) == 35
    assert len({item.task_id for item in artifact.tasks}) == 35
    assert {item.family for item in artifact.tasks} == {
        "arithmetic",
        "counting",
        "logic",
        "multiple-choice",
        "sequence",
        "string",
        "symbolic",
    }
    assert all(item.expected_answer and item.expected_answer.strip() for item in artifact.tasks)
    assert all(item.scorer_version == "exact-match-v1" for item in artifact.tasks)
    assert all(not validate_task(item) for item in artifact.tasks)


def test_validated_task_file_is_a_clean_candidate_subset() -> None:
    candidates = load_task_set(CANDIDATES)
    validated = load_task_set(VALIDATED)

    assert validated.task_set_version == "e01-validated-v1"
    assert validated.status == "validated"
    assert len(validated.tasks) == 12
    assert len({item.family for item in validated.tasks}) >= 4
    assert set(validated.tasks) <= set(candidates.tasks)
    assert all(not task_leakage_issues(item) for item in validated.tasks)


def test_loader_rejects_duplicate_ids_and_wrong_scorer(tmp_path: Path) -> None:
    payload = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    payload["tasks"][1]["task_id"] = payload["tasks"][0]["task_id"]
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(payload), encoding="utf-8")
    payload = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    payload["tasks"][0]["scorer_version"] = "other-v1"
    wrong_scorer = tmp_path / "wrong-scorer.json"
    wrong_scorer.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TaskSetValidationError, match="unique"):
        load_task_set(duplicate)
    with pytest.raises(TaskSetValidationError, match="scorer_version"):
        load_task_set(wrong_scorer)


def test_validation_detects_contract_id_family_and_leakage_issues() -> None:
    base = task()
    missing_contract = replace(base, prompt="Calculate 20 + 21.")
    wrong_family = replace(base, family="counting")
    leaked = replace(
        base,
        prompt="Agent_id chooses. The answer is 41." + CONTRACT,
    )

    assert any("response-contract" in issue for issue in validate_task(missing_contract))
    assert "task_id prefix does not match family" in validate_task(wrong_family)
    assert "agent_id" in task_leakage_issues(leaked)
    assert "expected answer appears in the question text" in task_leakage_issues(leaked)


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (0.0, "too_hard"),
        (0.10, "acceptable"),
        (0.20, "strong"),
        (0.50, "strong"),
        (0.80, "strong"),
        (0.90, "acceptable"),
        (1.0, "too_easy"),
    ],
)
def test_difficulty_classifier_uses_frozen_bands(rate: float, expected: str) -> None:
    assert classify_difficulty(rate) == expected


def test_task_attempt_diagnostics_preserve_raw_format_failures() -> None:
    item = task()
    responses = (
        response(item.task_id, 1, "41"),
        response(item.task_id, 2, "Answer: 41"),
        response(item.task_id, 3, "The result is 41"),
        response(item.task_id, 4, "42"),
    )

    diagnostic = analyze_task_attempts(item, responses)

    assert diagnostic["correct"] == 1
    assert diagnostic["exact_match_rate"] == 0.25
    assert diagnostic["unique_raw_outputs"] == 4
    assert diagnostic["format_error_patterns"] == {
        "answer_prefix": 1,
        "expected_answer_with_extra_text": 1,
        "wrong_single_token": 1,
    }
    assert classify_format_error("41", "41") is None
    assert diagnostic["format_compliance_rate"] == 0.5


def test_format_compliance_is_family_specific() -> None:
    integer = task()
    choice = replace(task("mcq-900", "multiple-choice"), expected_answer="B")
    boolean = replace(task("logic-900", "logic"), expected_answer="false")

    assert output_format_valid(integer, "-12") is True
    assert output_format_valid(integer, "twelve") is False
    assert output_format_valid(choice, "B") is True
    assert output_format_valid(choice, "b") is False
    assert output_format_valid(boolean, "false") is True
    assert output_format_valid(boolean, "False") is False


def test_final_selection_is_deterministic_balanced_and_rejects_extremes() -> None:
    tasks = tuple(
        task(f"arith-{index:03d}", "arithmetic")
        for index in range(1, 7)
    ) + tuple(
        replace(
            task(f"string-{index:03d}", "string"),
            prompt="Reverse DOG." + CONTRACT.replace("integer", "string"),
            expected_answer="GOD",
        )
        for index in range(1, 7)
    )
    rates = [0.0, 1.0, 0.50, 0.25, 0.75, 0.90] * 2
    diagnostics = {
        item.task_id: {"attempts": 16, "exact_match_rate": rate}
        for item, rate in zip(tasks, rates, strict=True)
    }

    selected, rejected = select_validated_tasks(tasks, diagnostics, maximum_tasks=6)

    assert [item.family for item in selected] == [
        "arithmetic",
        "string",
        "arithmetic",
        "string",
        "arithmetic",
        "string",
    ]
    assert rejected["arith-001"] == "too_hard"
    assert rejected["arith-002"] == "too_easy"
    assert selected == select_validated_tasks(tasks, diagnostics, maximum_tasks=6)[0]


def test_selection_uses_only_ceiling_tasks_as_minimum_count_fallback() -> None:
    tasks = (
        task("arith-001", "arithmetic"),
        task("arith-002", "arithmetic"),
        replace(task("logic-001", "logic"), expected_answer="false"),
        replace(task("string-001", "string"), expected_answer="ABC"),
    )
    diagnostics = {
        "arith-001": {"attempts": 16, "exact_match_rate": 0.5},
        "arith-002": {"attempts": 16, "exact_match_rate": 0.0},
        "logic-001": {"attempts": 16, "exact_match_rate": 1.0},
        "string-001": {"attempts": 16, "exact_match_rate": 1.0},
    }

    selected, rejected = select_validated_tasks(
        tasks, diagnostics, maximum_tasks=3, minimum_tasks=3
    )

    assert [item.task_id for item in selected] == [
        "arith-001",
        "logic-001",
        "string-001",
    ]
    assert rejected == {"arith-002": "too_hard"}


def test_objective_round_diagnostics_detect_degeneracy_and_ties() -> None:
    result = analyze_objective_rounds(
        (
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            (1.0,) * 8,
            (0.0,) * 8,
            (0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )
    )

    assert result["mixed_score_round_rate"] == 0.5
    assert result["all_correct_rate"] == 0.25
    assert result["all_incorrect_rate"] == 0.25
    assert result["degenerate_objective_round_rate"] == 0.5
    assert result["objective_tie_rate"] == 0.75
    assert result["mean_tied_lowest_count"] == 5


def test_readiness_gate_and_report_generation() -> None:
    selected = tuple(task(f"arith-{index:03d}") for index in range(1, 13))
    selected = tuple(
        replace(
            item,
            task_id=f"{prefix}-{index:03d}",
            family=family,
        )
        for index, (item, (prefix, family)) in enumerate(
            zip(
                selected,
                (("arith", "arithmetic"), ("count", "counting"), ("logic", "logic"), ("string", "string")) * 3,
                strict=True,
            ),
            start=1,
        )
    )
    objective = analyze_objective_rounds(((0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),) * 12)
    readiness = evaluate_task_set_readiness(
        integrity_issues=(),
        provider_failures=0,
        provider_requests=100,
        selected_tasks=selected,
        objective_rounds=objective,
    )
    report = {
        "readiness": readiness,
        "objective_round_simulation": objective,
        "candidate_task_count": 12,
        "selected_task_count": 12,
        "rejected_task_count": 0,
        "selected_families": sorted({item.family for item in selected}),
        "validation_attempt_count": 192,
        "selection_criteria": {},
        "provider": {"requests": 288, "failures": 0},
        "integrity": {"issues": []},
        "format_compliance": {"rate": 1.0, "patterns": {}},
        "task_diagnostics": [
            {
                "task_id": "arith-001",
                "family": "arithmetic",
                "attempts": 16,
                "correct": 8,
                "exact_match_rate": 0.5,
                "difficulty_class": "strong",
                "unique_raw_outputs": 2,
            }
        ],
    }

    assert readiness["decision"] == "OBJECTIVE TASK SET READY"
    assert "OBJECTIVE TASK SET READY" in render_validation_markdown(report)

    failed = evaluate_task_set_readiness(
        integrity_issues=("leak",),
        provider_failures=6,
        provider_requests=100,
        selected_tasks=selected[:3],
        objective_rounds=analyze_objective_rounds(((0.0,) * 8,)),
    )
    assert failed["decision"] == "REVISE TASK SET"
    assert len(failed["failures"]) == 5
