from __future__ import annotations

import pytest

from scripts.summarize_e02a_task_profile_revision import attach_exact_match_scores
from src.tasks import Task


def test_attach_exact_match_scores_reconstructs_raw_checkpoint_rows() -> None:
    task = Task(
        task_id="arith-999",
        family="arithmetic",
        prompt="Return only the integer.",
        expected_answer="4",
        scorer_version="exact-match-v1",
    )
    rows = [
        {"evaluation_id": "ok-1", "task_id": task.task_id, "status": "ok", "raw_output": " 4\n"},
        {"evaluation_id": "ok-2", "task_id": task.task_id, "status": "ok", "raw_output": "5"},
        {"evaluation_id": "err-1", "task_id": task.task_id, "status": "error"},
    ]

    attach_exact_match_scores(rows, {task.task_id: task})

    assert rows[0]["correct"] is True
    assert rows[1]["correct"] is False
    assert "correct" not in rows[2]


def test_attach_exact_match_scores_rejects_unknown_task_reference() -> None:
    rows = [
        {
            "evaluation_id": "unknown-1",
            "task_id": "missing-999",
            "status": "ok",
            "raw_output": "4",
        }
    ]

    with pytest.raises(ValueError, match="unknown task_id"):
        attach_exact_match_scores(rows, {})
