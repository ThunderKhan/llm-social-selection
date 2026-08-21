from __future__ import annotations

import pytest

from src.domain import Response
from src.scoring import ScoringError, normalize_answer, score_response
from src.tasks import Task


def response(content: str, *, task_id: str = "task-001") -> Response:
    return Response(
        response_id="response-001",
        trial_id="trial-001",
        round_index=0,
        task_id=task_id,
        agent_id="agent-001",
        content=content,
        provider_name="mock",
        model_name="deterministic-v1",
    )


def task(expected_answer: str | None = "Answer A") -> Task:
    return Task(
        "task-001",
        "logic",
        "Provide an answer.",
        expected_answer,
        "exact-match-v1",
    )


def test_exact_answer_scores_one() -> None:
    assert score_response(response("Answer A"), task()).value == 1.0


def test_wrong_answer_scores_zero() -> None:
    assert score_response(response("Answer B"), task()).value == 0.0


def test_normalization_collapses_whitespace_and_casefolds() -> None:
    assert normalize_answer("  ANSWER\n  A ") == "answer a"
    assert score_response(response("  ANSWER\n  A "), task()).value == 1.0


def test_scoring_rejects_mismatched_task_reference() -> None:
    with pytest.raises(ScoringError, match="does not match task"):
        score_response(response("Answer A", task_id="task-other"), task())


def test_scoring_requires_answer_key() -> None:
    with pytest.raises(ScoringError, match="has no expected answer"):
        score_response(response("Answer A"), task(None))


def test_scoring_rejects_task_scorer_version_mismatch() -> None:
    mismatched = Task(
        "task-001",
        "logic",
        "Provide an answer.",
        "Answer A",
        "other-v1",
    )

    with pytest.raises(ScoringError, match="task scorer version"):
        score_response(response("Answer A"), mismatched)


def test_scoring_is_deterministic() -> None:
    first = score_response(response("Answer A"), task())
    second = score_response(response("Answer A"), task())

    assert first == second
