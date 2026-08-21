from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from src.agents import AgentIdentity, PromptProfile
from src.domain import Ballot, Response, Score, SelectionEvent
from src.tasks import Task


def test_valid_agent_identity_is_immutable() -> None:
    agent = AgentIdentity(
        agent_id="agent-001",
        profile_id="profile-003",
        display_label="Participant 1",
        generation=0,
    )

    assert agent.agent_id == "agent-001"
    assert agent.profile_id == "profile-003"
    with pytest.raises(FrozenInstanceError):
        agent.generation = 1  # type: ignore[misc]


def test_negative_agent_generation_is_rejected() -> None:
    with pytest.raises(ValueError, match="generation must be a non-negative integer"):
        AgentIdentity("agent-001", "profile-003", "Participant 1", -1)


@pytest.mark.parametrize("field", ["agent_id", "profile_id", "display_label"])
def test_empty_agent_identity_fields_are_rejected(field: str) -> None:
    values = {
        "agent_id": "agent-001",
        "profile_id": "profile-003",
        "display_label": "Participant 1",
        "generation": 0,
    }
    values[field] = "  "

    with pytest.raises(ValueError, match=f"{field} must be a non-empty string"):
        AgentIdentity(**values)  # type: ignore[arg-type]


def test_prompt_profile_defensively_freezes_parameters() -> None:
    parameters = {"setting_a": 2, "setting_b": "low"}
    profile = PromptProfile("profile-003", parameters, "v1")
    parameters["setting_a"] = 9

    assert profile.parameters["setting_a"] == 2
    with pytest.raises(TypeError):
        profile.parameters["setting_a"] = 4  # type: ignore[index]


def test_valid_task_construction() -> None:
    task = Task(
        task_id="task-logic-004",
        family="logic",
        prompt="What follows from the premises?",
        expected_answer="A",
        scorer_version="exact-v1",
    )

    assert task.expected_answer == "A"


def test_empty_task_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="task_id must be a non-empty string"):
        Task("", "logic", "Question", "A", "exact-v1")


def test_valid_response_construction() -> None:
    response = Response(
        response_id="response-t001-r03-a07",
        trial_id="trial-001",
        round_index=3,
        task_id="task-logic-004",
        agent_id="agent-007",
        content="Answer A",
        provider_name="mock",
        model_name="deterministic-v1",
        request_id="request-008",
        seed=42,
    )

    assert response.round_index == 3
    assert response.latency_ms is None
    assert response.token_count is None


def test_empty_response_content_is_rejected() -> None:
    with pytest.raises(ValueError, match="content must be a non-empty string"):
        Response(
            "response-001",
            "trial-001",
            0,
            "task-001",
            "agent-001",
            "",
            "mock",
            "deterministic-v1",
        )


def test_negative_round_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="round_index must be a non-negative integer"):
        Ballot("ballot-001", "trial-001", -1, "agent-001", "agent-002")


def test_valid_support_ballot_construction() -> None:
    ballot = Ballot("ballot-001", "trial-001", 2, "agent-001", "agent-002")

    assert ballot.supported_agent_id == "agent-002"


def test_self_ballot_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot vote for the same agent"):
        Ballot("ballot-001", "trial-001", 2, "agent-001", "agent-001")


@pytest.mark.parametrize("value", [nan, inf, -inf, True])
def test_invalid_score_value_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="value must be a finite number"):
        Score(
            "score-001",
            "trial-001",
            0,
            "task-001",
            "agent-001",
            value,
            "exact-v1",
        )


def test_valid_score_construction() -> None:
    score = Score(
        "score-001", "trial-001", 0, "task-001", "agent-001", 1.0, "exact-v1"
    )

    assert score.value == 1.0


def test_valid_selection_event_construction() -> None:
    event = SelectionEvent(
        selection_id="selection-001",
        trial_id="trial-001",
        round_index=4,
        mechanism="peer_vote",
        selected_agent_id="agent-006",
        reason="Recorded support result",
    )

    assert event.selected_agent_id == "agent-006"
    assert event.mechanism == "peer_vote"
