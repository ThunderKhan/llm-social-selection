from __future__ import annotations

from dataclasses import replace

import pytest

from src.agents import PromptProfile
from src.domain import SelectionEvent
from src.models import MockModelProvider
from src.population import Population
from src.tasks import Task
from src.tournament import RoundContext, RoundEngine, RoundError


def context_for(
    condition: str,
    population: Population,
    profiles: dict[str, PromptProfile],
    task: Task,
) -> RoundContext:
    return RoundContext(
        experiment_id="experiment-e00",
        trial_id="trial-001",
        round_index=0,
        condition=condition,  # type: ignore[arg-type]
        seed=42,
        task=task,
        population=population,
        profiles=profiles,
    )


def test_round_context_rejects_missing_profile(
    population: Population,
    profiles: dict[str, PromptProfile],
    round_task: Task,
) -> None:
    del profiles["profile-008"]

    with pytest.raises(RoundError, match="missing prompt profiles: profile-008"):
        context_for("peer_vote", population, profiles, round_task)


def test_round_context_rejects_negative_index(
    population: Population,
    profiles: dict[str, PromptProfile],
    round_task: Task,
) -> None:
    context = context_for("peer_vote", population, profiles, round_task)

    with pytest.raises(RoundError, match="round_index must be a non-negative integer"):
        replace(context, round_index=-1)


@pytest.mark.parametrize("condition", ["peer_vote", "objective", "random"])
def test_all_conditions_complete_one_comparable_round(
    condition: str,
    population: Population,
    profiles: dict[str, PromptProfile],
    round_task: Task,
) -> None:
    result = RoundEngine().execute(
        context_for(condition, population, profiles, round_task),
        MockModelProvider(),
    )

    assert len(result.responses) == 8
    assert len(result.scores) == 8
    assert len(result.ballots) == 8
    assert isinstance(result.selection, SelectionEvent)
    assert result.selection.mechanism == condition
    assert {response.agent_id for response in result.responses} == {
        agent.agent_id for agent in population.agents
    }
    assert {ballot.voter_agent_id for ballot in result.ballots} == {
        agent.agent_id for agent in population.agents
    }


def test_repeated_round_execution_is_identical(
    population: Population,
    profiles: dict[str, PromptProfile],
    round_task: Task,
) -> None:
    context = context_for("peer_vote", population, profiles, round_task)
    engine = RoundEngine()

    first = engine.execute(context, MockModelProvider())
    second = engine.execute(context, MockModelProvider())

    assert first == second


def test_conditions_share_response_score_and_ballot_evidence(
    population: Population,
    profiles: dict[str, PromptProfile],
    round_task: Task,
) -> None:
    engine = RoundEngine()
    results = tuple(
        engine.execute(
            context_for(condition, population, profiles, round_task),
            MockModelProvider(),
        )
        for condition in ("peer_vote", "objective", "random")
    )

    assert results[0].responses == results[1].responses == results[2].responses
    assert results[0].scores == results[1].scores == results[2].scores
    assert results[0].ballots == results[1].ballots == results[2].ballots
    assert {result.selection.mechanism for result in results} == {
        "peer_vote",
        "objective",
        "random",
    }
