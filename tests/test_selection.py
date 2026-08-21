from __future__ import annotations

import pytest

from src.domain import Ballot, Score
from src.population import Population
from src.selection import (
    ObjectiveSelectionStrategy,
    PeerVoteSelectionStrategy,
    RandomSelectionStrategy,
    SelectionError,
)


def ballots_for_targets(targets: tuple[str, ...]) -> tuple[Ballot, ...]:
    return tuple(
        Ballot(
            ballot_id=f"ballot-{index:03d}",
            trial_id="trial-001",
            round_index=0,
            voter_agent_id=f"agent-{index:03d}",
            supported_agent_id=target,
        )
        for index, target in enumerate(targets, start=1)
    )


def scores_for_values(values: tuple[float, ...]) -> tuple[Score, ...]:
    return tuple(
        Score(
            score_id=f"score-{index:03d}",
            trial_id="trial-001",
            round_index=0,
            task_id="task-001",
            agent_id=f"agent-{index:03d}",
            value=value,
            scorer_version="exact-match-v1",
        )
        for index, value in enumerate(values, start=1)
    )


def select_arguments(population: Population) -> dict[str, object]:
    return {
        "population": population,
        "trial_id": "trial-001",
        "round_index": 0,
        "seed": 42,
    }


def test_peer_selects_unique_fewest_supported_agent(population: Population) -> None:
    # Vote totals are [2, 1, 1, 1, 1, 1, 1, 0].
    ballots = ballots_for_targets(
        (
            "agent-002",
            "agent-001",
            "agent-001",
            "agent-003",
            "agent-004",
            "agent-005",
            "agent-006",
            "agent-007",
        )
    )

    event = PeerVoteSelectionStrategy().select(
        **select_arguments(population), ballots=ballots, scores=()
    )

    assert event.selected_agent_id == "agent-008"
    assert event.reason == "fewest_support_votes;count=0"


def test_peer_all_equal_tie_is_deterministic(population: Population) -> None:
    ballots = ballots_for_targets(
        tuple(f"agent-{(index % 8) + 1:03d}" for index in range(1, 9))
    )
    strategy = PeerVoteSelectionStrategy()

    first = strategy.select(**select_arguments(population), ballots=ballots, scores=())
    second = strategy.select(**select_arguments(population), ballots=ballots, scores=())

    assert first == second
    assert first.selected_agent_id in {agent.agent_id for agent in population.agents}


def test_peer_rejects_ineligible_supported_agent(population: Population) -> None:
    ballots = list(
        ballots_for_targets(
            tuple(f"agent-{(index % 8) + 1:03d}" for index in range(1, 9))
        )
    )
    ballots[0] = Ballot(
        "ballot-001", "trial-001", 0, "agent-001", "agent-999"
    )

    with pytest.raises(SelectionError, match="ineligible supported agent"):
        PeerVoteSelectionStrategy().select(
            **select_arguments(population), ballots=ballots, scores=()
        )


def test_objective_selects_lowest_score(population: Population) -> None:
    scores = scores_for_values((1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0))

    event = ObjectiveSelectionStrategy().select(
        **select_arguments(population), ballots=(), scores=scores
    )

    assert event.selected_agent_id == "agent-003"
    assert event.reason == "lowest_objective_score;value=0"


def test_objective_tie_is_deterministic_and_ballots_are_ignored(
    population: Population,
) -> None:
    scores = scores_for_values((0.0,) * 8)
    strategy = ObjectiveSelectionStrategy()
    first = strategy.select(
        **select_arguments(population), ballots=(), scores=scores
    )
    second = strategy.select(
        **select_arguments(population),
        ballots=ballots_for_targets(
            tuple(f"agent-{(index % 8) + 1:03d}" for index in range(1, 9))
        ),
        scores=scores,
    )

    assert first == second


def test_random_selection_is_deterministic_eligible_and_ignores_evidence(
    population: Population,
) -> None:
    strategy = RandomSelectionStrategy()
    first = strategy.select(
        **select_arguments(population), ballots=(), scores=()
    )
    second = strategy.select(
        **select_arguments(population),
        ballots=ballots_for_targets(
            tuple(f"agent-{(index % 8) + 1:03d}" for index in range(1, 9))
        ),
        scores=scores_for_values(tuple(float(index) for index in range(8))),
    )

    assert first == second
    assert first.selected_agent_id in {agent.agent_id for agent in population.agents}
    assert first.reason == "seeded_uniform_random"
