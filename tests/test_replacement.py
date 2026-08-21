from __future__ import annotations

import random

import pytest

from src.agents import AgentIdentity, PromptProfile
from src.population import Population, build_initial_population
from src.replacement import (
    FixedReplacementQueue,
    ReplacementError,
    replace_selected_agent,
)


def profile_pool() -> dict[str, PromptProfile]:
    return {
        f"profile-{index:03d}": PromptProfile(
            f"profile-{index:03d}", {"setting": index}, "v1"
        )
        for index in range(1, 9)
    }


def test_fixed_queue_is_deterministic_and_condition_independent() -> None:
    profiles = profile_pool()
    first = FixedReplacementQueue.build(
        trial_id="trial-001", trial_seed=42, profiles=profiles, count=12
    )
    second = FixedReplacementQueue.build(
        trial_id="trial-001",
        trial_seed=42,
        profiles=dict(reversed(tuple(profiles.items()))),
        count=12,
    )

    assert first == second
    assert len(first.candidates) == 12
    assert len({candidate.agent.agent_id for candidate in first.candidates}) == 12
    assert tuple(candidate.queue_index for candidate in first.candidates) == tuple(
        range(12)
    )


def test_different_seed_changes_fixed_queue_profile_order() -> None:
    profiles = profile_pool()

    assert FixedReplacementQueue.build(
        trial_id="trial-001", trial_seed=42, profiles=profiles, count=8
    ).profile_ids != FixedReplacementQueue.build(
        trial_id="trial-001", trial_seed=43, profiles=profiles, count=8
    ).profile_ids


def test_fixed_queue_does_not_depend_on_global_random_state() -> None:
    profiles = profile_pool()
    random.seed(1)
    first = FixedReplacementQueue.build(
        trial_id="trial-001", trial_seed=42, profiles=profiles, count=9
    )
    random.seed(999)
    second = FixedReplacementQueue.build(
        trial_id="trial-001", trial_seed=42, profiles=profiles, count=9
    )

    assert first == second


def test_initial_population_is_deterministic_and_seeded() -> None:
    profiles = profile_pool()
    first = build_initial_population(
        trial_id="trial-001", trial_seed=42, profiles=profiles
    )
    second = build_initial_population(
        trial_id="trial-001", trial_seed=42, profiles=profiles
    )
    changed = build_initial_population(
        trial_id="trial-001", trial_seed=43, profiles=profiles
    )

    assert first == second
    assert len(first) == 8
    assert tuple(agent.profile_id for agent in first.agents) != tuple(
        agent.profile_id for agent in changed.agents
    )


def test_population_replacement_occurs_in_place(population: Population) -> None:
    replacement = AgentIdentity(
        "trial-001-replacement-000",
        "profile-001",
        "Replacement 1",
        0,
    )

    transitioned = replace_selected_agent(population, "agent-004", replacement)

    assert len(transitioned) == 8
    assert transitioned.agents[3] == replacement
    assert "agent-004" not in {agent.agent_id for agent in transitioned.agents}
    assert transitioned.agents[:3] == population.agents[:3]
    assert transitioned.agents[4:] == population.agents[4:]


def test_population_replacement_rejects_unknown_and_duplicate(
    population: Population,
) -> None:
    fresh = AgentIdentity("fresh", "profile-001", "Replacement", 0)
    with pytest.raises(ReplacementError, match="selected agent is not active"):
        replace_selected_agent(population, "missing", fresh)

    with pytest.raises(ReplacementError, match="already active"):
        replace_selected_agent(population, "agent-002", population.agents[0])
