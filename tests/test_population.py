from __future__ import annotations

import pytest

from src.agents import AgentIdentity
from src.population import Population, PopulationError


def test_exactly_eight_unique_agents_are_accepted(
    eight_agents: tuple[AgentIdentity, ...],
) -> None:
    population = Population(eight_agents)

    assert len(population) == 8
    assert population.agents == eight_agents


@pytest.mark.parametrize("count", [7, 9])
def test_population_rejects_non_v01_size(
    eight_agents: tuple[AgentIdentity, ...], count: int
) -> None:
    agents = list(eight_agents[:count])
    if count == 9:
        agents.append(AgentIdentity("agent-009", "profile-009", "Participant 9", 0))

    with pytest.raises(PopulationError, match="exactly 8 agents"):
        Population(tuple(agents))


def test_population_rejects_duplicate_agent_ids(
    eight_agents: tuple[AgentIdentity, ...],
) -> None:
    duplicate = AgentIdentity(
        eight_agents[0].agent_id,
        "profile-999",
        "Participant 9",
        0,
    )

    with pytest.raises(PopulationError, match="agent IDs must be unique"):
        Population((*eight_agents[:-1], duplicate))


def test_population_lookup_returns_identity(
    population: Population,
) -> None:
    assert population.get("agent-004") is population.agents[3]

    with pytest.raises(PopulationError, match="unknown agent ID"):
        population.get("agent-999")


def test_population_preserves_caller_order(
    eight_agents: tuple[AgentIdentity, ...],
) -> None:
    reversed_agents = tuple(reversed(eight_agents))

    assert Population(reversed_agents).agents == reversed_agents
