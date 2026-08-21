from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from ..agents import AgentIdentity


class PopulationError(ValueError):
    """A population violates the v0.1 apparatus invariants."""


@dataclass(frozen=True)
class Population:
    agents: tuple[AgentIdentity, ...]
    _by_id: Mapping[str, AgentIdentity] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        agents = tuple(self.agents)
        if len(agents) != 8:
            raise PopulationError(f"population must contain exactly 8 agents, got {len(agents)}")

        by_id = {agent.agent_id: agent for agent in agents}
        if len(by_id) != len(agents):
            raise PopulationError("population agent IDs must be unique")

        object.__setattr__(self, "agents", agents)
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))

    def get(self, agent_id: str) -> AgentIdentity:
        try:
            return self._by_id[agent_id]
        except KeyError as error:
            raise PopulationError(f"unknown agent ID: {agent_id}") from error

    def __len__(self) -> int:
        return len(self.agents)
