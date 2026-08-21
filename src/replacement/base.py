from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..agents import AgentIdentity


class ReplacementError(ValueError):
    """A fixed-profile population transition is invalid."""


@dataclass(frozen=True)
class ReplacementCandidate:
    queue_index: int
    profile_id: str
    agent: AgentIdentity


class ReplacementStrategy(ABC):
    @abstractmethod
    def replacement_for(self, queue_index: int) -> ReplacementCandidate:
        """Return the precomputed replacement at an explicit queue position."""
