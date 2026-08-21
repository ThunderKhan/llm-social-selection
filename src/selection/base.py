from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..domain import Ballot, Score, SelectionEvent
from ..population import Population


class SelectionError(ValueError):
    """Selection inputs violate the deterministic round protocol."""


def selection_id(trial_id: str, round_index: int) -> str:
    return f"selection-{trial_id}-r{round_index:03d}"


class SelectionStrategy(ABC):
    @abstractmethod
    def select(
        self,
        *,
        population: Population,
        ballots: Sequence[Ballot],
        scores: Sequence[Score],
        trial_id: str,
        round_index: int,
        seed: int,
    ) -> SelectionEvent:
        """Select one eligible agent without mutating round state."""
