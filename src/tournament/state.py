from dataclasses import dataclass

from ..domain import ReplacementEvent, SelectionMechanism
from ..population import Population
from .result import RoundResult


@dataclass(frozen=True)
class TrialState:
    experiment_id: str
    trial_id: str
    trial_seed: int
    condition: SelectionMechanism
    total_rounds: int
    next_round_index: int
    population: Population
    replacement_queue_position: int
    completed: bool


@dataclass(frozen=True)
class TrialStep:
    result: RoundResult
    replacement: ReplacementEvent | None
    state: TrialState
