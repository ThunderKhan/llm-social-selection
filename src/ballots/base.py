from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from ..agents import AgentIdentity
from ..domain import Ballot, BallotEvidence, Response
from ..models import ModelProvider
from ..population import Population
from ..tasks import Task


@dataclass(frozen=True)
class BallotGeneration:
    ballot: Ballot
    evidence: BallotEvidence | None = None


class BallotProvider(ABC):
    @abstractmethod
    def generate_ballot(
        self,
        *,
        trial_id: str,
        round_index: int,
        trial_seed: int,
        task: Task,
        voter: AgentIdentity,
        population: Population,
        responses: Sequence[Response],
        model_provider: ModelProvider,
    ) -> BallotGeneration:
        """Generate one support ballot for an eligible voter."""
