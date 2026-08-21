from collections.abc import Sequence

from ..agents import AgentIdentity
from ..domain import Response
from ..models import ModelProvider
from ..population import Population
from ..seeding import derive_seed
from ..tasks import Task
from .base import BallotGeneration, BallotProvider


class DeterministicBallotProvider(BallotProvider):
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
        from ..tournament.ballots import generate_support_ballot

        del model_provider
        ballot = generate_support_ballot(
            trial_id=trial_id,
            round_index=round_index,
            task_id=task.task_id,
            voter_agent_id=voter.agent_id,
            eligible_agent_ids=tuple(agent.agent_id for agent in population.agents),
            responses=responses,
            seed=derive_seed(
                trial_seed,
                round_index,
                "ballot",
                voter.agent_id,
            ),
        )
        return BallotGeneration(ballot=ballot)
