from ..agents import generate_agent_response
from ..models import ModelProvider
from ..scoring import score_response
from ..selection import (
    ObjectiveSelectionStrategy,
    PeerVoteSelectionStrategy,
    RandomSelectionStrategy,
    SelectionStrategy,
)
from .ballots import generate_support_ballot
from .context import RoundContext
from .result import RoundResult
from .seeds import derive_seed


def _strategy_for(context: RoundContext) -> SelectionStrategy:
    if context.condition == "peer_vote":
        return PeerVoteSelectionStrategy()
    if context.condition == "objective":
        return ObjectiveSelectionStrategy()
    return RandomSelectionStrategy()


class RoundEngine:
    def execute(self, context: RoundContext, provider: ModelProvider) -> RoundResult:
        responses = tuple(
            generate_agent_response(
                response_id=(
                    f"response-{context.trial_id}-r{context.round_index:03d}-{agent.agent_id}"
                ),
                trial_id=context.trial_id,
                round_index=context.round_index,
                agent=agent,
                profile=context.profiles[agent.profile_id],
                task=context.task,
                provider=provider,
                request_id=(
                    f"request-{context.trial_id}-r{context.round_index:03d}-{agent.agent_id}"
                ),
                seed=derive_seed(
                    context.seed,
                    context.round_index,
                    "response",
                    agent.agent_id,
                ),
            )
            for agent in context.population.agents
        )

        scores = tuple(score_response(response, context.task) for response in responses)
        eligible_ids = tuple(agent.agent_id for agent in context.population.agents)
        ballots = tuple(
            generate_support_ballot(
                trial_id=context.trial_id,
                round_index=context.round_index,
                task_id=context.task.task_id,
                voter_agent_id=agent.agent_id,
                eligible_agent_ids=eligible_ids,
                responses=responses,
                seed=derive_seed(
                    context.seed,
                    context.round_index,
                    "ballot",
                    agent.agent_id,
                ),
            )
            for agent in context.population.agents
        )

        selection_namespace = {
            "peer_vote": "peer_tiebreak",
            "objective": "objective_tiebreak",
            "random": "random_selection",
        }[context.condition]
        selection = _strategy_for(context).select(
            population=context.population,
            ballots=ballots,
            scores=scores,
            trial_id=context.trial_id,
            round_index=context.round_index,
            seed=derive_seed(
                context.seed,
                context.round_index,
                selection_namespace,
            ),
        )
        return RoundResult(
            trial_id=context.trial_id,
            round_index=context.round_index,
            task=context.task,
            responses=responses,
            scores=scores,
            ballots=ballots,
            selection=selection,
        )
