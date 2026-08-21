from collections.abc import Sequence
from random import Random

from ..domain import Ballot, Score, SelectionEvent
from ..population import Population
from .base import SelectionStrategy, selection_id


class RandomSelectionStrategy(SelectionStrategy):
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
        del ballots, scores
        eligible = sorted(agent.agent_id for agent in population.agents)
        selected = Random(seed).choice(eligible)
        return SelectionEvent(
            selection_id=selection_id(trial_id, round_index),
            trial_id=trial_id,
            round_index=round_index,
            mechanism="random",
            selected_agent_id=selected,
            reason="seeded_uniform_random",
        )
