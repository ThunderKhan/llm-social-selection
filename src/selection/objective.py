from collections.abc import Sequence
from random import Random

from ..domain import Ballot, Score, SelectionEvent
from ..population import Population
from .base import SelectionError, SelectionStrategy, selection_id


class ObjectiveSelectionStrategy(SelectionStrategy):
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
        del ballots
        eligible = {agent.agent_id for agent in population.agents}
        if len(scores) != len(population):
            raise SelectionError(
                f"objective requires exactly {len(population)} scores, got {len(scores)}"
            )

        values: dict[str, float] = {}
        for score in scores:
            if score.trial_id != trial_id or score.round_index != round_index:
                raise SelectionError("score references do not match the selection context")
            if score.agent_id not in eligible:
                raise SelectionError(f"score belongs to ineligible agent: {score.agent_id}")
            if score.agent_id in values:
                raise SelectionError(f"duplicate agent score: {score.agent_id}")
            values[score.agent_id] = score.value

        if set(values) != eligible:
            missing = ", ".join(sorted(eligible - set(values)))
            raise SelectionError(f"missing agent scores: {missing}")

        lowest = min(values.values())
        tied = sorted(agent_id for agent_id, value in values.items() if value == lowest)
        selected = tied[0] if len(tied) == 1 else Random(seed).choice(tied)
        return SelectionEvent(
            selection_id=selection_id(trial_id, round_index),
            trial_id=trial_id,
            round_index=round_index,
            mechanism="objective",
            selected_agent_id=selected,
            reason=f"lowest_objective_score;value={lowest:g}",
        )
