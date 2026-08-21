from collections import Counter
from collections.abc import Sequence
from random import Random

from ..domain import Ballot, Score, SelectionEvent
from ..population import Population
from .base import SelectionError, SelectionStrategy, selection_id


class PeerVoteSelectionStrategy(SelectionStrategy):
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
        del scores
        eligible = {agent.agent_id for agent in population.agents}
        if len(ballots) != len(population):
            raise SelectionError(
                f"peer_vote requires exactly {len(population)} ballots, got {len(ballots)}"
            )

        voters: set[str] = set()
        counts = Counter({agent_id: 0 for agent_id in eligible})
        for ballot in ballots:
            if ballot.trial_id != trial_id or ballot.round_index != round_index:
                raise SelectionError("ballot references do not match the selection context")
            if ballot.voter_agent_id not in eligible:
                raise SelectionError(f"ineligible voter: {ballot.voter_agent_id}")
            if ballot.supported_agent_id not in eligible:
                raise SelectionError(
                    f"ineligible supported agent: {ballot.supported_agent_id}"
                )
            if ballot.voter_agent_id in voters:
                raise SelectionError(f"duplicate voter ballot: {ballot.voter_agent_id}")
            voters.add(ballot.voter_agent_id)
            counts[ballot.supported_agent_id] += 1

        if voters != eligible:
            missing = ", ".join(sorted(eligible - voters))
            raise SelectionError(f"missing voter ballots: {missing}")

        fewest = min(counts.values())
        tied = sorted(agent_id for agent_id, count in counts.items() if count == fewest)
        selected = tied[0] if len(tied) == 1 else Random(seed).choice(tied)
        return SelectionEvent(
            selection_id=selection_id(trial_id, round_index),
            trial_id=trial_id,
            round_index=round_index,
            mechanism="peer_vote",
            selected_agent_id=selected,
            reason=f"fewest_support_votes;count={fewest}",
        )
