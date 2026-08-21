from dataclasses import dataclass

from ..domain import Ballot, BallotEvidence, Response, Score, SelectionEvent
from ..tasks import Task


@dataclass(frozen=True)
class RoundResult:
    trial_id: str
    round_index: int
    task: Task
    responses: tuple[Response, ...]
    scores: tuple[Score, ...]
    ballots: tuple[Ballot, ...]
    selection: SelectionEvent
    ballot_evidence: tuple[BallotEvidence, ...] = ()
