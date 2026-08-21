from ..domain import Response, Score
from ..tasks import Task


SCORER_VERSION = "exact-match-v1"


class ScoringError(ValueError):
    """A response cannot be scored against the supplied task."""


def normalize_answer(value: str) -> str:
    """Collapse whitespace and apply Unicode-aware case normalization."""
    return " ".join(value.split()).casefold()


def score_response(
    response: Response,
    task: Task,
    *,
    scorer_version: str = SCORER_VERSION,
) -> Score:
    if response.task_id != task.task_id:
        raise ScoringError(
            f"response task ID {response.task_id!r} does not match task {task.task_id!r}"
        )
    if task.expected_answer is None:
        raise ScoringError(f"task {task.task_id!r} has no expected answer")
    if scorer_version != SCORER_VERSION:
        raise ScoringError(f"unsupported scorer version: {scorer_version}")
    if task.scorer_version != scorer_version:
        raise ScoringError(
            f"task scorer version {task.scorer_version!r} does not match {scorer_version!r}"
        )

    value = 1.0 if normalize_answer(response.content) == normalize_answer(task.expected_answer) else 0.0
    return Score(
        score_id=f"score-{response.response_id}",
        trial_id=response.trial_id,
        round_index=response.round_index,
        task_id=task.task_id,
        agent_id=response.agent_id,
        value=value,
        scorer_version=scorer_version,
    )
