import hashlib
import json
from collections.abc import Sequence

from ..domain import Ballot, Response


class BallotError(ValueError):
    """A deterministic support ballot cannot be generated from the inputs."""


def generate_support_ballot(
    *,
    trial_id: str,
    round_index: int,
    task_id: str,
    voter_agent_id: str,
    eligible_agent_ids: Sequence[str],
    responses: Sequence[Response],
    seed: int,
) -> Ballot:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise BallotError("seed must be an integer")
    eligible = tuple(sorted(eligible_agent_ids))
    if len(eligible) != len(set(eligible)):
        raise BallotError("eligible agent IDs must be unique")
    if voter_agent_id not in eligible:
        raise BallotError(f"voter is not eligible: {voter_agent_id}")

    candidates = tuple(agent_id for agent_id in eligible if agent_id != voter_agent_id)
    if not candidates:
        raise BallotError("support ballot requires at least one other eligible agent")

    response_rows = sorted(
        (
            response.agent_id,
            response.response_id,
            response.content,
        )
        for response in responses
    )
    response_agent_ids = [row[0] for row in response_rows]
    if response_agent_ids != list(eligible):
        raise BallotError("responses must contain exactly one response per eligible agent")
    for response in responses:
        if (
            response.trial_id != trial_id
            or response.round_index != round_index
            or response.task_id != task_id
        ):
            raise BallotError("response references do not match the ballot context")

    payload = json.dumps(
        {
            "candidates": candidates,
            "responses": response_rows,
            "round_index": round_index,
            "seed": seed,
            "task_id": task_id,
            "trial_id": trial_id,
            "voter_agent_id": voter_agent_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    supported_agent_id = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
    return Ballot(
        ballot_id=f"ballot-{trial_id}-r{round_index:03d}-{voter_agent_id}",
        trial_id=trial_id,
        round_index=round_index,
        voter_agent_id=voter_agent_id,
        supported_agent_id=supported_agent_id,
    )
