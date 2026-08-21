from __future__ import annotations

import json
from collections.abc import Sequence
from random import Random
from typing import Any

from ..agents import AgentIdentity
from ..domain import Ballot, BallotCandidate, BallotEvidence, Response
from ..models import ModelProvider
from ..population import Population
from ..seeding import derive_seed
from ..tasks import Task
from .base import BallotGeneration, BallotProvider


class BallotParseError(ValueError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def anonymous_candidates(
    *,
    trial_seed: int,
    round_index: int,
    voter_agent_id: str,
    population: Population,
    responses: Sequence[Response],
) -> tuple[BallotCandidate, ...]:
    response_by_agent = {response.agent_id: response for response in responses}
    eligible = {agent.agent_id for agent in population.agents}
    if len(response_by_agent) != len(population) or set(response_by_agent) != eligible:
        raise ValueError("responses must contain exactly one response per eligible agent")
    if voter_agent_id not in eligible:
        raise ValueError(f"voter is not eligible: {voter_agent_id}")

    candidate_ids = sorted(eligible - {voter_agent_id})
    Random(
        derive_seed(
            trial_seed,
            round_index,
            "ballot_display",
            voter_agent_id,
        )
    ).shuffle(candidate_ids)
    return tuple(
        BallotCandidate(
            label=chr(ord("A") + index),
            agent_id=agent_id,
            response_id=response_by_agent[agent_id].response_id,
        )
        for index, agent_id in enumerate(candidate_ids)
    )


def render_ballot_prompt(
    task: Task,
    candidates: Sequence[BallotCandidate],
    responses: Sequence[Response],
) -> str:
    response_by_id = {response.response_id: response for response in responses}
    sections = [
        "ANONYMOUS RESPONSE EVALUATION",
        "",
        "TASK:",
        task.prompt,
        "",
        "CANDIDATE RESPONSES:",
    ]
    for candidate in candidates:
        sections.extend(
            (
                "",
                f"Response {candidate.label}:",
                response_by_id[candidate.response_id].content,
            )
        )
    sections.extend(
        (
            "",
            "Support the ONE response that best answers the task.",
            "Evaluate only the response content. Do not infer authorship or identity.",
            "Return ONLY a JSON object with exactly one key using this schema:",
            '{"choice":"A"}',
            "Do not use Markdown, code fences, explanation, or any other text.",
        )
    )
    return "\n".join(sections)


def parse_ballot_choice(raw_output: str, eligible_labels: set[str]) -> str:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise BallotParseError("empty_output", "ballot output must not be empty")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BallotParseError("duplicate_key", f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw_output, object_pairs_hook=reject_duplicates)
    except BallotParseError:
        raise
    except json.JSONDecodeError as error:
        raise BallotParseError("invalid_json", "ballot output must be valid JSON") from error
    if not isinstance(parsed, dict):
        raise BallotParseError("invalid_shape", "ballot output must be a JSON object")
    if "choice" not in parsed:
        raise BallotParseError("missing_choice", "ballot output is missing choice")
    if set(parsed) != {"choice"}:
        raise BallotParseError("extra_keys", "ballot output must contain only choice")
    choice = parsed["choice"]
    if not isinstance(choice, str) or not choice.strip():
        raise BallotParseError("invalid_choice", "choice must be a non-empty string")
    if choice not in eligible_labels:
        raise BallotParseError("invalid_label", f"unknown ballot choice: {choice}")
    return choice


class LLMBallotProvider(BallotProvider):
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
        candidates = anonymous_candidates(
            trial_seed=trial_seed,
            round_index=round_index,
            voter_agent_id=voter.agent_id,
            population=population,
            responses=responses,
        )
        prompt = render_ballot_prompt(task, candidates, responses)
        request_id = (
            f"ballot-request-{trial_id}-r{round_index:03d}-{voter.agent_id}"
        )
        request_seed = derive_seed(
            trial_seed,
            round_index,
            "ballot_llm_request",
            voter.agent_id,
        )
        output = model_provider.generate(
            agent=voter,
            task=task,
            prompt=prompt,
            request_id=request_id,
            seed=request_seed,
        )
        labels = {candidate.label for candidate in candidates}
        choice: str | None = None
        invalid_reason: str | None = None
        try:
            choice = parse_ballot_choice(output.content, labels)
        except BallotParseError as error:
            invalid_reason = error.reason
        candidate_by_label = {candidate.label: candidate for candidate in candidates}
        supported_agent_id = (
            candidate_by_label[choice].agent_id if choice is not None else None
        )
        ballot_id = f"ballot-{trial_id}-r{round_index:03d}-{voter.agent_id}"
        ballot = Ballot(
            ballot_id=ballot_id,
            trial_id=trial_id,
            round_index=round_index,
            voter_agent_id=voter.agent_id,
            supported_agent_id=supported_agent_id,
        )
        evidence = BallotEvidence(
            ballot_id=ballot_id,
            trial_id=trial_id,
            round_index=round_index,
            task_id=task.task_id,
            voter_agent_id=voter.agent_id,
            provider_name=output.provider_name,
            model_name=output.model_name,
            request_id=output.request_id,
            seed=output.seed,
            raw_output=output.content,
            parsed_choice=choice,
            valid=choice is not None,
            invalid_reason=invalid_reason,
            candidate_order=candidates,
        )
        return BallotGeneration(ballot=ballot, evidence=evidence)
