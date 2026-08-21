from dataclasses import dataclass
from math import isfinite
from typing import Literal


SelectionMechanism = Literal["peer_vote", "objective", "random"]


def _require_non_empty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_round_index(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("round_index must be a non-negative integer")


@dataclass(frozen=True)
class Response:
    response_id: str
    trial_id: str
    round_index: int
    task_id: str
    agent_id: str
    content: str
    provider_name: str
    model_name: str
    request_id: str | None = None
    seed: int | None = None
    latency_ms: float | None = None
    token_count: int | None = None

    def __post_init__(self) -> None:
        for field, value in (
            ("response_id", self.response_id),
            ("trial_id", self.trial_id),
            ("task_id", self.task_id),
            ("agent_id", self.agent_id),
            ("content", self.content),
            ("provider_name", self.provider_name),
            ("model_name", self.model_name),
        ):
            _require_non_empty(value, field)
        _require_round_index(self.round_index)
        if self.request_id is not None:
            _require_non_empty(self.request_id, "request_id")
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
        ):
            raise ValueError("seed must be an integer or None")
        if self.latency_ms is not None and (
            not isinstance(self.latency_ms, (int, float))
            or isinstance(self.latency_ms, bool)
            or not isfinite(self.latency_ms)
            or self.latency_ms < 0
        ):
            raise ValueError("latency_ms must be a non-negative finite number or None")
        if self.token_count is not None and (
            not isinstance(self.token_count, int)
            or isinstance(self.token_count, bool)
            or self.token_count < 0
        ):
            raise ValueError("token_count must be a non-negative integer or None")


@dataclass(frozen=True)
class Ballot:
    ballot_id: str
    trial_id: str
    round_index: int
    voter_agent_id: str
    supported_agent_id: str | None

    def __post_init__(self) -> None:
        for field, value in (
            ("ballot_id", self.ballot_id),
            ("trial_id", self.trial_id),
            ("voter_agent_id", self.voter_agent_id),
        ):
            _require_non_empty(value, field)
        _require_round_index(self.round_index)
        if self.supported_agent_id is not None:
            _require_non_empty(self.supported_agent_id, "supported_agent_id")
        if self.voter_agent_id == self.supported_agent_id:
            raise ValueError("support ballot cannot vote for the same agent")


@dataclass(frozen=True)
class BallotCandidate:
    label: str
    agent_id: str
    response_id: str

    def __post_init__(self) -> None:
        _require_non_empty(self.label, "label")
        _require_non_empty(self.agent_id, "agent_id")
        _require_non_empty(self.response_id, "response_id")


@dataclass(frozen=True)
class BallotEvidence:
    ballot_id: str
    trial_id: str
    round_index: int
    task_id: str
    voter_agent_id: str
    provider_name: str
    model_name: str
    request_id: str
    seed: int | None
    raw_output: str
    parsed_choice: str | None
    valid: bool
    invalid_reason: str | None
    candidate_order: tuple[BallotCandidate, ...]
    latency_ms: float | None = None
    token_count: int | None = None

    def __post_init__(self) -> None:
        for field, value in (
            ("ballot_id", self.ballot_id),
            ("trial_id", self.trial_id),
            ("task_id", self.task_id),
            ("voter_agent_id", self.voter_agent_id),
            ("provider_name", self.provider_name),
            ("model_name", self.model_name),
            ("request_id", self.request_id),
            ("raw_output", self.raw_output),
        ):
            _require_non_empty(value, field)
        _require_round_index(self.round_index)
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
        ):
            raise ValueError("seed must be an integer or None")
        if self.latency_ms is not None and (
            not isinstance(self.latency_ms, (int, float))
            or isinstance(self.latency_ms, bool)
            or not isfinite(self.latency_ms)
            or self.latency_ms < 0
        ):
            raise ValueError("latency_ms must be a non-negative finite number or None")
        if self.token_count is not None and (
            not isinstance(self.token_count, int)
            or isinstance(self.token_count, bool)
            or self.token_count < 0
        ):
            raise ValueError("token_count must be a non-negative integer or None")
        candidates = tuple(self.candidate_order)
        if len(candidates) != 7:
            raise ValueError("candidate_order must contain exactly 7 candidates")
        if len({candidate.label for candidate in candidates}) != 7:
            raise ValueError("candidate labels must be unique")
        if len({candidate.agent_id for candidate in candidates}) != 7:
            raise ValueError("candidate agent IDs must be unique")
        if self.voter_agent_id in {candidate.agent_id for candidate in candidates}:
            raise ValueError("candidate_order must exclude the voter")
        if self.valid:
            if self.parsed_choice not in {candidate.label for candidate in candidates}:
                raise ValueError("valid ballot evidence requires an eligible parsed_choice")
            if self.invalid_reason is not None:
                raise ValueError("valid ballot evidence cannot have invalid_reason")
        else:
            if self.parsed_choice is not None:
                raise ValueError("invalid ballot evidence cannot have parsed_choice")
            _require_non_empty(self.invalid_reason, "invalid_reason")
        object.__setattr__(self, "candidate_order", candidates)


@dataclass(frozen=True)
class Score:
    score_id: str
    trial_id: str
    round_index: int
    task_id: str
    agent_id: str
    value: float
    scorer_version: str

    def __post_init__(self) -> None:
        for field, value in (
            ("score_id", self.score_id),
            ("trial_id", self.trial_id),
            ("task_id", self.task_id),
            ("agent_id", self.agent_id),
            ("scorer_version", self.scorer_version),
        ):
            _require_non_empty(value, field)
        _require_round_index(self.round_index)
        if (
            not isinstance(self.value, (int, float))
            or isinstance(self.value, bool)
            or not isfinite(self.value)
        ):
            raise ValueError("value must be a finite number")


@dataclass(frozen=True)
class SelectionEvent:
    selection_id: str
    trial_id: str
    round_index: int
    mechanism: SelectionMechanism
    selected_agent_id: str
    reason: str

    def __post_init__(self) -> None:
        for field, value in (
            ("selection_id", self.selection_id),
            ("trial_id", self.trial_id),
            ("selected_agent_id", self.selected_agent_id),
            ("reason", self.reason),
        ):
            _require_non_empty(value, field)
        _require_round_index(self.round_index)
        if self.mechanism not in ("peer_vote", "objective", "random"):
            raise ValueError(
                "mechanism must be one of peer_vote, objective, or random"
            )


@dataclass(frozen=True)
class ReplacementEvent:
    replacement_id: str
    trial_id: str
    round_index: int
    removed_agent_id: str
    added_agent_id: str
    profile_id: str
    queue_index: int
    reason: Literal["fixed_profile_pool"] = "fixed_profile_pool"

    def __post_init__(self) -> None:
        for field, value in (
            ("replacement_id", self.replacement_id),
            ("trial_id", self.trial_id),
            ("removed_agent_id", self.removed_agent_id),
            ("added_agent_id", self.added_agent_id),
            ("profile_id", self.profile_id),
        ):
            _require_non_empty(value, field)
        _require_round_index(self.round_index)
        if self.removed_agent_id == self.added_agent_id:
            raise ValueError("replacement agent ID must differ from removed agent ID")
        if (
            not isinstance(self.queue_index, int)
            or isinstance(self.queue_index, bool)
            or self.queue_index < 0
        ):
            raise ValueError("queue_index must be a non-negative integer")
        if self.reason != "fixed_profile_pool":
            raise ValueError("reason must be fixed_profile_pool")
