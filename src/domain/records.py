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
    supported_agent_id: str

    def __post_init__(self) -> None:
        for field, value in (
            ("ballot_id", self.ballot_id),
            ("trial_id", self.trial_id),
            ("voter_agent_id", self.voter_agent_id),
            ("supported_agent_id", self.supported_agent_id),
        ):
            _require_non_empty(value, field)
        _require_round_index(self.round_index)
        if self.voter_agent_id == self.supported_agent_id:
            raise ValueError("support ballot cannot vote for the same agent")


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
