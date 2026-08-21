from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agents.models import AgentIdentity
    from ..tasks import Task


def _require_non_empty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


@dataclass(frozen=True)
class ModelOutput:
    content: str
    provider_name: str
    model_name: str
    request_id: str
    seed: int | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None
    token_count: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.content, "content")
        _require_non_empty(self.provider_name, "provider_name")
        _require_non_empty(self.model_name, "model_name")
        _require_non_empty(self.request_id, "request_id")
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
        ):
            raise ValueError("seed must be an integer or None")
        if self.finish_reason is not None:
            _require_non_empty(self.finish_reason, "finish_reason")
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


class ModelProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Stable provider identifier."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable model identifier."""

    @abstractmethod
    def generate(
        self,
        *,
        agent: AgentIdentity,
        task: Task,
        prompt: str,
        request_id: str,
        seed: int | None = None,
        response_schema: Mapping[str, Any] | None = None,
    ) -> ModelOutput:
        """Generate model output for one explicit request."""
