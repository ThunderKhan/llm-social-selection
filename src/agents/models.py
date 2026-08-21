from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping, TypeAlias


PromptValue: TypeAlias = str | int | float | bool


def _require_non_empty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


@dataclass(frozen=True)
class AgentIdentity:
    agent_id: str
    profile_id: str
    display_label: str
    generation: int

    def __post_init__(self) -> None:
        _require_non_empty(self.agent_id, "agent_id")
        _require_non_empty(self.profile_id, "profile_id")
        _require_non_empty(self.display_label, "display_label")
        if not isinstance(self.generation, int) or isinstance(self.generation, bool):
            raise ValueError("generation must be a non-negative integer")
        if self.generation < 0:
            raise ValueError("generation must be a non-negative integer")


@dataclass(frozen=True)
class PromptProfile:
    profile_id: str
    parameters: Mapping[str, PromptValue]
    template_version: str

    def __post_init__(self) -> None:
        _require_non_empty(self.profile_id, "profile_id")
        _require_non_empty(self.template_version, "template_version")

        parameters = dict(self.parameters)
        for key, value in parameters.items():
            _require_non_empty(key, "parameter key")
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"parameters.{key} must be a simple scalar value")
            if isinstance(value, float) and not isfinite(value):
                raise ValueError(f"parameters.{key} must be finite")

        object.__setattr__(self, "parameters", MappingProxyType(parameters))
