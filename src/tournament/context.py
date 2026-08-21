from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..agents import PromptProfile
from ..domain import SelectionMechanism
from ..population import Population
from ..tasks import Task


class RoundError(ValueError):
    """A round cannot be constructed or executed as requested."""


def _require_non_empty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RoundError(f"{field} must be a non-empty string")


@dataclass(frozen=True)
class RoundContext:
    experiment_id: str
    trial_id: str
    round_index: int
    condition: SelectionMechanism
    seed: int
    task: Task
    population: Population
    profiles: Mapping[str, PromptProfile]

    def __post_init__(self) -> None:
        _require_non_empty(self.experiment_id, "experiment_id")
        _require_non_empty(self.trial_id, "trial_id")
        if (
            not isinstance(self.round_index, int)
            or isinstance(self.round_index, bool)
            or self.round_index < 0
        ):
            raise RoundError("round_index must be a non-negative integer")
        if self.condition not in ("peer_vote", "objective", "random"):
            raise RoundError("condition must be one of peer_vote, objective, or random")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise RoundError("seed must be an integer")

        profiles = dict(self.profiles)
        for profile_id, profile in profiles.items():
            if profile_id != profile.profile_id:
                raise RoundError(
                    f"profile mapping key {profile_id!r} does not match profile.profile_id"
                )
        missing = sorted(
            {
                agent.profile_id
                for agent in self.population.agents
                if agent.profile_id not in profiles
            }
        )
        if missing:
            raise RoundError(f"missing prompt profiles: {', '.join(missing)}")

        object.__setattr__(self, "profiles", MappingProxyType(profiles))
