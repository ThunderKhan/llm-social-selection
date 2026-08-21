import hashlib
import json
from dataclasses import dataclass
from random import Random
from typing import Mapping

from ..agents import AgentIdentity, PromptProfile
from ..seeding import derive_seed
from .base import ReplacementCandidate, ReplacementError, ReplacementStrategy


FIXED_QUEUE_VERSION = "fixed-profile-pool-v1"


def profile_pool_hash(profiles: Mapping[str, PromptProfile]) -> str:
    rows = []
    for profile_id in sorted(profiles):
        profile = profiles[profile_id]
        if profile.profile_id != profile_id:
            raise ReplacementError(
                f"profile mapping key {profile_id!r} does not match profile ID"
            )
        rows.append(
            {
                "parameters": dict(profile.parameters),
                "profile_id": profile.profile_id,
                "template_version": profile.template_version,
            }
        )
    payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FixedReplacementQueue(ReplacementStrategy):
    trial_id: str
    trial_seed: int
    candidates: tuple[ReplacementCandidate, ...]
    version: str = FIXED_QUEUE_VERSION

    @classmethod
    def build(
        cls,
        *,
        trial_id: str,
        trial_seed: int,
        profiles: Mapping[str, PromptProfile],
        count: int,
        agent_id_namespace: str | None = None,
    ) -> "FixedReplacementQueue":
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ReplacementError("replacement count must be a non-negative integer")
        profile_ids = sorted(profiles)
        if not profile_ids and count:
            raise ReplacementError("fixed profile pool must not be empty")
        profile_pool_hash(profiles)

        ordered: list[str] = []
        block_index = 0
        while len(ordered) < count:
            block = profile_ids.copy()
            Random(
                derive_seed(
                    trial_seed,
                    block_index,
                    "fixed_replacement_queue",
                    FIXED_QUEUE_VERSION,
                )
            ).shuffle(block)
            ordered.extend(block)
            block_index += 1

        identity_namespace = agent_id_namespace or trial_id
        candidates = tuple(
            ReplacementCandidate(
                queue_index=index,
                profile_id=profile_id,
                agent=AgentIdentity(
                    agent_id=f"{identity_namespace}-replacement-{index:03d}",
                    profile_id=profile_id,
                    display_label=f"Replacement {index + 1}",
                    generation=0,
                ),
            )
            for index, profile_id in enumerate(ordered[:count])
        )
        return cls(trial_id=trial_id, trial_seed=trial_seed, candidates=candidates)

    def replacement_for(self, queue_index: int) -> ReplacementCandidate:
        if (
            not isinstance(queue_index, int)
            or isinstance(queue_index, bool)
            or queue_index < 0
            or queue_index >= len(self.candidates)
        ):
            raise ReplacementError(f"replacement queue index out of range: {queue_index}")
        return self.candidates[queue_index]

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(candidate.profile_id for candidate in self.candidates)
