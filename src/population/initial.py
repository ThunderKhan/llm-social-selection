from random import Random
from typing import Mapping

from ..agents import AgentIdentity, PromptProfile
from ..seeding import derive_seed
from .models import Population, PopulationError


INITIAL_POPULATION_VERSION = "initial-population-v1"


def build_initial_population(
    *,
    trial_id: str,
    trial_seed: int,
    profiles: Mapping[str, PromptProfile],
    agent_id_namespace: str | None = None,
) -> Population:
    profile_ids = sorted(profiles)
    if len(profile_ids) < 8:
        raise PopulationError(
            f"initial profile pool must contain at least 8 profiles, got {len(profile_ids)}"
        )
    for profile_id, profile in profiles.items():
        if profile_id != profile.profile_id:
            raise PopulationError(f"profile mapping key mismatch: {profile_id}")
    Random(
        derive_seed(
            trial_seed,
            0,
            "initial_population",
            INITIAL_POPULATION_VERSION,
        )
    ).shuffle(profile_ids)
    identity_namespace = agent_id_namespace or trial_id
    return Population(
        tuple(
            AgentIdentity(
                agent_id=f"{identity_namespace}-agent-{index:03d}",
                profile_id=profile_id,
                display_label=f"Participant {index}",
                generation=0,
            )
            for index, profile_id in enumerate(profile_ids[:8], start=1)
        )
    )
