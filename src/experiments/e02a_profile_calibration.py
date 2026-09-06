from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..agents import AgentIdentity, PromptProfile, render_prompt
from ..tasks import Task
from .e01 import e01_profiles


E02A_PROFILE_TASK_PROTOCOL = "e02a-task-profile-calibration-v3"


def calibration_profiles() -> tuple[PromptProfile, ...]:
    """Return the frozen eight E01 profiles in stable profile-id order."""
    profiles = e01_profiles()
    return tuple(profiles[key] for key in sorted(profiles))


def profile_manifest(profiles: Sequence[PromptProfile]) -> list[dict[str, Any]]:
    return [
        {
            "profile_id": profile.profile_id,
            "parameters": dict(profile.parameters),
            "template_version": profile.template_version,
        }
        for profile in profiles
    ]


def profile_task_prompt(
    task: Task,
    profile: PromptProfile,
    *,
    phase: str,
    repeat_index: int,
) -> tuple[AgentIdentity, str]:
    """Render a task exactly through an experimental profile prompt."""
    if repeat_index < 0:
        raise ValueError("repeat_index must be non-negative")
    if not phase.strip():
        raise ValueError("phase must be non-empty")
    agent = AgentIdentity(
        agent_id=f"e02a-{phase}-{profile.profile_id}-r{repeat_index:02d}",
        profile_id=profile.profile_id,
        display_label=f"E02A {phase} participant",
        generation=0,
    )
    return agent, render_prompt(profile, task)
