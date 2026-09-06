from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..agents import AgentIdentity, PromptProfile, render_prompt
from ..tasks import Task

E02A_TASK_REVISION_PROTOCOL = "e02a-task-calibration-v2"


def neutral_task_prompt(task: Task, participant_index: int) -> tuple[AgentIdentity, PromptProfile, str]:
    """Render task calibration through the same profile/task prompt path used by trials."""
    if participant_index < 0:
        raise ValueError("participant_index must be non-negative")
    profile_id = f"e02a-neutral-profile-{participant_index:03d}"
    agent = AgentIdentity(
        agent_id=f"e02a-neutral-agent-{participant_index:03d}",
        profile_id=profile_id,
        display_label=f"E02A Neutral Participant {participant_index}",
        generation=0,
    )
    profile = PromptProfile(
        profile_id=profile_id,
        parameters={},
        template_version=E02A_TASK_REVISION_PROTOCOL,
    )
    return agent, profile, render_prompt(profile, task)


def evaluate_task_revision_readiness(
    *,
    final_task_count: int,
    holdout_summary: Mapping[str, Any],
    task_provider_requests: int,
    task_provider_failures: int,
    e01_hash_unchanged: bool,
    prior_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen E02A gates without adding post-hoc hard thresholds."""
    failures: list[str] = []

    def require(condition: bool, reason: str) -> None:
        if not condition:
            failures.append(reason)

    require(e01_hash_unchanged, "E01 database hash changed")
    require(final_task_count >= 16, "fewer than 16 holdout-mixed tasks")
    require(
        float(holdout_summary["mixed_score_round_rate"]) >= 0.75,
        "holdout mixed-score rate below 75%",
    )
    require(
        float(holdout_summary["degenerate_round_rate"]) < 0.25,
        "holdout degenerate-round rate is not below 25%",
    )

    position = prior_report["position_diagnostic"]
    require(
        float(position["strict_validity_rate"]) >= 0.95,
        "position ballot validity below 95%",
    )
    require(
        float(position["max_display_share"]) <= 0.35,
        "maximum display-position share exceeds 35%",
    )
    require(
        float(position["max_label_share"]) <= 0.35,
        "maximum anonymous-label share exceeds 35%",
    )

    peer = prior_report["peer_diagnostic"]
    require(
        peer["recommendation"] != "PEER MECHANISM STILL UNRESOLVED",
        "peer mechanism remains unresolved",
    )

    task_failure_rate = (
        task_provider_failures / task_provider_requests if task_provider_requests else 1.0
    )
    require(task_failure_rate <= 0.05, "task provider failure/timeout rate exceeds 5%")

    repeatability = prior_report["repeatability"]
    require(
        int(repeatability["response"]["prompts"]) >= 20
        and int(repeatability["ballot"]["prompts"]) >= 20,
        "repeatability probe has fewer than 20 prompts per type",
    )

    return {
        "decision": (
            "E02A READY — PROCEED TO PROFILE MANIPULATION CHECK"
            if not failures
            else "REVISE E02A"
        ),
        "failures": failures,
        "task_provider_failure_rate": task_failure_rate,
        "preferred_task_gate_passed": final_task_count >= 20
        and float(holdout_summary["mixed_score_round_rate"]) >= 0.85,
        "position_ratio_retained_as_diagnostic_only": True,
        "position_ratio_value": position.get("max_to_min_label_count_ratio"),
    }
