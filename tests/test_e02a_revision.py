from __future__ import annotations

from src.experiments.e02a_revision import (
    E02A_TASK_REVISION_PROTOCOL,
    evaluate_task_revision_readiness,
    neutral_task_prompt,
)
from src.tasks import Task


def test_neutral_task_prompt_uses_experiment_renderer() -> None:
    task = Task(
        task_id="arith-999",
        family="arithmetic",
        prompt="Compute 2 + 3.\n\nOUTPUT REQUIREMENT:\nReturn only the integer.\nDo not explain.\nDo not repeat the question.\nDo not use Markdown.\nDo not prepend \"Answer:\".",
        expected_answer="5",
        scorer_version="exact-match-v1",
    )

    agent, profile, prompt = neutral_task_prompt(task, 7)

    assert agent.profile_id == profile.profile_id
    assert profile.parameters == {}
    assert profile.template_version == E02A_TASK_REVISION_PROTOCOL
    assert "SYSTEM ROLE:" in prompt
    assert "TASK:" in prompt
    assert task.prompt in prompt
    assert "OUTPUT CONTRACT:" in prompt


def test_readiness_uses_frozen_max_share_gates_not_posthoc_label_ratio() -> None:
    prior_report = {
        "position_diagnostic": {
            "strict_validity_rate": 1.0,
            "max_display_share": 0.265,
            "max_label_share": 0.184,
            "max_to_min_label_count_ratio": float("inf"),
        },
        "peer_diagnostic": {
            "recommendation": "ADOPT REVISED PEER BALLOT FOR E03"
        },
        "repeatability": {
            "response": {"prompts": 20},
            "ballot": {"prompts": 20},
        },
    }

    result = evaluate_task_revision_readiness(
        final_task_count=16,
        holdout_summary={
            "mixed_score_round_rate": 0.75,
            "degenerate_round_rate": 0.20,
        },
        task_provider_requests=100,
        task_provider_failures=0,
        e01_hash_unchanged=True,
        prior_report=prior_report,
    )

    assert result["decision"] == "E02A READY — PROCEED TO PROFILE MANIPULATION CHECK"
    assert result["failures"] == []
    assert result["position_ratio_retained_as_diagnostic_only"] is True


def test_readiness_still_rejects_actual_position_gate_failure() -> None:
    prior_report = {
        "position_diagnostic": {
            "strict_validity_rate": 1.0,
            "max_display_share": 0.36,
            "max_label_share": 0.20,
            "max_to_min_label_count_ratio": 1.2,
        },
        "peer_diagnostic": {
            "recommendation": "ADOPT REVISED PEER BALLOT FOR E03"
        },
        "repeatability": {
            "response": {"prompts": 20},
            "ballot": {"prompts": 20},
        },
    }

    result = evaluate_task_revision_readiness(
        final_task_count=16,
        holdout_summary={
            "mixed_score_round_rate": 0.80,
            "degenerate_round_rate": 0.10,
        },
        task_provider_requests=100,
        task_provider_failures=0,
        e01_hash_unchanged=True,
        prior_report=prior_report,
    )

    assert result["decision"] == "REVISE E02A"
    assert "maximum display-position share exceeds 35%" in result["failures"]
