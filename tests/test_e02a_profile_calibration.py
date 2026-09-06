from pathlib import Path

from src.experiments.e02a_profile_calibration import (
    E02A_PROFILE_TASK_PROTOCOL,
    calibration_profiles,
    profile_manifest,
    profile_task_prompt,
)
from src.tasks.calibration import load_task_set

ROOT = Path(__file__).resolve().parents[1]


def test_calibration_uses_exactly_the_frozen_e01_profile_set() -> None:
    profiles = calibration_profiles()

    assert len(profiles) == 8
    assert [profile.profile_id for profile in profiles] == [
        f"e01-profile-{index:03d}" for index in range(1, 9)
    ]
    assert [profile.parameters["approach"] for profile in profiles] == [
        "literal",
        "constraint-first",
        "pattern-first",
        "calculation-first",
        "independent-check",
        "skeptical",
        "deliberate",
        "concise",
    ]


def test_profile_task_prompt_preserves_experimental_profile_parameters() -> None:
    task = load_task_set(ROOT / "tasks" / "e02a_candidates_v2.json").tasks[0]
    profile = calibration_profiles()[3]

    agent, prompt = profile_task_prompt(
        task,
        profile,
        phase="calibration",
        repeat_index=1,
    )

    assert agent.profile_id == profile.profile_id
    assert profile.profile_id in prompt
    assert profile.template_version in prompt
    assert '"approach":"calculation-first"' in prompt
    assert task.prompt in prompt
    assert "OUTPUT CONTRACT:" in prompt


def test_profile_manifest_is_complete_and_protocol_is_versioned() -> None:
    profiles = calibration_profiles()
    manifest = profile_manifest(profiles)

    assert E02A_PROFILE_TASK_PROTOCOL == "e02a-task-profile-calibration-v3"
    assert len(manifest) == 8
    assert {row["profile_id"] for row in manifest} == {
        f"e01-profile-{index:03d}" for index in range(1, 9)
    }
    assert all(row["parameters"].get("approach") for row in manifest)
    assert all(row["template_version"] == "e01-profile-prompt-v1" for row in manifest)
