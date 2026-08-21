from __future__ import annotations

import random

import pytest

from src.agents import AgentIdentity, PromptProfile, generate_agent_response, render_prompt
from src.domain import Response
from src.models import MockModelProvider, ModelOutput, ModelProvider
from src.tasks import Task


@pytest.fixture
def agent() -> AgentIdentity:
    return AgentIdentity("agent-003", "profile-003", "Participant 3", 0)


@pytest.fixture
def profile() -> PromptProfile:
    return PromptProfile("profile-003", {"setting_b": 0.5, "setting_a": 2}, "v1")


@pytest.fixture
def task() -> Task:
    return Task("task-logic-002", "logic", "Choose A or B.", "A", "exact-v1")


def test_mock_provider_implements_provider_interface() -> None:
    provider = MockModelProvider()

    assert isinstance(provider, ModelProvider)
    assert provider.provider_name == "mock"
    assert provider.model_name == "deterministic-v1"


def test_mock_output_is_deterministic(
    agent: AgentIdentity, task: Task
) -> None:
    provider = MockModelProvider()
    arguments = {
        "agent": agent,
        "task": task,
        "prompt": "Effective prompt",
        "request_id": "request-008",
        "seed": 42,
    }

    first = provider.generate(**arguments)
    second = provider.generate(**arguments)

    assert first == second
    assert isinstance(first, ModelOutput)
    assert first.content.startswith("MOCK_RESPONSE:")


def test_mock_output_is_unchanged_by_optional_response_schema(
    agent: AgentIdentity, task: Task
) -> None:
    provider = MockModelProvider()
    arguments = {
        "agent": agent,
        "task": task,
        "prompt": "Effective prompt",
        "request_id": "request-008",
        "seed": 42,
    }

    plain = provider.generate(**arguments)
    structured = provider.generate(
        **arguments,
        response_schema={"type": "object", "additionalProperties": False},
    )

    assert structured == plain


def test_different_agent_changes_mock_output(task: Task) -> None:
    provider = MockModelProvider()
    first_agent = AgentIdentity("agent-001", "profile-001", "Participant 1", 0)
    second_agent = AgentIdentity("agent-002", "profile-001", "Participant 2", 0)

    first = provider.generate(
        agent=first_agent,
        task=task,
        prompt="Effective prompt",
        request_id="request-008",
    )
    second = provider.generate(
        agent=second_agent,
        task=task,
        prompt="Effective prompt",
        request_id="request-008",
    )

    assert first.content != second.content


def test_different_task_changes_mock_output(agent: AgentIdentity) -> None:
    provider = MockModelProvider()
    first_task = Task("task-001", "logic", "Question one", "A", "exact-v1")
    second_task = Task("task-002", "logic", "Question two", "B", "exact-v1")

    first = provider.generate(
        agent=agent,
        task=first_task,
        prompt="Effective prompt",
        request_id="request-008",
    )
    second = provider.generate(
        agent=agent,
        task=second_task,
        prompt="Effective prompt",
        request_id="request-008",
    )

    assert first.content != second.content


def test_mock_provider_does_not_depend_on_global_random_state(
    agent: AgentIdentity, task: Task
) -> None:
    provider = MockModelProvider()
    arguments = {
        "agent": agent,
        "task": task,
        "prompt": "Effective prompt",
        "request_id": "request-008",
        "seed": 42,
    }

    random.seed(1)
    first = provider.generate(**arguments)
    random.seed(999)
    second = provider.generate(**arguments)

    assert first == second


def test_prompt_rendering_is_stable_across_parameter_order(
    task: Task,
) -> None:
    first = PromptProfile("profile-003", {"setting_a": 2, "setting_b": 0.5}, "v1")
    second = PromptProfile("profile-003", {"setting_b": 0.5, "setting_a": 2}, "v1")

    assert render_prompt(first, task) == render_prompt(second, task)


def test_prompt_rendering_enforces_concise_output_contract(
    profile: PromptProfile, task: Task
) -> None:
    prompt = render_prompt(profile, task)

    assert "Do not repeat the task" in prompt
    assert "Do not include explanations, reasoning, Markdown, units" in prompt
    assert "labels such as 'Answer:'" in prompt
    assert "unless the task explicitly requests them" in prompt


def test_agent_response_flow_preserves_references_and_metadata(
    agent: AgentIdentity, profile: PromptProfile, task: Task
) -> None:
    provider = MockModelProvider()
    arguments = {
        "response_id": "response-t001-r03-a03",
        "trial_id": "trial-001",
        "round_index": 3,
        "agent": agent,
        "profile": profile,
        "task": task,
        "provider": provider,
        "request_id": "request-008",
        "seed": 42,
    }

    first = generate_agent_response(**arguments)
    second = generate_agent_response(**arguments)

    assert isinstance(first, Response)
    assert first == second
    assert first.agent_id == agent.agent_id
    assert first.task_id == task.task_id
    assert first.provider_name == "mock"
    assert first.model_name == "deterministic-v1"
    assert first.request_id == "request-008"
    assert first.seed == 42
    assert first.latency_ms is None
    assert first.token_count is None


def test_agent_response_flow_rejects_mismatched_profile(
    agent: AgentIdentity, task: Task
) -> None:
    with pytest.raises(ValueError, match="agent.profile_id must match"):
        generate_agent_response(
            response_id="response-001",
            trial_id="trial-001",
            round_index=0,
            agent=agent,
            profile=PromptProfile("profile-999", {}, "v1"),
            task=task,
            provider=MockModelProvider(),
            request_id="request-001",
        )
