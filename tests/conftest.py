from __future__ import annotations

import pytest

from src.agents import AgentIdentity, PromptProfile
from src.population import Population
from src.tasks import Task


@pytest.fixture
def eight_agents() -> tuple[AgentIdentity, ...]:
    return tuple(
        AgentIdentity(
            agent_id=f"agent-{index:03d}",
            profile_id=f"profile-{index:03d}",
            display_label=f"Participant {index}",
            generation=0,
        )
        for index in range(1, 9)
    )


@pytest.fixture
def population(eight_agents: tuple[AgentIdentity, ...]) -> Population:
    return Population(eight_agents)


@pytest.fixture
def profiles(eight_agents: tuple[AgentIdentity, ...]) -> dict[str, PromptProfile]:
    return {
        agent.profile_id: PromptProfile(
            profile_id=agent.profile_id,
            parameters={"setting": index},
            template_version="v1",
        )
        for index, agent in enumerate(eight_agents, start=1)
    }


@pytest.fixture
def round_task() -> Task:
    return Task(
        task_id="task-logic-001",
        family="logic",
        prompt="Choose the correct answer: A or B.",
        expected_answer="A",
        scorer_version="exact-match-v1",
    )
