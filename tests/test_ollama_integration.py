from __future__ import annotations

import os

import pytest

from src.agents import AgentIdentity, PromptProfile, generate_agent_response
from src.ballots import LLMBallotProvider
from src.domain import Response
from src.models import OllamaProvider
from src.population import Population
from src.tasks import Task
from src.tournament import RoundContext, RoundEngine


pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(
        os.environ.get("LLMSS_RUN_OLLAMA_TESTS") != "1",
        reason="set LLMSS_RUN_OLLAMA_TESTS=1 to run local Ollama tests",
    ),
]


def test_local_qwen_generates_typed_response() -> None:
    provider = OllamaProvider(model="qwen3:0.6b", num_predict=32)
    version = provider.check_health()
    provider.ensure_model_available()
    task = Task(
        "m2-smoke-task",
        "arithmetic",
        "Answer directly. Return only the final answer. Do not include explanation. "
        "What is 2 + 2?",
        "4",
        "exact-match-v1",
    )
    agent = AgentIdentity("m2-agent-001", "m2-profile-001", "Participant 1", 0)
    profile = PromptProfile("m2-profile-001", {}, "m2-smoke-v1")

    response = generate_agent_response(
        response_id="m2-response-001",
        trial_id="m2-integration",
        round_index=0,
        agent=agent,
        profile=profile,
        task=task,
        provider=provider,
        request_id="m2-request-001",
        seed=42,
    )

    assert version
    assert response.content.strip()
    assert response.provider_name == "ollama"
    assert response.model_name == "qwen3:0.6b"


def live_population() -> tuple[Population, dict[str, PromptProfile]]:
    agents = tuple(
        AgentIdentity(
            f"live-agent-{index:03d}",
            f"live-profile-{index:03d}",
            f"Participant {index}",
            0,
        )
        for index in range(1, 9)
    )
    profiles = {
        agent.profile_id: PromptProfile(agent.profile_id, {}, "m2-ballot-smoke-v1")
        for agent in agents
    }
    return Population(agents), profiles


def ballot_task() -> Task:
    return Task(
        "m2-ballot-task",
        "arithmetic",
        "Answer directly. Return only the final answer to: 2 + 2",
        "4",
        "exact-match-v1",
    )


def test_local_qwen_generates_one_anonymous_ballot() -> None:
    provider = OllamaProvider(model="qwen3:0.6b", num_predict=32)
    provider.check_health()
    provider.ensure_model_available()
    population, _ = live_population()
    task = ballot_task()
    responses = tuple(
        Response(
            response_id=f"live-response-{index:03d}",
            trial_id="live-ballot-trial",
            round_index=0,
            task_id=task.task_id,
            agent_id=agent.agent_id,
            content=content,
            provider_name="ollama",
            model_name="qwen3:0.6b",
        )
        for index, (agent, content) in enumerate(
            zip(
                population.agents,
                ("4", "5", "2 + 2 = 4", "3", "four", "22", "0", "The answer is 4."),
                strict=True,
            ),
            start=1,
        )
    )

    generated = LLMBallotProvider().generate_ballot(
        trial_id="live-ballot-trial",
        round_index=0,
        trial_seed=42,
        task=task,
        voter=population.agents[0],
        population=population,
        responses=responses,
        model_provider=provider,
    )

    assert generated.evidence is not None
    assert generated.evidence.raw_output.strip()
    assert len(generated.evidence.candidate_order) == 7
    assert population.agents[0].agent_id not in {
        candidate.agent_id for candidate in generated.evidence.candidate_order
    }


def test_local_qwen_completes_real_response_and_ballot_round() -> None:
    provider = OllamaProvider(model="qwen3:0.6b", num_predict=32)
    provider.check_health()
    provider.ensure_model_available()
    population, profiles = live_population()
    task = ballot_task()
    context = RoundContext(
        experiment_id="live-ballot-experiment",
        trial_id="live-ballot-round",
        round_index=0,
        condition="peer_vote",
        seed=42,
        task=task,
        population=population,
        profiles=profiles,
    )

    result = RoundEngine(ballot_provider=LLMBallotProvider()).execute(
        context, provider
    )

    assert len(result.responses) == 8
    assert len(result.ballots) == 8
    assert len(result.ballot_evidence) == 8
    assert all(evidence.raw_output.strip() for evidence in result.ballot_evidence)
    assert result.selection.mechanism == "peer_vote"
