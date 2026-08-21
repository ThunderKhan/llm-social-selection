from __future__ import annotations

import os

import pytest

from src.agents import AgentIdentity, PromptProfile, generate_agent_response
from src.models import OllamaProvider
from src.tasks import Task


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
