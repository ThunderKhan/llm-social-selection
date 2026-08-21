from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from src.agents import AgentIdentity
from src.models import (
    OllamaConnectionError,
    OllamaModelError,
    OllamaProvider,
    OllamaResponseError,
    OllamaTimeoutError,
)
from src.tasks import Task


@dataclass
class FakeTransport:
    results: list[bytes | Exception]
    requests: list[Request] = field(default_factory=list)
    timeouts: list[float] = field(default_factory=list)

    def send(self, request: Request, timeout: float) -> bytes:
        self.requests.append(request)
        self.timeouts.append(timeout)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def agent() -> AgentIdentity:
    return AgentIdentity("agent-001", "profile-001", "Participant 1", 0)


@pytest.fixture
def task() -> Task:
    return Task("task-001", "arithmetic", "Return 2 + 2.", "4", "exact-match-v1")


def encoded(value: object) -> bytes:
    return json.dumps(value).encode("utf-8")


def generate(
    provider: OllamaProvider, agent: AgentIdentity, task: Task, *, seed: int = 42
):
    return provider.generate(
        agent=agent,
        task=task,
        prompt="Return only the final answer to 2 + 2.",
        request_id="request-001",
        seed=seed,
    )


def test_provider_metadata_and_configuration() -> None:
    provider = OllamaProvider(
        model="custom:latest",
        base_url="http://localhost:9999/",
        timeout_seconds=17,
        temperature=0.25,
        num_predict=12,
        transport=FakeTransport([]),
    )

    assert provider.provider_name == "ollama"
    assert provider.model_name == "custom:latest"
    assert provider.base_url == "http://localhost:9999"
    assert provider.timeout_seconds == 17.0
    assert provider.temperature == 0.25
    assert provider.num_predict == 12


def test_generate_builds_expected_request_and_parses_metadata(
    agent: AgentIdentity, task: Task
) -> None:
    transport = FakeTransport(
        [
            encoded(
                {
                    "response": "4",
                    "thinking": "ignored separate reasoning",
                    "done_reason": "stop",
                    "total_duration": 2_500_000,
                    "eval_count": 3,
                }
            )
        ]
    )
    provider = OllamaProvider(
        transport=transport,
        timeout_seconds=33,
        temperature=0,
        num_predict=16,
    )

    output = generate(provider, agent, task, seed=2**63 + 5)

    request = transport.requests[0]
    body = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://127.0.0.1:11434/api/generate"
    assert request.method == "POST"
    assert transport.timeouts == [33.0]
    assert body == {
        "model": "qwen3:0.6b",
        "options": {
            "num_predict": 16,
            "seed": (2**63 + 5) % (2**31 - 1),
            "temperature": 0.0,
        },
        "prompt": "Return only the final answer to 2 + 2.",
        "stream": False,
        "think": False,
    }
    assert output.content == "4"
    assert output.provider_name == "ollama"
    assert output.model_name == "qwen3:0.6b"
    assert output.finish_reason == "stop"
    assert output.latency_ms == 2.5
    assert output.token_count == 3
    assert output.seed == (2**63 + 5) % (2**31 - 1)


@pytest.mark.parametrize("body", [b"not-json", b"[]"])
def test_invalid_json_or_shape_is_rejected(
    body: bytes, agent: AgentIdentity, task: Task
) -> None:
    provider = OllamaProvider(transport=FakeTransport([body]))

    with pytest.raises(OllamaResponseError):
        generate(provider, agent, task)


def test_missing_or_empty_final_response_is_rejected(
    agent: AgentIdentity, task: Task
) -> None:
    missing = OllamaProvider(
        transport=FakeTransport([encoded({"thinking": "not primary output"})])
    )
    empty = OllamaProvider(transport=FakeTransport([encoded({"response": "  "})]))

    with pytest.raises(OllamaResponseError, match="missing"):
        generate(missing, agent, task)
    with pytest.raises(OllamaResponseError, match="empty"):
        generate(empty, agent, task)


def test_http_error_is_typed(agent: AgentIdentity, task: Task) -> None:
    error = HTTPError(
        "http://localhost/api/generate",
        500,
        "server error",
        None,
        BytesIO(encoded({"error": "generation failed"})),
    )
    provider = OllamaProvider(transport=FakeTransport([error]))

    with pytest.raises(OllamaResponseError, match="HTTP 500"):
        generate(provider, agent, task)


def test_missing_model_http_error_is_typed(agent: AgentIdentity, task: Task) -> None:
    error = HTTPError(
        "http://localhost/api/generate",
        404,
        "not found",
        None,
        BytesIO(encoded({"error": "model 'missing' not found"})),
    )
    provider = OllamaProvider(model="missing", transport=FakeTransport([error]))

    with pytest.raises(OllamaModelError, match="missing"):
        generate(provider, agent, task)


def test_connection_error_is_typed(agent: AgentIdentity, task: Task) -> None:
    provider = OllamaProvider(
        transport=FakeTransport([URLError(ConnectionRefusedError("refused"))])
    )

    with pytest.raises(OllamaConnectionError, match="Could not reach Ollama"):
        generate(provider, agent, task)


@pytest.mark.parametrize("error", [TimeoutError(), socket.timeout()])
def test_timeout_is_typed(
    error: Exception, agent: AgentIdentity, task: Task
) -> None:
    provider = OllamaProvider(
        timeout_seconds=5,
        transport=FakeTransport([error]),
    )

    with pytest.raises(OllamaTimeoutError, match="5 seconds"):
        generate(provider, agent, task)


def test_health_check_and_availability() -> None:
    healthy = OllamaProvider(
        transport=FakeTransport([encoded({"version": "0.11.4"})])
    )
    unavailable = OllamaProvider(
        transport=FakeTransport([URLError(ConnectionRefusedError("refused"))])
    )

    assert healthy.check_health() == "0.11.4"
    assert unavailable.is_available() is False


def test_model_presence_and_missing_model_message() -> None:
    present = OllamaProvider(
        transport=FakeTransport(
            [encoded({"models": [{"name": "qwen3:0.6b"}]})]
        )
    )
    missing = OllamaProvider(
        model="missing:latest",
        transport=FakeTransport(
            [encoded({"models": [{"name": "qwen3:0.6b"}]})]
        ),
    )

    assert present.has_model() is True
    with pytest.raises(OllamaModelError, match="ollama pull missing:latest"):
        missing.ensure_model_available()
