from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Any

from .base import ModelOutput, ModelProvider, _require_non_empty

if TYPE_CHECKING:
    from ..agents.models import AgentIdentity
    from ..tasks import Task


class MockModelProvider(ModelProvider):
    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "deterministic-v1"

    def generate(
        self,
        *,
        agent: AgentIdentity,
        task: Task,
        prompt: str,
        request_id: str,
        seed: int | None = None,
        response_schema: Mapping[str, Any] | None = None,
    ) -> ModelOutput:
        del response_schema
        _require_non_empty(prompt, "prompt")
        _require_non_empty(request_id, "request_id")
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool)
        ):
            raise ValueError("seed must be an integer or None")

        payload = json.dumps(
            {
                "agent_id": agent.agent_id,
                "profile_id": agent.profile_id,
                "task_id": task.task_id,
                "prompt": prompt,
                "request_id": request_id,
                "seed": seed,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return ModelOutput(
            content=f"MOCK_RESPONSE:{digest}",
            provider_name=self.provider_name,
            model_name=self.model_name,
            request_id=request_id,
            seed=seed,
        )
