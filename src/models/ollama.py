from __future__ import annotations

import json
import socket
from collections.abc import Mapping
from math import isfinite
from typing import TYPE_CHECKING, Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import ModelOutput, ModelProvider, _require_non_empty
from .errors import (
    ModelProviderError,
    OllamaConnectionError,
    OllamaModelError,
    OllamaResponseError,
    OllamaTimeoutError,
)

if TYPE_CHECKING:
    from ..agents import AgentIdentity
    from ..tasks import Task


DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_NUM_PREDICT = 32
MAX_OLLAMA_SEED = 2**31 - 1


class OllamaTransport(Protocol):
    def send(self, request: Request, timeout: float) -> bytes:
        """Send one HTTP request and return its complete response body."""


class UrllibOllamaTransport:
    def send(self, request: Request, timeout: float) -> bytes:
        with urlopen(request, timeout=timeout) as response:
            return response.read()


class OllamaProvider(ModelProvider):
    def __init__(
        self,
        *,
        model: str = "qwen3:0.6b",
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        temperature: float = 0.0,
        num_predict: int = DEFAULT_NUM_PREDICT,
        transport: OllamaTransport | None = None,
    ) -> None:
        _require_non_empty(model, "model")
        _require_non_empty(base_url, "base_url")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError("temperature must be a non-negative finite number")
        if (
            not isinstance(num_predict, int)
            or isinstance(num_predict, bool)
            or num_predict <= 0
        ):
            raise ValueError("num_predict must be a positive integer")

        self._model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.temperature = float(temperature)
        self.num_predict = num_predict
        self._transport = transport or UrllibOllamaTransport()

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

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
        del agent, task
        _require_non_empty(prompt, "prompt")
        _require_non_empty(request_id, "request_id")
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool)
        ):
            raise ValueError("seed must be an integer or None")
        effective_seed = seed % MAX_OLLAMA_SEED if seed is not None else None
        options: dict[str, int | float] = {
            "temperature": self.temperature,
            "num_predict": self.num_predict,
        }
        if effective_seed is not None:
            options["seed"] = effective_seed
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": options,
        }
        if response_schema is not None:
            payload["format"] = dict(response_schema)
        data = self._request_json(
            "/api/generate",
            method="POST",
            payload=payload,
        )
        content = data.get("response")
        if not isinstance(content, str):
            raise OllamaResponseError("Ollama response is missing the final 'response' field")
        if not content.strip():
            raise OllamaResponseError("Ollama returned an empty final response")

        finish_reason = data.get("done_reason")
        if not isinstance(finish_reason, str) or not finish_reason.strip():
            finish_reason = None
        total_duration = data.get("total_duration")
        latency_ms = (
            total_duration / 1_000_000
            if isinstance(total_duration, (int, float))
            and not isinstance(total_duration, bool)
            and total_duration >= 0
            else None
        )
        eval_count = data.get("eval_count")
        token_count = (
            eval_count
            if isinstance(eval_count, int)
            and not isinstance(eval_count, bool)
            and eval_count >= 0
            else None
        )
        return ModelOutput(
            content=content,
            provider_name=self.provider_name,
            model_name=self.model_name,
            request_id=request_id,
            seed=effective_seed,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            token_count=token_count,
        )

    def check_health(self) -> str:
        data = self._request_json("/api/version", method="GET")
        version = data.get("version")
        if not isinstance(version, str) or not version.strip():
            raise OllamaResponseError("Ollama health response is missing 'version'")
        return version

    def is_available(self) -> bool:
        try:
            self.check_health()
        except ModelProviderError:
            return False
        return True

    def available_models(self) -> tuple[str, ...]:
        data = self._request_json("/api/tags", method="GET")
        models = data.get("models")
        if not isinstance(models, list):
            raise OllamaResponseError("Ollama model response is missing 'models'")
        names: set[str] = set()
        for model in models:
            if not isinstance(model, dict):
                continue
            for field in ("name", "model"):
                value = model.get(field)
                if isinstance(value, str) and value.strip():
                    names.add(value)
        return tuple(sorted(names))

    def has_model(self) -> bool:
        return self.model_name in self.available_models()

    def ensure_model_available(self) -> None:
        if not self.has_model():
            raise OllamaModelError(
                f"Ollama model {self.model_name!r} is not available locally.\n"
                f"Run:\nollama pull {self.model_name}"
            )

    def _request_json(
        self,
        path: str,
        *,
        method: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            raw = self._transport.send(request, self.timeout_seconds)
        except HTTPError as error:
            detail = self._http_error_detail(error)
            if error.code == 404 and "model" in detail.casefold():
                raise OllamaModelError(
                    f"Ollama model {self.model_name!r} is not available locally: {detail}"
                ) from error
            raise OllamaResponseError(
                f"Ollama HTTP {error.code} for {path}: {detail}"
            ) from error
        except (TimeoutError, socket.timeout) as error:
            raise OllamaTimeoutError(
                f"Ollama request to {path} timed out after {self.timeout_seconds:g} seconds"
            ) from error
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise OllamaTimeoutError(
                    f"Ollama request to {path} timed out after {self.timeout_seconds:g} seconds"
                ) from error
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.base_url}: {error.reason}"
            ) from error
        except OSError as error:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.base_url}: {error}"
            ) from error

        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OllamaResponseError("Ollama returned invalid JSON") from error
        if not isinstance(parsed, dict):
            raise OllamaResponseError("Ollama returned a non-object JSON response")
        message = parsed.get("error")
        if isinstance(message, str) and message.strip():
            if "model" in message.casefold() and (
                "not found" in message.casefold()
                or "not available" in message.casefold()
            ):
                raise OllamaModelError(
                    f"Ollama model {self.model_name!r} is not available locally: {message}"
                )
            raise OllamaResponseError(f"Ollama returned an error: {message[:500]}")
        return parsed

    @staticmethod
    def _http_error_detail(error: HTTPError) -> str:
        try:
            raw = error.read(2048)
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
                return parsed["error"][:500]
            return raw.decode("utf-8", errors="replace")[:500] or error.reason
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return str(error.reason)
