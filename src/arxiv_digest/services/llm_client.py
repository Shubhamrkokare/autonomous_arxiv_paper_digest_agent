"""A small local-Ollama boundary for structured, testable LLM generation."""

import json
from typing import Any, Protocol

import httpx


class LLMClient(Protocol):
    """Interface used by briefing and QA services; tests can provide a fake."""

    def generate_json(
        self, prompt: str, *, schema: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...


class LLMUnavailableError(RuntimeError):
    """Raised when the configured local Ollama service cannot be used."""


class OllamaClient:
    """Generate one JSON object through Ollama's local HTTP API."""

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    def generate_json(
        self, prompt: str, *, schema: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            request_body: dict[str, Any] = {
                "model": self._model,
                "prompt": prompt,
                # Ollama accepts a JSON Schema here. This constrains the model's
                # shape at generation time instead of trusting a prompt alone.
                "format": schema or "json",
                "stream": False,
                "options": {"temperature": 0.1},
            }
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json=request_body,
                timeout=120.0,
            )
            response.raise_for_status()
            return json.loads(response.json()["response"])
        except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise LLMUnavailableError(
                "Local Ollama generation failed. Install Ollama, start its service, "
                f"and pull the configured model '{self._model}'. Details: {error}"
            ) from error
