from typing import Any

import httpx

from app.errors import (
    ModelNotFoundError,
    ModelProviderError,
    ModelTimeoutError,
    ModelUnavailableError,
)
from app.models import GenerationOptions
from app.services.model_client import ModelClient


class OllamaModelClient(ModelClient):
    """Model client that generates outputs through Ollama."""

    def __init__(
        self,
        base_url: str,
        options: dict[str, Any] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._options = options

    def generate(
        self,
        model_name: str,
        inputs: list[str],
        options: GenerationOptions | None = None,
    ) -> list[str]:
        """Generate one output for each input."""

        merged_options = {
            **(self._options or {}),
            **(options.to_provider_options() if options else {}),
        }

        outputs: list[str] = []

        for input_text in inputs:
            payload: dict[str, Any] = {
                "model": model_name,
                "prompt": input_text,
                "stream": False,
            }

            if merged_options:
                payload["options"] = merged_options

            outputs.append(
                self._post_generate(
                    model_name=model_name,
                    payload=payload,
                )
            )

        return outputs

    def _post_generate(
        self,
        model_name: str,
        payload: dict[str, Any],
    ) -> str:
        """Call Ollama and translate failures into evaluation errors."""

        try:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            # Ollama answers 404 when the model has not been pulled.
            if exc.response.status_code == 404:
                raise ModelNotFoundError(model_name, str(exc)) from exc

            raise ModelProviderError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(str(exc)) from exc

        try:
            return str(response.json()["response"]).strip()
        except (ValueError, KeyError, TypeError) as exc:
            raise ModelProviderError(
                f"Unexpected Ollama response: {response.text[:200]}"
            ) from exc
