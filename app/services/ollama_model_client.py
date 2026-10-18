from typing import Any

import httpx

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
    ) -> list[str]:
        """Generate one output for each input."""

        outputs: list[str] = []

        for input_text in inputs:
            payload: dict[str, Any] = {
                "model": model_name,
                "prompt": input_text,
                "stream": False,
            }

            if self._options:
                payload["options"] = self._options

            response = httpx.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=60.0,
            )

            response.raise_for_status()

            data = response.json()
            outputs.append(data["response"].strip())

        return outputs
