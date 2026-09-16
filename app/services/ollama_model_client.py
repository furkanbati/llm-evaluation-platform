import httpx

from app.services.model_client import ModelClient


class OllamaModelClient(ModelClient):
    """Model client that generates outputs through Ollama."""

    def __init__(
        self,
        base_url: str,
    ) -> None:
        self._base_url = base_url.rstrip("/")

    def generate(
        self,
        model_name: str,
        inputs: list[str],
    ) -> list[str]:
        """Generate one output for each input."""

        outputs: list[str] = []

        for input_text in inputs:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": input_text,
                    "stream": False,
                },
                timeout=60.0,
            )

            response.raise_for_status()

            data = response.json()
            outputs.append(data["response"].strip())

        return outputs