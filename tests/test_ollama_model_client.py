import httpx
import pytest

from app.services.ollama_model_client import OllamaModelClient


def test_ollama_model_client_generates_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_data = {
        "response": "Paris",
    }

    def mock_post(
        *args,
        **kwargs,
    ) -> httpx.Response:
        request = httpx.Request(
            method="POST",
            url="http://ollama:11434/api/generate",
        )

        return httpx.Response(
            status_code=200,
            json=response_data,
            request=request,
        )

    monkeypatch.setattr(
        httpx,
        "post",
        mock_post,
    )

    client = OllamaModelClient(
        base_url="http://ollama:11434",
    )

    outputs = client.generate(
        model_name="llama3",
        inputs=[
            "What is the capital of France?",
        ],
    )

    assert outputs == ["Paris"]


@pytest.mark.integration
def test_ollama_model_client_connects_to_real_ollama() -> None:
    client = OllamaModelClient(
        base_url="http://ollama:11434",
    )

    outputs = client.generate(
        model_name="llama3",
        inputs=[
            "Reply with exactly one word: Paris",
        ],
    )

    assert len(outputs) == 1
    assert outputs[0]