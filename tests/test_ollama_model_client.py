import httpx
import pytest

from app.config import OLLAMA_BASE_URL
from app.models import GenerationOptions
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
        base_url=OLLAMA_BASE_URL,
    )

    outputs = client.generate(
        model_name="llama3",
        inputs=[
            "Reply with exactly one word: Paris",
        ],
    )

    assert len(outputs) == 1
    assert outputs[0]


def _capture_post(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict]:
    payloads: list[dict] = []

    def mock_post(
        *args,
        **kwargs,
    ) -> httpx.Response:
        payloads.append(kwargs["json"])

        return httpx.Response(
            status_code=200,
            json={"response": "Paris"},
            request=httpx.Request(
                method="POST",
                url="http://ollama:11434/api/generate",
            ),
        )

    monkeypatch.setattr(
        httpx,
        "post",
        mock_post,
    )

    return payloads


def test_ollama_model_client_sends_options_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _capture_post(monkeypatch)

    client = OllamaModelClient(
        base_url="http://ollama:11434",
        options={"temperature": 0.0},
    )

    client.generate(
        model_name="llama3",
        inputs=["question"],
    )

    assert payloads[0]["options"] == {"temperature": 0.0}


def test_ollama_model_client_omits_options_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _capture_post(monkeypatch)

    client = OllamaModelClient(
        base_url="http://ollama:11434",
    )

    client.generate(
        model_name="llama3",
        inputs=["question"],
    )

    assert "options" not in payloads[0]


def test_ollama_model_client_sends_call_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _capture_post(monkeypatch)

    client = OllamaModelClient(
        base_url="http://ollama:11434",
    )

    client.generate(
        model_name="llama3",
        inputs=["question"],
        options=GenerationOptions(temperature=0.7, seed=42),
    )

    assert payloads[0]["options"] == {"temperature": 0.7, "seed": 42}


def test_ollama_model_client_call_options_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _capture_post(monkeypatch)

    client = OllamaModelClient(
        base_url="http://ollama:11434",
        options={"temperature": 0.0, "num_ctx": 4096},
    )

    client.generate(
        model_name="llama3",
        inputs=["question"],
        options=GenerationOptions(temperature=0.5),
    )

    assert payloads[0]["options"] == {"temperature": 0.5, "num_ctx": 4096}


def test_ollama_model_client_omits_unset_call_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _capture_post(monkeypatch)

    client = OllamaModelClient(
        base_url="http://ollama:11434",
    )

    client.generate(
        model_name="llama3",
        inputs=["question"],
        options=GenerationOptions(),
    )

    assert "options" not in payloads[0]
