import httpx
import pytest

from app.config import OLLAMA_BASE_URL, SEMANTIC_SIMILARITY_MODEL
from app.errors import ModelNotFoundError, ModelProviderError
from app.evaluators.semantic_similarity import (
    SemanticSimilarityEvaluator,
    cosine_similarity,
)
from app.services.embedding_client import EmbeddingClient
from app.services.ollama_embedding_client import OllamaEmbeddingClient


class FakeEmbeddingClient(EmbeddingClient):
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[tuple[str, list[str]]] = []

    def embed(
        self,
        model_name: str,
        texts: list[str],
    ) -> list[list[float]]:
        self.calls.append((model_name, texts))
        return [self._vectors[text] for text in texts]


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ([1.0, 2.0], [1.0, 2.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
        ([3.0, 4.0], [6.0, 8.0], 1.0),
        ([0.0, 0.0], [1.0, 1.0], 0.0),
    ],
)
def test_cosine_similarity(
    first: list[float],
    second: list[float],
    expected: float,
) -> None:
    assert cosine_similarity(first, second) == pytest.approx(expected)


def test_cosine_similarity_rejects_different_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        cosine_similarity([1.0], [1.0, 2.0])


def make_evaluator(
    vectors: dict[str, list[float]],
    threshold: float = 0.8,
) -> tuple[SemanticSimilarityEvaluator, FakeEmbeddingClient]:
    client = FakeEmbeddingClient(vectors)

    evaluator = SemanticSimilarityEvaluator(
        embedding_client=client,
        model_name="embed-model",
        threshold=threshold,
    )

    return evaluator, client


def test_semantic_similarity_embeds_both_outputs_in_one_call() -> None:
    evaluator, client = make_evaluator(
        {"Paris": [1.0, 0.0], "The capital is Paris.": [1.0, 0.0]},
    )

    result = evaluator.evaluate(
        expected_output="Paris",
        generated_output="The capital is Paris.",
    )

    assert client.calls == [("embed-model", ["Paris", "The capital is Paris."])]
    assert result.metric_name == "semantic_similarity"
    assert result.score == pytest.approx(1.0)
    assert result.passed is True


def test_semantic_similarity_applies_threshold() -> None:
    # cos(45 degrees) is about 0.707
    vectors = {"a": [1.0, 0.0], "b": [1.0, 1.0]}

    passing, _ = make_evaluator(vectors, threshold=0.7)
    failing, _ = make_evaluator(vectors, threshold=0.75)

    assert passing.evaluate("a", "b").passed is True
    assert failing.evaluate("a", "b").passed is False


def test_semantic_similarity_clips_negative_scores() -> None:
    evaluator, _ = make_evaluator({"a": [1.0, 0.0], "b": [-1.0, 0.0]})

    result = evaluator.evaluate("a", "b")

    assert result.score == 0.0
    assert result.passed is False


def _mock_post(
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response,
) -> list[dict]:
    payloads: list[dict] = []

    def mock_post(*args, **kwargs) -> httpx.Response:
        payloads.append(kwargs["json"])
        return response

    monkeypatch.setattr(httpx, "post", mock_post)

    return payloads


REQUEST = httpx.Request(method="POST", url="http://ollama:11434/api/embed")


def test_ollama_embedding_client_sends_texts_and_reads_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _mock_post(
        monkeypatch,
        httpx.Response(
            status_code=200,
            json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
            request=REQUEST,
        ),
    )

    vectors = OllamaEmbeddingClient("http://ollama:11434").embed(
        model_name="nomic-embed-text",
        texts=["a", "b"],
    )

    assert payloads == [{"model": "nomic-embed-text", "input": ["a", "b"]}]
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.parametrize(
    "body",
    [
        {"embeddings": [[0.1, 0.2]]},
        {"embeddings": "not a list"},
        {"something": "else"},
    ],
)
def test_ollama_embedding_client_rejects_unexpected_responses(
    monkeypatch: pytest.MonkeyPatch,
    body: dict,
) -> None:
    _mock_post(
        monkeypatch,
        httpx.Response(status_code=200, json=body, request=REQUEST),
    )

    with pytest.raises(ModelProviderError):
        OllamaEmbeddingClient("http://ollama:11434").embed(
            model_name="nomic-embed-text",
            texts=["a", "b"],
        )


def test_ollama_embedding_client_reports_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_post(
        monkeypatch,
        httpx.Response(status_code=404, json={"error": "x"}, request=REQUEST),
    )

    with pytest.raises(ModelNotFoundError) as exc_info:
        OllamaEmbeddingClient("http://ollama:11434").embed(
            model_name="missing-embed",
            texts=["a"],
        )

    assert "missing-embed" in exc_info.value.public_message


@pytest.mark.integration
def test_ollama_embedding_client_connects_to_real_ollama() -> None:
    vectors = OllamaEmbeddingClient(OLLAMA_BASE_URL).embed(
        model_name=SEMANTIC_SIMILARITY_MODEL,
        texts=["Paris", "The capital of France is Paris."],
    )

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 0
