import math

from app.models import MetricResult
from app.services.embedding_client import EmbeddingClient

from .base import Evaluator


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Return the cosine similarity of two vectors, or 0.0 for a zero vector."""

    if len(first) != len(second):
        raise ValueError("embeddings have different dimensions")

    dot = sum(a * b for a, b in zip(first, second, strict=True))
    norm = math.sqrt(sum(a * a for a in first)) * math.sqrt(sum(b * b for b in second))

    if norm == 0:
        return 0.0

    return dot / norm


class SemanticSimilarityEvaluator(Evaluator):
    """Scores the cosine similarity of the answers' embeddings.

    Negative similarities are clipped to 0.0 so the score stays in [0, 1].
    The threshold depends on the embedding model and should be calibrated
    when the model changes.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        model_name: str,
        threshold: float,
    ) -> None:
        self._embedding_client = embedding_client
        self._model_name = model_name
        self._threshold = threshold

    def evaluate(
        self,
        expected_output: str,
        generated_output: str,
        input_text: str | None = None,
    ) -> MetricResult:
        expected, generated = self._embedding_client.embed(
            model_name=self._model_name,
            texts=[expected_output, generated_output],
        )

        score = min(1.0, max(0.0, cosine_similarity(expected, generated)))

        return MetricResult(
            metric_name="semantic_similarity",
            score=score,
            passed=score >= self._threshold,
        )
