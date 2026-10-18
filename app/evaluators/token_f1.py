from collections import Counter

from app.models import MetricResult

from .base import Evaluator
from .normalized_match import normalize_text

PASS_THRESHOLD = 0.5


def token_f1(expected_output: str, generated_output: str) -> float:
    """Return the SQuAD-style F1 of the normalized word overlap."""

    expected_tokens = normalize_text(expected_output).split()
    generated_tokens = normalize_text(generated_output).split()

    if not expected_tokens or not generated_tokens:
        # Both empty after normalization counts as agreement.
        return 1.0 if expected_tokens == generated_tokens else 0.0

    common = sum((Counter(expected_tokens) & Counter(generated_tokens)).values())

    if common == 0:
        return 0.0

    precision = common / len(generated_tokens)
    recall = common / len(expected_tokens)

    return 2 * precision * recall / (precision + recall)


class TokenF1Evaluator(Evaluator):
    """Scores the word overlap between the answers, giving partial credit.

    Long answers lower precision, so "The capital of France is Paris."
    scores about 0.33 against "Paris". Passes at PASS_THRESHOLD or above.
    """

    def evaluate(
        self,
        expected_output: str,
        generated_output: str,
        input_text: str | None = None,
    ) -> MetricResult:
        score = token_f1(expected_output, generated_output)

        return MetricResult(
            metric_name="token_f1",
            score=score,
            passed=score >= PASS_THRESHOLD,
        )
