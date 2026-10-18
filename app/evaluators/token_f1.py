from collections import Counter

from app.models import MetricResult

from .base import Evaluator
from .normalized_match import normalize_text

PASS_THRESHOLD = 0.5


def _token_overlap(
    expected_output: str,
    generated_output: str,
) -> tuple[int, int, int]:
    """Return (common, expected, generated) normalized token counts."""

    expected_tokens = normalize_text(expected_output).split()
    generated_tokens = normalize_text(generated_output).split()

    common = sum((Counter(expected_tokens) & Counter(generated_tokens)).values())

    return common, len(expected_tokens), len(generated_tokens)


def token_f1(expected_output: str, generated_output: str) -> float:
    """Return the SQuAD-style F1 of the normalized word overlap."""

    common, expected_count, generated_count = _token_overlap(
        expected_output,
        generated_output,
    )

    if not expected_count or not generated_count:
        # Both empty after normalization counts as agreement.
        return 1.0 if expected_count == generated_count else 0.0

    if common == 0:
        return 0.0

    precision = common / generated_count
    recall = common / expected_count

    return 2 * precision * recall / (precision + recall)


def token_recall(expected_output: str, generated_output: str) -> float:
    """Return the share of normalized expected tokens found in the answer."""

    common, expected_count, generated_count = _token_overlap(
        expected_output,
        generated_output,
    )

    if not expected_count:
        # Nothing to recall: agree only if the answer is empty too.
        return 1.0 if not generated_count else 0.0

    return common / expected_count


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


class TokenRecallEvaluator(Evaluator):
    """Scores how much of the expected answer appears in the generated one.

    Extra words do not lower the score, so full-sentence answers are not
    penalized, but wrong answers that mention the expected words score
    high. Passes at PASS_THRESHOLD or above.
    """

    def evaluate(
        self,
        expected_output: str,
        generated_output: str,
        input_text: str | None = None,
    ) -> MetricResult:
        score = token_recall(expected_output, generated_output)

        return MetricResult(
            metric_name="token_recall",
            score=score,
            passed=score >= PASS_THRESHOLD,
        )
