import re
import unicodedata

from app.models import MetricResult

from .base import Evaluator

# Punctuation, except "." or "," between digits so "4.0" and "1,000" survive.
_PUNCTUATION = re.compile(r"(?<!\d)[.,]|[.,](?!\d)|[^\w\s.,]")

_ARTICLES = re.compile(r"\b(?:a|an|the)\b")


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation and English articles, collapse spaces."""

    text = unicodedata.normalize("NFKC", text).casefold()
    text = _PUNCTUATION.sub(" ", text)
    text = _ARTICLES.sub(" ", text)

    return " ".join(text.split())


class NormalizedMatchEvaluator(Evaluator):
    """Matches answers after normalization, or when the answer contains them.

    Passes when the normalized outputs are equal or the normalized expected
    output appears as whole words in the generated output, so
    "The capital is Paris." matches "Paris" but "Parisian" does not.
    """

    def evaluate(
        self,
        expected_output: str,
        generated_output: str,
        input_text: str | None = None,
    ) -> MetricResult:
        expected = normalize_text(expected_output)
        generated = normalize_text(generated_output)

        if expected:
            match = expected == generated or f" {expected} " in f" {generated} "
        else:
            # Nothing is left after normalization, e.g. an expected output of
            # only punctuation or "The"; fall back to a case-insensitive match.
            match = expected_output.strip().casefold() == (
                generated_output.strip().casefold()
            )

        return MetricResult(
            metric_name="normalized_match",
            score=1.0 if match else 0.0,
            passed=match,
        )
