from difflib import SequenceMatcher

from app.models import MetricResult

from .base import Evaluator


class SimilarityEvaluator(Evaluator):

    def evaluate(
        self,
        expected_output: str,
        generated_output: str,
    ) -> MetricResult:

        score = SequenceMatcher(
            None,
            expected_output.strip(),
            generated_output.strip(),
        ).ratio()

        return MetricResult(
            metric_name="similarity",
            score=score,
            passed=score >= 0.8,
        )

