from app.models import MetricResult

from .base import Evaluator


class ExactMatchEvaluator(Evaluator):

    def evaluate(
        self,
        expected_output: str,
        generated_output: str,
    ) -> MetricResult:

        match = (
            expected_output.strip()
            == generated_output.strip()
        )

        return MetricResult(
            metric_name="exact_match",
            score=1.0 if match else 0.0,
            passed=match,
        )